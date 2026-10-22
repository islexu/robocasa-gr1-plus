"""Validated initial-state interventions. Invalid cases roll back and remain visible."""
from __future__ import annotations
import numpy as np
from .catalogue_v2 import InvalidPerturbation, local_rng
from .scene_v2 import CLUTTER
from .shifts import penetration_map, array_digest, state_snapshot
from .visual_v2 import workspace, rotation, quat_matrix, matrix_quat

def positive_relations(suite):
    """Keep positive source/support relations from the official task predicates."""
    from robocasa.utils import object_utils as ou
    relations = {}
    for name in ("obj_container", "container"):
        if name in suite.objects:
            relations["receptacle:" + name] = bool(ou.check_obj_in_receptacle(suite, "obj", name))
    relations["counter_contact"] = bool(ou.check_obj_fixture_contact(suite, "obj", suite.counter))
    return relations

def _joint(sim, name, expected_type):
    i = sim.model.joint_name2id(name)
    if i < 0 or int(sim.model.jnt_type[i]) != expected_type:
        raise InvalidPerturbation(f"missing_or_wrong_joint_type:{name}")
    return i

def _zero_velocity(sim, joint_id, size):
    start = sim.model.jnt_dofadr[joint_id]
    sim.data.qvel[start:start+size] = 0

def physical_initial_shift_v2(suite, spec, relation_reader=None):
    sim, p = suite.sim, spec["parameters"]
    before, contacts = sim.get_state(), penetration_map(sim)
    changed, relations = [], {}
    relation_reader = relation_reader or positive_relations
    sub = spec["subtype"]
    try:
        if spec["dimension"] == "robot_state":
            for name, delta in zip(p["joints"], p["delta"]):
                i = _joint(sim, name, 3)  # mjJNT_HINGE
                old = np.asarray(sim.data.get_joint_qpos(name)).copy()
                new = old + delta
                low, high = sim.model.jnt_range[i]
                if sim.model.jnt_limited[i] and not np.all((new > low) & (new < high)):
                    raise InvalidPerturbation(f"joint_limit:{name}")
                sim.data.set_joint_qpos(name, new)
                _zero_velocity(sim, i, 1)
                changed.append({"joint": name, "delta_rad": delta})
        elif sub == "target_pose":
            obj = suite.objects.get(p["object_name"])
            if obj is None or len(obj.joints) != 1:
                raise InvalidPerturbation("target_requires_one_free_joint")
            joint = obj.joints[0]
            i = _joint(sim, joint, 0)  # mjJNT_FREE
            relations = relation_reader(suite)
            # A missing initial support relation cannot establish preservation.
            if not any(relations.values()):
                raise InvalidPerturbation("no_initial_support_or_receptacle_relation")
            q = np.asarray(sim.data.get_joint_qpos(joint)).copy()
            q[:2] += p["xy_delta_m"]
            q[3:] = matrix_quat(rotation([0, 0, p["yaw_deg"]]) @ quat_matrix(q[3:]))
            sim.data.set_joint_qpos(joint, q)
            _zero_velocity(sim, i, 6)
            changed.append({"object": p["object_name"], "joint": joint,
                            "xy_delta_m": p["xy_delta_m"], "yaw_deg": p["yaw_deg"]})
        elif sub == "clutter":
            center, points = workspace(suite)
            r = rotation([0, 0, np.rad2deg(suite.counter.rot)])
            local = (points - center) @ r
            low, high = local.min(axis=0), local.max(axis=0)
            margin = p["half_size_m"] + .035
            if np.any(high[:2] - low[:2] < 2 * margin):
                raise InvalidPerturbation("counter_too_small_for_clutter")
            grid = np.array([[x, y, high[2]] for x in np.linspace(low[0]+margin, high[0]-margin, 7)
                             for y in np.linspace(low[1]+margin, high[1]-margin, 7)])
            rng = local_rng(p["placement_seed"], "clutter_positions")
            grid = grid[rng.permutation(len(grid))] @ r.T + center
            occupied = []
            for obj in suite.objects.values():
                for joint in obj.joints:
                    q = np.asarray(sim.data.get_joint_qpos(joint))
                    if q.shape == (7,):
                        occupied.append(q[:2].copy())
            for n in range(p["count"]):
                # A fixed geometric clearance rule, independent of policy results.
                candidates = [v for v in grid if all(np.linalg.norm(v[:2] - o) >= p["target_clearance_m"] for o in occupied)]
                if not candidates:
                    raise InvalidPerturbation("no_clutter_position_with_clearance")
                pos = candidates[0].copy()
                pos[2] += p["half_size_m"] + .001
                occupied.append(pos[:2])
                joint = CLUTTER + str(n) + "_joint"
                i = _joint(sim, joint, 0)
                sim.data.set_joint_qpos(joint, np.r_[pos, 1., 0., 0., 0.])
                _zero_velocity(sim, i, 6)
                changed.append({"joint": joint, "position_m": pos.tolist(), "shape": p["shape"]})
        else:
            return {}
        sim.forward()
        for key, dist in penetration_map(sim).items():
            if dist < -.002 and dist < contacts.get(key, 0.) - .001:
                raise InvalidPerturbation(f"new_penetration:{key}:{dist:.6f}")
        if relations:
            after = relation_reader(suite)
            lost = [key for key, value in relations.items() if value and not after.get(key, False)]
            if lost:
                raise InvalidPerturbation(f"lost_initial_relation:{lost}")
        if not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all():
            raise InvalidPerturbation("nonfinite_initial_state")
        if suite._check_success():
            raise InvalidPerturbation("initial_state_already_successful")
        if spec["dimension"] == "robot_state":
            for robot in suite.robots:
                robot.composite_controller.reset()
        return {"kind": spec["dimension"], "subtype": sub, "changed": changed,
                "positive_relations": relations, "state_sha256": array_digest(state_snapshot(sim)),
                "validation": "limits, finite state, penetration, positive relations, initial non-success; reachability unverified"}
    except Exception:
        sim.set_state(before)
        sim.forward()
        raise
