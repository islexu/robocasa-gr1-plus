from __future__ import annotations
from contextlib import contextmanager
import hashlib
import numpy as np
from .protocol import canonical

VISUAL_FIELDS = ("cam_pos", "cam_quat", "cam_fovy", "geom_rgba", "geom_matid",
                 "light_diffuse", "light_ambient", "light_specular")
PHYSICAL_FIELDS = ("body_mass", "body_inertia", "body_pos", "body_quat", "geom_size",
                   "geom_friction", "geom_contype", "geom_conaffinity", "jnt_range",
                   "jnt_type", "dof_damping", "actuator_gainprm", "actuator_biasprm")

def array_digest(arrays):
    h = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        h.update(name.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def physical_digest(sim):
    m = sim.model
    arrays = {key: getattr(m, key) for key in PHYSICAL_FIELDS}
    arrays["geom_bodyid"] = m.geom_bodyid
    return array_digest(arrays)

def state_snapshot(sim):
    data = sim.data
    arrays = {"state": sim.get_state().flatten().copy(),
              "mocap_pos": data.mocap_pos.copy(), "mocap_quat": data.mocap_quat.copy()}
    return arrays

def background_geoms(model):
    # Only surfaces identified by their fixture prefix, never robot or movable object geoms.
    selected = []
    for i in range(model.ngeom):
        name = model.geom_id2name(i) or ""
        if name.startswith(("wall_", "floor_", "table_main_")):
            selected.append(i)
    if not selected:
        raise ValueError("No named background surfaces found; inspect model geom names")
    return selected

@contextmanager
def visual_shift(sim, spec):
    """Temporarily alter only renderer fields; restore even after a render failure."""
    dim, index, level = spec["dimension"], spec["index"], spec["severity"]
    saved = {key: getattr(sim.model, key).copy() for key in VISUAL_FIELDS}
    try:
        m = sim.model
        if dim == "camera":
            camera = m.camera_name2id("egoview")
            delta = np.zeros(3)
            delta[index // 2] = (1 if index % 2 == 0 else -1) * 0.015 * level
            m.cam_pos[camera] += delta
        elif dim == "background":
            palettes = np.array([[.25,.45,.65], [.65,.35,.2], [.25,.6,.35],
                                 [.6,.3,.6], [.8,.75,.4], [.25,.25,.25]])
            ids = background_geoms(m)
            strength = (.35, .65, 1.)[level - 1]
            m.geom_rgba[ids, :3] = ((1-strength) * m.geom_rgba[ids, :3]
                                      + strength * palettes[index])
            m.geom_matid[ids] = -1
        elif dim == "light":
            tints = np.array([[1,.8,.65], [.65,.8,1], [.75,1,.75],
                              [1,.75,.95], [1,1,1], [1,1,1]])
            factor = 1 + ((.25 * level) if index == 5 else (-.18 * level))
            for key in ("light_diffuse", "light_ambient", "light_specular"):
                getattr(m,key)[:] *= factor * tints[index]
        sim.forward()
        yield
    finally:
        for key, value in saved.items():
            getattr(sim.model, key)[:] = value
        sim.forward()

def corrupt_image(image, spec, seed, step):
    if spec["dimension"] != "noise":
        return np.asarray(image).copy()
    import cv2
    x = np.asarray(image)
    index, level = spec["index"], spec["severity"]
    # Do not touch global NumPy RNG: paired render must not change the next simulator reset.
    entropy = int(hashlib.sha256(canonical([seed, step, spec]).encode()).hexdigest()[:16],16)
    rng = np.random.default_rng(entropy)
    if index == 0:
        return np.clip(x.astype(float) + rng.normal(0, 6*level, x.shape),0,255).astype(np.uint8)
    if index == 1:
        mask = rng.random(x.shape[:2]) < .01 * level
        out = x.copy()
        out[mask] = rng.integers(0, 2, size=(int(mask.sum()),1)) * 255
        return out
    if index == 2:
        return cv2.GaussianBlur(x, (2*level+1, 2*level+1), 0)
    if index == 3:
        kernel = np.zeros((2*level+3, 2*level+3), np.float32)
        kernel[len(kernel)//2] = 1 / len(kernel)
        return cv2.filter2D(x, -1, kernel)
    if index == 4:
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR),
                                 [cv2.IMWRITE_JPEG_QUALITY, (65,40,20)[level-1]])
        if not ok:
            raise RuntimeError("JPEG corruption encoding failed")
        return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    small = cv2.resize(x, (x.shape[1]//(level+1), x.shape[0]//(level+1)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_NEAREST)

def language_shift(language, spec):
    if spec["dimension"] != "language":
        return language
    prefix = "unlocked_waist: "
    if not language.startswith(prefix):
        raise ValueError("Missing GR1 language prefix")
    instruction = language[len(prefix):]
    templates = ("Please {task}", "Your task is to {task}", "Carefully {task}",
                 "Could you {task}", "Follow this instruction: {task}", "Now {task}")
    # Keep every entity, relation, and action in the original instruction verbatim.
    return prefix + templates[spec["index"]].format(task=instruction)

def penetration_map(sim):
    result = {}
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        key = tuple(sorted((int(c.geom1),int(c.geom2))))
        result[key] = min(result.get(key,0.), float(c.dist))
    return result

def physical_initial_shift(suite, spec):
    """Apply an explicit initial-pose intervention. Invalid poses fail, never resample."""
    sim = suite.sim
    before, contacts = sim.get_state(), penetration_map(sim)
    changed = []
    dim, index, level = spec["dimension"], spec["index"], spec["severity"]
    try:
        if dim == "robot_state":
            names = (["robot0_l_elbow_pitch"] if index < 2 else
                     ["robot0_r_elbow_pitch"] if index < 4 else
                     ["robot0_l_elbow_pitch", "robot0_r_elbow_pitch"])
            delta = (1 if index % 2 == 0 else -1) * .03 * level
            for name in names:
                i = sim.model.joint_name2id(name)
                old = np.asarray(sim.data.get_joint_qpos(name))
                new = old + delta
                low, high = sim.model.jnt_range[i]
                if sim.model.jnt_limited[i] and not np.all((new > low) & (new < high)):
                    raise ValueError(f"Initial joint shift outside limits: {name}")
                sim.data.set_joint_qpos(name,new)
                changed.append({"joint":name,"delta":delta})
        elif dim == "layout":
            direction = np.array([(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1)][index],dtype=float)
            direction /= np.linalg.norm(direction)
            delta = np.r_[direction * .01 * level, 0.]
            for name, obj in sorted(suite.objects.items()):
                for joint in obj.joints:
                    qpos = np.asarray(sim.data.get_joint_qpos(joint))
                    if qpos.shape != (7,):
                        continue
                    qpos = qpos.copy(); qpos[:3] += delta
                    sim.data.set_joint_qpos(joint,qpos)
                    changed.append({"object":name,"joint":joint,"delta":delta.tolist()})
        else:
            return {}
        if not changed:
            raise ValueError(f"No coordinates changed for {dim}")
        sim.forward()
        for key, dist in penetration_map(sim).items():
            # Reject new or materially worsened deep penetration; existing support contact is allowed.
            if dist < -.002 and dist < contacts.get(key,0.) - .001:
                raise ValueError(f"Initial shift caused penetration at {key}: {dist}")
        if not np.isfinite(sim.data.qpos).all() or suite._check_success():
            raise ValueError("Initial shifted state is invalid or already successful")
        if dim == "robot_state":
            for robot in suite.robots:
                robot.composite_controller.reset()
        return {"kind":dim, "changed":changed,
                "state_sha256":array_digest(state_snapshot(sim)),
                "validation":"joint limits, finite state, penetration, initial non-success; reachability unverified"}
    except Exception:
        sim.set_state(before); sim.forward()
        raise
