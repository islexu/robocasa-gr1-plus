"""Real MuJoCo fixtures, not official GR1 assets or measured benchmark scores."""
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import pytest
mujoco = pytest.importorskip("mujoco")
from test_mujoco_pair import XML, SimBridge, PhysicalSimBridge
from robocasa_gr1_plus.catalogue_v2 import SUBTYPES, ARM_JOINTS, make_variant, InvalidPerturbation
from robocasa_gr1_plus.scene_v2 import inject_scene_xml, CLUTTER
from robocasa_gr1_plus.environment import GR1PlusEnvironment
from robocasa_gr1_plus.embodiment import IMAGE_KEY, LANGUAGE_KEY, PARTS
from robocasa_gr1_plus.shifts import VISUAL_FIELDS, array_digest, state_snapshot, physical_digest
from robocasa_gr1_plus.visual_v2 import visual_shift_v2, corrupt_image_v2, rotation, quat_matrix, matrix_quat
from robocasa_gr1_plus.physical_v2 import physical_initial_shift_v2

def scene(spec, tmp_path):
    tree = ET.fromstring(XML)
    ET.SubElement(tree.find("worldbody"),"geom",name="wall_test",type="box",
                  size="2 .03 1",pos="0 .8 .8",rgba=".4 .6 .5 1")
    return inject_scene_xml(ET.tostring(tree,encoding="unicode"),spec,tmp_path)

VISUAL_CASES = [(d,s) for d in ("camera","background","light","noise") for s in SUBTYPES[d]]

@pytest.mark.parametrize("dimension,subtype", VISUAL_CASES)
def test_v2_every_visual_subtype_renders_and_restores_same_state(dimension,subtype,tmp_path):
    spec = make_variant(dimension,subtype,severity=1 if subtype=="shadow" else 3)
    sim = SimBridge(scene(spec,tmp_path))
    renderer = mujoco.Renderer(sim.raw,height=256,width=256)
    wrapper = object.__new__(GR1PlusEnvironment)
    wrapper.v2=True; wrapper.sim=sim; wrapper.spec=spec; wrapper.steps=0
    wrapper.camera_pivot=np.array([0.,0.,.05]); wrapper.provenance={}
    wrapper.episode={"pairing":"gr1_visual_same_state","base_task_id":"fixture","variant_name":subtype,"seed":42}
    def native():
        renderer.update_scene(sim.data,camera="egoview")
        raw={IMAGE_KEY:renderer.render().copy(),LANGUAGE_KEY:"unlocked_waist: pick the block from the board and place it in the basket"}
        raw.update({"state."+name:np.zeros(n) for name,n in PARTS})
        return raw
    wrapper._render_native=native
    before=array_digest(state_snapshot(sim)); physics=physical_digest(sim)
    arrays={k:getattr(sim.model,k).copy() for k in VISUAL_FIELDS+("light_dir","light_castshadow")}
    try:
        wrapper._latest=wrapper._render(True)
        clean,shifted=wrapper.pair()
        assert not np.array_equal(clean["primary"],shifted["primary"])
        assert array_digest(state_snapshot(sim))==before and physical_digest(sim)==physics
        for k,v in arrays.items(): np.testing.assert_array_equal(getattr(sim.model,k),v)
        wrapper.episode.pop("pairing")
        with pytest.raises(ValueError,match="calibration-only"): wrapper.pair()
    finally:
        renderer.close()

@pytest.mark.parametrize("subtype",SUBTYPES["noise"])
def test_corruption_has_local_rng_and_replays_across_policy_methods(subtype):
    image=np.random.default_rng(9).integers(0,256,(64,64,3),dtype=np.uint8)
    spec=make_variant("noise",subtype,severity=5)
    np.random.seed(19); expected=np.random.random(); np.random.seed(19)
    a=corrupt_image_v2(image,spec,8)
    assert np.random.random()==expected
    np.testing.assert_array_equal(a,corrupt_image_v2(image,spec,8))
    assert a.dtype==np.uint8 and a.shape==image.shape and not np.array_equal(a,image)

def test_camera_distance_uses_world_coordinates_of_a_rotated_parent_and_rolls_back():
    tree=ET.fromstring(XML); world=tree.find("worldbody"); camera=world.find("camera")
    world.remove(camera)
    body=ET.SubElement(world,"body",name="camera_parent",pos=".2 .3 .4",quat=".92387953 0 0 .38268343")
    body.append(camera)
    sim=SimBridge(ET.tostring(tree,encoding="unicode"))
    before=sim.data.cam_xpos[0].copy(); orient=sim.data.cam_xmat[0].copy()
    pivot=np.array([0.,0.,.05]); spec=make_variant("camera","distance",severity=4)
    with pytest.raises(RuntimeError):
        with visual_shift_v2(sim,spec,pivot):
            np.testing.assert_allclose(sim.data.cam_xpos[0],pivot+1.35*(before-pivot),atol=1e-8)
            np.testing.assert_allclose(sim.data.cam_xmat[0],orient,atol=1e-8)
            raise RuntimeError("render failed")
    np.testing.assert_allclose(sim.data.cam_xpos[0],before,atol=1e-8)
    for angle in (0., 90., 179.999, 180.):
        r=rotation([angle,0,0]); np.testing.assert_allclose(quat_matrix(matrix_quat(r)),r,atol=1e-8)

def suite_for(sim):
    points=np.array([[x,y,z] for x in (-.8,.8) for y in (-.6,.6) for z in (-.05,.05)])
    return SimpleNamespace(sim=sim,_check_success=lambda:False,robots=[],
                           objects={"obj":SimpleNamespace(joints=["object_joint"])},
                           counter=SimpleNamespace(rot=0.,get_bbox_points=lambda:points))

def test_target_shift_preserves_other_objects_and_rejects_lost_relations_with_exact_rollback():
    sim=PhysicalSimBridge(); suite=suite_for(sim)
    spec=make_variant("layout","target_pose",severity=2)
    before=sim.data.qpos.copy()
    proof=physical_initial_shift_v2(suite,spec,lambda s:{"source_container":True})
    np.testing.assert_allclose(sim.data.qpos[:2]-before[:2],spec["parameters"]["xy_delta_m"])
    np.testing.assert_array_equal(sim.data.qpos[7:],before[7:])
    assert proof["positive_relations"]=={"source_container":True}
    sim=PhysicalSimBridge(); suite=suite_for(sim); sim.data.qvel[:]=.2
    before=array_digest(state_snapshot(sim))
    with pytest.raises(InvalidPerturbation,match="lost_initial_relation"):
        physical_initial_shift_v2(suite,spec,lambda s:{"source_container":np.linalg.norm(s.sim.data.qpos[:2]) < .005})
    assert array_digest(state_snapshot(sim))==before

def test_clutter_is_a_real_free_body_with_mass_collision_and_seeded_counter_placement(tmp_path):
    spec=make_variant("layout","clutter",severity=3)
    xml=scene(spec,tmp_path); sim=PhysicalSimBridge(xml); suite=suite_for(sim)
    before=sim.data.qpos[:7].copy()
    proof=physical_initial_shift_v2(suite,spec)
    assert len(proof["changed"])==2 and sim.raw.nq==23
    for n in range(2):
        body=mujoco.mj_name2id(sim.raw,mujoco.mjtObj.mjOBJ_BODY,CLUTTER+str(n))
        geom=mujoco.mj_name2id(sim.raw,mujoco.mjtObj.mjOBJ_GEOM,CLUTTER+str(n)+"_geom")
        assert sim.raw.body_mass[body]>0 and sim.raw.geom_contype[geom]==1
        assert sim.raw.geom_group[geom]==1  # native GR1 hides collision-only group 0
        assert sim.data.xpos[body,2]==pytest.approx(.073)
    np.testing.assert_array_equal(sim.data.qpos[:7],before)
    again=PhysicalSimBridge(xml)
    assert physical_initial_shift_v2(suite_for(again),spec)["changed"]==proof["changed"]

def test_joint_vector_applies_all_14_coordinates_and_rejects_joint_limit_with_rollback():
    tree=ET.fromstring(XML); world=tree.find("worldbody")
    for body in list(world.findall("body")):
        if body.find("joint") is not None: world.remove(body)
    for n,name in enumerate(ARM_JOINTS):
        b=ET.SubElement(world,"body",pos=f"{2+n*.1} 0 .4")
        ET.SubElement(b,"joint",name=name,type="hinge",range="-90 90")
        ET.SubElement(b,"geom",type="sphere",size=".02")
    sim=PhysicalSimBridge(ET.tostring(tree,encoding="unicode")); suite=suite_for(sim)
    spec=make_variant("robot_state","joint_vector",severity=5)
    before=sim.data.qpos.copy()
    proof=physical_initial_shift_v2(suite,spec)
    np.testing.assert_allclose(sim.data.qpos[7:]-before[7:],spec["parameters"]["delta"])
    assert len(proof["changed"])==14
    sim.model.jnt_range[sim.model.joint_name2id(ARM_JOINTS[0])]=[-.001,.001]
    before=array_digest(state_snapshot(sim))
    with pytest.raises(InvalidPerturbation,match="joint_limit"):
        physical_initial_shift_v2(suite,spec)
    assert array_digest(state_snapshot(sim))==before
