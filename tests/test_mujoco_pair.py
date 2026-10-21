"""Real MuJoCo renderer fixture. This is not a GR1 task or a policy score."""
from types import SimpleNamespace
import numpy as np
import pytest

mujoco=pytest.importorskip('mujoco')
from robocasa_gr1_plus.environment import GR1PlusEnvironment
from robocasa_gr1_plus.embodiment import IMAGE_KEY, LANGUAGE_KEY, PARTS
from robocasa_gr1_plus.shifts import state_snapshot, array_digest, physical_initial_shift
from robocasa_gr1_plus.protocol import variant

XML='''<mujoco>
<visual><global offwidth="256" offheight="256"/></visual>
<worldbody>
<light pos="0 -1 3" diffuse=".7 .7 .7" ambient=".2 .2 .2"/>
<camera name="egoview" pos="0 -1.6 1.4" xyaxes="1 0 0 0 .65 .75"/>
<geom name="table_main_top" type="box" size=".8 .6 .05" pos="0 0 0" rgba=".6 .45 .3 1"/>
<body name="object" pos="0 0 .15"><freejoint name="object_joint"/>
<geom name="obj_main_visual" type="box" size=".08 .08 .08" rgba=".8 .1 .1 1"/></body>
<body pos="-.4 0 .2"><joint name="robot0_l_elbow_pitch" type="hinge" range="-90 90"/>
<geom type="capsule" size=".03 .15"/></body>
<body pos=".4 0 .2"><joint name="robot0_r_elbow_pitch" type="hinge" range="-90 90"/>
<geom type="capsule" size=".03 .15"/></body>
</worldbody></mujoco>'''

class ModelBridge:
    def __init__(self,model): self.raw=model
    def __getattr__(self,name): return getattr(self.raw,name)
    def camera_name2id(self,name): return mujoco.mj_name2id(self.raw,mujoco.mjtObj.mjOBJ_CAMERA,name)
    def joint_name2id(self,name): return mujoco.mj_name2id(self.raw,mujoco.mjtObj.mjOBJ_JOINT,name)
    def geom_id2name(self,i): return mujoco.mj_id2name(self.raw,mujoco.mjtObj.mjOBJ_GEOM,i)

class SimBridge:
    def __init__(self):
        self.raw=mujoco.MjModel.from_xml_string(XML); self.data=mujoco.MjData(self.raw)
        self.model=ModelBridge(self.raw); self.forward()
    def forward(self): mujoco.mj_forward(self.raw,self.data)
    def get_state(self): return SimpleNamespace(flatten=lambda:np.r_[self.data.time,self.data.qpos,self.data.qvel,self.data.act])

class PhysicalSimBridge(SimBridge):
    def __init__(self):
        super().__init__()
        self.native_data=self.data
        owner=self
        class DataBridge:
            def __getattr__(self,key): return getattr(owner.native_data,key)
            def get_joint_qpos(self,name):
                i=owner.model.joint_name2id(name)
                start=owner.raw.jnt_qposadr[i]
                size=7 if owner.raw.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE else 1
                return owner.native_data.qpos[start:start+size].copy()
            def set_joint_qpos(self,name,value):
                i=owner.model.joint_name2id(name)
                start=owner.raw.jnt_qposadr[i]
                value=np.atleast_1d(value)
                owner.native_data.qpos[start:start+len(value)]=value
        self.data=DataBridge()
    def forward(self): mujoco.mj_forward(self.raw,getattr(self,'native_data',self.data))
    def get_state(self):
        state=np.r_[self.data.time,self.data.qpos,self.data.qvel,self.data.act].copy()
        return SimpleNamespace(flatten=lambda:state.copy())
    def set_state(self,state):
        values=state.flatten(); self.native_data.time=values[0]
        self.native_data.qpos[:]=values[1:1+self.raw.nq]
        self.native_data.qvel[:]=values[1+self.raw.nq:1+self.raw.nq+self.raw.nv]

@pytest.mark.parametrize('dimension',['layout','robot_state'])
def test_real_mujoco_initial_pose_shifts_change_only_intended_joint_coordinates(dimension):
    sim=PhysicalSimBridge(); before=sim.data.qpos.copy()
    resets=[]
    suite=SimpleNamespace(sim=sim,_check_success=lambda:False,
                          objects={'block':SimpleNamespace(joints=['object_joint'])},
                          robots=[SimpleNamespace(composite_controller=SimpleNamespace(reset=lambda:resets.append(1)))])
    proof=physical_initial_shift(suite,variant(dimension,0,1))
    assert proof['changed'] and not np.array_equal(before,sim.data.qpos)
    if dimension=='layout':
        assert sim.data.qpos[0]-before[0]==pytest.approx(.01)
        np.testing.assert_array_equal(sim.data.qpos[1:],before[1:])
        assert not resets
    else:
        np.testing.assert_array_equal(sim.data.qpos[:7],before[:7])
        assert sim.data.get_joint_qpos('robot0_l_elbow_pitch')[0]==pytest.approx(.03)
        assert resets==[1]

@pytest.mark.parametrize('dimension',['camera','background','light','noise'])
def test_real_render_visual_pairs_hold_state_fixed_and_do_not_expose_test_oracles(dimension):
    sim=SimBridge()
    renderer=mujoco.Renderer(sim.raw,height=256,width=256)
    wrapper=object.__new__(GR1PlusEnvironment)
    wrapper.sim=sim; wrapper.steps=0; wrapper.spec=variant(dimension,0,3)
    wrapper.episode={'pairing':'gr1_visual_same_state','base_task_id':'fixture',
                     'variant_name':dimension,'seed':42}
    def render_native():
        renderer.update_scene(sim.data,camera='egoview')
        raw={IMAGE_KEY:renderer.render().copy(),LANGUAGE_KEY:'unlocked_waist: move the block'}
        raw.update({'state.'+name:np.zeros(n) for name,n in PARTS})
        return raw
    wrapper._render_native=render_native
    before=array_digest(state_snapshot(sim))
    try:
        wrapper._latest=wrapper._render(True)
        clean,shifted=wrapper.pair()
        assert not np.array_equal(clean['primary'],shifted['primary'])
        assert before==array_digest(state_snapshot(sim))
        assert wrapper.pair_metadata['clean_rollout_executed'] is False
        wrapper.episode.pop('pairing')
        with pytest.raises(ValueError,match='calibration-only'): wrapper.pair()
    finally:
        renderer.close()
