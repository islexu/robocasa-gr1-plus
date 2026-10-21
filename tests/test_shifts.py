from types import SimpleNamespace
import numpy as np
import pytest
from robocasa_gr1_plus.shifts import visual_shift, corrupt_image, language_shift, VISUAL_FIELDS, physical_initial_shift
from robocasa_gr1_plus.protocol import variant

class Model:
    def __init__(self):
        self.ngeom=3
        self.cam_pos=np.zeros((1,3)); self.cam_quat=np.array([[1.,0,0,0]])
        self.cam_fovy=np.array([60.]); self.geom_rgba=np.ones((3,4))
        self.geom_matid=np.zeros(3,dtype=int)
        for x in ('light_diffuse','light_ambient','light_specular'): setattr(self,x,np.ones((1,3)))
    def camera_name2id(self,name): assert name=='egoview'; return 0
    def geom_id2name(self,i): return ['table_main_top','robot0_arm','obj_main_visual'][i]

@pytest.mark.parametrize('dimension',['camera','background','light'])
def test_renderer_only_changes_restore_on_exception_and_do_not_recolor_task_objects(dimension):
    model=Model(); sim=SimpleNamespace(model=model,forward=lambda:None)
    saved={key:getattr(model,key).copy() for key in VISUAL_FIELDS}
    with pytest.raises(RuntimeError):
        with visual_shift(sim,variant(dimension)):
            assert any(not np.array_equal(getattr(model,k),v) for k,v in saved.items())
            if dimension=='background':
                np.testing.assert_array_equal(model.geom_rgba[1:],saved['geom_rgba'][1:])
                np.testing.assert_array_equal(model.geom_matid[1:],saved['geom_matid'][1:])
            raise RuntimeError('renderer failed')
    for k,v in saved.items(): np.testing.assert_array_equal(getattr(model,k),v)

@pytest.mark.parametrize('index',range(6))
def test_noise_replays_exactly_without_consuming_global_rng(index):
    image=np.random.default_rng(1).integers(0,256,size=(64,64,3),dtype=np.uint8)
    spec=variant('noise',index,2)
    np.random.seed(19); expected=np.random.rand()
    np.random.seed(19)
    a=corrupt_image(image,spec,42,5)
    assert np.random.rand()==expected
    np.testing.assert_array_equal(a,corrupt_image(image,spec,42,5))
    assert not np.array_equal(a,image) and a.dtype==np.uint8

def test_language_variants_preserve_every_original_entity_relation_and_action():
    instruction='put the milk into the microwave and close it'
    for i in range(6):
        shifted=language_shift('unlocked_waist: '+instruction,variant('language',i))
        assert shifted.startswith('unlocked_waist: ') and instruction in shifted

def test_physical_shift_rolls_back_when_penetration_is_introduced():
    q={'robot0_l_elbow_pitch':np.array([.5])}
    class Data:
        ncon=0; contact=[]
        @property
        def qpos(self): return np.array([q['robot0_l_elbow_pitch'][0]])
        def get_joint_qpos(self,name): return q[name].copy()
        def set_joint_qpos(self,name,value): q[name]=np.asarray(value).copy()
    class Sim:
        data=Data()
        model=SimpleNamespace(joint_name2id=lambda n:0,jnt_range=np.array([[-1.,1.]]),jnt_limited=np.array([True]))
        def get_state(self): return q['robot0_l_elbow_pitch'].copy()
        def set_state(self,s): q['robot0_l_elbow_pitch']=s.copy()
        def forward(self):
            invalid=q['robot0_l_elbow_pitch'][0]>.51
            self.data.ncon=int(invalid)
            self.data.contact=[SimpleNamespace(geom1=0,geom2=1,dist=-.1)] if invalid else []
    sim=Sim(); suite=SimpleNamespace(sim=sim,_check_success=lambda:False)
    with pytest.raises(ValueError,match='penetration'): physical_initial_shift(suite,variant('robot_state',0))
    np.testing.assert_array_equal(q['robot0_l_elbow_pitch'],[.5])
