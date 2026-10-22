"""Real MuJoCo behind a small Gym-contract fixture; official GR1 assets are absent."""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import numpy as np
import pytest
mujoco=pytest.importorskip("mujoco")
gym=pytest.importorskip("gymnasium")
from test_mujoco_pair import XML,SimBridge
from test_mujoco_v2 import suite_for
from robocasa_gr1_plus.protocol import UPSTREAM_COMMIT,canonical
from robocasa_gr1_plus.protocol_v3 import make_plan_v3
from robocasa_gr1_plus.environment import GR1PlusEnvironment
from robocasa_gr1_plus.embodiment import IMAGE_KEY,LANGUAGE_KEY,PARTS

@pytest.mark.parametrize("split,dimension",[("calibration","background"),("confirmation","composition")])
def test_constructor_hard_reset_render_step_and_pair_contract(split,dimension,tmp_path,monkeypatch):
    if not os.getenv("GR1_PLUS_ASSET_CACHE"):pytest.skip("Prepare photo pack")
    make_plan_v3(tmp_path/"plan")
    row=next(json.loads(s) for s in (tmp_path/"plan"/(split+".jsonl")).read_text().splitlines()
             if json.loads(s)["dimension"]==dimension)
    source=tmp_path/"upstream";source.mkdir()
    for name in ("robocasa","robocasa.utils","robocasa.utils.gym_utils","robosuite","robosuite.environments","robosuite.environments.base"):
        module=ModuleType(name);module.__path__=[]
        if name=="robocasa":module.__file__=str(source/"robocasa/__init__.py")
        monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setattr("robocasa_gr1_plus.environment.source_provenance",lambda path:{"commit":UPSTREAM_COMMIT,"python_sources_sha256":"fixture"})
    class Task:
        viewer_get_obs=False;control_freq=20
        def edit_model_xml(self,xml):return xml
        def set_xml_processor(self,processor):self._xml_processors.append(processor)
        def __init__(self):
            self._xml_processors=[self.edit_model_xml];self.renderer=None
            self.counter=suite_for(SimBridge()).counter
            self.hard_reset()
        def hard_reset(self):
            if self.renderer:self.renderer.close()
            xml=XML.replace("</worldbody>",'<geom name="wall_fixture" type="box" pos="0 .8 .8" size="2 .03 1"/></worldbody>')
            for processor in self._xml_processors:xml=processor(xml)
            self.sim=SimBridge(xml);self.renderer=mujoco.Renderer(self.sim.raw,height=256,width=256)
        def _get_observations(self,force_update=True):
            self.renderer.update_scene(self.sim.data,camera="egoview")
            raw={IMAGE_KEY:self.renderer.render().copy(),LANGUAGE_KEY:"unlocked_waist: pick the block and place it in the basket"}
            raw.update({"state."+name:np.zeros(n) for name,n in PARTS})
            return raw
        def _check_success(self):return False
    original=Task.edit_model_xml
    sys.modules["robosuite.environments.base"].REGISTERED_ENVS={row["base_task_id"]:Task}
    class NativeGym:
        def __init__(self):self.env=Task();self.unwrapped=self
        def get_basic_observation(self,raw):return raw
        def get_groot_observation(self,raw):return raw
        def reset(self,seed=None):self.env.hard_reset();return self.env._get_observations(),{}
        def step(self,action):
            assert set(action)=={"action."+name for name,n in PARTS}
            mujoco.mj_step(self.env.sim.raw,self.env.sim.data)
            return self.env._get_observations(),0.,False,False,{"success":False}
        def close(self):self.env.renderer.close()
    monkeypatch.setattr(gym,"make",lambda *args,**kwargs:NativeGym())
    env=GR1PlusEnvironment({"methods":["frozen"],"environment":{"expected_package_root":str(source),"max_steps":720}},row)
    try:
        first=env.reset(); old_sim=env.sim
        assert Task.edit_model_xml is original
        assert first["state"].shape==(29,) and first["primary"].shape==(256,256,3)
        if split=="calibration":assert not np.array_equal(*[o["primary"] for o in env.pair()])
        else:
            with pytest.raises(ValueError,match="calibration-only"):env.pair()
        again=env.reset();assert env.sim is not old_sim
        np.testing.assert_array_equal(first["primary"],again["primary"])
        observation,success,done=env.step(first["state"])
        assert not success and not done and observation["state"].shape==(29,)
    finally:env.close()
    # Actual CLI preflight/report gate through the same native API fixture.
    from robocasa_gr1_plus.cli import validate
    from robocasa_gr1_plus.preflight import verify_validation
    manifest=tmp_path/"one-row-fixture.jsonl";manifest.write_text(canonical(row)+"\n")
    report=validate(manifest,str(source),tmp_path/"validation",hold_steps=2)
    assert report["complete_valid_coverage"]
    assert verify_validation(manifest,tmp_path/"validation/validation.json")["episodes"]==1
