"""Original photo textures and composition fixtures; not GR1 checkpoint evaluation."""
import os
from types import SimpleNamespace
import numpy as np
import pytest
mujoco=pytest.importorskip("mujoco")
from test_mujoco_pair import XML,SimBridge,PhysicalSimBridge
from test_mujoco_v2 import suite_for
from robocasa_gr1_plus.assets import catalogue,asset_pool,asset_reference
from robocasa_gr1_plus.catalogue_v3 import COMPOSITIONS,make_variant_v3
from robocasa_gr1_plus.environment import GR1PlusEnvironment
from robocasa_gr1_plus.scene_v3 import inject_scene_v3,HOUSEHOLD_GEOMS,photo_texture
from robocasa_gr1_plus.embodiment import IMAGE_KEY,LANGUAGE_KEY,PARTS
from robocasa_gr1_plus.shifts import state_snapshot,array_digest,physical_digest,VISUAL_FIELDS
from robocasa_gr1_plus.physical_v2 import physical_initial_shift_v2

def wrapper_for(spec,xml):
    sim=SimBridge(xml); renderer=mujoco.Renderer(sim.raw,height=256,width=256)
    wrapper=object.__new__(GR1PlusEnvironment)
    wrapper.v2=True; wrapper.v3=True; wrapper.spec=spec; wrapper.sim=sim;wrapper.steps=0;wrapper.provenance={}
    wrapper.camera_pivot=np.array([0.,0.,.05])
    wrapper.episode={"protocol":"robocasa-gr1-plus-v0.3","seed":42,"base_task_id":"fixture","variant_name":spec["subtype"]}
    def native():
        renderer.update_scene(sim.data,camera="egoview")
        raw={IMAGE_KEY:renderer.render().copy(),LANGUAGE_KEY:"unlocked_waist: pick the block and place it in the basket"}
        raw.update({"state."+name:np.zeros(n) for name,n in PARTS})
        return raw
    wrapper._render_native=native
    return wrapper,renderer

@pytest.mark.parametrize("record",catalogue()["assets"],ids=lambda r:r["id"])
def test_all_eight_real_photo_assets_compile_render_and_preserve_physics(record):
    if not os.getenv("GR1_PLUS_ASSET_CACHE"): pytest.skip("Prepare the CC0 photo pack first")
    spec=make_variant_v3((("background","surface_texture"),),split=record["split"],asset_id=record["id"],severity=3)
    wrapper,renderer=wrapper_for(spec,inject_scene_v3(XML,spec["components"]))
    before=array_digest(state_snapshot(wrapper.sim)); physics=physical_digest(wrapper.sim)
    try:
        clean=wrapper._render(False); shifted=wrapper._render(True)
        assert not np.array_equal(clean["primary"],shifted["primary"])
        assert before==array_digest(state_snapshot(wrapper.sim)) and physics==physical_digest(wrapper.sim)
        assert photo_texture(spec["components"][0]["parameters"])[0].is_file()
    finally: renderer.close()

@pytest.mark.parametrize("recipes",COMPOSITIONS)
def test_each_fixed_composition_changes_images_restores_arrays_and_exposes_no_pair(recipes):
    if any(d=="background" for d,s in recipes) and not os.getenv("GR1_PLUS_ASSET_CACHE"): pytest.skip("Prepare photo pack first")
    spec=make_variant_v3(recipes,asset_id=asset_pool("confirmation")[0],severity=3)
    xml=XML.replace("</worldbody>",'<geom name="wall_test" type="box" pos="0 .8 .8" size="2 .03 1"/></worldbody>')
    wrapper,renderer=wrapper_for(spec,inject_scene_v3(xml,spec["components"]))
    arrays={k:getattr(wrapper.sim.model,k).copy() for k in VISUAL_FIELDS+("light_dir","light_castshadow")}
    before=array_digest(state_snapshot(wrapper.sim))
    try:
        clean=wrapper._render(False); wrapper._latest=wrapper._render(True)
        assert not np.array_equal(clean["primary"],wrapper._latest["primary"])
        with pytest.raises(ValueError,match="calibration-only"): wrapper.pair()
        assert array_digest(state_snapshot(wrapper.sim))==before
        for k,v in arrays.items(): np.testing.assert_array_equal(getattr(wrapper.sim.model,k),v)
        with pytest.raises(RuntimeError):
            with wrapper._visual_context(spec): raise RuntimeError("renderer failure")
        for k,v in arrays.items(): np.testing.assert_array_equal(getattr(wrapper.sim.model,k),v)
    finally: renderer.close()

@pytest.mark.parametrize("split",["development","confirmation"])
def test_household_clutter_uses_multiple_physical_geoms_and_split_specific_shapes(split):
    spec=make_variant_v3((("layout","clutter"),),severity=5,split=split)
    component=spec["components"][0]
    sim=PhysicalSimBridge(inject_scene_v3(XML,spec["components"]))
    proof=physical_initial_shift_v2(suite_for(sim),component)
    assert len(proof["changed"])==4
    assert all(shape in HOUSEHOLD_GEOMS for shape in component["parameters"]["asset_ids"])
    assert sim.raw.nbody>SimBridge().raw.nbody
    assert all(sim.model.geom_group[i]==1 for i in range(sim.raw.ngeom) if (sim.model.geom_id2name(i) or "").startswith("gr1plus_v2_clutter"))
