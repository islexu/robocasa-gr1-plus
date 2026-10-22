import copy
import json
import pytest
from robocasa_gr1_plus.assets import asset_pool,asset_lock_sha256,prepare_assets
from robocasa_gr1_plus.catalogue_v3 import CORE,COMPOSITIONS,make_variant_v3,validate_variant_v3
from robocasa_gr1_plus.protocol_v3 import make_plan_v3

def read(path): return [json.loads(s) for s in path.read_text().splitlines()]

def test_task_and_texture_holdouts_are_disjoint_and_calibration_covers_every_visual_dimension(tmp_path):
    plan=make_plan_v3(tmp_path,trials=4)
    assert plan["episodes"]=={"calibration":84,"development":504,"confirmation":1008}
    tasks=[set(x) for x in plan["base_task_splits"].values()]
    textures=[set(asset_pool(s)) for s in ("calibration","development","confirmation")]
    for groups in (tasks,textures): assert not any(a&b for i,a in enumerate(groups) for b in groups[:i])
    calibration=read(tmp_path/"calibration.jsonl")
    assert {r["dimension"] for r in calibration}=={"camera","background","light","noise"}
    for split in ("calibration","development","confirmation"):
        rows=read(tmp_path/(split+".jsonl"))
        assert len(rows)==len({r["episode_id"] for r in rows})
        for r in rows:
            assert r["difficulty"] is None and r["asset_lock_sha256"]==asset_lock_sha256()
            assert (r.get("pairing")=="gr1_visual_same_state")== (split=="calibration")
            assert validate_variant_v3(r["variant"],split)==r["variant"]
            for c in r["variant"]["components"]:
                if c["dimension"]=="background": assert c["parameters"]["asset"]["id"] in asset_pool(split)
        for task in plan["base_task_splits"][split]:
            used={c["parameters"]["asset"]["id"] for r in rows if r["base_task_id"]==task
                  for c in r["variant"]["components"] if c["dimension"]=="background"}
            assert used<=set(asset_pool(split))
            if split!="calibration": assert used==set(asset_pool(split))

def test_compositions_reuse_exact_single_shift_parameters_and_nominal_scene_seeds(tmp_path):
    make_plan_v3(tmp_path,severities=(1,3),trials=1)
    rows=read(tmp_path/"development.jsonl")
    for row in rows:
        if row["dimension"]!="composition": continue
        assert "pairing" not in row
        for component in row["variant"]["components"]:
            single=next(r for r in rows if r["base_task_id"]==row["base_task_id"] and r["init_id"]==row["init_id"]
                        and r["dimension"]==component["dimension"] and r["subtype"]==component["subtype"] and r["severity"]==row["severity"])
            assert single["seed"]==row["seed"]
            assert single["variant"]["components"][0]==component

def test_profile_counts_checkpoint_mode_immutability_and_no_duplicate_fixed_textures(tmp_path):
    core=make_plan_v3(tmp_path/"core",mode="checkpoint")
    assert core["episodes"]=={"confirmation":504}
    full=make_plan_v3(tmp_path/"full",mode="checkpoint",profile="full")
    assert full["episodes"]=={"confirmation":600}
    assert not (tmp_path/"core/calibration.jsonl").exists()
    before=(tmp_path/"core/plan.json").read_bytes()
    with pytest.raises(ValueError,match="Different"):
        make_plan_v3(tmp_path/"core",mode="checkpoint",profile="full")
    assert (tmp_path/"core/plan.json").read_bytes()==before
    make_plan_v3(tmp_path/"many",variants=7)
    rows=read(tmp_path/"many/calibration.jsonl")
    subset=[r for r in rows if r["base_task_id"]==rows[0]["base_task_id"] and r["subtype"]=="surface_texture"]
    assert len(subset)==len(asset_pool("calibration"))
    assert len({json.dumps(r["variant"]["components"][0]["parameters"],sort_keys=True) for r in subset})==len(subset)

def test_asset_and_component_tampering_cannot_relabel_a_case_as_v3():
    spec=make_variant_v3((("background","surface_texture"),),asset_id=asset_pool("confirmation")[0])
    wrong=copy.deepcopy(spec); wrong["components"][0]["parameters"]["asset"]["sha256"]="0"*64
    with pytest.raises(ValueError): validate_variant_v3(wrong,"confirmation")
    with pytest.raises(ValueError): validate_variant_v3(spec,"calibration")
    with pytest.raises(ValueError): make_variant_v3((("layout","target_pose"),("camera","orbit")))
    with pytest.raises(ValueError): make_variant_v3((),severity=2)
