import json
import numpy as np
import pytest
from robocasa_gr1_plus.catalogue_v2 import SUBTYPES, make_variant, validate_variant
from robocasa_gr1_plus.protocol_v2 import make_plan_v2

def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]

def test_v2_default_covers_every_subtype_and_pairs_only_heldout_visual_calibration(tmp_path):
    plan = make_plan_v2(tmp_path)
    assert plan["episodes"] == {"calibration":48, "development":126, "confirmation":252}
    splits = [set(s) for s in plan["base_task_splits"].values()]
    assert not any(a & b for i, a in enumerate(splits) for b in splits[:i])
    expected = {(d, s) for d, sub in SUBTYPES.items() for s in sub} | {("clean", "nominal")}
    for split in ("development", "confirmation"):
        manifest = rows(tmp_path / (split + ".jsonl"))
        for task in plan["base_task_splits"][split]:
            subset = [r for r in manifest if r["base_task_id"] == task]
            assert {(r["dimension"],r["subtype"]) for r in subset} == expected
            assert len({r["seed"] for r in subset}) == 1
            assert all(r["difficulty"] is None and "pairing" not in r for r in subset)
    assert all(r["pairing"] == "gr1_visual_same_state" and r["dimension"] in ("camera", "noise")
               for r in rows(tmp_path / "calibration.jsonl"))
    assert len({r["episode_id"] for r in rows(tmp_path / "confirmation.jsonl")}) == 252

def test_all_strengths_share_directions_and_fixed_cases_are_not_duplicated(tmp_path):
    make_plan_v2(tmp_path, variants=4, severities=(1,3,5), trials=2)
    manifest = rows(tmp_path / "development.jsonl")
    subset = [r for r in manifest if r["base_task_id"] == manifest[0]["base_task_id"] and r["init_id"] == 0]
    assert len([r for r in subset if r["subtype"] == "shadow"]) == 1
    assert len([r for r in subset if r["dimension"] == "language"]) == 7  # 3+3+1
    assert len([r for r in subset if r["subtype"] == "distance"]) == 3
    robot = [r["variant"] for r in subset if r["dimension"] == "robot_state" and r["variant"]["index"] == 0]
    for spec in robot:
        delta = np.array(spec["parameters"]["delta"])
        assert len(delta) == 14 and np.linalg.norm(delta) == pytest.approx(spec["parameters"]["l2_radius_rad"])
        np.testing.assert_allclose(delta / np.linalg.norm(delta), np.array(robot[0]["parameters"]["delta"]) / .05)

def test_checkpoint_mode_has_all_24_tasks_no_calibration_or_oracles(tmp_path):
    plan = make_plan_v2(tmp_path, mode="checkpoint", trials=2)
    assert plan["episodes"] == {"confirmation": 24 * 21 * 2}
    assert not (tmp_path / "calibration.jsonl").exists()
    assert all("pairing" not in r and r["evaluation_mode"] == "checkpoint" for r in rows(tmp_path / "confirmation.jsonl"))
    before = (tmp_path / "confirmation.jsonl").read_bytes()
    with pytest.raises(ValueError, match="different"):
        make_plan_v2(tmp_path, mode="checkpoint", trials=3)
    assert before == (tmp_path / "confirmation.jsonl").read_bytes()

def test_variant_parameters_are_tamper_evident_and_no_global_rng_is_used():
    np.random.seed(73)
    expected = np.random.random()
    np.random.seed(73)
    spec = make_variant("robot_state", "joint_vector", severity=4)
    assert np.random.random() == expected
    assert validate_variant(spec) == spec
    spec["parameters"]["delta"][0] += .01
    with pytest.raises(ValueError, match="recipe"):
        validate_variant(spec)

def test_numeric_presets_do_not_repeat_identical_parameters():
    from robocasa_gr1_plus.catalogue_v2 import CATEGORICAL
    for dimension,types in SUBTYPES.items():
        for subtype in types:
            if (dimension,subtype) not in CATEGORICAL:
                parameters=[json.dumps(make_variant(dimension,subtype,severity=s)["parameters"],sort_keys=True) for s in range(1,6)]
                assert len(set(parameters))==5, (dimension,subtype)
