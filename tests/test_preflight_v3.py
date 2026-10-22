import json
from types import SimpleNamespace
import numpy as np
import pytest
from robocasa_gr1_plus.protocol import source_fingerprint
from robocasa_gr1_plus.protocol_v3 import make_plan_v3
from robocasa_gr1_plus.preflight import verify_validation,manifest_hash,check_hold_stability
from robocasa_gr1_plus.catalogue_v2 import InvalidPerturbation
from robocasa_gr1_plus.runtime import runtime_versions

def test_preflight_rejects_partial_invalid_stale_and_relabelled_coverage(tmp_path):
    make_plan_v3(tmp_path/"plan",mode="checkpoint")
    manifest=tmp_path/"plan/confirmation.jsonl"
    rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    path=tmp_path/"validation.json"
    report={"extension_sha256":source_fingerprint(),"manifest_sha256":manifest_hash(manifest),
            "runtime_versions":runtime_versions(),
            "complete_valid_coverage":True,"errors":0,"invalid":0,
            "records":[{"episode_id":r["episode_id"],"variant":r["variant"],"status":"passed",
                        "initial_state_sha256":"0"*64,"initial_image_rgb_sha256":"1"*64,"hold_steps":10,
                        "environment_provenance":{"commit":r["upstream_commit"],"env_name":r["env_name"]}} for r in rows]}
    path.write_text(json.dumps(report)); assert verify_validation(manifest,path)["episodes"]==504
    report["records"].pop();path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match="coverage"): verify_validation(manifest,path)
    report["extension_sha256"]="changed";path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match="source"): verify_validation(manifest,path)

def test_passive_task_completion_and_large_free_body_drift_are_not_valid_initializations():
    model=SimpleNamespace(njnt=1,jnt_type=np.array([0]),jnt_qposadr=np.array([0]))
    sim=SimpleNamespace(model=model,data=SimpleNamespace(qpos=np.array([.01,0,0,1,0,0,0])))
    before={0:np.zeros(3)}
    assert check_hold_stability(before,sim,False)=={"0":.01}
    with pytest.raises(InvalidPerturbation,match="passive"): check_hold_stability(before,sim,True)
    sim.data.qpos[0]=.04
    with pytest.raises(InvalidPerturbation,match="drift"): check_hold_stability(before,sim,False)
