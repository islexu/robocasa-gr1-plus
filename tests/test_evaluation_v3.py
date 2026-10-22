import hashlib
import json
from types import SimpleNamespace
import numpy as np
from robocasa_gr1_plus.evaluation import evaluate_dit4dit
from robocasa_gr1_plus.protocol_v3 import make_plan_v3
from robocasa_gr1_plus.protocol import canonical,source_fingerprint
from robocasa_gr1_plus.preflight import manifest_hash
from robocasa_gr1_plus.runtime import runtime_versions
from robocasa_gr1_plus.shifts import array_digest,state_snapshot

def test_frozen_runner_separates_policy_failures_from_inference_errors_and_resumes_once(tmp_path):
    make_plan_v3(tmp_path/"plan",mode="checkpoint")
    rows=[json.loads(s) for s in (tmp_path/"plan/confirmation.jsonl").read_text().splitlines()][:3]
    manifest=tmp_path/"subset.jsonl";manifest.write_text("".join(canonical(r)+"\n" for r in rows))
    sim=SimpleNamespace(data=SimpleNamespace(mocap_pos=np.zeros((0,3)),mocap_quat=np.zeros((0,4))),
                        get_state=lambda:SimpleNamespace(flatten=lambda:np.zeros(8)))
    image=np.zeros((256,256,3),np.uint8)
    validation=tmp_path/"validation.json"
    validation.write_text(json.dumps({"manifest_sha256":manifest_hash(manifest),"extension_sha256":source_fingerprint(),
        "runtime_versions":runtime_versions(),"complete_valid_coverage":True,"errors":0,"invalid":0,
        "records":[{"episode_id":r["episode_id"],"variant":r["variant"],"status":"passed","hold_steps":10,
                    "initial_state_sha256":array_digest(state_snapshot(sim)),"initial_image_rgb_sha256":hashlib.sha256(image.tobytes()).hexdigest(),
                    "environment_provenance":{"commit":r["upstream_commit"],"env_name":r["env_name"]}} for r in rows]}))
    class Policy:
        fingerprint="synthetic-fixture-only"
        def __init__(self,fail=True):self.fail=fail;self.resets=[];self.closed=False
        def reset(self,seed):self.resets.append(seed)
        def predict(self,obs):
            if self.fail and obs["case"]==rows[2]["episode_id"]:raise RuntimeError("inference fixture error")
            return np.zeros((12,29),np.float32)
        def close(self):self.closed=True
    class Environment:
        def __init__(self,cfg,row):self.row=row;self.sim=sim;self.provenance={"claim":"synthetic_fixture"}
        def reset(self):return {"primary":image,"state":np.zeros(29),"language":"unlocked_waist: fixture","case":self.row["episode_id"]}
        def step(self,action):return self.reset(),self.row["episode_id"]!=rows[1]["episode_id"],True
        def close(self):pass
    policy=Policy()
    args=dict(manifest=manifest,validation=validation,output=tmp_path/"results",root="fixture",dit4dit_root="fixture",checkpoint="fixture",
              _environment=Environment,_policy=policy)
    report=evaluate_dit4dit(**args)
    total=report["overall"]["frozen"]
    assert total["completed"]==2 and total["successes"]==1 and total["error"]==1 and total["pending"]==0
    assert policy.closed and len(policy.resets)==3
    resumed=Policy(fail=False);args["_policy"]=resumed
    report=evaluate_dit4dit(**args,resume=True)
    assert report["overall"]["frozen"]["complete_valid_coverage"]
    assert report["recovered_error_records"]==1 and len(resumed.resets)==1
