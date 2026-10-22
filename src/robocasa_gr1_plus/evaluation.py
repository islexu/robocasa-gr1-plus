"""Frozen DiT4DiT evaluation without a LoTTA dependency, using the same manifests."""
import json
from pathlib import Path
import random
import tempfile
import time
import numpy as np
from .protocol import digest
from .preflight import verify_validation
from .report_v2 import summarize

def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w",dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name); json.dump(value,stream,indent=2,allow_nan=False); stream.write("\n")
    try: temporary.replace(path)
    finally: temporary.unlink(missing_ok=True)

def evaluate_dit4dit(manifest,validation,output,root,dit4dit_root,checkpoint,host="127.0.0.1",port=6398,
                     unnorm_key=None,execute_steps=12,max_steps=720,resume=False,
                     _policy=None,_environment=None):
    from .environment import GR1PlusEnvironment
    from .dit4dit import DiT4DiTClient
    from .shifts import array_digest,state_snapshot
    gate=verify_validation(manifest,validation)
    rows=[json.loads(s) for s in Path(manifest).read_text().splitlines() if s.strip()]
    if any(r["split"]=="calibration" or r.get("pairing") for r in rows):
        raise ValueError("Frozen evaluation requires development/confirmation without reference pairs")
    if min(execute_steps,max_steps)<1: raise ValueError("Use positive action execution/horizon")
    policy=_policy or DiT4DiTClient(dit4dit_root,checkpoint,host,port,unnorm_key)
    output=Path(output)
    cfg={"methods":["frozen"],"environment":{"expected_package_root":root,"max_steps":max_steps}}
    identity={"preflight":gate,"model_fingerprint":policy.fingerprint,"execute_steps":execute_steps,"max_steps":max_steps}
    run_id=digest(identity)[:20]
    checks={r["episode_id"]:r for r in json.loads(Path(validation).read_text())["records"]}
    runtime_errors=0
    try:
        if (output/"run.json").exists():
            old=json.loads((output/"run.json").read_text())
            if old["run_id"]!=run_id: raise ValueError("Different checkpoint, manifest, validation or rollout settings in output")
            if not resume: raise FileExistsError("Output already exists; use --resume")
        atomic_json(output/"run.json",{"run_id":run_id,"backend":"official_dit4dit_websocket","manifest":rows,
                    "methods":["frozen"],"identity":identity,"policy_identity":getattr(policy,"identity",None)})
        for row in rows:
            destination=output/"episodes"/("frozen-"+row["episode_id"]+".json")
            if destination.exists() and resume:
                old=json.loads(destination.read_text())
                if old.get("run_id")!=run_id or old.get("status")!="completed" or type(old.get("success")) is not bool:
                    raise ValueError("Resume found a mismatched or incomplete result")
                continue
            env=None; stage="reset"; steps=0; begin=time.monotonic(); timings=[]
            record={**row,"method":"frozen","run_id":run_id,"model_fingerprint":policy.fingerprint}
            try:
                random.seed(row["seed"]); np.random.seed(row["seed"])
                policy.reset(row["seed"])
                env=(_environment or GR1PlusEnvironment)(cfg,row)
                obs=env.reset()
                if array_digest(state_snapshot(env.sim))!=checks[row["episode_id"]]["initial_state_sha256"]:
                    raise RuntimeError("Runtime reset differs from validated initial state")
                image_sha=__import__("hashlib").sha256(obs["primary"].tobytes()).hexdigest()
                if image_sha!=checks[row["episode_id"]].get("initial_image_rgb_sha256"):
                    raise RuntimeError("Runtime image differs from validated initial image")
                success,done=False,False
                while not success and not done and steps<max_steps:
                    stage="inference"; t=time.monotonic(); actions=np.asarray(policy.predict(obs))
                    timings.append(time.monotonic()-t)
                    if actions.ndim!=2 or actions.shape[1]!=29 or len(actions)<1 or not np.isfinite(actions).all():
                        raise ValueError("Policy must return a finite absolute [chunk,29] action array")
                    stage="environment_step"
                    for action in actions[:min(execute_steps,max_steps-steps)]:
                        obs,success,done=env.step(action); steps+=1
                        if success or done: break
                record.update({"status":"completed","success":bool(success),"steps":steps,
                               "environment_provenance":env.provenance,"seconds":time.monotonic()-begin,
                               "inference_seconds_mean":sum(timings)/len(timings) if timings else None,
                               "updates":[],"adapter_parameters":0,"sites":[]})
                record.update({"timings":{"inference":timings,"update":[]},"episode_wall_seconds":record["seconds"],
                               "peak_cuda_bytes":None,"observed_frames":steps+1,"available_segments":0})
                atomic_json(destination,record)
                print(f"completed {row['base_task_id']} {row['variant_name']} success={success}",flush=True)
            except Exception as exc:
                runtime_errors+=1
                atomic_json(output/"errors"/("frozen-"+row["episode_id"]+".json"),
                            {**record,"status":"error","error_type":type(exc).__name__,"error":str(exc),"stage":stage})
                print(f"error {row['episode_id']} {stage} {type(exc).__name__}: {exc}",flush=True)
            finally:
                if env is not None: env.close()
        report=summarize(manifest,output,output/"coverage.json",["frozen"])
        report["runtime_errors"]=runtime_errors
        return report
    finally:
        policy.close()
