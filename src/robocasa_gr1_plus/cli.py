from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from .protocol import make_plan, PROTOCOL, UPSTREAM_COMMIT
from .catalogue_v2 import PROTOCOL_V2, SUBTYPES, make_variant, InvalidPerturbation

def doctor(root, require_model=False):
    names = ["robocasa", "robosuite", "mujoco", "gymnasium", "cv2"]
    if require_model:
        names += ["lotta", "torch", "diffusers", "omegaconf", "transformers"]
    dependencies = {name: importlib.util.find_spec(name) is not None for name in names}
    errors = [f"Missing dependency: {name}" for name, exists in dependencies.items() if not exists]
    source = None
    if root:
        try:
            from .environment import source_provenance
            source = source_provenance(root)
            if dependencies["robocasa"]:
                found = importlib.util.find_spec("robocasa")
                if Path(root).resolve() not in Path(found.origin).resolve().parents:
                    errors.append("Imported robocasa is not the expected checkout")
            assets = Path(root)/"robocasa/models/assets"
            # Asset presence alone is not proof that the simulator can load all tasks.
            source["assets_directory_exists"] = assets.is_dir()
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(str(exc))
    else:
        errors.append("Set GR1_PLUS_UPSTREAM to the pinned robocasa checkout")
    if require_model and dependencies.get("torch"):
        import torch
        if not torch.cuda.is_available():
            errors.append("No CUDA GPU visible")
    return {"supported_protocols":[PROTOCOL, PROTOCOL_V2],"dependencies":dependencies,"source":source,"errors":errors}

def validate(manifest, root, output, hold_steps=10, limit=None):
    import numpy as np
    from PIL import Image
    from .environment import GR1PlusEnvironment
    from .shifts import array_digest, state_snapshot
    rows = [json.loads(s) for s in Path(manifest).read_text().splitlines() if s.strip()]
    if not rows or hold_steps < 1 or (limit is not None and limit < 1):
        raise ValueError("Use a nonempty manifest and positive hold_steps/limit")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    selected = rows if limit is None else rows[:limit]
    records = []
    cfg = {"environment":{"expected_package_root":root,"max_steps":max(hold_steps,1)}}
    for row in selected:
        record = {"episode_id":row["episode_id"], "base_task_id":row["base_task_id"],
                  "protocol":row["protocol"], "dimension":row["dimension"],
                  "subtype":row.get("subtype"), "variant":row["variant"]}
        env = None
        try:
            env = GR1PlusEnvironment(cfg,row)
            obs = env.reset()
            state_hash = array_digest(state_snapshot(env.sim))
            initial_image = obs["primary"].copy()
            image = initial_image
            if row.get("pairing"):
                clean, shifted = env.pair()
                image = np.concatenate([clean["primary"],shifted["primary"]],axis=1)
                record["pair_proof"] = env.pair_metadata
            elif row["protocol"] == PROTOCOL_V2 and row["dimension"] in ("camera", "background", "light", "noise"):
                # Validation-only visibility check, never supplied to the policy.
                clean_image = env._render(False)["primary"]
                if np.array_equal(clean_image, initial_image):
                    raise InvalidPerturbation("visual_intervention_not_visible_in_ego_image")
                image = np.concatenate([clean_image, initial_image], axis=1)
            Image.fromarray(image).save(output/(row["episode_id"]+".png"))
            hold = obs["state"].copy()
            for _ in range(hold_steps):
                obs, success, _ = env.step(hold)
                if not np.isfinite(obs["state"]).all():
                    raise RuntimeError("Non-finite simulation state during hold test")
            record["success_after_hold"] = success
            record["environment_provenance"] = env.provenance
            env.close(); env = None
            # Fresh instance, same episode seed: check the reset used by each policy method.
            env = GR1PlusEnvironment(cfg,row)
            again = env.reset()
            if state_hash != array_digest(state_snapshot(env.sim)) or not np.array_equal(initial_image,again["primary"]):
                raise RuntimeError("Fresh seeded GR1 reset is not reproducible")
            record.update({"status":"passed","initial_state_sha256":state_hash,"hold_steps":hold_steps})
        except Exception as exc:
            record.update({"status":"invalid" if isinstance(exc, InvalidPerturbation) else "error",
                           "error_type":type(exc).__name__,"error":str(exc)})
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    record["close_error"] = f"{type(exc).__name__}: {exc}"
                    if record.get("status") == "passed":
                        record.update({"status":"error", "error_type":type(exc).__name__, "error":str(exc)})
        records.append(record)
        print(f"{record['status']} {row['base_task_id']} {row['variant_name']}",flush=True)
    report = {"protocols":sorted({r["protocol"] for r in rows}),"upstream_commit":UPSTREAM_COMMIT,
              "manifest_sha256":__import__("hashlib").sha256(Path(manifest).read_bytes()).hexdigest(),
              "requested":len(rows),"validated":len(records),"complete_coverage":len(records)==len(rows),
              "errors":sum(r["status"]=="error" for r in records),
              "invalid":sum(r["status"]=="invalid" for r in records),
              "complete_valid_coverage":len(records)==len(rows) and all(r["status"]=="passed" for r in records),
              "records":records,
              "note":"Initial-state mechanics and reset checks; does not establish task reachability or policy success."}
    (output/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    return report

def main(argv=None):
    parser = argparse.ArgumentParser(description="RoboCasa-GR1-Plus robustness benchmark")
    sub = parser.add_subparsers(dest="command",required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--output",required=True)
    plan.add_argument("--protocol",choices=("v1", "v2"),default="v2")
    plan.add_argument("--mode",choices=("lotta", "checkpoint"),default="lotta")
    plan.add_argument("--seed",type=int,default=42)
    plan.add_argument("--calibration-tasks",type=int,default=6)
    plan.add_argument("--development-tasks",type=int,default=6)
    plan.add_argument("--variants",type=int,default=1)
    plan.add_argument("--severities",default="1")
    plan.add_argument("--trials",type=int,default=1)
    plan.add_argument("--calibration-dimensions",default="camera,noise")
    d = sub.add_parser("doctor")
    d.add_argument("--root",default=os.getenv("GR1_PLUS_UPSTREAM"))
    d.add_argument("--require-model",action="store_true")
    v = sub.add_parser("validate")
    v.add_argument("--root",default=os.getenv("GR1_PLUS_UPSTREAM"),required=not os.getenv("GR1_PLUS_UPSTREAM"))
    v.add_argument("--manifest",required=True)
    v.add_argument("--output",required=True)
    v.add_argument("--hold-steps",type=int,default=10)
    v.add_argument("--limit",type=int)
    sub.add_parser("catalogue", help="Print every v2 subtype and its parameter presets")
    s = sub.add_parser("summarize", help="Report subtype/severity rates and missing/invalid coverage")
    s.add_argument("--manifest",required=True)
    s.add_argument("--results",required=True)
    s.add_argument("--output",required=True)
    s.add_argument("--methods",help="Comma-separated; defaults to plan mode")
    args = vars(parser.parse_args(argv)); command = args.pop("command")
    if command=="plan":
        args["severities"] = tuple(int(x) for x in args["severities"].split(","))
        args["calibration_dimensions"] = tuple(args["calibration_dimensions"].split(","))
        protocol = args.pop("protocol")
        if protocol == "v2":
            from .protocol_v2 import make_plan_v2
            result = make_plan_v2(**args)
        else:
            if args.pop("mode") != "lotta":
                raise ValueError("Checkpoint mode requires v2")
            result = make_plan(**args)
    elif command=="doctor":
        result = doctor(**args)
    elif command=="catalogue":
        from .catalogue_v2 import CATEGORICAL
        result = {f"{d}/{s}": [make_variant(d,s,severity=k) for k in ((1,) if (d,s) in CATEGORICAL else range(1,6))]
                  for d, types in SUBTYPES.items() for s in types}
    elif command=="summarize":
        from .report_v2 import summarize
        args["methods"] = args["methods"].split(",") if args["methods"] else None
        result = summarize(**args)
        print(json.dumps(result["overall"],indent=2))
        return 0
    else:
        result = validate(**args)
        print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))
        return 2 if result["errors"] or result["invalid"] else 0
    print(json.dumps(result,indent=2))
    return 2 if result.get("errors") else 0

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (ValueError,FileNotFoundError,RuntimeError) as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        raise SystemExit(2)
