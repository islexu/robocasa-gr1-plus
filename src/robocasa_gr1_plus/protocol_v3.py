"""Task/asset-disjoint plans shared by original DiT4DiT and LoTTA."""
import json
from pathlib import Path
import random
from .assets import asset_pool, asset_lock_sha256
from .catalogue_v2 import SUBTYPES, CATEGORICAL, SINGLE_SAMPLE
from .catalogue_v3 import PROTOCOL_V3, CORE, COMPOSITIONS, make_variant_v3
from .protocol import TASKS, VISUAL, UPSTREAM_COMMIT, canonical, digest, env_name, source_fingerprint

def make_plan_v3(output, seed=42, calibration_tasks=6, development_tasks=6, variants=1,
                 severities=(1,), trials=1, calibration_dimensions=("camera","background","light","noise"),
                 mode="lotta", profile="core", compositions=True,calibration_trials=1):
    if any(type(v) is not int for v in (seed,calibration_tasks,development_tasks,variants,trials,calibration_trials)) or min(variants,trials,calibration_trials)<1:
        raise ValueError("Use integer seed/split sizes and positive variants/trials")
    if not severities or len(set(severities))!=len(severities) or any(type(s) is not int or s not in range(1,6) for s in severities):
        raise ValueError("Use distinct strengths in 1..5")
    if not calibration_dimensions or not set(calibration_dimensions)<=VISUAL or len(set(calibration_dimensions))!=len(calibration_dimensions):
        raise ValueError("Calibration dimensions must be distinct and purely visual")
    if mode not in ("lotta","checkpoint") or profile not in ("core","full"):
        raise ValueError("Select lotta/checkpoint and core/full")
    if mode=="lotta" and (min(calibration_tasks,development_tasks)<1 or calibration_tasks+development_tasks>=len(TASKS)):
        raise ValueError("Reserve nonempty task-disjoint splits")
    bases=list(TASKS); random.Random(seed).shuffle(bases)
    splits=({"calibration":bases[:calibration_tasks],"development":bases[calibration_tasks:calibration_tasks+development_tasks],
             "confirmation":bases[calibration_tasks+development_tasks:]} if mode=="lotta" else {"confirmation":list(TASKS)})
    settings={"seed":seed,"calibration_tasks":calibration_tasks,"development_tasks":development_tasks,"variants":variants,
              "severities":list(severities),"trials":trials,"calibration_dimensions":list(calibration_dimensions),
              "calibration_trials":calibration_trials,
              "mode":mode,"profile":profile,"compositions":bool(compositions)}
    source,assets=source_fingerprint(),asset_lock_sha256()
    plan_id=digest({"protocol":PROTOCOL_V3,"settings":settings,"extension":source,"asset_lock":assets})
    payload,counts={},{}
    for split,tasks in splits.items():
        rows=[]
        for task in tasks:
            matrix=[] if split=="calibration" else [((),0,1)]
            catalogue={d:SUBTYPES[d] for d in calibration_dimensions} if split=="calibration" else (CORE if profile=="core" else SUBTYPES)
            for dimension,subtypes in catalogue.items():
                for subtype in subtypes:
                    levels=(1,) if (dimension,subtype) in CATEGORICAL else severities
                    n=1 if (dimension,subtype) in SINGLE_SAMPLE else min(3,variants) if dimension=="language" else variants
                    if dimension=="background": n=min(n,len(asset_pool(split)))
                    matrix.extend((((dimension,subtype),),i,s) for i in range(n) for s in levels)
            if split!="calibration" and compositions:
                matrix.extend((recipes,0,s) for recipes in COMPOSITIONS for s in severities)
            for recipes,index,severity in matrix:
                for trial in range(calibration_trials if split=="calibration" else trials):
                    # A composition reuses exactly the corresponding single-shift parameters.
                    sample_seed=int(digest([seed,task,trial,index])[:8],16)
                    pool=asset_pool(split)
                    asset_id=pool[(tasks.index(task)+trial*variants+index)%len(pool)] if any(d=="background" for d,s in recipes) else None
                    spec=make_variant_v3(recipes,index,severity,sample_seed,split,asset_id)
                    dimension,subtype=spec["dimension"],spec["subtype"]
                    row={"suite":"robocasa_gr1","benchmark":"robocasa_gr1_plus","task_id":TASKS.index(task),"base_task_id":task,
                         "env_name":env_name(task),"variant":spec,"variant_name":f"{dimension}-{subtype}-v{index}-s{severity}",
                         "dimension":dimension,"subtype":subtype,"severity":severity,"difficulty":None,
                         "difficulty_kind":"not_empirically_calibrated","condition":"clean","init_id":trial,
                         "seed":(seed+int(digest([task,trial])[:8],16))%(2**31),"split":split,"plan_id":plan_id,
                         "protocol":PROTOCOL_V3,"evaluation_mode":mode,"profile":profile,"upstream_commit":UPSTREAM_COMMIT,
                         "extension_sha256":source,"asset_lock_sha256":assets,"asset_split":split,
                         "stream_id":f"{task}/{dimension}/{subtype}/{index}/{severity}"}
                    if split=="calibration": row["pairing"]="gr1_visual_same_state"
                    row["episode_id"]=digest(row)[:20]; rows.append(row)
        counts[split]=len(rows); payload[split+".jsonl"]="".join(canonical(r)+"\n" for r in rows)
    plan={"protocol":PROTOCOL_V3,"plan_id":plan_id,"settings":settings,"base_task_splits":splits,"episodes":counts,
          "texture_asset_splits":{s:asset_pool(s) for s in ("calibration","development","confirmation")},
          "asset_lock_sha256":assets,"extension_sha256":source,"upstream_commit":UPSTREAM_COMMIT,
          "note":"Fixed task-preserving GR1 shifts; mechanics/semantic review and checkpoint evaluation required."}
    payload["plan.json"]=json.dumps(plan,indent=2)+"\n"
    output=Path(output)
    for name,content in payload.items():
        if (output/name).exists() and (output/name).read_text()!=content:
            raise ValueError("Different source, asset pack or plan already exists; use a new output")
    output.mkdir(parents=True,exist_ok=True)
    for name,content in payload.items(): (output/name).write_text(content)
    return plan
