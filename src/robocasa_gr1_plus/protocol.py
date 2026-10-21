from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random

UPSTREAM_COMMIT = "4840e671596f93ca03651524b9f72ffb1aadfeff"
PROTOCOL = "robocasa-gr1-plus-v0.1"
DIMENSIONS = ("camera", "background", "light", "noise", "language", "robot_state", "layout")
VISUAL = frozenset(DIMENSIONS[:4])
TASKS = (
    "PnPCupToDrawerClose", "PnPPotatoToMicrowaveClose", "PnPMilkToMicrowaveClose",
    "PnPBottleToCabinetClose", "PnPWineToCabinetClose", "PnPCanToDrawerClose",
    "PosttrainPnPNovelFromCuttingboardToBasketSplitA",
    "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA",
    "PosttrainPnPNovelFromCuttingboardToPanSplitA",
    "PosttrainPnPNovelFromCuttingboardToPotSplitA",
    "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA",
    "PosttrainPnPNovelFromPlacematToBasketSplitA",
    "PosttrainPnPNovelFromPlacematToBowlSplitA",
    "PosttrainPnPNovelFromPlacematToPlateSplitA",
    "PosttrainPnPNovelFromPlacematToTieredshelfSplitA",
    "PosttrainPnPNovelFromPlateToBowlSplitA",
    "PosttrainPnPNovelFromPlateToCardboardboxSplitA",
    "PosttrainPnPNovelFromPlateToPanSplitA",
    "PosttrainPnPNovelFromPlateToPlateSplitA",
    "PosttrainPnPNovelFromTrayToCardboardboxSplitA",
    "PosttrainPnPNovelFromTrayToPlateSplitA",
    "PosttrainPnPNovelFromTrayToPotSplitA",
    "PosttrainPnPNovelFromTrayToTieredbasketSplitA",
    "PosttrainPnPNovelFromTrayToTieredshelfSplitA",
)

def env_name(task):
    if task not in TASKS:
        raise ValueError(f"Unknown GR1 tabletop task: {task}")
    return f"gr1_unified/{task}_GR1ArmsAndWaistFourierHands_Env"

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def source_fingerprint():
    root = Path(__file__).resolve().parent
    return digest({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))})

def variant(dimension, index=0, severity=1):
    if dimension not in ("clean", *DIMENSIONS):
        raise ValueError(f"Unknown dimension: {dimension}")
    if type(index) is not int or not 0 <= index < 6:
        raise ValueError("variant index must be an integer in [0, 5]")
    if type(severity) is not int or severity not in (1, 2, 3):
        raise ValueError("severity must be 1, 2 or 3")
    return {"dimension": dimension, "index": index, "severity": severity}

def make_plan(output, seed=42, calibration_tasks=6, development_tasks=6,
              variants=1, severities=(1,), trials=1, calibration_dimensions=("camera", "noise")):
    if not 1 <= calibration_tasks or not 1 <= development_tasks:
        raise ValueError("Calibration and development sizes must be positive")
    if calibration_tasks + development_tasks >= len(TASKS):
        raise ValueError("Reserve independent confirmation tasks")
    if not 1 <= variants <= 6 or trials < 1:
        raise ValueError("Use 1..6 variants and positive trials")
    if not calibration_dimensions or not set(calibration_dimensions) <= VISUAL:
        raise ValueError("Calibration accepts pure visual dimensions only")
    if not severities or len(set(severities)) != len(severities):
        raise ValueError("Severities must be nonempty and distinct")
    for level in severities:
        variant("camera", 0, level)
    bases = list(TASKS)
    random.Random(seed).shuffle(bases)
    splits = {"calibration": bases[:calibration_tasks],
              "development": bases[calibration_tasks:calibration_tasks + development_tasks],
              "confirmation": bases[calibration_tasks + development_tasks:]}
    output = Path(output)
    manifests, counts = {}, {}
    settings = {"seed": seed, "calibration_tasks": calibration_tasks,
                "development_tasks": development_tasks, "variants": variants,
                "severities": list(severities), "trials": trials,
                "calibration_dimensions": list(calibration_dimensions)}
    extension_sha256 = source_fingerprint()
    plan_id = digest({"settings": settings, "protocol": PROTOCOL, "source": UPSTREAM_COMMIT,
                      "extension_sha256":extension_sha256})
    for split, tasks in splits.items():
        rows = []
        dims = calibration_dimensions if split == "calibration" else ("clean", *DIMENSIONS)
        for task in tasks:
            for dim in dims:
                for level in ((1,) if dim in ("clean", "language") else severities):
                    for v in range(1 if dim == "clean" else variants):
                        spec = variant(dim, v, level)
                        for trial in range(trials):
                            # Identical nominal episode seeds across clean/shift and methods.
                            episode_seed = (seed + int(digest([task, trial])[:8], 16)) % (2**31)
                            row = {"suite": "robocasa_gr1", "benchmark": "robocasa_gr1_plus",
                                   "task_id": TASKS.index(task), "base_task_id": task,
                                   "env_name": env_name(task), "variant": spec,
                                   "variant_name": f"{dim}-v{v}-s{level}", "dimension": dim,
                                   "difficulty": level, "condition": "clean", "init_id": trial,
                                   "seed": episode_seed, "split": split, "plan_id": plan_id,
                                   "extension_sha256":extension_sha256,
                                   "protocol": PROTOCOL, "upstream_commit": UPSTREAM_COMMIT,
                                   "stream_id": f"{task}/{dim}/{v}/{level}"}
                            if split == "calibration":
                                row["pairing"] = "gr1_visual_same_state"
                            row["episode_id"] = digest(row)[:20]
                            rows.append(row)
        manifests[split] = "".join(canonical(r) + "\n" for r in rows)
        counts[split] = len(rows)
    plan = {"protocol": PROTOCOL, "plan_id": plan_id, "settings": settings,
            "extension_sha256":extension_sha256,
            "base_task_splits": splits, "episodes": counts, "upstream_commit": UPSTREAM_COMMIT,
            "status": "protocol_implementation; task reachability and policy scores require simulation",
            "note": "Custom GR1 robustness extension; not official LIBERO-Plus or RoboCasa leaderboard."}
    payload = {**{k + ".jsonl": v for k,v in manifests.items()},
               "plan.json": json.dumps(plan, indent=2) + "\n"}
    # Check every existing file before writing any, to avoid partially replacing a locked plan.
    for name, content in payload.items():
        path = output/name
        if path.exists() and path.read_text() != content:
            raise ValueError("Output contains a different plan; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in payload.items():
        (output/name).write_text(content)
    return plan
