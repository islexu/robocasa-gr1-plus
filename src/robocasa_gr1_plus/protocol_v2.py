"""Subtype-complete, task-disjoint LoTTA and checkpoint plans."""
from __future__ import annotations
import json
from pathlib import Path
import random
from .catalogue_v2 import SUBTYPES, CATEGORICAL, SINGLE_SAMPLE, PROTOCOL_V2, make_variant
from .protocol import TASKS, VISUAL, UPSTREAM_COMMIT, canonical, digest, env_name, source_fingerprint

def make_plan_v2(output, seed=42, calibration_tasks=6, development_tasks=6, variants=1,
                 severities=(1,), trials=1, calibration_dimensions=("camera", "noise"),
                 mode="lotta"):
    if type(seed) is not int or type(variants) is not int or variants < 1 or type(trials) is not int or trials < 1:
        raise ValueError("Use an integer seed and positive variants/trials")
    if not severities or len(set(severities)) != len(severities) or any(type(x) is not int or x not in range(1,6) for x in severities):
        raise ValueError("Select distinct severity presets from 1..5")
    if not calibration_dimensions or not set(calibration_dimensions) <= VISUAL or len(set(calibration_dimensions)) != len(calibration_dimensions):
        raise ValueError("Calibration requires distinct visual dimensions")
    if mode not in ("lotta", "checkpoint"):
        raise ValueError("mode must be lotta or checkpoint")
    bases = list(TASKS)
    if mode == "lotta":
        if calibration_tasks < 1 or development_tasks < 1 or calibration_tasks + development_tasks >= len(bases):
            raise ValueError("Reserve nonempty disjoint calibration/development/confirmation tasks")
        random.Random(seed).shuffle(bases)
        splits = {"calibration": bases[:calibration_tasks],
                  "development": bases[calibration_tasks:calibration_tasks+development_tasks],
                  "confirmation": bases[calibration_tasks+development_tasks:]}
    else:
        # A checkpoint-only audit never supplies calibration observations or a LoTTA selection.
        splits = {"confirmation": bases}
    settings = {"seed": seed, "variants": variants, "severities": list(severities), "trials": trials,
                "calibration_tasks": calibration_tasks, "development_tasks": development_tasks,
                "calibration_dimensions": list(calibration_dimensions), "mode": mode}
    source = source_fingerprint()
    plan_id = digest({"protocol": PROTOCOL_V2, "settings": settings, "source": source})
    payload, counts = {}, {}
    for split, tasks in splits.items():
        rows = []
        for task in tasks:
            recipes = [] if split == "calibration" else [("clean", "nominal", 0, 1)]
            dimensions = calibration_dimensions if split == "calibration" else tuple(SUBTYPES)
            for dimension in dimensions:
                for subtype in SUBTYPES[dimension]:
                    levels = (1,) if (dimension, subtype) in CATEGORICAL else severities
                    samples = 1 if (dimension, subtype) in SINGLE_SAMPLE else min(variants, 3) if dimension == "language" else variants
                    recipes.extend((dimension, subtype, index, level) for index in range(samples) for level in levels)
            for dimension, subtype, index, severity in recipes:
                for trial in range(trials):
                    episode_seed = (seed + int(digest([task, trial])[:8],16)) % (2**31)
                    # Recipe sampling is independent of severity: larger presets share a direction.
                    sample_seed = int(digest([seed, task, trial, dimension, subtype, index])[:8],16)
                    spec = make_variant(dimension, subtype, index, severity, sample_seed)
                    row = {"suite": "robocasa_gr1", "benchmark": "robocasa_gr1_plus",
                           "task_id": TASKS.index(task), "base_task_id": task, "env_name": env_name(task),
                           "variant": spec, "variant_name": f"{dimension}-{subtype}-v{index}-s{severity}",
                           "dimension": dimension, "subtype": subtype, "severity": severity,
                           "difficulty": None, "difficulty_kind": "not_empirically_calibrated",
                           "condition": "clean", "init_id": trial, "seed": episode_seed,
                           "split": split, "plan_id": plan_id, "protocol": PROTOCOL_V2,
                           "evaluation_mode": mode, "upstream_commit": UPSTREAM_COMMIT,
                           "extension_sha256": source,
                           "stream_id": f"{task}/{dimension}/{subtype}/{index}/{severity}"}
                    if split == "calibration":
                        row["pairing"] = "gr1_visual_same_state"
                    row["episode_id"] = digest(row)[:20]
                    rows.append(row)
        counts[split] = len(rows)
        payload[f"{split}.jsonl"] = "".join(canonical(r)+"\n" for r in rows)
    plan = {"protocol": PROTOCOL_V2, "plan_id": plan_id, "settings": settings,
            "base_task_splits": splits, "episodes": counts, "subtypes": SUBTYPES,
            "extension_sha256": source, "upstream_commit": UPSTREAM_COMMIT,
            "note": "GR1-specific parameter presets inspired by LIBERO-Plus; success and difficulty uncalibrated."}
    payload["plan.json"] = json.dumps(plan, indent=2)+"\n"
    output = Path(output)
    for name, text in payload.items():
        if (output/name).exists() and (output/name).read_text() != text:
            raise ValueError("Output contains a different protocol/source/plan; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, text in payload.items(): (output/name).write_text(text)
    return plan
