"""Coverage-aware summaries of native LoTTA episode/error artifacts."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from .protocol import digest

def summarize(manifest, results, output, methods=None):
    manifest = Path(manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    expected = {r["episode_id"]: r for r in rows}
    if not rows or len(expected) != len(rows):
        raise ValueError("Manifest is empty or has duplicate episode IDs")
    if len({r["protocol"] for r in rows}) != 1 or len({r["plan_id"] for r in rows}) != 1:
        raise ValueError("Cannot combine protocols or plans")
    if any(r["episode_id"] != digest({k:v for k,v in r.items() if k != "episode_id"})[:20] for r in rows):
        raise ValueError("Manifest episode identities differ from their content")
    modes = {r.get("evaluation_mode", "lotta") for r in rows}
    methods = methods or (["frozen"] if modes == {"checkpoint"} else ["frozen", "random", "localized"])
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("Select distinct policy methods")
    if modes == {"checkpoint"} and methods != ["frozen"]:
        raise ValueError("Checkpoint mode reports frozen only; use a task-disjoint LoTTA plan for adaptation")
    completed, errors = {}, {}
    fingerprints = set()
    for kind, target in (("episodes", completed), ("errors", errors)):
        for path in sorted(Path(results).rglob("*.json")):
            if path.parent.name != kind:
                continue
            record = json.loads(path.read_text())
            key = (record.get("method"), record.get("episode_id"))
            if key[0] not in methods:
                continue
            if key[1] not in expected:
                raise ValueError(f"Result is outside the requested manifest: {path}")
            row = expected[key[1]]
            if any(record.get(k) != v for k, v in row.items()):
                raise ValueError(f"Result provenance differs from manifest: {path}")
            run_path = path.parent.parent / "run.json"
            if run_path.exists():
                run = json.loads(run_path.read_text())
                if record.get("run_id") != run.get("run_id"):
                    raise ValueError(f"Result belongs to a different run: {path}")
            if key in target:
                raise ValueError(f"Duplicate result for {key}; do not merge reruns")
            if record.get("model_fingerprint") is not None:
                fingerprints.add(record["model_fingerprint"])
                if len(fingerprints) > 1:
                    raise ValueError("Results contain different model fingerprints; report each checkpoint separately")
            if kind == "episodes" and (record.get("status") != "completed" or type(record.get("success")) is not bool):
                raise ValueError(f"Malformed completed result: {path}")
            target[key] = record
    groups = defaultdict(list)
    totals = defaultdict(list)
    for method in methods:
        for row in rows:
            key = (method, row["episode_id"])
            success, failure = completed.get(key), errors.get(key)
            status = "completed" if success else ("invalid" if failure and failure.get("error_type") == "InvalidPerturbation"
                                                    else "error" if failure else "pending")
            item = {"status": status, "success": bool(success and success["success"]),
                    "base_task_id": row["base_task_id"]}
            groups[(method, row["dimension"], row.get("subtype", "legacy"), row["variant"]["severity"])].append(item)
            totals[method].append(item)
    def counts(items):
        out = {name: sum(i["status"] == name for i in items) for name in ("completed", "invalid", "error", "pending")}
        out.update({"planned": len(items), "successes": sum(i["success"] for i in items)})
        out["success_rate"] = out["successes"] / out["completed"] if out["completed"] else None
        out["complete_valid_coverage"] = out["completed"] == out["planned"]
        return out
    table = [{"method": k[0], "dimension": k[1], "subtype": k[2], "severity": k[3], **counts(v)}
             for k, v in sorted(groups.items())]
    overall = {m: counts(items) for m, items in totals.items()}
    for method in methods:
        rates = []
        for dim in sorted({r["dimension"] for r in rows} - {"clean"}):
            items = [i for k, v in groups.items() if k[0] == method and k[1] == dim for i in v]
            rate = counts(items)["success_rate"]
            if rate is not None:
                rates.append(rate)
        overall[method]["dimension_macro_rate"] = sum(rates) / len(rates) if rates else None
    report = {"protocol": rows[0]["protocol"], "plan_id": rows[0]["plan_id"],
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "methods": methods, "overall": overall, "by_subtype_severity": table,
              "model_fingerprints": sorted(fingerprints),
              "recovered_error_records": len(set(completed) & set(errors)),
              "rate_denominator": "completed valid episodes only; errors, invalid cases and pending shown separately",
              "aggregation": "dimension_macro_rate gives equal weight to observed shifted dimensions; inspect coverage before comparison"}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
