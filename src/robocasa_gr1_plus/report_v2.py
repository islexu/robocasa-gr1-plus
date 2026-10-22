"""Coverage-aware summaries of native LoTTA episode/error artifacts."""
from collections import defaultdict
import hashlib
import json
import numpy as np
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
    reset_proofs=defaultdict(set)
    missing_reset_proofs=0
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
            if kind=="episodes":
                provenance=record.get("environment_provenance",{})
                proof=(provenance.get("initial_state_sha256"),provenance.get("physical_model_sha256"))
                if all(proof):
                    reset_proofs[key[1]].add(proof)
                    if len(reset_proofs[key[1]])>1:
                        raise ValueError("Methods have different initial simulation states or physics for the same episode")
                else: missing_reset_proofs+=1
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
        singles=[counts([i for k,v in groups.items() if k[0]==method and k[1]==dim for i in v])["success_rate"]
                 for dim in sorted({r["dimension"] for r in rows}-{"clean","composition"})]
        singles=[r for r in singles if r is not None]
        overall[method]["single_dimension_macro_rate"]=sum(singles)/len(singles) if singles else None
        composed=[i for k,v in groups.items() if k[0]==method and k[1]=="composition" for i in v]
        overall[method]["composition_rate"]=counts(composed)["success_rate"] if composed else None
    asset_groups=defaultdict(list)
    clean_pairs=[]
    for method in methods:
        clean={(r["base_task_id"],r["init_id"]):completed.get((method,r["episode_id"])) for r in rows if r["dimension"]=="clean"}
        for row in rows:
            key=(method,row["episode_id"]); record=completed.get(key); failure=errors.get(key)
            for component in row["variant"].get("components",[row["variant"]]):
                asset=component.get("parameters",{}).get("asset")
                if asset:
                    status="completed" if record else "invalid" if failure and failure.get("error_type")=="InvalidPerturbation" else "error" if failure else "pending"
                    asset_groups[(method,asset["id"])].append({"status":status,"success":bool(record and record["success"])})
        for dimension in sorted({r["dimension"] for r in rows}-{"clean"}):
            shifted=[r for r in rows if r["dimension"]==dimension]
            pairs=[(clean.get((r["base_task_id"],r["init_id"])),completed.get((method,r["episode_id"]))) for r in shifted]
            observed=[(a,b) for a,b in pairs if a is not None and b is not None]
            clean_pairs.append({"method":method,"dimension":dimension,"planned_pairs":len(pairs),"completed_pairs":len(observed),
                                "complete_pairs":len(observed)==len(pairs),
                                "clean_minus_shifted_rate":sum(int(a["success"])-int(b["success"]) for a,b in observed)/len(observed) if observed else None})
    paired={}
    for method in sorted(set(methods)-{"frozen"}):
        if "frozen" not in methods or not overall[method]["complete_valid_coverage"] or not overall["frozen"]["complete_valid_coverage"]:
            paired[method]={"status":"incomplete_coverage","paired_gain":None}; continue
        if missing_reset_proofs:
            paired[method]={"status":"missing_reset_proof","paired_gain":None};continue
        clusters=defaultdict(list)
        for row in rows:
            eid=row["episode_id"]
            clusters[row["base_task_id"]].append(int(completed[(method,eid)]["success"])-int(completed[("frozen",eid)]["success"]))
        means=np.array([np.mean(v) for v in clusters.values()])
        rng=np.random.default_rng(42)
        bootstrap=means[rng.integers(0,len(means),(2000,len(means)))].mean(axis=1)
        paired[method]={"status":"complete","paired_task_macro_gain":float(means.mean()),
                        "task_cluster_bootstrap_95":np.quantile(bootstrap,[.025,.975]).tolist(),"task_clusters":len(means),
                        "few_task_clusters":len(means)<10}
    report = {"protocol": rows[0]["protocol"], "plan_id": rows[0]["plan_id"],
              "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "methods": methods, "overall": overall, "by_subtype_severity": table,
              "model_fingerprints": sorted(fingerprints),
              "missing_reset_proof_records":missing_reset_proofs,
              "by_texture_asset":[{"method":k[0],"asset_id":k[1],**counts(v)} for k,v in sorted(asset_groups.items())],
              "matched_clean_drop":clean_pairs,"paired_vs_frozen":paired,
              "recovered_error_records": len(set(completed) & set(errors)),
              "rate_denominator": "completed valid episodes only; errors, invalid cases and pending shown separately",
              "aggregation": "dimension_macro_rate gives equal weight to observed shifted dimensions; inspect coverage before comparison"}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
