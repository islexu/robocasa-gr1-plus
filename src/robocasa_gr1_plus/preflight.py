"""A manifest-bound validation gate, shared by both evaluation entry points."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .protocol import source_fingerprint
from .catalogue_v2 import InvalidPerturbation
from .runtime import runtime_versions

def manifest_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def free_positions(sim):
    return {int(i):sim.data.qpos[int(sim.model.jnt_qposadr[i]):int(sim.model.jnt_qposadr[i])+3].copy()
            for i in range(sim.model.njnt) if int(sim.model.jnt_type[i])==0}

def check_hold_stability(before,sim,success):
    if success: raise InvalidPerturbation("passive_hold_completes_task")
    after=free_positions(sim)
    displacements={str(k):float(np.linalg.norm(after[k]-v)) for k,v in before.items()}
    if any(d>.03 for d in displacements.values()):
        raise InvalidPerturbation("free_object_drift_exceeds_3cm_during_hold")
    return displacements

def verify_validation(manifest,validation):
    rows=[json.loads(s) for s in Path(manifest).read_text().splitlines() if s.strip()]
    report=json.loads(Path(validation).read_text())
    if not rows or report.get("manifest_sha256")!=manifest_hash(manifest):
        raise ValueError("Validation report is for a different manifest")
    source=source_fingerprint()
    if report.get("extension_sha256")!=source or any(r.get("extension_sha256")!=source for r in rows):
        raise ValueError("Validation source changed; revalidate this manifest on its original source")
    if report.get("runtime_versions")!=runtime_versions():
        raise ValueError("Simulator/image dependency versions changed; revalidate")
    records=report.get("records",[])
    if (not report.get("complete_valid_coverage") or report.get("errors",0) or report.get("invalid",0)
            or len(records)!=len(rows) or len({r["episode_id"] for r in records})!=len(rows)
            or {r["episode_id"] for r in records}!={r["episode_id"] for r in rows}
            or any(r.get("status")!="passed" for r in records)):
        raise ValueError("Full valid preflight coverage is required; inspect errors/invalid/pending without dropping cases")
    by_id={r["episode_id"]:r for r in records}
    if any(r.get("variant")!=by_id[r["episode_id"]].get("variant") for r in rows):
        raise ValueError("Validated variant parameters differ from the manifest")
    for row in rows:
        proof=by_id[row["episode_id"]]
        provenance=proof.get("environment_provenance",{})
        if (len(proof.get("initial_state_sha256",""))!=64 or len(proof.get("initial_image_rgb_sha256",""))!=64
                or proof.get("hold_steps",0)<1 or provenance.get("commit")!=row["upstream_commit"]
                or provenance.get("env_name")!=row["env_name"]):
            raise ValueError("Validation lacks native source, reset/image hashes or hold proof")
        if row.get("pairing"):
            pair=proof.get("pair_proof",{})
            if (pair.get("kind")!="gr1_visual_same_state" or pair.get("state_sha256")!=proof["initial_state_sha256"]
                    or not pair.get("native_wrapper_observation") or pair.get("clean_rollout_executed") is not False):
                raise ValueError("Calibration validation lacks same-state native visual pair proof")
    return {"manifest_sha256":manifest_hash(manifest),"validation_sha256":manifest_hash(validation),
            "extension_sha256":source,"runtime_versions":runtime_versions(),"episodes":len(rows)}
