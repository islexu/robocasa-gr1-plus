import json
import pytest
from robocasa_gr1_plus.protocol_v2 import make_plan_v2
from robocasa_gr1_plus.report_v2 import summarize

def test_report_keeps_invalid_errors_pending_and_resolved_retries_visible(tmp_path):
    make_plan_v2(tmp_path/"plan",mode="checkpoint")
    manifest=tmp_path/"plan/confirmation.jsonl"
    rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    root=tmp_path/"results"
    (root/"episodes").mkdir(parents=True); (root/"errors").mkdir()
    for n,kind in enumerate(("completed","invalid","error","recovered")):
        row={**rows[n],"method":"frozen"}
        if kind in ("invalid","error","recovered"):
            (root/"errors"/f"{n}.json").write_text(json.dumps({**row,"status":"error","error_type":"InvalidPerturbation" if kind=="invalid" else "RuntimeError"}))
        if kind in ("completed","recovered"):
            (root/"episodes"/f"{n}.json").write_text(json.dumps({**row,"status":"completed","success":True}))
    report=summarize(manifest,root,tmp_path/"report.json")
    total=report["overall"]["frozen"]
    assert total["completed"]==2 and total["invalid"]==1 and total["error"]==1
    assert total["pending"]==500 and total["success_rate"]==1.
    assert not total["complete_valid_coverage"] and report["recovered_error_records"]==1
    (root/"episodes/duplicate.json").write_text((root/"episodes/0.json").read_text())
    with pytest.raises(ValueError,match="Duplicate"):
        summarize(manifest,root,tmp_path/"report.json")

def test_report_rejects_plan_mixing_and_adaptation_on_checkpoint_plan(tmp_path):
    make_plan_v2(tmp_path/"plan",mode="checkpoint")
    manifest=tmp_path/"plan/confirmation.jsonl"
    with pytest.raises(ValueError,match="frozen only"):
        summarize(manifest,tmp_path/"results",tmp_path/"report.json",["localized"])
    row=json.loads(manifest.read_text().splitlines()[0])
    root=tmp_path/"results/episodes"; root.mkdir(parents=True)
    (root/"bad.json").write_text(json.dumps({**row,"protocol":"different","method":"frozen","status":"completed","success":True}))
    with pytest.raises(ValueError,match="provenance"):
        summarize(manifest,root.parent,tmp_path/"report.json")
