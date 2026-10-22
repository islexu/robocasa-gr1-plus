import json
import pytest
from robocasa_gr1_plus.protocol import canonical
from robocasa_gr1_plus.protocol_v3 import make_plan_v3
from robocasa_gr1_plus.report_v2 import summarize

def test_complete_matched_method_rates_assets_clean_drop_and_task_bootstrap(tmp_path):
    make_plan_v3(tmp_path/"plan")
    rows=[json.loads(s) for s in (tmp_path/"plan/development.jsonl").read_text().splitlines()]
    task=rows[0]["base_task_id"]
    chosen=[next(r for r in rows if r["base_task_id"]==task and r["dimension"]==d) for d in ("clean","camera","background")]
    manifest=tmp_path/"subset.jsonl";manifest.write_text("".join(canonical(r)+"\n" for r in chosen))
    output=tmp_path/"results/episodes";output.mkdir(parents=True)
    for method in ("frozen","localized"):
        for n,row in enumerate(chosen):
            record={**row,"method":method,"status":"completed","success":n==0 or n==1 and method=="localized",
                    "model_fingerprint":"fixture","environment_provenance":{"initial_state_sha256":"0"*64,"physical_model_sha256":"1"*64}}
            (output/(method+str(n)+".json")).write_text(json.dumps(record))
    report=summarize(manifest,output.parent,tmp_path/"report.json",["frozen","localized"])
    assert report["paired_vs_frozen"]["localized"]["paired_task_macro_gain"]==pytest.approx(1/3)
    assert report["paired_vs_frozen"]["localized"]["few_task_clusters"]
    assert len(report["by_texture_asset"])==2
    assert report["overall"]["localized"]["single_dimension_macro_rate"]==.5
    camera=next(r for r in report["matched_clean_drop"] if r["method"]=="frozen" and r["dimension"]=="camera")
    assert camera["clean_minus_shifted_rate"]==1. and camera["complete_pairs"]
    path=output/"localized1.json";record=json.loads(path.read_text());record["environment_provenance"]["initial_state_sha256"]="different"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match="different initial"):summarize(manifest,output.parent,tmp_path/"report.json",["frozen","localized"])
