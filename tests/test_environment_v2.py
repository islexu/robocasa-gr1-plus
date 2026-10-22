"""Fail-fast protocol guards, before any simulator assets or model are loaded."""
import copy
import json
import pytest
from robocasa_gr1_plus.protocol import digest
from robocasa_gr1_plus.protocol_v2 import make_plan_v2
from robocasa_gr1_plus.environment import GR1PlusEnvironment

def test_environment_rejects_checkpoint_adaptation_and_manifest_label_tampering(tmp_path):
    make_plan_v2(tmp_path,mode="checkpoint")
    row=json.loads((tmp_path/"confirmation.jsonl").read_text().splitlines()[1])
    with pytest.raises(ValueError,match="frozen-only"):
        GR1PlusEnvironment({"methods":["localized"]},row)
    changed=copy.deepcopy(row)
    changed["dimension"]="noise"
    changed["episode_id"]=digest({k:v for k,v in changed.items() if k!="episode_id"})[:20]
    with pytest.raises(ValueError,match="labels"):
        GR1PlusEnvironment({},changed)
    changed=copy.deepcopy(row); changed["seed"]+=1
    with pytest.raises(ValueError,match="identity"):
        GR1PlusEnvironment({},changed)
