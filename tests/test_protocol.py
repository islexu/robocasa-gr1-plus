import json
import numpy as np
import pytest
from robocasa_gr1_plus.protocol import make_plan, TASKS, VISUAL, variant

def test_plan_split_disjointness_method_pairing_and_nominal_seeds(tmp_path):
    result=make_plan(tmp_path)
    splits=[set(v) for v in result['base_task_splits'].values()]
    assert set.union(*splits)==set(TASKS)
    assert all(not splits[i]&splits[j] for i in range(3) for j in range(i))
    assert result['episodes']=={'calibration':12,'development':48,'confirmation':96}
    for split in result['base_task_splits']:
        rows=[json.loads(x) for x in (tmp_path/f'{split}.jsonl').read_text().splitlines()]
        ids=[r['episode_id'] for r in rows]
        assert len(ids)==len(set(ids))
        if split=='calibration':
            assert all(r['dimension'] in VISUAL and r['pairing']=='gr1_visual_same_state' for r in rows)
        else:
            for task in result['base_task_splits'][split]:
                subset=[r for r in rows if r['base_task_id']==task]
                assert len({r['seed'] for r in subset})==1
                assert all('pairing' not in r for r in subset)

def test_plan_lock_is_transactional_and_full_protocol_has_no_language_duplicates(tmp_path):
    make_plan(tmp_path)
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError,match='different plan'): make_plan(tmp_path,seed=123)
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    make_plan(tmp_path)
    other=tmp_path/'full'
    plan=make_plan(other,variants=6,severities=(1,2,3),trials=2)
    rows=[json.loads(x) for x in (other/'development.jsonl').read_text().splitlines()]
    # Six physical/visual dimensions x18, six language prompts, one nominal episode.
    assert len(rows)==6*(6*18+6+1)*2
    assert all(r['variant']['severity']==1 for r in rows if r['dimension']=='language')

@pytest.mark.parametrize('kwargs',[{'calibration_dimensions':('layout',)}, {'variants':7},
                                    {'trials':0},{'severities':(1,1)}, {'severities':(4,)}])
def test_invalid_protocol_requests_fail(tmp_path,kwargs):
    with pytest.raises(ValueError): make_plan(tmp_path,**kwargs)
