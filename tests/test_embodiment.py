import numpy as np
import pytest
from robocasa_gr1_plus.embodiment import (PARTS, encode_gr1_state, decode_gr1_action,
                                         action_dict, format_observation, IMAGE_KEY, LANGUAGE_KEY)

def test_gr1_preprocessing_matches_official_client_formula():
    raw=np.linspace(-1,1,29,dtype=np.float32)
    chunks=np.split(raw,[7,14,20,26])
    reference=np.concatenate([np.concatenate([np.sin(x),np.cos(x)]) for x in chunks])
    encoded=encode_gr1_state(raw)
    np.testing.assert_array_equal(encoded[0,:58],reference)
    assert encoded.shape==(1,64) and np.all(encoded[0,58:]==0)
    # Regression: LIBERO's interleaved sin/cos format is wrong for GR1.
    assert not np.array_equal(reference,np.stack([np.sin(raw),np.cos(raw)],axis=-1).reshape(-1))

def test_continuous_hands_and_all_29_absolute_targets_are_preserved():
    norm=np.linspace(-2,2,64,dtype=np.float32)[None]
    stats={'min':[-2]*29,'max':[3]*29,'mask':[True]*28+[False]}
    result=decode_gr1_action(norm,stats)
    reference=(np.clip(norm[:,:29],-1,1)+1)/2*5-2
    reference[:,-1]=np.clip(norm[:,28],-1,1)
    np.testing.assert_allclose(result,reference,atol=1e-6)
    output=action_dict(result[0])
    np.testing.assert_array_equal(np.concatenate(list(output.values())),result[0])
    assert [v.shape for v in output.values()]==[(7,),(7,),(6,),(6,),(3,)]
    assert len(set(result[0,14:26]))>2
    with pytest.raises(ValueError,match='29 action'): decode_gr1_action(norm,{'min':[0]*7,'max':[1]*7})
    with pytest.raises(ValueError): action_dict(np.zeros(7))

def test_exact_official_crop_key_and_prefix_required():
    raw={IMAGE_KEY:np.zeros((256,256,3),np.uint8),LANGUAGE_KEY:'unlocked_waist: put the cup in the drawer'}
    raw.update({'state.'+name:np.zeros(n) for name,n in PARTS})
    obs=format_observation(raw)
    assert obs['wrist'] is None and obs['state'].shape==(29,)
    raw[LANGUAGE_KEY]='put cup'
    with pytest.raises(ValueError): format_observation(raw)
