import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from robocasa_gr1_plus.dit4dit import DiT4DiTClient,checkpoint_identity,SERVER_PROTOCOL,reset_model
from robocasa_gr1_plus.embodiment import encode_gr1_state,decode_gr1_action

def checkpoint(tmp_path):
    root=tmp_path/"run"; (root/"checkpoints").mkdir(parents=True)
    path=root/"checkpoints/fixture.pt";path.write_bytes(b"test fixture, not model weights")
    (root/"config.yaml").write_text("fixture: true\n")
    stats={"min":[-2.]*29,"max":[3.]*29,"mask":[True]*28+[False]}
    (root/"dataset_statistics.json").write_text(json.dumps({"gr1":{"action":stats}}))
    return path,stats

def root_fixture(tmp_path):
    root=tmp_path/"dit4dit"; (root/"deployment/model_server/tools").mkdir(parents=True)
    (root/"deployment/model_server/tools/websocket_policy_client.py").write_text("# fixture\n")
    return root

class Client:
    def __init__(self,path):
        self.metadata={"protocol":SERVER_PROTOCOL,"embodiment":"robocasa_gr1","action_dim":29,
                       "checkpoint":checkpoint_identity(path),"sources_sha256":"1"*64,"precision":"bfloat16"}
        self.requests=[];self.closed=False;self.failure=False
        self.actions=np.linspace(-1,1,16*32).reshape(1,16,32)
    def get_server_metadata(self): return self.metadata
    def predict_action(self,request):
        self.requests.append(request)
        if request.get("type")=="reset": return {"ok":True}
        return {"ok":False,"error":"fixture failure"} if self.failure else {"ok":True,"data":{"normalized_actions":self.actions}}
    def close(self): self.closed=True

def test_official_socket_request_matches_gr1_preprocessing_without_exposing_clean_or_future_frames(tmp_path):
    path,stats=checkpoint(tmp_path); transport=Client(path)
    policy=DiT4DiTClient(root_fixture(tmp_path),path,client=transport)
    state=np.linspace(-1,1,29,dtype=np.float32)
    obs={"primary":np.full((256,256,3),93,np.uint8),"state":state,"language":"unlocked_waist: pick the cup",
         "clean_oracle":np.zeros((256,256,3),np.uint8),"future_image":"forbidden"}
    policy.reset(42); actions=policy.predict(obs)
    query=transport.requests[-1]
    assert set(query)=={"examples"}
    example=query["examples"][0]
    assert set(example)=={"image","lang","state"} and example["image"][0].shape==(224,224,3)
    np.testing.assert_array_equal(example["state"],encode_gr1_state(state))
    np.testing.assert_array_equal(actions,decode_gr1_action(transport.actions[0],stats))
    assert actions.shape==(16,29) and len(set(actions[:,14]))>2
    transport.failure=True
    with pytest.raises(RuntimeError,match="inference error"): policy.predict(obs)
    policy.close(); assert transport.closed

def test_wrong_checkpoint_or_libero_server_is_rejected_before_first_action(tmp_path):
    path,_=checkpoint(tmp_path);transport=Client(path); transport.metadata["action_dim"]=7
    with pytest.raises(ValueError,match="Server identity"): DiT4DiTClient(root_fixture(tmp_path),path,client=transport)
    assert transport.closed

def test_episode_reset_clears_video_cache_and_replays_cpu_random_state():
    seed_calls=[]
    torch=SimpleNamespace(manual_seed=lambda seed:seed_calls.append(seed),cuda=SimpleNamespace(is_available=lambda:False))
    extractor=SimpleNamespace(_cached_hidden={"old":"frame"},_capture_hidden_enabled=False)
    model=SimpleNamespace(backbone_interface=SimpleNamespace(extractor=extractor))
    reset_model(model,42,torch); value=np.random.random()
    reset_model(model,42,torch)
    assert np.random.random()==value and seed_calls==[42,42]
    assert extractor._cached_hidden=={} and extractor._capture_hidden_enabled
