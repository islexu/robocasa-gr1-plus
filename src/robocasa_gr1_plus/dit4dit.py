"""Original DiT4DiT websocket protocol, GR1 normalization, and an identified server.

No LoTTA import or adapter is required for this frozen-policy path.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
from .embodiment import encode_gr1_state,decode_gr1_action
from .protocol import digest

SERVER_PROTOCOL="gr1plus-dit4dit-frozen-v1"

def file_hash(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def checkpoint_identity(checkpoint):
    path=Path(checkpoint).expanduser().resolve()
    if path.suffix!=".pt" or not path.is_file(): raise FileNotFoundError("Use the GR1 .pt checkpoint")
    run=path.parents[1]
    return {"checkpoint_sha256":file_hash(path),"config_sha256":file_hash(run/"config.yaml"),
            "statistics_sha256":file_hash(run/"dataset_statistics.json")}

def action_statistics(checkpoint,unnorm_key=None):
    path=Path(checkpoint).expanduser().resolve().parents[1]/"dataset_statistics.json"
    statistics=json.loads(path.read_text())
    if unnorm_key is None:
        if len(statistics)!=1: raise ValueError("Set unnorm_key for checkpoints with multiple datasets")
        unnorm_key=next(iter(statistics))
    if unnorm_key not in statistics: raise ValueError("Unknown normalization key")
    stats=statistics[unnorm_key]["action"]
    decode_gr1_action(np.zeros((1,29)),stats)
    return stats

def upstream_path(root):
    root=Path(root).expanduser().resolve()
    if not (root/"deployment/model_server/tools/websocket_policy_client.py").is_file():
        raise FileNotFoundError("Set dit4dit_root to the official DiT4DiT checkout")
    sys.path.insert(0,str(root))
    return root

def sources_hash(root):
    root=Path(root)
    paths=sorted(list((root/"DiT4DiT").rglob("*.py"))+list((root/"deployment/model_server").rglob("*.py")))
    return digest({str(p.relative_to(root)):file_hash(p) for p in paths})

class DiT4DiTClient:
    def __init__(self,dit4dit_root,checkpoint,host="127.0.0.1",port=6398,unnorm_key=None,
                 client=None):
        self.stats=action_statistics(checkpoint,unnorm_key)
        identity=checkpoint_identity(checkpoint)
        root=upstream_path(dit4dit_root)
        if client is None:
            from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
            proxy_names=("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy","ALL_PROXY","all_proxy")
            proxies={k:os.environ[k] for k in proxy_names if k in os.environ}
            try: client=WebsocketClientPolicy(host,port)
            finally: os.environ.update(proxies)  # upstream client removes proxy vars while connecting
        self.client=client
        try:
            metadata=client.get_server_metadata()
            if (metadata.get("protocol")!=SERVER_PROTOCOL or metadata.get("embodiment")!="robocasa_gr1"
                    or metadata.get("action_dim")!=29 or metadata.get("checkpoint")!=identity):
                raise ValueError("Server identity differs; start gr1-plus serve-dit4dit with the same GR1 checkpoint")
            self.identity={"checkpoint":identity,"server_sources_sha256":metadata["sources_sha256"],
                           "client_sources_sha256":sources_hash(root),"sampling":"checkpoint action-model configuration",
                           "server_precision":metadata.get("precision"),"server_runtime":metadata.get("runtime"),
                           "server_parameter_dtypes":metadata.get("parameter_dtypes"),
                           "embodiment":"robocasa_gr1","normalization":self.stats}
            self.fingerprint=digest(self.identity)
        except Exception:
            client.close(); raise

    def reset(self,seed):
        response=self.client.predict_action({"type":"reset","seed":int(seed)})
        if not response.get("ok"): raise RuntimeError("Policy server reset failed")

    def predict(self,observation):
        import cv2
        image=np.asarray(observation["primary"])
        if image.shape!=(256,256,3) or image.dtype!=np.uint8: raise ValueError("Expected native cropped GR1 RGB")
        if not observation["language"].startswith("unlocked_waist: "): raise ValueError("Missing GR1 embodiment prefix")
        example={"image":[cv2.resize(image,(224,224),interpolation=cv2.INTER_AREA)],
                 "lang":observation["language"],"state":encode_gr1_state(observation["state"])}
        # Pinned framework ignores client DDIM kwargs. Use the same call as LoTTA:
        # action sampling is defined by the checkpoint's action-model configuration.
        result=self.client.predict_action({"examples":[example]})
        if not result.get("ok"): raise RuntimeError(f"DiT4DiT inference error: {result.get('error')}")
        actions=np.asarray(result["data"]["normalized_actions"])
        if actions.ndim!=3 or actions.shape[0]!=1 or actions.shape[1]<1 or actions.shape[2]<29:
            raise ValueError("Invalid GR1 action chunk from server")
        return decode_gr1_action(actions[0],self.stats)

    def close(self): self.client.close()

def reset_model(policy,seed,torch_module):
    if type(seed) is not int or not 0<=seed<2**31: raise ValueError("Invalid episode seed")
    random.seed(seed); np.random.seed(seed); torch_module.manual_seed(seed)
    if torch_module.cuda.is_available(): torch_module.cuda.manual_seed_all(seed)
    extractor=policy.backbone_interface.extractor
    if hasattr(extractor,"_cached_hidden"): extractor._cached_hidden.clear()
    if hasattr(extractor,"_capture_hidden_enabled"): extractor._capture_hidden_enabled=True

def identified_server_type(base_server,torch_module):
    class IdentifiedServer(base_server):
        _active=False
        async def _handler(self,websocket):
            if self._active:
                await websocket.close(code=1013,reason="Use one evaluation client per policy server")
                return
            self._active=True
            try: await super()._handler(websocket)
            finally: self._active=False
        def _route_message(self,msg):
            if msg.get("type")=="reset":
                reset_model(self._policy,msg["seed"],torch_module)
                return {"ok":True,"type":"reset"}
            return super()._route_message(msg)
    return IdentifiedServer

def serve_dit4dit(dit4dit_root,checkpoint,host="127.0.0.1",port=6398,use_bf16=False,unnorm_key=None):
    root=upstream_path(dit4dit_root)
    identity=checkpoint_identity(checkpoint)
    action_statistics(checkpoint,unnorm_key)
    import torch
    from DiT4DiT.model.framework.base_framework import baseframework
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    if not torch.cuda.is_available(): raise RuntimeError("DiT4DiT model serving requires CUDA")
    policy=baseframework.from_pretrained(checkpoint)
    if use_bf16: policy=policy.to(torch.bfloat16)
    policy=policy.to("cuda").eval().requires_grad_(False)
    server=identified_server_type(WebsocketPolicyServer,torch)(policy,host=host,port=port,idle_timeout=-1,
                            metadata={"protocol":SERVER_PROTOCOL,"embodiment":"robocasa_gr1","action_dim":29,
                                      "precision":"model_cast_bfloat16" if use_bf16 else "checkpoint_parameter_dtypes",
                                      "parameter_dtypes":sorted({str(p.dtype) for p in policy.parameters()}),
                                      "runtime":{"torch":torch.__version__,"numpy":np.__version__},
                                      "checkpoint":identity,"sources_sha256":sources_hash(root)})
    server.serve_forever()
