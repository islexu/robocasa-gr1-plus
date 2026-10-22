"""Pinned, attributable assets. Downloads happen only in an explicit prepare step."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request
from .protocol import digest

def catalogue():
    path = Path(__file__).parent / "data/textures.lock.json"
    return json.loads(path.read_text())

def asset_lock_sha256():
    return digest(catalogue())

def asset_record(asset_id):
    records = [r for r in catalogue()["assets"] if r["id"] == asset_id]
    if len(records) != 1:
        raise ValueError(f"Asset not in the pinned pack: {asset_id}")
    return records[0]

def asset_reference(asset_id):
    r = asset_record(asset_id)
    return {k:r[k] for k in ("id", "sha256", "family", "license", "split")}

def asset_pool(split):
    return [r["id"] for r in catalogue()["assets"] if r["split"] == split]

def cache_root(cache=None):
    return Path(cache or os.getenv("GR1_PLUS_ASSET_CACHE", "~/.cache/robocasa-gr1-plus/assets-v1")).expanduser().resolve()

def asset_path(reference, cache=None):
    if reference != asset_reference(reference["id"]):
        raise ValueError("Asset reference differs from the pinned pack")
    record = asset_record(reference["id"])
    path = cache_root(cache) / (record["sha256"] + ".jpg")
    if not path.is_file():
        raise FileNotFoundError(f"Missing asset {record['id']}; run gr1-plus assets fetch --cache {cache_root(cache)}")
    data = path.read_bytes()
    if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"Asset checksum mismatch: {record['id']}")
    return path

def prepare_assets(cache=None, fetch=False):
    root = cache_root(cache)
    records = []
    for record in catalogue()["assets"]:
        reference = asset_reference(record["id"])
        try:
            path = asset_path(reference, root)
        except FileNotFoundError:
            if not fetch:
                raise
            root.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(record["url"], headers={"User-Agent":"robocasa-gr1-plus/0.3 (Powered by Poly Haven)"})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(record["bytes"] + 1)
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError(f"Downloaded asset checksum mismatch: {record['id']}")
            with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            try:
                temporary.replace(root / (record["sha256"] + ".jpg"))
            finally:
                temporary.unlink(missing_ok=True)
            path = asset_path(reference, root)
        records.append({"id": record["id"], "sha256": record["sha256"], "bytes": record["bytes"]})
    return {"pack_id": catalogue()["pack_id"], "asset_lock_sha256": asset_lock_sha256(),
            "credit": "Powered by Poly Haven", "cache": str(root), "verified": records}
