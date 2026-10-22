import hashlib
import json
import os
import pytest
from robocasa_gr1_plus.assets import catalogue,asset_pool,asset_reference,asset_path,prepare_assets

def test_asset_registry_has_byte_hashes_cc0_provenance_and_disjoint_families():
    pack=catalogue(); assert len(pack["assets"])==8
    assert sum(a["bytes"] for a in pack["assets"])==4424108
    families=[]
    for split in ("calibration","development","confirmation"):
        families.append({a["family"] for a in pack["assets"] if a["split"]==split})
    assert not any(a&b for i,a in enumerate(families) for b in families[:i])
    for a in pack["assets"]:
        assert a["license"]=="CC0-1.0" and len(a["sha256"])==64 and a["url"].startswith("https://dl.polyhaven.org/")
        assert a["source_url"].startswith("https://polyhaven.com/a/")

def test_runtime_never_downloads_missing_assets_and_rejects_corrupt_cache(tmp_path):
    asset=asset_reference(asset_pool("confirmation")[0])
    with pytest.raises(FileNotFoundError,match="assets fetch"): asset_path(asset,tmp_path)
    (tmp_path/(asset["sha256"]+".jpg")).write_bytes(b"corrupted")
    with pytest.raises(ValueError,match="checksum"): asset_path(asset,tmp_path)

def test_real_downloaded_pack_hashes_when_explicitly_prepared():
    if not os.getenv("GR1_PLUS_ASSET_CACHE"): pytest.skip("Set GR1_PLUS_ASSET_CACHE to test real downloaded assets")
    report=prepare_assets(fetch=False)
    assert len(report["verified"])==8
