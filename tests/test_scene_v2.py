import xml.etree.ElementTree as ET
import numpy as np
import pytest
from robocasa_gr1_plus.catalogue_v2 import make_variant
from robocasa_gr1_plus.scene_v2 import inject_scene_xml, patched_model_xml, texture_asset

def test_actual_bitmap_assets_reproduce_and_detect_cache_corruption(tmp_path):
    spec = make_variant("background", "surface_texture")
    path, sha = texture_asset(spec["parameters"],tmp_path)
    assert path.suffix == ".png" and len(sha) == 64
    assert texture_asset(spec["parameters"],tmp_path) == (path,sha)
    from PIL import Image
    Image.fromarray(np.zeros((256,256,3),np.uint8)).save(path)
    with pytest.raises(ValueError,match="cache content"):
        texture_asset(spec["parameters"],tmp_path)

def test_xml_hook_restores_class_even_on_failure_but_keeps_instance_hard_reset_hook(tmp_path):
    class Parent:
        def edit_model_xml(self,xml): return xml
    class Task(Parent): pass
    spec=make_variant("layout","clutter",severity=3)
    original=Task.edit_model_xml
    with pytest.raises(RuntimeError):
        with patched_model_xml(Task,spec,tmp_path):
            instance=Task()
            xml=instance.edit_model_xml("<mujoco><worldbody/></mujoco>")
            assert len(ET.fromstring(xml).findall(".//freejoint"))==2
            raise RuntimeError("constructor failure")
    assert Task.edit_model_xml is original and "edit_model_xml" not in Task.__dict__
    assert len(ET.fromstring(instance.edit_model_xml("<mujoco><worldbody/></mujoco>")).findall(".//freejoint"))==2
    assert "freejoint" not in Task().edit_model_xml("<mujoco><worldbody/></mujoco>")
