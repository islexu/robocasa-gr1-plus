"""Authored bitmap materials and physical primitives, injected before MJCF compilation."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import threading
from types import MethodType
import xml.etree.ElementTree as ET
import numpy as np
from .catalogue_v2 import local_rng

MATERIAL = "gr1plus_v2_material"
CLUTTER = "gr1plus_v2_clutter_"
_MODEL_LOCK = threading.RLock()

def texture_pixels(parameters, size=256):
    """Small original procedural texture set; no external benchmark assets."""
    p = parameters
    rng = local_rng(p["texture_seed"], p["pattern"])
    y, x = np.mgrid[:size, :size] / size
    noise = rng.normal(0, .035, (size, size))
    pattern = p["pattern"]
    if pattern == "brick":
        row = np.floor(y * 12)
        mortar = ((y * 12 % 1) < .09) | (((x * 6 + row % 2 / 2) % 1) < .05)
        value = np.where(mortar, .9, .38) + noise
        tint = np.array([1., .55, .35])
    elif pattern == "wood":
        value = .5 + .22 * np.sin(x * 150 + 5 * np.sin(y * 25)) + noise
        tint = np.array([1., .72, .4])
    elif pattern == "tile":
        mortar = (x * 10 % 1 < .06) | (y * 10 % 1 < .06)
        value = np.where(mortar, .18, .8) + noise
        tint = np.array([.5, .8, 1.])
    elif pattern == "fabric":
        value = .5 + .16 * np.sin(x * 650) * np.cos(y * 650) + noise
        tint = np.array([.7, 1., .65])
    elif pattern == "marble":
        value = .6 + .25 * np.sin(35 * x + 7 * np.sin(10 * y)) + noise
        tint = np.array([.85, .8, 1.])
    elif pattern == "checker":
        value = .25 + .5 * ((np.floor(x * 12) + np.floor(y * 12)) % 2) + noise
        tint = np.ones(3)
    else:
        raise ValueError(f"Unknown texture: {pattern}")
    image = value[..., None] * tint
    image = .5 + p["contrast"] * (image - .5)
    return np.uint8(np.clip(image * 255, 0, 255))

def texture_asset(parameters, cache=None):
    from PIL import Image
    pixels = texture_pixels(parameters)
    sha = hashlib.sha256(pixels.tobytes()).hexdigest()
    cache = Path(cache or os.getenv("GR1_PLUS_TEXTURE_CACHE", "~/.cache/robocasa-gr1-plus/v2")).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (sha + ".png")
    if not path.exists():
        import tempfile
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".png", delete=False) as f:
            temporary = Path(f.name)
        try:
            Image.fromarray(pixels).save(temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    # Do not accept a corrupted or user-replaced cached file under a content hash.
    with Image.open(path) as image:
        if hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest() != sha:
            raise ValueError(f"Texture cache content differs from recipe: {path}")
    return path.resolve(), sha

def inject_scene_xml(xml, spec, cache=None):
    dim, subtype, p = spec["dimension"], spec["subtype"], spec["parameters"]
    if dim != "background" and (dim, subtype) != ("layout", "clutter"):
        return xml
    root = ET.fromstring(xml)
    if root.find(f".//*[@name='{MATERIAL}']") is not None or root.find(f".//*[@name='{CLUTTER}0']") is not None:
        raise ValueError("v2 assets were already injected")
    if dim == "background":
        path, _ = texture_asset(p, cache)
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "texture", name="gr1plus_v2_texture", type="2d", file=str(path))
        ET.SubElement(asset, "material", name=MATERIAL, texture="gr1plus_v2_texture",
                      texrepeat="2 2", texuniform="true", rgba="1 1 1 1", reflectance="0")
    else:
        world = root.find("worldbody")
        if world is None:
            raise ValueError("Missing worldbody")
        for i in range(p["count"]):
            name = CLUTTER + str(i)
            body = ET.SubElement(world, "body", name=name, pos=f"{20+i} 20 5")
            ET.SubElement(body, "freejoint", name=name + "_joint")
            n = {"box": 3, "cylinder": 2, "sphere": 1}[p["shape"]]
            ET.SubElement(body, "geom", name=name + "_geom", type=p["shape"],
                          size=" ".join([str(p["half_size_m"])] * n), density="150",
                          friction="1 .005 .0001", contype="1", conaffinity="1",
                          group="1", rgba=".15 .55 .85 1")
    return ET.tostring(root, encoding="unicode")

@contextmanager
def patched_model_xml(task_class, spec, cache=None):
    """Patch only construction; retain the hook on that instance for hard resets.

    No upstream file changes, global RNG draws, or changes to task object samplers.
    The lock serializes constructions through this helper within one process.
    """
    if spec["dimension"] != "background" and (spec["dimension"], spec["subtype"]) != ("layout", "clutter"):
        yield
        return
    with _MODEL_LOCK:
        original = task_class.edit_model_xml
        own = task_class.__dict__.get("edit_model_xml")
        def edited(instance, xml):
            result = inject_scene_xml(original(instance, xml), spec, cache)
            instance.edit_model_xml = MethodType(edited, instance)
            return result
        task_class.edit_model_xml = edited
        try:
            yield
        finally:
            if own is None:
                delattr(task_class, "edit_model_xml")
            else:
                task_class.edit_model_xml = own
