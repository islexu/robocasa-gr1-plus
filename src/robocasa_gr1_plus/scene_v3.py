"""Photo-texture materials and authored household clutter in native MuJoCo."""
from pathlib import Path
import hashlib
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
from .assets import asset_path, cache_root
from .scene_v2 import MATERIAL, CLUTTER

def photo_texture(parameters, cache=None):
    from PIL import Image
    source=asset_path(parameters["asset"],cache)
    with Image.open(source) as image:
        pixels=np.asarray(image.convert("RGB").resize((512,512),Image.Resampling.LANCZOS),dtype=float)
    pixels=np.uint8(np.clip(127.5+parameters["contrast"]*(pixels-127.5),0,255))
    sha=hashlib.sha256(pixels.tobytes()).hexdigest()
    root=cache_root(cache)/"derived"; root.mkdir(parents=True,exist_ok=True)
    path=root/(sha+".png")
    if not path.exists():
        with tempfile.NamedTemporaryFile(dir=root,suffix=".png",delete=False) as f: temporary=Path(f.name)
        try:
            Image.fromarray(pixels).save(temporary); temporary.replace(path)
        finally: temporary.unlink(missing_ok=True)
    with Image.open(path) as image:
        if hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()!=sha:
            raise ValueError("Derived photo texture cache is corrupted")
    return path,sha

# unit-shape boxes/cylinders fit within [-1,1]^3; scale is half_size_m.
HOUSEHOLD_GEOMS={
    "block":(("box",(1,1,1),(0,0,0)),),
    "carton":(("box",(.6,.45,1),(0,0,0)),),
    "can":(("cylinder",(.65,1),(0,0,0)),),
    "bottle":(("cylinder",(.45,.7),(0,0,-.3)),("cylinder",(.22,.3),(0,0,.7))),
    "jar":(("cylinder",(.75,.85),(0,0,-.15)),("cylinder",(.8,.15),(0,0,.85))),
    "mug":(("cylinder",(.6,1),(0,0,0)),("box",(.2,.15,.65),(.8,0,0))),
}

def inject_scene_v3(xml,components,cache=None):
    root=ET.fromstring(xml)
    for spec in components:
        dim,sub,p=spec["dimension"],spec["subtype"],spec["parameters"]
        if dim=="background":
            path,_=photo_texture(p,cache)
            asset=root.find("asset")
            if asset is None: asset=ET.SubElement(root,"asset")
            if root.find(f".//*[@name='{MATERIAL}']") is not None: raise ValueError("Duplicate v3 material")
            ET.SubElement(asset,"texture",name="gr1plus_v3_photo",type="2d",file=str(path))
            ET.SubElement(asset,"material",name=MATERIAL,texture="gr1plus_v3_photo",texrepeat="2 2",texuniform="true",rgba="1 1 1 1")
        elif (dim,sub)==("layout","clutter"):
            world=root.find("worldbody")
            if world is None: raise ValueError("Missing worldbody")
            h=p["half_size_m"]
            for n,asset_id in enumerate(p["asset_ids"]):
                name=CLUTTER+str(n)
                if root.find(f".//*[@name='{name}']") is not None: raise ValueError("Duplicate clutter body")
                body=ET.SubElement(world,"body",name=name,pos=f"{20+n} 20 5")
                ET.SubElement(body,"freejoint",name=name+"_joint")
                for k,(kind,size,pos) in enumerate(HOUSEHOLD_GEOMS[asset_id]):
                    ET.SubElement(body,"geom",name=name+f"_geom{k}",type=kind,
                                  size=" ".join(str(h*x) for x in size),pos=" ".join(str(h*x) for x in pos),
                                  density="150",friction="1 .005 .0001",contype="1",conaffinity="1",group="1",rgba=".18 .55 .8 1")
    return ET.tostring(root,encoding="unicode")

