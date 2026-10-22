"""A fixed GR1 evaluation matrix with real assets and predeclared compositions."""
from .catalogue_v2 import SUBTYPES, CATEGORICAL, SINGLE_SAMPLE, make_variant, local_rng
from .assets import asset_reference, asset_pool
from .protocol import canonical

PROTOCOL_V3 = "robocasa-gr1-plus-v0.3"
CORE = {d:tuple(s for s in types if d not in ("layout", "robot_state") and (d,s) != ("language","commonsense"))
        for d,types in SUBTYPES.items()}
CORE = {d:s for d,s in CORE.items() if s}
COMPOSITIONS = (
    (("camera","orbit"), ("background","surface_texture")),
    (("background","scene_texture"), ("light","diffuse")),
    (("camera","orientation"), ("noise","gaussian_blur")),
    (("background","surface_texture"), ("noise","fog")),
)
HOUSEHOLD = {"development": ("bottle", "carton", "block"), "confirmation": ("mug", "jar", "can")}

def make_component(dimension, subtype, index, severity, sample_seed, split, asset_id=None):
    spec = make_variant(dimension, subtype, index, severity, sample_seed)
    if dimension == "background":
        if asset_id is None:
            raise ValueError("v3 background requires an explicit locked asset")
        if asset_id not in asset_pool(split):
            raise ValueError("Texture asset belongs to another split")
        spec["parameters"] = {"asset": asset_reference(asset_id), "contrast": [.2,.4,.6,.8,1.][severity-1],
                              "source": "polyhaven_cc0", "texrepeat": [2.,2.]}
    elif (dimension,subtype) == ("layout","clutter"):
        pool = HOUSEHOLD.get(split)
        if pool is None:
            raise ValueError("Physical clutter cannot be calibration data")
        p = spec["parameters"]
        rng = local_rng(sample_seed, "authored_household_assets", index)
        p["asset_ids"] = [pool[int(rng.integers(len(pool)))] for _ in range(p["count"])]
        p["shape"] = "authored_household"
        p["asset_license"] = "CC0-1.0"
    return spec

def make_variant_v3(recipes, index=0, severity=1, sample_seed=42, split="confirmation", asset_id=None):
    if type(sample_seed) is not int or sample_seed<0 or split not in ("calibration","development","confirmation"):
        raise ValueError("Use an integer recipe seed and known split")
    recipes = tuple(tuple(r) for r in recipes)
    if not recipes:
        if index!=0 or severity!=1: raise ValueError("Clean has one nominal variant")
        components = [make_variant("clean", sample_seed=sample_seed)]
        dimension, subtype = "clean", "nominal"
    else:
        dimensions = [d for d,s in recipes]
        if len(set(dimensions)) != len(dimensions) or len(recipes) > 2:
            raise ValueError("Use at most two distinct perturbation dimensions")
        if len(recipes)>1 and recipes not in COMPOSITIONS:
            raise ValueError("Composition is not in the fixed v3 matrix")
        components = [make_component(d,s,index,severity,sample_seed,split,asset_id if d=="background" else None) for d,s in recipes]
        dimension = dimensions[0] if len(recipes)==1 else "composition"
        subtype = recipes[0][1] if len(recipes)==1 else "+".join(d+"_"+s for d,s in recipes)
    return {"dimension":dimension,"subtype":subtype,"index":index,"severity":severity,
            "sample_seed":sample_seed,"components":components}

def validate_variant_v3(spec, split):
    keys = {"dimension","subtype","index","severity","sample_seed","components"}
    if set(spec) != keys or not spec["components"]:
        raise ValueError("Invalid v3 variant schema")
    components = spec["components"]
    recipes = [(c["dimension"],c["subtype"]) for c in components if c["dimension"]!="clean"]
    asset_ids = [c["parameters"]["asset"]["id"] for c in components if c["dimension"]=="background"]
    rebuilt = make_variant_v3(recipes, spec["index"],spec["severity"],spec["sample_seed"],split,asset_ids[0] if asset_ids else None)
    if canonical(spec) != canonical(rebuilt):
        raise ValueError("v3 parameters do not match their recipe, split or pinned assets")
    return rebuilt
