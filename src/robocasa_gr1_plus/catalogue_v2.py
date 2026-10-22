"""Explicit GR1 presets inspired by LIBERO-Plus; strength is not empirical difficulty."""
from __future__ import annotations
import hashlib
import numpy as np
from .protocol import canonical

PROTOCOL_V2 = "robocasa-gr1-plus-v0.2"
SUBTYPES = {
    "camera": ("distance", "orbit", "orientation"),
    "background": ("scene_texture", "surface_texture"),
    "light": ("diffuse", "direction", "specular", "shadow"),
    "noise": ("motion_blur", "gaussian_blur", "zoom_blur", "fog", "glass_blur"),
    "language": ("paraphrase", "distraction", "commonsense"),
    "robot_state": ("joint_vector",),
    "layout": ("target_pose", "clutter"),
}
CATEGORICAL = {("language", s) for s in SUBTYPES["language"]} | {("light", "shadow")}
SINGLE_SAMPLE = {("camera", "distance"), ("light", "specular"), ("light", "shadow"),
                 ("noise", "gaussian_blur"), ("noise", "zoom_blur"), ("language", "commonsense")}
ARM_JOINTS = tuple(f"robot0_{side}_{joint}" for side in ("l", "r") for joint in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch",
    "wrist_yaw", "wrist_roll", "wrist_pitch"))

class InvalidPerturbation(ValueError):
    """An invalid or ineffective benchmark case, not a policy failure."""

def local_rng(*items):
    seed = int(hashlib.sha256(canonical(items).encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed)

def unit(rng, n):
    x = rng.normal(size=n)
    return x / np.linalg.norm(x)

def make_variant(dimension, subtype=None, index=0, severity=1, sample_seed=42):
    if type(index) is not int or index < 0 or type(severity) is not int or severity not in range(1, 6):
        raise ValueError("v2 requires index>=0 and severity in 1..5")
    if dimension == "clean":
        if subtype not in (None, "nominal") or severity != 1 or index != 0:
            raise ValueError("Clean has one nominal variant")
        return {"dimension": "clean", "subtype": "nominal", "index": 0,
                "severity": 1, "sample_seed": int(sample_seed), "parameters": {}}
    if dimension not in SUBTYPES or subtype not in SUBTYPES[dimension]:
        raise ValueError(f"Unknown v2 subtype: {dimension}/{subtype}")
    if (dimension, subtype) in CATEGORICAL and severity != 1:
        raise ValueError("Categorical language/shadow variants have severity=1 only")
    if (dimension, subtype) in SINGLE_SAMPLE and index != 0:
        raise ValueError("This subtype has one realization per strength")
    if dimension == "language" and index >= 3:
        raise ValueError("Language has at most three authored templates")
    rng = local_rng(sample_seed, dimension, subtype, index)
    k = severity - 1
    if dimension == "camera":
        if subtype == "distance":
            params = {"distance_scale": [1.05, 1.10, 1.20, 1.35, 1.50][k]}
        elif subtype == "orbit":
            angles = unit(rng, 2) * [5., 10., 15., 25., 35.][k]
            params = {"yaw_deg": float(angles[0]), "elevation_deg": float(angles[1])}
        else:
            params = {"rotation_deg": (unit(rng, 3) * [2., 4., 6., 8., 10.][k]).tolist()}
    elif dimension == "background":
        patterns = ("brick", "wood", "tile", "fabric", "marble", "checker")
        params = {"pattern": patterns[int(rng.integers(len(patterns)))],
                  "texture_seed": int(sample_seed), "contrast": [.2, .4, .6, .8, 1.][k],
                  "source": "authored_procedural_bitmap"}
    elif dimension == "light":
        if subtype == "diffuse":
            tint = rng.uniform(.35, 1., 3)
            tint /= tint.max()
            params = {"rgb_multiplier": ((1 - .16 * severity) * tint).tolist()}
        elif subtype == "direction":
            params = {"rotation_deg": (unit(rng, 3) * [10., 20., 30., 45., 60.][k]).tolist()}
        elif subtype == "specular":
            params = {"gain": [1.5, 2., 3., 4., 6.][k], "minimum_reference": .05}
        else:
            params = {"operation": "invert_nominal_castshadow"}
    elif dimension == "noise":
        if subtype == "motion_blur":
            params = {"radius": [5, 8, 15, 25, 35][k], "sigma": [2, 3, 6, 12, 20][k],
                      "angle_deg": float(rng.uniform(-45, 45))}
        elif subtype == "gaussian_blur":
            params = {"sigma": [1., 2., 4., 6., 10.][k]}
        elif subtype == "zoom_blur":
            params = {"max_scale": [1.11, 1.21, 1.31, 1.46, 1.56][k], "scale_step": .02}
        elif subtype == "fog":
            params = {"density": [.5, 1.5, 2.5, 3.5, 5.][k], "decay": [3., 2.5, 2., 1.6, 1.3][k]}
        else:
            params = {"sigma": [.5, .9, 1.3, 1.8, 2.5][k],
                      "max_displacement": [1, 2, 3, 4, 5][k], "iterations": [3, 3, 2, 2, 1][k]}
    elif dimension == "language":
        params = {"template_index": index % 3, "preserve_order": True,
                  "preserve_original_instruction_in_provenance": True}
    elif dimension == "robot_state":
        radius = [.05, .10, .15, .20, .25][k]
        params = {"joints": list(ARM_JOINTS), "delta": (radius * unit(rng, len(ARM_JOINTS))).tolist(),
                  "l2_radius_rad": radius}
    elif subtype == "target_pose":
        params = {"object_name": "obj", "xy_delta_m": (unit(rng, 2) * [.01, .02, .03, .04, .05][k]).tolist(),
                  "yaw_deg": float(rng.choice([-1, 1]) * [3., 6., 10., 15., 20.][k]),
                  "preserve_positive_support_and_receptacle_relations": True}
    else:
        params = {"count": [1, 1, 2, 3, 4][k], "shape": ("box", "cylinder", "sphere")[index % 3],
                  "half_size_m": [.016, .022, .022, .022, .022][k], "placement_seed": int(sample_seed),
                  "target_clearance_m": .10}
    return {"dimension": dimension, "subtype": subtype, "index": index, "severity": severity,
            "sample_seed": int(sample_seed), "parameters": params}

def validate_variant(spec):
    required = {"dimension", "subtype", "index", "severity", "sample_seed", "parameters"}
    if set(spec) != required:
        raise ValueError("Invalid v2 variant schema")
    rebuilt = make_variant(**{key: spec[key] for key in required - {"parameters"}})
    if canonical(spec) != canonical(rebuilt):
        raise ValueError("Variant parameters do not match their deterministic v2 recipe")
    return rebuilt
