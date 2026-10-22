"""Renderer-only interventions and deterministic image corruptions."""
from __future__ import annotations
from contextlib import contextmanager
import numpy as np
from .catalogue_v2 import InvalidPerturbation, local_rng
from .scene_v2 import MATERIAL
from .shifts import VISUAL_FIELDS

def rotation(vector_deg):
    v = np.deg2rad(np.asarray(vector_deg, dtype=float))
    angle = np.linalg.norm(v)
    if angle == 0:
        return np.eye(3)
    x, y, z = v / angle
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)

def quat_matrix(q):
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def matrix_quat(m):
    # Symmetric eigenvector formulation handles rotations close to 180 degrees.
    k = np.array([[m[0,0]-m[1,1]-m[2,2], m[0,1]+m[1,0], m[0,2]+m[2,0], m[2,1]-m[1,2]],
                  [m[0,1]+m[1,0], m[1,1]-m[0,0]-m[2,2], m[1,2]+m[2,1], m[0,2]-m[2,0]],
                  [m[0,2]+m[2,0], m[1,2]+m[2,1], m[2,2]-m[0,0]-m[1,1], m[1,0]-m[0,1]],
                  [m[2,1]-m[1,2], m[0,2]-m[2,0], m[1,0]-m[0,1], np.trace(m)]]) / 3
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1][[3, 0, 1, 2]]
    return q if q[0] >= 0 else -q

def workspace(suite):
    points = np.asarray(suite.counter.get_bbox_points())
    return (points.min(axis=0) + points.max(axis=0)) / 2, points

@contextmanager
def visual_shift_v2(sim, spec, pivot=None):
    m, p = sim.model, spec["parameters"]
    fields = VISUAL_FIELDS + ("light_dir", "light_castshadow")
    saved = {key: getattr(m, key).copy() for key in fields}
    try:
        dim, sub = spec["dimension"], spec["subtype"]
        if dim == "camera":
            i = m.camera_name2id("egoview")
            if i < 0:
                raise InvalidPerturbation("missing_egoview_camera")
            body = m.cam_bodyid[i]
            positions = getattr(sim.data, "body_xpos", None)
            matrices = getattr(sim.data, "body_xmat", None)
            positions = sim.data.xpos if positions is None else positions
            matrices = sim.data.xmat if matrices is None else matrices
            parent_r = np.asarray(matrices[body]).reshape(3, 3)
            parent_p = positions[body]
            pos = parent_p + parent_r @ m.cam_pos[i]
            orient = parent_r @ quat_matrix(m.cam_quat[i])
            if sub == "orientation":
                orient = orient @ rotation(p["rotation_deg"])
            else:
                if pivot is None:
                    raise InvalidPerturbation("camera_requires_workspace_pivot")
                pivot = np.asarray(pivot)
                if np.linalg.norm(pos - pivot) < .01:
                    raise InvalidPerturbation("camera_at_workspace_pivot")
                if sub == "distance":
                    pos = pivot + p["distance_scale"] * (pos - pivot)
                else:
                    r = rotation([p["elevation_deg"], 0, 0]) @ rotation([0, 0, p["yaw_deg"]])
                    pos, orient = pivot + r @ (pos - pivot), r @ orient
            m.cam_pos[i] = parent_r.T @ (pos - parent_p)
            m.cam_quat[i] = matrix_quat(parent_r.T @ orient)
        elif dim == "background":
            prefixes = ("wall_", "floor_") if sub == "scene_texture" else ("table_main_",)
            ids = [i for i in range(m.ngeom) if (m.geom_id2name(i) or "").startswith(prefixes)]
            if not ids:
                raise InvalidPerturbation(f"missing_named_surfaces:{prefixes}")
            material = m.mat_name2id(MATERIAL)
            if material < 0:
                raise InvalidPerturbation("texture_material_not_compiled")
            m.geom_matid[ids] = material
            m.geom_rgba[ids] = [1, 1, 1, 1]
        elif dim == "light":
            if not m.nlight:
                raise InvalidPerturbation("missing_lights")
            if sub == "diffuse":
                m.light_diffuse[:] *= p["rgb_multiplier"]
            elif sub == "direction":
                m.light_dir[:] = m.light_dir @ rotation(p["rotation_deg"]).T
            elif sub == "specular":
                m.light_specular[:] = np.maximum(m.light_specular, p["minimum_reference"]) * p["gain"]
            else:
                m.light_castshadow[:] = 1 - m.light_castshadow
        sim.forward()
        yield
    finally:
        for key, value in saved.items():
            getattr(m, key)[:] = value
        sim.forward()

def corrupt_image_v2(image, spec, step=0):
    x = np.asarray(image)
    if spec["dimension"] != "noise":
        return x.copy()
    import cv2
    if x.dtype != np.uint8 or x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("Noise expects RGB uint8 HWC")
    p, sub = spec["parameters"], spec["subtype"]
    rng = local_rng(spec["sample_seed"], spec["index"], sub, step)
    h, w = x.shape[:2]
    if sub == "gaussian_blur":
        out = cv2.GaussianBlur(x, (0, 0), p["sigma"])
    elif sub == "motion_blur":
        radius = p["radius"]
        kernel = np.zeros((2 * radius + 1, 2 * radius + 1), np.float32)
        t = np.arange(-radius, radius + 1)
        a = np.deg2rad(p["angle_deg"])
        weights = np.exp(-t*t / (2 * p["sigma"]**2))
        np.add.at(kernel, (np.rint(radius + t*np.sin(a)).astype(int),
                          np.rint(radius + t*np.cos(a)).astype(int)), weights)
        out = cv2.filter2D(x, -1, kernel / kernel.sum())
    elif sub == "zoom_blur":
        images = [x.astype(float)]
        for scale in np.arange(1 + p["scale_step"], p["max_scale"] + 1e-9, p["scale_step"]):
            hh, ww = max(1, round(h / scale)), max(1, round(w / scale))
            y0, x0 = (h-hh)//2, (w-ww)//2
            images.append(cv2.resize(x[y0:y0+hh, x0:x0+ww], (w, h)).astype(float))
        out = np.mean(images, axis=0)
    elif sub == "fog":
        field = np.zeros((h, w))
        weight = 0.
        for k, n in enumerate((2, 4, 8, 16, 32)):
            gain = p["decay"] ** (-k)
            field += gain * cv2.resize(rng.random((n, n)), (w, h), interpolation=cv2.INTER_CUBIC)
            weight += gain
        alpha = np.clip((field / weight) * p["density"] / 6 + p["density"] / 12, 0, .95)
        out = x * (1 - alpha[..., None]) + 255 * alpha[..., None]
    else:
        out = cv2.GaussianBlur(x, (0, 0), p["sigma"])
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        d = p["max_displacement"]
        for _ in range(p["iterations"]):
            dx = rng.integers(-d, d + 1, (h, w)).astype(np.float32)
            dy = rng.integers(-d, d + 1, (h, w)).astype(np.float32)
            out = cv2.remap(out, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        out = cv2.GaussianBlur(out, (0, 0), p["sigma"])
    return np.uint8(np.clip(out, 0, 255))
