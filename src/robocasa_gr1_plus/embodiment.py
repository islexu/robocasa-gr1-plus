"""The official DiT4DiT GR1 client contract, deliberately separate from LIBERO."""
import numpy as np

PARTS = (("left_arm", 7), ("right_arm", 7), ("left_hand", 6), ("right_hand", 6), ("waist", 3))
IMAGE_KEY = "video.ego_view_bg_crop_pad_res256_freq20"
LANGUAGE_KEY = "annotation.human.coarse_action"

def state_parts(observation):
    result = []
    for name, size in PARTS:
        part = np.asarray(observation["state." + name], dtype=np.float32)
        if part.shape != (size,) or not np.isfinite(part).all():
            raise ValueError(f"Expected unbatched state.{name} shape {(size,)}")
        result.append(part)
    return result

def encode_gr1_state(raw, state_dim=64):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (29,) or state_dim < 58 or not np.isfinite(raw).all():
        raise ValueError("GR1 requires 29 finite joint coordinates and at least 58 encoded channels")
    encoded, offset = [], 0
    for _, size in PARTS:
        part = raw[offset:offset + size]
        # Official GR1: [sin(part),cos(part)] per body part, not interleaved sin/cos.
        encoded.extend((np.sin(part), np.cos(part)))
        offset += size
    return np.pad(np.concatenate(encoded)[None], ((0, 0), (0, state_dim - 58)))

def decode_gr1_action(normalized, stats):
    x = np.asarray(normalized, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 29 or not np.isfinite(x).all():
        raise ValueError("GR1 expects finite [chunk, >=29] normalized actions")
    low, high = np.asarray(stats["min"]), np.asarray(stats["max"])
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    if low.shape != (29,) or high.shape != (29,) or mask.shape != (29,):
        raise ValueError("Select GR1 checkpoint normalization with exactly 29 action dimensions")
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high < low):
        raise ValueError("Invalid action min/max statistics")
    x = np.clip(x[:, :29], -1, 1)
    return np.where(mask, (x + 1) * 0.5 * (high - low) + low, x).astype(np.float32)

def action_dict(action):
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (29,) or not np.isfinite(action).all():
        raise ValueError("GR1 executes a single finite 29-D absolute joint target")
    result, offset = {}, 0
    for name, size in PARTS:
        result["action." + name] = action[offset:offset+size].copy()
        offset += size
    return result

def format_observation(raw):
    image = np.asarray(raw[IMAGE_KEY])
    if image.dtype != np.uint8 or image.shape != (256, 256, 3):
        raise ValueError(f"Official cropped GR1 image required at {IMAGE_KEY}")
    lang = raw[LANGUAGE_KEY]
    if not isinstance(lang, str) or not lang.startswith("unlocked_waist: "):
        raise ValueError("Expected official GR1 arms-and-waist language prefix")
    return {"primary": image.copy(), "wrist": None, "state": np.concatenate(state_parts(raw)),
            "language": lang}
