"""Per-environment canvas (curiosity_plan §3.1): every image embedder, world model and decoder
sees the unlabelled frame resized to this fixed H x W x C, so the patch grid is fixed per env.

    canvas = to_canvas(frame=frame, env_name="android")      # (256, 112, 3) uint8
    frame = frame_for_embedding(obs=obs, info=info)           # info["raw_frame"] if the env has one
"""
import numpy as np
from PIL import Image

# (H, W, C). GameBoy: native grey; Android / Web: RGB, downscaled.
CANVAS = {"gameboy": (144, 160, 1), "android": (256, 112, 3), "web": (192, 256, 3)}


def canvas_shape(*, env_name: str) -> tuple:
    if env_name not in CANVAS:
        raise ValueError(f"No canvas for env {env_name!r}; known: {sorted(CANVAS)}")
    return CANVAS[env_name]


def frame_for_embedding(*, obs: dict, info: dict):
    """The unlabelled screenshot where the env has one (set-of-mark labels are not content)."""
    raw = (info or {}).get("raw_frame")
    return raw if raw is not None else obs["frame"]


def _as_array(frame) -> np.ndarray:
    if hasattr(frame, "pil"):           # cusi_practice.records.EncodedImage
        frame = frame.pil()
    if isinstance(frame, Image.Image):
        frame = np.asarray(frame.convert("RGB"))
    return np.asarray(frame, dtype=np.uint8)


def to_canvas(*, frame, env_name: str) -> np.ndarray:
    """frame (HxW, HxWx1/3/4 uint8, PIL or EncodedImage) -> the env's canvas, H x W x C uint8."""
    h, w, c = canvas_shape(env_name=env_name)
    x = _as_array(frame)
    if x.ndim == 2:
        x = x[:, :, None]
    if x.shape[2] == 4:
        x = x[:, :, :3]
    if c == 1:
        # GameBoy frames are grey repeated to 3 channels: take one channel (exact, no rounding).
        x = x[:, :, :1] if x.shape[2] == 1 or np.array_equal(x[:, :, 0], x[:, :, 1]) else \
            np.asarray(Image.fromarray(x).convert("L"))[:, :, None]
    elif x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    if x.shape[:2] != (h, w):
        img = Image.fromarray(x[:, :, 0] if c == 1 else x)
        x = np.asarray(img.resize((w, h), Image.BOX), dtype=np.uint8)
        if x.ndim == 2:
            x = x[:, :, None]
    return x
