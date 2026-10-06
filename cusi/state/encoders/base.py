"""Image embedder interface (curiosity_plan §3.2.1). Inference only.

    emb = SomeEmbedder(env_name="gameboy", ...)
    canvas = emb.preprocess(frame)            # the env's canvas (cusi.state.canvas)
    vecs = emb.embed(frames=[f1, f2])         # torch float32 (N, output_dim), unit norm, on CPU

Embeddings are returned on the CPU so archives and replays never hold device tensors.
Checkpoints (trained cnn, fine-tuned siglip) carry metadata (embedder, env, canvas, base model),
checked on load by check_meta.
"""
import json
import os
from abc import ABC, abstractmethod
import numpy as np
import torch
from cusi.state.canvas import canvas_shape, to_canvas

META_FILENAME = "embedder_meta.json"


class ImageEmbedder(ABC):
    name: str = "image"

    def __init__(self, *, env_name: str) -> None:
        self.env_name = env_name
        self.canvas = canvas_shape(env_name=env_name)
        self.output_dim: int = 0

    def preprocess(self, frame) -> np.ndarray:
        return to_canvas(frame=frame, env_name=self.env_name)

    def canvases(self, *, frames: list) -> torch.Tensor:
        """Frames -> float32 (N, C, H, W) in [0, 255]."""
        x = np.stack([self.preprocess(f) for f in frames])
        return torch.from_numpy(x).permute(0, 3, 1, 2).float()

    @abstractmethod
    def embed(self, *, frames: list) -> torch.Tensor:
        """float32 (N, output_dim), L2-normalised, on the CPU."""

    def meta(self) -> dict:
        return {"embedder": self.name, "env": self.env_name, "canvas": list(self.canvas)}


def write_meta(*, directory: str, meta: dict) -> None:
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, META_FILENAME), "w") as f:
        json.dump(meta, f, indent=1)


def check_meta(*, directory: str, expected: dict) -> dict:
    """Error if the checkpoint in `directory` was made for another embedder / env / canvas / base model."""
    path = os.path.join(directory, META_FILENAME)
    if not os.path.exists(path):
        raise ValueError(f"{directory} has no {META_FILENAME}; not a cusi.state checkpoint")
    with open(path) as f:
        meta = json.load(f)
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"{directory}: checkpoint {key}={meta.get(key)!r}, but this run needs {value!r}")
    return meta
