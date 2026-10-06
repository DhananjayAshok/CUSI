"""StateEmbedding, the similarity between states, and StateEncoder (curiosity_plan §3.2.1).

    encoder = StateEncoder(image=build_image_embedder(...), text=build_text_embedder(...), w_image=0.5)
    [e1, e2] = encoder.encode(obs_list=[obs1, obs2], info_list=[info1, info2])
    encoder.similarity(a=e1, b=e2)

similarity(a, b) = w_image * sim_image(a, b) + (1 - w_image) * sim_text(a, b), w_image forced to 1
when the text embedder is `none`. sim_image by metric (as GameBoyRL's EmbedBuffer):
    cosine    dot product of the unit vectors
    distance  1 - ||a - b|| / 2  (unit vectors: in [0, 1]; novelty with w_image = 1 is the raw min
              distance, exactly as EmbedBuffer)
    hinge     share of dimensions within 0.01
"""
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import torch
from cusi.state.canvas import frame_for_embedding
from cusi.state.encoders import ImageEmbedder, TextEmbedder
from cusi.state.text import element_lines

SIMILARITY_METRICS = ("cosine", "distance", "hinge")


@dataclass
class StateEmbedding:
    image: torch.Tensor            # (D,) float32, unit norm, CPU
    text: Any                      # the text embedder's representation (None for `none`)
    image_name: str
    text_name: str


@dataclass
class StateRecord:
    """What a scorer sees of one state: the observation, its info, and its embedding.
    prev_image: the image vector of the state before it (world-model frame stack), if any."""
    obs: dict
    info: dict
    embedding: Optional[StateEmbedding] = None
    prev_image: Optional[torch.Tensor] = None
    extra: dict = field(default_factory=dict)


def check_metric(metric: str) -> None:
    if metric not in SIMILARITY_METRICS:
        raise ValueError(f"similarity_metric must be one of {SIMILARITY_METRICS}, got {metric!r}")


def image_similarities(*, query: torch.Tensor, matrix: torch.Tensor, metric: str) -> torch.Tensor:
    """(N,) similarity of query (D,) to every row of matrix (N, D)."""
    if metric == "cosine":
        return (query.unsqueeze(0) @ matrix.T).squeeze(0)
    if metric == "distance":
        return 1.0 - torch.cdist(query.unsqueeze(0), matrix).squeeze(0) / 2.0
    return ((matrix - query.unsqueeze(0)).abs() < 0.01).float().mean(dim=-1)


def image_novelty(*, query: torch.Tensor, matrix: torch.Tensor, metric: str) -> float:
    """EmbedBuffer.score for one frame, bit for bit: cosine 1 - max dot, distance min distance,
    hinge 1 - max share-within-0.01."""
    q = query.unsqueeze(0)
    if metric == "cosine":
        return float((1 - (q @ matrix.T).max(dim=-1).values).mean())
    if metric == "distance":
        return float(torch.cdist(q, matrix).min(dim=-1).values.mean())
    close = ((q.unsqueeze(1) - matrix.unsqueeze(0)).abs() < 0.01).float().mean(dim=-1)
    return float((1 - close.max(dim=-1).values).mean())


class StateEncoder:
    def __init__(self, *, image: ImageEmbedder, text: TextEmbedder, w_image: float = 0.5,
                 metric: str = "cosine") -> None:
        check_metric(metric)
        self.image, self.text, self.metric = image, text, metric
        self.w_image = 1.0 if text.name == "none" else float(w_image)
        if not 0.0 <= self.w_image <= 1.0:
            raise ValueError(f"w_image must be in [0, 1], got {w_image}")

    def encode(self, *, obs_list: list, info_list: list) -> list:
        frames = [frame_for_embedding(obs=o, info=i) for o, i in zip(obs_list, info_list)]
        images = self.image.embed(frames=frames)
        texts = self.text.represent_many(lines_list=[element_lines(texts=o.get("texts", {})) for o in obs_list])
        return [StateEmbedding(image=images[k], text=texts[k], image_name=self.image.name, text_name=self.text.name)
                for k in range(len(frames))]

    def encode_one(self, *, obs: dict, info: dict) -> StateEmbedding:
        return self.encode(obs_list=[obs], info_list=[info])[0]

    def similarity(self, *, a: StateEmbedding, b: StateEmbedding) -> float:
        s = float(image_similarities(query=a.image, matrix=b.image.unsqueeze(0), metric=self.metric)[0])
        if self.w_image < 1.0:
            s = self.w_image * s + (1 - self.w_image) * float(self.text.similarities(query=a.text, reps=[b.text])[0])
        return s

    def config(self) -> dict:
        return {"image_embedder": self.image.name, "text_embedder": self.text.name, "w_image": self.w_image,
                "similarity_metric": self.metric, "image_meta": self.image.meta()}
