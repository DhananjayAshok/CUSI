"""
State embeddings, the weighted image/text similarity between them, and the encoder that makes them.
The image channel sees the unlabelled frame with its text boxes masked out (mask_text), the text channel the boxes
themselves, so a text change moves only the text channel.
"""
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import torch
from cusi.envs.text_boxes import text_boxes_of
from cusi.state.canvas import frame_for_embedding
from cusi.state.encoders import ImageEmbedder, TextEmbedder

SIMILARITY_METRICS = ("cosine", "distance", "hinge")


@dataclass
class StateEmbedding:
    image: torch.Tensor            # (D,) float32, unit norm, CPU
    text: Any                      # the text embedder's representation (a BoxSet; None for `none`)
    image_name: str
    text_name: str


@dataclass
class StateRecord:
    """What a scorer sees of one state; prev_image is the previous state's image vector (world-model frame stack)."""
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
    """GameBoyRL's EmbedBuffer.score for one frame, bit for bit."""
    q = query.unsqueeze(0)
    if metric == "cosine":
        return float((1 - (q @ matrix.T).max(dim=-1).values).mean())
    if metric == "distance":
        return float(torch.cdist(q, matrix).min(dim=-1).values.mean())
    close = ((q.unsqueeze(1) - matrix.unsqueeze(0)).abs() < 0.01).float().mean(dim=-1)
    return float((1 - close.max(dim=-1).values).mean())


def combine(*, image_sim, text_sim, w_image: float):
    """w_image · image + (1 − w_image) · text; where text is undefined (both screens without text: None or NaN),
    the image similarity alone."""
    if text_sim is None or w_image >= 1.0:
        return image_sim
    if isinstance(text_sim, torch.Tensor):
        mixed = w_image * image_sim + (1 - w_image) * text_sim
        return torch.where(torch.isnan(text_sim), image_sim, mixed)
    return w_image * image_sim + (1 - w_image) * text_sim


def mask_boxes(*, frame, boxes: list) -> np.ndarray:
    """A copy of frame with every text box (with text) filled by the median colour of the 1-pixel ring around it."""
    x = np.array(frame, dtype=np.uint8, copy=True)
    if x.ndim == 2:
        x = x[:, :, None]
    h, w = x.shape[:2]
    for b in boxes:
        if not b.text:
            continue
        x0, y0, x1, y1 = (int(round(v)) for v in b.box)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        ring = []
        if y0 > 0:
            ring.append(x[y0 - 1, max(0, x0 - 1):min(w, x1 + 1)])
        if y1 < h:
            ring.append(x[y1, max(0, x0 - 1):min(w, x1 + 1)])
        if x0 > 0:
            ring.append(x[y0:y1, x0 - 1])
        if x1 < w:
            ring.append(x[y0:y1, x1])
        if not ring:
            continue
        colour = np.median(np.concatenate(ring, axis=0), axis=0)
        x[y0:y1, x0:x1] = colour.astype(np.uint8)
    return x if np.asarray(frame).ndim == 3 else x[:, :, 0]


class StateEncoder:
    def __init__(self, *, image: ImageEmbedder, text: TextEmbedder, w_image: float = 0.5,
                 metric: str = "cosine", mask_text: bool = True) -> None:
        check_metric(metric)
        self.image, self.text, self.metric, self.mask_text = image, text, metric, mask_text
        self.w_image = 1.0 if text.name == "none" else float(w_image)
        if not 0.0 <= self.w_image <= 1.0:
            raise ValueError(f"w_image must be in [0, 1], got {w_image}")

    def encode(self, *, obs_list: list, info_list: list) -> list:
        frames, boxes_list = [], []
        for o, i in zip(obs_list, info_list):
            boxes = text_boxes_of(obs=o)
            frame = frame_for_embedding(obs=o, info=i)
            if self.mask_text and boxes:
                frame = mask_boxes(frame=frame, boxes=boxes)
            frames.append(frame)
            boxes_list.append(boxes)
        images = self.image.embed(frames=frames)
        texts = self.text.represent_many(boxes_list=boxes_list)
        return [StateEmbedding(image=images[k], text=texts[k], image_name=self.image.name, text_name=self.text.name)
                for k in range(len(frames))]

    def encode_one(self, *, obs: dict, info: dict) -> StateEmbedding:
        return self.encode(obs_list=[obs], info_list=[info])[0]

    def similarity(self, *, a: StateEmbedding, b: StateEmbedding) -> float:
        s = float(image_similarities(query=a.image, matrix=b.image.unsqueeze(0), metric=self.metric)[0])
        if self.w_image < 1.0:
            s = combine(image_sim=s, text_sim=self.text.similarity(a=a.text, b=b.text), w_image=self.w_image)
        return float(s)

    def config(self) -> dict:
        return {"image_embedder": self.image.name, "text_embedder": self.text.name, "w_image": self.w_image,
                "similarity_metric": self.metric, "mask_text": self.mask_text, "image_meta": self.image.meta()}
