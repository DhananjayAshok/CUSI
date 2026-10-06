"""Layer 2 of cusi.state: NoveltyArchive (curiosity_plan §3.2.2), GameBoyRL's EmbedBuffer storage
generalised to StateEmbedding. It knows nothing about episodes or searches: consumers decide its
lifetime (curiosity: restore the prior every episode; search: one archive, never reset).

    archive = NoveltyArchive(metric="cosine", text_embedder=encoder.text, w_image=encoder.w_image)
    archive.novelty(e)          # 1 - max similarity (w_image = 1: EmbedBuffer's per-metric score)
    archive.add(e)              # skipped if a stored state is a near-duplicate
    snap = archive.copy(); ...; archive.restore(snap)
    archive.save(directory=d, merge=True); archive = NoveltyArchive.load(directory=d, ...)
    cell = archive.cell_of(e)   # needs cell_threshold; per-cell visit counts in archive.cell_counts

- Dedup: a state is not stored if a stored one is within dedup_threshold (0.001) in every image
  dimension and has an identical text representation.
- Compaction: above max_size (10,000) the store is reduced to compact_to (max_size // 2) by
  KMeans on the image vectors (random_state 42, n_init 1). With no text part the KMeans centres
  are kept (renormalised for cosine), exactly as EmbedBuffer; with a text part, each cluster
  keeps its member nearest the centre (a medoid), so image and text stay paired.
- Cells: cell_of assigns a state to the most similar existing cell if the similarity passes
  cell_threshold, else opens a new cell represented by that state. Cells are not compacted.
- regions: named region-novelty stores (cusi.state.regions) for the `region` scorer, copied,
  restored and saved with the archive.

Save format: <dir>/archive.pt (images, texts, cells) + the region stores' files. load() also reads
the pre-migration prior format (embed_buffer.pt + text_buffer.txt).
"""
import copy
import os
from typing import Optional
import numpy as np
import torch
from cusi.state.encoders.text import TextEmbedder
from cusi.state.regions import REGION_TYPES, TEXT_LINES, TextLineBuffer, OCRRegionBuffer
from cusi.state.state import StateEmbedding, check_metric, image_novelty, image_similarities

ARCHIVE_FILE = "archive.pt"
OLD_EMBED_FILE = "embed_buffer.pt"


class NoveltyArchive:
    def __init__(self, *, metric: str = "cosine", text_embedder: Optional[TextEmbedder] = None,
                 w_image: float = 1.0, dedup_threshold: float = 0.001, max_size: int = 10_000,
                 compact_to: Optional[int] = None, cell_threshold: Optional[float] = None) -> None:
        check_metric(metric)
        self.metric = metric
        self.text_embedder = text_embedder or TextEmbedder()
        self.w_image = 1.0 if self.text_embedder.name == "none" else float(w_image)
        self.dedup_threshold, self.max_size = dedup_threshold, max_size
        self.compact_to = compact_to if compact_to is not None else max_size // 2
        self.cell_threshold = cell_threshold
        self.images: Optional[torch.Tensor] = None     # (N, D)
        self.texts: list = []
        self.cells: list = []                          # StateEmbedding per cell
        self.cell_counts: list = []
        self.regions: dict = {}
        self.n_compactions = 0

    def __len__(self) -> int:
        return 0 if self.images is None else int(self.images.shape[0])

    # ------------------------------------------------------------ similarity

    def _similarities(self, *, e: StateEmbedding, images: torch.Tensor, texts: list) -> torch.Tensor:
        sims = image_similarities(query=e.image, matrix=images, metric=self.metric)
        if self.w_image < 1.0:
            text_sims = torch.from_numpy(self.text_embedder.similarities(query=e.text, reps=texts))
            sims = self.w_image * sims + (1 - self.w_image) * text_sims
        return sims

    def similarities(self, e: StateEmbedding) -> torch.Tensor:
        if len(self) == 0:
            return torch.zeros(0)
        return self._similarities(e=e, images=self.images, texts=self.texts)

    def max_similarity(self, e: StateEmbedding) -> float:
        return float(self.similarities(e).max()) if len(self) else 0.0

    def nearest(self, e: StateEmbedding) -> tuple:
        """(index, similarity) of the most similar stored state; (None, 0.0) if empty."""
        if len(self) == 0:
            return None, 0.0
        sims = self.similarities(e)
        i = int(sims.argmax())
        return i, float(sims[i])

    def novelty(self, e: StateEmbedding) -> float:
        """1.0 if empty. w_image = 1: EmbedBuffer.score (cosine 1 - max dot, distance min distance,
        hinge 1 - max share within 0.01). Else 1 - max combined similarity."""
        if len(self) == 0:
            return 1.0
        if self.w_image == 1.0:
            return image_novelty(query=e.image, matrix=self.images, metric=self.metric)
        return 1.0 - self.max_similarity(e)

    # ------------------------------------------------------------ storage

    def _is_duplicate(self, e: StateEmbedding) -> bool:
        if len(self) == 0:
            return False
        close = (self.images - e.image.unsqueeze(0)).abs().amax(dim=-1) < self.dedup_threshold
        if not bool(close.any()):
            return False
        if self.text_embedder.name == "none":
            return True
        return any(self.text_embedder.identical(a=e.text, b=self.texts[i]) for i in torch.where(close)[0].tolist())

    def add(self, e: StateEmbedding) -> bool:
        """Store e unless it is a near-duplicate; compact above max_size. Returns whether it was stored."""
        if len(self) == 0:
            self.images = e.image.unsqueeze(0).clone()
            self.texts = [e.text]
            return True
        if self._is_duplicate(e):
            return False
        self.images = torch.cat([self.images, e.image.unsqueeze(0).to(self.images.dtype)], dim=0)
        self.texts.append(e.text)
        if len(self) > self.max_size:
            self.compact()
        return True

    def compact(self) -> None:
        from sklearn.cluster import KMeans
        x = self.images.cpu().numpy()
        kmeans = KMeans(n_clusters=self.compact_to, random_state=42, n_init=1).fit(x)
        self.n_compactions += 1
        if self.text_embedder.name == "none":
            centres = torch.tensor(kmeans.cluster_centers_, dtype=self.images.dtype)
            if self.metric == "cosine":
                centres = torch.nn.functional.normalize(centres, dim=-1)
            self.images, self.texts = centres, [None] * len(centres)
            return
        keep = []
        for c in range(self.compact_to):
            members = np.where(kmeans.labels_ == c)[0]
            if len(members):
                d = ((x[members] - kmeans.cluster_centers_[c]) ** 2).sum(-1)
                keep.append(int(members[d.argmin()]))
        self.images = self.images[keep]
        self.texts = [self.texts[i] for i in keep]

    # ------------------------------------------------------------ cells

    def cell_of(self, e: StateEmbedding, *, count: bool = True) -> int:
        """The cell e falls in (opening a new one if no cell passes cell_threshold)."""
        if self.cell_threshold is None:
            raise ValueError("cell_of needs a NoveltyArchive built with cell_threshold")
        cid = None
        if self.cells:
            images = torch.stack([c.image for c in self.cells])
            sims = self._similarities(e=e, images=images, texts=[c.text for c in self.cells])
            best = int(sims.argmax())
            if float(sims[best]) >= self.cell_threshold:
                cid = best
        if cid is None:
            self.cells.append(e)
            self.cell_counts.append(0)
            cid = len(self.cells) - 1
        if count:
            self.cell_counts[cid] += 1
        return cid

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    # ------------------------------------------------------------ regions

    def region(self, name: str):
        """The named region store, created empty on first use."""
        if name not in self.regions:
            self.regions[name] = REGION_TYPES[name]()
        return self.regions[name]

    # ------------------------------------------------------------ snapshots

    def copy(self) -> dict:
        return {"images": None if self.images is None else self.images.clone(), "texts": list(self.texts),
                "cells": list(self.cells), "cell_counts": list(self.cell_counts),
                "regions": {k: v.copy() for k, v in self.regions.items()}}

    def restore(self, snapshot: dict) -> None:
        self.images = None if snapshot["images"] is None else snapshot["images"].clone()
        self.texts = list(snapshot["texts"])
        self.cells = list(snapshot["cells"])
        self.cell_counts = list(snapshot["cell_counts"])
        self.regions = {k: v.copy() for k, v in snapshot["regions"].items()}

    # ------------------------------------------------------------ files

    def save(self, *, directory: str, merge: bool = False) -> None:
        """Write the archive; merge=True first adds the entries of an archive already saved there that
        this one does not have (GameBoyRL's iterative_save)."""
        os.makedirs(directory, exist_ok=True)
        images, texts = self.images, list(self.texts)
        path = os.path.join(directory, ARCHIVE_FILE)
        if merge and os.path.exists(path) and images is not None:
            old = torch.load(path, weights_only=False)
            if old["images"] is not None:
                max_diff = (old["images"].unsqueeze(1) - images.unsqueeze(0)).abs().amax(dim=-1)
                unseen = (max_diff.amin(dim=1) >= self.dedup_threshold).tolist()
                keep = [i for i, u in enumerate(unseen) if u]
                if keep:
                    images = torch.cat([old["images"][keep], images], dim=0)
                    texts = [old["texts"][i] for i in keep] + texts
        torch.save({"images": None if images is None else images.cpu(), "texts": texts, "cells": self.cells,
                    "cell_counts": self.cell_counts, "metric": self.metric,
                    "text_embedder": self.text_embedder.name}, path)
        for buf in self.regions.values():
            buf.save(directory=directory, merge=merge)

    def load_from(self, *, directory: str) -> None:
        """Replace this archive's contents with the one saved in `directory` (new or pre-migration format)."""
        path = os.path.join(directory, ARCHIVE_FILE)
        if os.path.exists(path):
            data = torch.load(path, weights_only=False)
            if data.get("text_embedder", "none") != self.text_embedder.name:
                raise ValueError(f"{directory}: archive saved with text embedder {data.get('text_embedder')!r}, "
                                 f"this run uses {self.text_embedder.name!r}")
            self.images, self.texts = data["images"], list(data["texts"])
            self.cells, self.cell_counts = list(data.get("cells", [])), list(data.get("cell_counts", []))
        elif os.path.exists(os.path.join(directory, OLD_EMBED_FILE)):
            self.images = torch.load(os.path.join(directory, OLD_EMBED_FILE)).float().cpu()
            self.texts = [None] * len(self.images)
        else:
            raise ValueError(f"No archive found at {directory}")
        self.regions = {}
        for name, cls in REGION_TYPES.items():
            buf = cls.load(directory=directory)
            if buf is not None:
                self.regions[name] = buf
