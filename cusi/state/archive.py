"""
The novelty archive: a deduplicated, compacting store of visited states (image vector + interned text-box keys), with
cells and region stores. Text similarity to every stored state is one sparse product plus a padded gather
(plans/next_plan.md §0.2).
"""
import os
from typing import Optional
import numpy as np
import torch
from cusi.state.encoders.text import EMPTY, BoxSet, TextEmbedder, chamfer
from cusi.state.keys import KeyStore
from cusi.state.regions import REGION_TYPES
from cusi.state.state import StateEmbedding, check_metric, combine, image_novelty, image_similarities

ARCHIVE_FILE = "archive.pt"
FORMAT = 2
#: Floats per chunk of the (query boxes x states x keys per state) gather.
GATHER_CHUNK = 20_000_000


class NoveltyArchive:
    def __init__(self, *, metric: str = "cosine", text_embedder: Optional[TextEmbedder] = None,
                 w_image: float = 1.0, dedup_threshold: float = 0.001, max_size: int = 10_000,
                 compact_to: Optional[int] = None, cell_threshold: Optional[float] = None,
                 text_seen_threshold: float = 0.9) -> None:
        check_metric(metric)
        self.metric = metric
        self.text_embedder = text_embedder or TextEmbedder()
        self.has_text = self.text_embedder.name != "none"
        self.w_image = 1.0 if not self.has_text else float(w_image)
        self.dedup_threshold, self.max_size = dedup_threshold, max_size
        self.compact_to = compact_to if compact_to is not None else max_size // 2
        self.cell_threshold = cell_threshold
        self.text_seen_threshold = text_seen_threshold
        self.images: Optional[torch.Tensor] = None     # (N, D)
        self.store = KeyStore()
        self.state_keys: list = []                     # per state: int64 key ids
        self.state_weights: list = []                  # per state: float32 weights
        self._padded = None                            # (ids (N, L) with pad = len(keys), weights (N, L))
        self.cells: list = []                          # StateEmbedding per cell
        self.cell_counts: list = []
        self.regions: dict = {}
        self.n_compactions = 0

    def __len__(self) -> int:
        return 0 if self.images is None else int(self.images.shape[0])

    # ------------------------------------------------------------ text

    def text_of(self, i: int) -> BoxSet:
        """State i's BoxSet."""
        return BoxSet(keys=tuple(self.store.keys[k] for k in self.state_keys[i]),
                      weights=tuple(float(w) for w in self.state_weights[i]))

    @property
    def texts(self) -> list:
        return [self.text_of(i) for i in range(len(self))]

    def _pad(self) -> tuple:
        if self._padded is None or self._padded[0].shape[0] != len(self):
            n = len(self)
            width = max([len(k) for k in self.state_keys] + [1])
            ids = np.full((n, width), len(self.store), dtype=np.int64)
            ws = np.zeros((n, width), dtype=np.float32)
            for i, (k, w) in enumerate(zip(self.state_keys, self.state_weights)):
                ids[i, :len(k)], ws[i, :len(w)] = k, w
            self._padded = (ids, ws)
        ids, ws = self._padded
        ids = np.where(ids >= len(self.store), len(self.store), ids)     # the pad id tracks the key count
        return ids, ws

    def text_similarities(self, text: BoxSet) -> np.ndarray:
        """(N,) chamfer text similarity of a BoxSet to every stored state; NaN where both have no text."""
        n = len(self)
        empty_state = np.array([len(k) == 0 for k in self.state_keys], dtype=bool)
        if not text.keys:
            return np.where(empty_state, np.nan, 0.0).astype(np.float32)
        q = len(text.keys)
        sims = self.store.cosines(keys=list(text.keys))                       # (q, K)
        sims = np.concatenate([sims, np.zeros((q, 1), dtype=np.float32)], axis=1)   # pad column
        ids, ws = self._pad()
        qw = np.asarray(text.weights, dtype=np.float32)
        rev = (sims.max(axis=0)[ids] * ws).sum(axis=1)                    # sim(state -> query)
        fwd = np.empty(n, dtype=np.float32)                               # sim(query -> state)
        step = max(1, GATHER_CHUNK // max(1, q * ids.shape[1]))
        for s0 in range(0, n, step):
            g = sims[:, ids[s0:s0 + step]]                                # (q, chunk, L)
            fwd[s0:s0 + step] = qw @ g.max(axis=2)
        out = 0.5 * (fwd + rev)
        out[empty_state] = 0.0
        return out.astype(np.float32)

    # ------------------------------------------------------------ similarity

    def _similarities(self, *, e: StateEmbedding, images: torch.Tensor, texts: list) -> torch.Tensor:
        """Against an explicit list of states (cells)."""
        sims = image_similarities(query=e.image, matrix=images, metric=self.metric)
        if self.w_image < 1.0:
            t = [chamfer(a=e.text or EMPTY, b=x or EMPTY) for x in texts]
            text_sims = torch.tensor([np.nan if v is None else v for v in t], dtype=sims.dtype)
            sims = combine(image_sim=sims, text_sim=text_sims, w_image=self.w_image)
        return sims

    def similarities(self, e: StateEmbedding) -> torch.Tensor:
        if len(self) == 0:
            return torch.zeros(0)
        sims = image_similarities(query=e.image, matrix=self.images, metric=self.metric)
        if self.w_image < 1.0:
            text_sims = torch.from_numpy(self.text_similarities(e.text or EMPTY)).to(sims.dtype)
            sims = combine(image_sim=sims, text_sim=text_sims, w_image=self.w_image)
        return sims

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
        """1 - max similarity to a stored state (1.0 if empty)."""
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
        if not self.has_text:
            return True
        keys = set((e.text or EMPTY).keys)
        return any({self.store.keys[k] for k in self.state_keys[i]} == keys for i in torch.where(close)[0].tolist())

    def _append_text(self, text: Optional[BoxSet]) -> None:
        text = text or EMPTY
        self.state_keys.append(self.store.add_many(keys=text.keys) if self.has_text else np.zeros(0, dtype=np.int64))
        self.state_weights.append(np.asarray(text.weights if self.has_text else (), dtype=np.float32))
        self._padded = None

    def add(self, e: StateEmbedding) -> bool:
        """Store e unless it is a near-duplicate; compact above max_size. Returns whether it was stored."""
        if len(self) == 0:
            self.images = e.image.unsqueeze(0).clone()
            self.state_keys, self.state_weights = [], []
            self._append_text(e.text)
            return True
        if self._is_duplicate(e):
            return False
        self.images = torch.cat([self.images, e.image.unsqueeze(0).to(self.images.dtype)], dim=0)
        self._append_text(e.text)
        if len(self) > self.max_size:
            self.compact()
        return True

    def _keep_states(self, keep: list) -> None:
        """Keep these states; drop keys no kept state uses (ids are remapped)."""
        self.images = self.images[keep]
        state_keys = [self.state_keys[i] for i in keep]
        self.state_weights = [self.state_weights[i] for i in keep]
        used = sorted({int(k) for ks in state_keys for k in ks})
        remap = {old: new for new, old in enumerate(used)}
        self.store = KeyStore([self.store.keys[k] for k in used])
        self.state_keys = [np.array([remap[int(k)] for k in ks], dtype=np.int64) for ks in state_keys]
        self._padded = None

    def compact(self) -> None:
        from sklearn.cluster import KMeans
        x = self.images.cpu().numpy()
        kmeans = KMeans(n_clusters=self.compact_to, random_state=42, n_init=1).fit(x)
        self.n_compactions += 1
        if not self.has_text:
            centres = torch.tensor(kmeans.cluster_centers_, dtype=self.images.dtype)
            if self.metric == "cosine":
                centres = torch.nn.functional.normalize(centres, dim=-1)
            self.images = centres
            self.state_keys = [np.zeros(0, dtype=np.int64) for _ in range(len(centres))]
            self.state_weights = [np.zeros(0, dtype=np.float32) for _ in range(len(centres))]
            self._padded = None
            return
        keep = []
        for c in range(self.compact_to):
            members = np.where(kmeans.labels_ == c)[0]
            if len(members):
                d = ((x[members] - kmeans.cluster_centers_[c]) ** 2).sum(-1)
                keep.append(int(members[d.argmin()]))
        self._keep_states(keep)

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
            self.regions[name] = REGION_TYPES[name](seen_threshold=self.text_seen_threshold)
        return self.regions[name]

    # ------------------------------------------------------------ snapshots

    def copy(self) -> dict:
        return {"images": None if self.images is None else self.images.clone(), "texts": self.texts,
                "cells": list(self.cells), "cell_counts": list(self.cell_counts),
                "regions": {k: v.copy() for k, v in self.regions.items()}}

    def restore(self, snapshot: dict) -> None:
        self.images = None if snapshot["images"] is None else snapshot["images"].clone()
        self._set_texts(snapshot["texts"])
        self.cells = list(snapshot["cells"])
        self.cell_counts = list(snapshot["cell_counts"])
        self.regions = {k: v.copy() for k, v in snapshot["regions"].items()}

    def _set_texts(self, texts: list) -> None:
        self.store = KeyStore()
        self.state_keys, self.state_weights = [], []
        for t in texts:
            self._append_text(t)

    # ------------------------------------------------------------ files

    def save(self, *, directory: str, merge: bool = False) -> None:
        """Write the archive; merge=True also keeps entries already saved there that this one lacks."""
        os.makedirs(directory, exist_ok=True)
        images, texts = self.images, self.texts
        path = os.path.join(directory, ARCHIVE_FILE)
        if merge and os.path.exists(path) and images is not None:
            old = _read(path=path, text_embedder=self.text_embedder.name, directory=directory)
            if old["images"] is not None:
                max_diff = (old["images"].unsqueeze(1) - images.unsqueeze(0)).abs().amax(dim=-1)
                unseen = (max_diff.amin(dim=1) >= self.dedup_threshold).tolist()
                keep = [i for i, u in enumerate(unseen) if u]
                if keep:
                    images = torch.cat([old["images"][keep], images], dim=0)
                    texts = [old["texts"][i] for i in keep] + texts
        keys = KeyStore()
        state_keys = [keys.add_many(keys=t.keys).tolist() for t in texts]
        torch.save({"format": FORMAT, "images": None if images is None else images.cpu(), "keys": list(keys.keys),
                    "state_keys": state_keys, "state_weights": [list(t.weights) for t in texts],
                    "cells": self.cells, "cell_counts": self.cell_counts, "metric": self.metric,
                    "text_embedder": self.text_embedder.name}, path)
        for buf in self.regions.values():
            buf.save(directory=directory, merge=merge)

    def load_from(self, *, directory: str) -> None:
        """Replace this archive's contents with the one saved in `directory`."""
        path = os.path.join(directory, ARCHIVE_FILE)
        if not os.path.exists(path):
            raise ValueError(f"No archive found at {directory}")
        data = _read(path=path, text_embedder=self.text_embedder.name, directory=directory)
        self.images = data["images"]
        self._set_texts(data["texts"])
        self.cells, self.cell_counts = list(data["cells"]), list(data["cell_counts"])
        self.regions = {}
        for name, cls in REGION_TYPES.items():
            buf = cls.load(directory=directory, seen_threshold=self.text_seen_threshold)
            if buf is not None:
                self.regions[name] = buf


def _read(*, path: str, text_embedder: str, directory: str) -> dict:
    data = torch.load(path, weights_only=False)
    if data.get("format") != FORMAT:
        raise ValueError(f"{directory}: archive format {data.get('format')!r}, this code reads format {FORMAT} "
                         "(older archives scored text differently; regenerate the run)")
    if data["text_embedder"] != text_embedder:
        raise ValueError(f"{directory}: archive saved with text embedder {data['text_embedder']!r}, "
                         f"this run uses {text_embedder!r}")
    keys = data["keys"]
    texts = [BoxSet(keys=tuple(keys[k] for k in ks), weights=tuple(ws))
             for ks, ws in zip(data["state_keys"], data["state_weights"])]
    return {"images": data["images"], "texts": texts, "cells": data.get("cells", []),
            "cell_counts": data.get("cell_counts", [])}
