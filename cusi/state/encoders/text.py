"""
The text channel: a screen's text boxes as a set of (role, normalised text) keys, each a hashed character-trigram
vector (plans/next_plan.md §0.2). Two screens compare by chamfer similarity over their boxes (same role only, area
weighted); `none` turns the channel off.
"""
import re
import zlib
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import sparse

TEXT_EMBEDDERS = ("none", "trigram")
DIM = 1 << 20
MAX_KEY_CHARS = 500
#: A screen keeps at most this many keys (the largest boxes), so set comparisons stay bounded on dense web pages.
MAX_BOXES = 64
#: No box weighs more than this share of its screen (unless the screen has fewer than 1 / MAX_WEIGHT boxes).
MAX_WEIGHT = 0.25
_DIGIT_RE = re.compile(r"\d")


def normalise(text: str) -> str:
    """The novelty key's text: casefolded, whitespace collapsed, capped, digits replaced by '#'."""
    return _DIGIT_RE.sub("#", " ".join(str(text or "").casefold().split())[:MAX_KEY_CHARS])


def box_key(*, role: str, text: str) -> Optional[str]:
    """'role\\ttext' (None for boxes without text)."""
    t = normalise(text)
    return f"{role}\t{t}" if t else None


def key_role(key: str) -> str:
    return key.split("\t", 1)[0]


def key_text(key: str) -> str:
    return key.split("\t", 1)[1]


_VECTOR_CACHE: dict = {}


def trigram_vector(text: str) -> tuple:
    """(sorted bucket indices int64, L2-normalised float32 values) of the text's trigrams, padded "  "+text+" "."""
    hit = _VECTOR_CACHE.get(text)
    if hit is not None:
        return hit
    t = f"  {text} "
    counts: dict = {}
    for i in range(len(t) - 2):
        b = zlib.crc32(t[i:i + 3].encode("utf-8")) & (DIM - 1)
        counts[b] = counts.get(b, 0) + 1
    idx = np.array(sorted(counts), dtype=np.int64)
    val = np.array([counts[i] for i in idx], dtype=np.float32)
    val /= np.linalg.norm(val)
    if len(_VECTOR_CACHE) > 200_000:
        _VECTOR_CACHE.clear()
    _VECTOR_CACHE[text] = (idx, val)
    return idx, val


def key_matrix(keys: list) -> sparse.csr_matrix:
    """(len(keys), DIM) CSR of the keys' trigram vectors."""
    if not keys:
        return sparse.csr_matrix((0, DIM), dtype=np.float32)
    vecs = [trigram_vector(key_text(k)) for k in keys]
    indptr = np.cumsum([0] + [len(v[0]) for v in vecs])
    return sparse.csr_matrix((np.concatenate([v[1] for v in vecs]), np.concatenate([v[0] for v in vecs]), indptr),
                             shape=(len(keys), DIM))


def key_cosines(*, a: list, b: list, b_matrix: Optional[sparse.csr_matrix] = None) -> np.ndarray:
    """(len(a), len(b)) trigram cosine between keys, 0 where the roles differ."""
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    bm = key_matrix(b) if b_matrix is None else b_matrix
    sims = (key_matrix(a) @ bm.T).toarray().astype(np.float32)
    roles_a = np.array([key_role(k) for k in a])
    roles_b = np.array([key_role(k) for k in b])
    sims[roles_a[:, None] != roles_b[None, :]] = 0.0
    return np.minimum(sims, 1.0)


def capped_weights(areas: np.ndarray, cap: float = MAX_WEIGHT) -> np.ndarray:
    """Area shares summing to 1 with no share above cap (water-filling); uniform if the cap cannot be met."""
    n = len(areas)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if n * cap <= 1.0:
        return np.full(n, 1.0 / n, dtype=np.float32)
    a = np.maximum(np.asarray(areas, dtype=np.float64), 1e-9)
    w = a / a.sum()
    fixed = np.zeros(n, dtype=bool)
    for _ in range(n):
        over = (w > cap + 1e-12) & ~fixed
        if not over.any():
            break
        fixed |= over
        w[fixed] = cap
        free = ~fixed
        w[free] = a[free] / a[free].sum() * (1.0 - cap * fixed.sum())
    return w.astype(np.float32)


@dataclass(frozen=True)
class BoxSet:
    """A screen's text: distinct box keys with their weights (summing to 1; empty for a screen without text)."""
    keys: tuple
    weights: tuple

    def __len__(self) -> int:
        return len(self.keys)


EMPTY = BoxSet(keys=(), weights=())


def box_set(boxes: list) -> BoxSet:
    """TextBox items (or dicts) -> BoxSet; boxes without text are ignored, repeated keys add their areas."""
    from cusi.envs.text_boxes import as_boxes
    areas: dict = {}
    for b in as_boxes(boxes=boxes):
        k = box_key(role=b.role, text=b.text)
        if k is not None:
            areas[k] = areas.get(k, 0.0) + max(b.area, 1.0)
    if not areas:
        return EMPTY
    if len(areas) > MAX_BOXES:
        areas = dict(sorted(areas.items(), key=lambda kv: -kv[1])[:MAX_BOXES])
    keys = tuple(sorted(areas))
    w = capped_weights(np.array([areas[k] for k in keys]))
    return BoxSet(keys=keys, weights=tuple(float(x) for x in w))


def chamfer(*, a: BoxSet, b: BoxSet, sims: Optional[np.ndarray] = None) -> Optional[float]:
    """½(Σ w_a max cos + Σ w_b max cos); None if both screens have no text, 0 if one has none."""
    if not a.keys and not b.keys:
        return None
    if not a.keys or not b.keys:
        return 0.0
    s = key_cosines(a=list(a.keys), b=list(b.keys)) if sims is None else sims
    return 0.5 * (float(np.dot(a.weights, s.max(axis=1))) + float(np.dot(b.weights, s.max(axis=0))))


class TextEmbedder:
    """`none`: no text channel."""
    name = "none"

    def represent(self, *, boxes: list):
        return None

    def represent_many(self, *, boxes_list: list) -> list:
        return [self.represent(boxes=b) for b in boxes_list]

    def similarity(self, *, a, b) -> Optional[float]:
        return None


class TrigramText(TextEmbedder):
    name = "trigram"

    def represent(self, *, boxes: list) -> BoxSet:
        return box_set(boxes)

    def similarity(self, *, a: BoxSet, b: BoxSet) -> Optional[float]:
        return chamfer(a=a, b=b)


def build_text_embedder(*, text_embedder: str) -> TextEmbedder:
    if text_embedder == "none":
        return TextEmbedder()
    if text_embedder == "trigram":
        return TrigramText()
    raise ValueError(f"--text_embedder must be one of {TEXT_EMBEDDERS}, got {text_embedder!r}")
