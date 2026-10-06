"""Text embedders (curiosity_plan §3.2.1). Each turns a state's element lines (cusi.state.text)
into a representation and compares a query against many stored ones.

    none      no text part (GameBoy for now); w_image is forced to 1.
    overlap   Jaccard overlap of the sets of lines. The cheapest option.
    tfidf     bag of lower-cased word tokens, tf-idf weighted, cosine. Document frequencies are
              counted online by the embedder over every state it has represented (so the idf
              keeps moving; an archive restore does not rewind it).
    dense     a sentence-embedding model (id required, e.g. sentence-transformers/all-MiniLM-L6-v2)
              through plain transformers + mean pooling, over the lines joined by newlines;
              unit vector, dot product.

    t = build_text_embedder(text_embedder="tfidf")
    rep = t.represent(lines=[...]);  sims = t.similarities(query=rep, reps=[rep1, rep2])   # np (N,)
"""
import math
import re
from collections import Counter
from typing import Optional
import numpy as np

TEXT_EMBEDDERS = ("none", "overlap", "tfidf", "dense")
_WORD_RE = re.compile(r"\w+")


class TextEmbedder:
    name = "none"

    def represent(self, *, lines: list):
        return None

    def represent_many(self, *, lines_list: list) -> list:
        return [self.represent(lines=lines) for lines in lines_list]

    def similarities(self, *, query, reps: list) -> np.ndarray:
        return np.zeros(len(reps), dtype=np.float32)

    def identical(self, *, a, b) -> bool:
        return True


class OverlapText(TextEmbedder):
    name = "overlap"

    def represent(self, *, lines: list) -> frozenset:
        return frozenset(lines)

    def similarities(self, *, query: frozenset, reps: list) -> np.ndarray:
        out = np.empty(len(reps), dtype=np.float32)
        for i, r in enumerate(reps):
            union = len(query | r)
            out[i] = 1.0 if union == 0 else len(query & r) / union
        return out

    def identical(self, *, a, b) -> bool:
        return a == b


class TfidfText(TextEmbedder):
    name = "tfidf"

    def __init__(self) -> None:
        self.df: Counter = Counter()
        self.n_docs = 0

    def represent(self, *, lines: list) -> dict:
        counts = Counter(w.lower() for line in lines for w in _WORD_RE.findall(line))
        self.df.update(counts.keys())
        self.n_docs += 1
        return dict(counts)

    def _weights(self, rep: dict) -> tuple:
        w = {t: c * (math.log((1 + self.n_docs) / (1 + self.df.get(t, 0))) + 1.0) for t, c in rep.items()}
        return w, math.sqrt(sum(v * v for v in w.values()))

    def similarities(self, *, query: dict, reps: list) -> np.ndarray:
        qw, qn = self._weights(query)
        out = np.zeros(len(reps), dtype=np.float32)
        for i, r in enumerate(reps):
            rw, rn = self._weights(r)
            if qn == 0 and rn == 0:
                out[i] = 1.0
            elif qn > 0 and rn > 0:
                small, big = (qw, rw) if len(qw) < len(rw) else (rw, qw)
                out[i] = sum(v * big.get(t, 0.0) for t, v in small.items()) / (qn * rn)
        return out

    def identical(self, *, a, b) -> bool:
        return a == b


class DenseText(TextEmbedder):
    name = "dense"

    def __init__(self, *, model_name: str, device: Optional[str] = None, max_length: int = 256) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        if not model_name:
            raise ValueError("--text_embedder dense needs --text_embedder_model")
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_length = max_length

    def represent_many(self, *, lines_list: list) -> list:
        import torch
        texts = ["\n".join(lines) for lines in lines_list]
        with torch.no_grad():
            batch = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                                   return_tensors="pt").to(self.device)
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy().astype(np.float32)
        return list(pooled)

    def represent(self, *, lines: list) -> np.ndarray:
        return self.represent_many(lines_list=[lines])[0]

    def similarities(self, *, query: np.ndarray, reps: list) -> np.ndarray:
        if not reps:
            return np.zeros(0, dtype=np.float32)
        return (np.stack(reps) @ query).astype(np.float32)

    def identical(self, *, a, b) -> bool:
        return bool(np.abs(a - b).max() < 1e-3)


def build_text_embedder(*, text_embedder: str, text_embedder_model: Optional[str] = None,
                        device: Optional[str] = None) -> TextEmbedder:
    if text_embedder == "none":
        return TextEmbedder()
    if text_embedder == "overlap":
        return OverlapText()
    if text_embedder == "tfidf":
        return TfidfText()
    if text_embedder == "dense":
        return DenseText(model_name=text_embedder_model, device=device)
    raise ValueError(f"--text_embedder must be one of {TEXT_EMBEDDERS}, got {text_embedder!r}")
