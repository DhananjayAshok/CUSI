"""
Region novelty kept inside a NoveltyArchive, the same on every env: the area-weighted share of a screen's text boxes
whose (role, normalised text) was never seen (plans/archive/next_plan.md §0.2).
"""
import copy
import json
import os
from typing import Optional
import numpy as np
from cusi.state.encoders.text import box_set, key_text
from cusi.state.keys import KeyStore

UNSEEN_TEXT = "unseen_text"
SEEN_FILE = "seen_text.json"


class UnseenTextBuffer:
    """A box is seen if a stored key with its role has trigram cosine >= seen_threshold, or its text is a prefix of a
    stored key's (streaming text replaying a known line). A box that extends a box of the previous screen (the same
    line still being typed) is credited only for the share of its characters that are new."""

    def __init__(self, *, seen_threshold: float = 0.9, max_size: int = 50_000, seed: int = 42) -> None:
        self.seen_threshold, self.max_size = seen_threshold, max_size
        self._rng = np.random.default_rng(seed)
        self.store = KeyStore()

    def __len__(self) -> int:
        return len(self.store)

    def _seen(self, keys: list) -> list:
        """Per key: seen by the cosine threshold or the prefix rule."""
        best = self.store.cosines(keys=keys).max(axis=1) if len(self.store) else np.zeros(len(keys))
        return [b >= self.seen_threshold or self.store.is_prefix_of_stored(key=k) for k, b in zip(keys, best)]

    def score(self, *, boxes: list, prev_boxes: Optional[list] = None, add: bool = True,
              base: Optional["UnseenTextBuffer"] = None) -> float:
        """base: a read-only buffer whose keys also count as seen (a curiosity session's B0)."""
        current = box_set(boxes)
        if not current.keys:
            return 0.0
        keys = list(current.keys)
        seen = self._seen(keys)
        if base is not None:
            seen = [s or t for s, t in zip(seen, base._seen(keys))]
        prev = [k for k in box_set(prev_boxes or []).keys]
        total = 0.0
        for k, w, is_seen in zip(keys, current.weights, seen):
            if is_seen:
                continue
            credit, text = 1.0, key_text(k)
            for p in prev:
                pt = key_text(p)
                if p.split("\t", 1)[0] == k.split("\t", 1)[0] and len(pt) < len(text) and text.startswith(pt):
                    credit = min(credit, (len(text) - len(pt)) / len(text))
            total += w * credit
        if add:
            self._store(keys=keys)
        return float(total)

    def add(self, *, boxes: list) -> None:
        """Mark the boxes' text as seen without scoring it."""
        self._store(keys=list(box_set(boxes).keys))

    def _store(self, *, keys: list) -> None:
        self.store.add_many(keys=keys)
        if len(self.store) > self.max_size:
            kept = self._rng.choice(len(self.store), self.max_size // 2, replace=False)
            self.store = KeyStore([self.store.keys[i] for i in sorted(kept)])

    def copy(self) -> "UnseenTextBuffer":
        out = UnseenTextBuffer(seen_threshold=self.seen_threshold, max_size=self.max_size)
        out.store = KeyStore(list(self.store.keys))
        out._rng = copy.deepcopy(self._rng)
        return out

    def save(self, *, directory: str, merge: bool = False) -> None:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, SEEN_FILE)
        keys = list(self.store.keys)
        if merge and os.path.exists(path):
            with open(path) as f:
                old = json.load(f)["keys"]
            have = set(keys)
            keys = [k for k in old if k not in have] + keys
        with open(path, "w") as f:
            json.dump({"keys": keys}, f)

    @classmethod
    def load(cls, *, directory: str, seen_threshold: float = 0.9) -> Optional["UnseenTextBuffer"]:
        path = os.path.join(directory, SEEN_FILE)
        if not os.path.exists(path):
            return None
        buf = cls(seen_threshold=seen_threshold)
        with open(path) as f:
            buf.store = KeyStore(json.load(f)["keys"])
        return buf


REGION_TYPES = {UNSEEN_TEXT: UnseenTextBuffer}
