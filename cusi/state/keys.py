"""
Interned text-box keys with their trigram vectors as one growing sparse matrix, shared by the archive's text channel
and the unseen-text buffer.
"""
import bisect
from typing import Optional
import numpy as np
from scipy import sparse
from cusi.state.encoders.text import DIM, key_matrix, key_role, key_text


class KeyStore:
    def __init__(self, keys: Optional[list] = None) -> None:
        self.keys: list = []
        self.index: dict = {}
        self._roles: list = []
        self._blocks: list = []          # CSR blocks of new keys, stacked on demand
        self._matrix = sparse.csr_matrix((0, DIM), dtype=np.float32)
        self._role_array = np.zeros(0, dtype=object)
        self._by_role: dict = {}         # role -> sorted key texts (prefix lookups)
        for k in keys or []:
            self.add(k)

    def __len__(self) -> int:
        return len(self.keys)

    def add(self, key: str) -> int:
        i = self.index.get(key)
        if i is not None:
            return i
        i = len(self.keys)
        self.keys.append(key)
        self.index[key] = i
        self._roles.append(key_role(key))
        self._blocks.append(key)
        bisect.insort(self._by_role.setdefault(key_role(key), []), key_text(key))
        return i

    def add_many(self, *, keys) -> np.ndarray:
        return np.array([self.add(k) for k in keys], dtype=np.int64)

    @property
    def matrix(self) -> sparse.csr_matrix:
        if self._blocks:
            self._matrix = sparse.vstack([self._matrix, key_matrix(self._blocks)], format="csr")
            self._role_array = np.array(self._roles, dtype=object)
            self._blocks = []
        return self._matrix

    def cosines(self, *, keys: list) -> np.ndarray:
        """(len(keys), len(self)) cosine of each query key to every stored key, 0 across roles."""
        if not keys or not self.keys:
            return np.zeros((len(keys), len(self.keys)), dtype=np.float32)
        m = self.matrix
        sims = (key_matrix(keys) @ m.T).toarray().astype(np.float32)
        roles = np.array([key_role(k) for k in keys], dtype=object)
        sims[roles[:, None] != self._role_array[None, :]] = 0.0
        return np.minimum(sims, 1.0)

    def is_prefix_of_stored(self, *, key: str) -> bool:
        """Some stored key with the same role starts with this key's text (and is longer)."""
        texts = self._by_role.get(key_role(key), [])
        t = key_text(key)
        j = bisect.bisect_left(texts, t)
        while j < len(texts) and texts[j].startswith(t):
            if len(texts[j]) > len(t):
                return True
            j += 1
        return False
