"""
Selection statistics: Q(n), a moving average of the yields of expansions in n's subtree; N(n), the expansions
from n; N_total, the expansions in the run. They depend only on the tree and the order of its expansions.
"""


class TreeStats:
    def __init__(self, *, q_ema: float = 0.3, root_q: float = 1.0) -> None:
        """q_ema: weight of the newest yield in Q. root_q: Q of an unexpanded root, which has no creating step."""
        if not 0 < q_ema <= 1:
            raise ValueError(f"q_ema must be in (0, 1], got {q_ema}")
        self.q_ema, self.root_q = q_ema, root_q
        self._q: dict = {}
        self._n_total = 0

    def q(self, node) -> float:
        """The moving average once n's subtree has a yield; before that, the novelty of the step that created n."""
        if node.id in self._q:
            return self._q[node.id]
        return self.root_q if node.value is None else float(node.value)

    def n(self, node) -> float:
        return float(node.n_expanded)

    def n_total(self) -> float:
        return float(self._n_total)

    def expansion_yield(self, *, expansion) -> float:
        """Mean novelty of the expansion's steps."""
        steps = expansion.steps
        return sum(s.value for s in steps) / len(steps) if steps else 0.0

    def record_yield(self, *, tree, node_id: int, value: float) -> None:
        """Credit an expansion from node_id to it and every ancestor."""
        self._n_total += 1
        for nid in tree.ancestors(node_id):
            tree.nodes[nid].yields.append(value)
            old = self._q.get(nid)
            self._q[nid] = value if old is None else (1 - self.q_ema) * old + self.q_ema * value
