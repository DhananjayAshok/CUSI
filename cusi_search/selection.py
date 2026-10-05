"""Energy-based selection over all nodes (skill_discovery.md §1.4 C).

    E(n) = Q(n) + c * P(n) * sqrt(N_total) / (1 + N(n)),   p(n) ∝ exp(E(n) / τ)

τ -> 0 is greedy PUCT; large τ is uniform. roots_only=True is the "always restart from the scene
start" special case. Q, N and N_total come from a stats provider (any object with
q(node), n(node), n_total()); P(n) is the node's VLM prior (default_prior when it has none).
Unrestorable nodes are never selected.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from cusi_search.tree import Tree


@dataclass
class Selection:
    node_id: int
    candidates: list
    energies: list
    probs: list
    q: list
    n: list
    p: list
    n_total: float


class EnergySelector:
    def __init__(self, *, stats, c: float = 1.0, tau: float = 1.0, roots_only: bool = False,
                 default_prior: float = 1.0, seed: int = 0) -> None:
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        self.stats, self.c, self.tau, self.roots_only = stats, c, tau, roots_only
        self.default_prior = default_prior
        self.rng = np.random.default_rng(seed)

    def select(self, *, tree: Tree) -> Optional[Selection]:
        nodes = [n for n in (tree.roots if self.roots_only else tree.nodes) if n.restorable]
        if not nodes:
            return None
        n_total = float(self.stats.n_total())
        q = np.array([self.stats.q(node) for node in nodes], dtype=np.float64)
        n = np.array([self.stats.n(node) for node in nodes], dtype=np.float64)
        p = np.array([node.prior if node.prior is not None else self.default_prior for node in nodes])
        energies = q + self.c * p * np.sqrt(n_total) / (1.0 + n)
        logits = energies / self.tau
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        k = int(self.rng.choice(len(nodes), p=probs))
        return Selection(node_id=nodes[k].id, candidates=[x.id for x in nodes], energies=energies.tolist(),
                         probs=probs.tolist(), q=q.tolist(), n=n.tolist(), p=p.tolist(), n_total=n_total)
