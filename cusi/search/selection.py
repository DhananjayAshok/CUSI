"""
Energy-based node selection: p(n) ∝ exp(E(n) / τ), E(n) = Q(n) + c * P(n) * sqrt(N_total) / (1 + N(n)).
P(n) is the node's VLM decision score (plans/archive/next_plan.md §B) when scoring is on, else default_prior. With
normalise_p (the default) P is normalised over the candidates, as a PUCT prior: otherwise every unexpanded node gets
the full sqrt(N_total) bonus and the search only ever chains fresh nodes (calibrated on gameboy/gemma_vlm_v1 with
cusi/search/calibrate.py).
"""
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
from cusi.search.tree import Tree


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
    scores: list = field(default_factory=list)      # decision-score entries computed for this decision


class EnergySelector:
    def __init__(self, *, stats, c: float = 1.0, tau: float = 1.0, roots_only: bool = False,
                 default_prior: float = 1.0, normalise_p: bool = True, seed: int = 0) -> None:
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        self.stats, self.c, self.tau, self.roots_only = stats, c, tau, roots_only
        self.default_prior, self.normalise_p, self.seed = default_prior, normalise_p, seed
        self.rng = np.random.default_rng(seed)

    def reseed(self, *, iteration: int) -> None:
        """Draw from a stream fixed by (seed, iteration), so a resumed search selects as it would have."""
        self.rng = np.random.default_rng([self.seed, iteration])

    eligible: Optional[Callable] = None     # set by the search when its expander expands only some nodes

    def candidates(self, *, tree: Tree) -> list:
        return [n for n in (tree.roots if self.roots_only else tree.nodes)
                if n.restorable and (self.eligible is None or self.eligible(n))]

    def energies(self, *, nodes: list, p: np.ndarray) -> tuple:
        n_total = float(self.stats.n_total())
        q = np.array([self.stats.q(node) for node in nodes], dtype=np.float64)
        n = np.array([self.stats.n(node) for node in nodes], dtype=np.float64)
        prior = p / max(p.sum(), 1e-12) if self.normalise_p else p
        return q + self.c * prior * np.sqrt(n_total) / (1.0 + n), q, n, n_total

    def select(self, *, tree: Tree, p_of: Optional[Callable] = None,
               rescore: Optional[Callable] = None) -> Optional[Selection]:
        """p_of(node) -> P(n) (default: default_prior). rescore(nodes ranked by energy) -> score entries, after which
        P is read again: the decision scorer re-scores the top candidates before sampling."""
        nodes = self.candidates(tree=tree)
        if not nodes:
            return None
        p_of = p_of or (lambda node: self.default_prior)
        p = np.array([p_of(node) for node in nodes], dtype=np.float64)
        energies, q, n, n_total = self.energies(nodes=nodes, p=p)
        scores = []
        if rescore is not None:
            ranked = [nodes[i] for i in np.argsort(-energies, kind="stable")]
            scores = rescore(ranked)
            p = np.array([p_of(node) for node in nodes], dtype=np.float64)
            energies, q, n, n_total = self.energies(nodes=nodes, p=p)
        logits = energies / self.tau
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        k = int(self.rng.choice(len(nodes), p=probs))
        return Selection(node_id=nodes[k].id, candidates=[x.id for x in nodes], energies=energies.tolist(),
                         probs=probs.tolist(), q=q.tolist(), n=n.tolist(), p=p.tolist(), n_total=n_total,
                         scores=scores)
