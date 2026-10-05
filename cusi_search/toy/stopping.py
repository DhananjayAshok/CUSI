# TOY (skill_discovery.md §1.5): throwaway first-build scaffolding, not part of the kept code.
# Strip it by deleting cusi_search/toy/, run_preexplore_toy.py, slurm/preexplore_toy.sh,
# scripts/slurm/preexplore_toy.sh, tests/*toy* and the TOY block in skill_discovery.md §1.5.
# Nothing outside those files imports this.
"""Toy stopping rule: stop at max expansions, total env steps, wall time or node count."""
from typing import Optional


class StopRule:
    def __init__(self, *, max_expansions: Optional[int] = None, max_steps: Optional[int] = None,
                 max_seconds: Optional[float] = None, max_nodes: Optional[int] = None) -> None:
        self.max_expansions, self.max_steps = max_expansions, max_steps
        self.max_seconds, self.max_nodes = max_seconds, max_nodes

    def __call__(self, search) -> Optional[str]:
        if self.max_expansions is not None and search.iteration >= self.max_expansions:
            return f"max_expansions {self.max_expansions}"
        if self.max_steps is not None and search.total_steps >= self.max_steps:
            return f"max_steps {self.max_steps}"
        if self.max_seconds is not None and search.elapsed >= self.max_seconds:
            return f"max_seconds {self.max_seconds}"
        if self.max_nodes is not None and len(search.tree.nodes) >= self.max_nodes:
            return f"max_nodes {self.max_nodes}"
        return None
