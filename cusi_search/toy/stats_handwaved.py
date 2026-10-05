# TOY (skill_discovery.md §1.5): throwaway first-build scaffolding, not part of the kept code.
# Strip it by deleting cusi_search/toy/, run_preexplore_toy.py, slurm/preexplore_toy.sh,
# scripts/slurm/preexplore_toy.sh, tests/*toy* and the TOY block in skill_discovery.md §1.5.
# Nothing outside those files imports this.
"""HANDWAVED search statistics. REVISIT before any real use (skill_discovery.md §1.5): picked
only so the energy of §1.4 C is computable.

    N(n)     = every observed step (in any segment) whose state fell in n's cell
               (archive.cell_counts[n.cell]), not only expansions from n
    N_total  = all observed steps (sum of the cell counts)
    Q(n)     = 0 until n is first expanded; then the running mean of the yields n has received
    yield    = new cells found + summed novelty of the expansion's steps; added to the expanded
               node and to every ancestor up to the root
    restore  = expanding n also counts as one observation of n's cell (otherwise re-expanding n
               never raises N(n), and a node with one good yield is picked forever)
Cells only; no clustering of cells (the open question in §1.4 A).
"""


class HandwavedStats:
    def __init__(self, *, archive, new_cell_weight: float = 1.0, novelty_weight: float = 1.0) -> None:
        self.archive = archive
        self.new_cell_weight, self.novelty_weight = new_cell_weight, novelty_weight

    def n(self, node) -> float:
        if node.cell is None or node.cell >= len(self.archive.cell_counts):
            return 0.0
        return float(self.archive.cell_counts[node.cell])

    def n_total(self) -> float:
        return float(sum(self.archive.cell_counts))

    def q(self, node) -> float:
        return float(sum(node.yields) / len(node.yields)) if node.yields else 0.0

    def expansion_yield(self, *, expansion) -> float:
        return (self.new_cell_weight * sum(s.new_cell for s in expansion.steps)
                + self.novelty_weight * sum(s.value for s in expansion.steps))

    def record_yield(self, *, tree, node_id: int, value: float) -> None:
        cell = tree.nodes[node_id].cell
        if cell is not None and cell < len(self.archive.cell_counts):
            self.archive.cell_counts[cell] += 1
        for nid in tree.ancestors(node_id):
            tree.nodes[nid].yields.append(value)
