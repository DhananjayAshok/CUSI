# TOY (skill_discovery.md §1.5): throwaway first-build scaffolding, not part of the kept code.
# Strip it by deleting cusi_search/toy/, run_preexplore_toy.py, slurm/preexplore_toy.sh,
# scripts/slurm/preexplore_toy.sh, tests/*toy* and the TOY block in skill_discovery.md §1.5.
# Nothing outside those files imports this.
"""Toy on-disk tree format (JSONL + PNG), rewritten in full by TreeWriter.write(search):

    <out>/config.json       run flags, thresholds, calibration
    <out>/nodes.jsonl       one line per node: parent, depth, env_key, cell, state_id, segment,
                            stats (n_expanded, Q, N, yields), prior + reasoning, flags, frame path
    <out>/expansions.jsonl  one line per expansion: start node, children, yield, stats, and every
                            step (action, valid/error, value + components, cell, new_cell, node,
                            frame path, texts truncated)
    <out>/selection.jsonl   one line per iteration: the chosen node, its energy and probability,
                            and all candidates with E, p, Q, N, P
    <out>/frames/           PNGs (node_<id>.png, exp_<e>_<t>.png), written once
    <out>/summary.json      counts and the stop reason
Not resumable.
"""
import json
import os
from typing import Optional

MAX_TEXT = 2000


def _texts(texts: dict) -> dict:
    return {k: str(v)[:MAX_TEXT] for k, v in (texts or {}).items()}


class TreeWriter:
    def __init__(self, *, out_dir: str, config: dict, write_every: int = 1) -> None:
        self.out_dir, self.write_every = out_dir, write_every
        self.frames_dir = os.path.join(out_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self._written: set = set()
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=1, default=str)

    def _png(self, name: str, image) -> str:
        rel = os.path.join("frames", name)
        if name not in self._written:
            image.pil().save(os.path.join(self.out_dir, rel))
            self._written.add(name)
        return rel

    def on_iteration(self, *, search, record: dict) -> None:
        if search.iteration % self.write_every == 0:
            self.write(search=search)

    def write(self, *, search, stop_reason: Optional[str] = None) -> None:
        tree, stats = search.tree, search.stats
        tmp = {}
        lines = []
        for n in tree.nodes:
            lines.append({"id": n.id, "parent": n.parent, "depth": n.depth, "env_key": n.env_key, "cell": n.cell,
                          "state_id": n.state_id, "segment": n.segment, "created_at": n.created_at,
                          "n_expanded": n.n_expanded, "q": stats.q(n), "n": stats.n(n), "yields": n.yields,
                          "prior": n.prior, "prior_reason": n.prior_reason, "prior_raw": n.prior_raw,
                          "flags": n.flags, "children": tree.children.get(n.id, []),
                          "frame": self._png(f"node_{n.id}.png", n.frame), "texts": _texts(n.texts)})
        tmp["nodes.jsonl"] = lines
        lines = []
        for e in tree.expansions:
            steps = [{"t": t, "action": s.action, "info": s.info, "value": s.value, "components": s.components,
                      "cell": s.cell, "new_cell": s.new_cell, "node_id": s.node_id, "extra": s.extra,
                      "frame": self._png(f"exp_{e.id}_{t}.png", s.frame), "texts": _texts(s.texts)}
                     for t, s in enumerate(e.steps)]
            lines.append({"id": e.id, "node_id": e.node_id, "iteration": e.iteration, "children": e.children,
                          "yield": e.yield_value, "stats": e.stats, "steps": steps})
        tmp["expansions.jsonl"] = lines
        tmp["selection.jsonl"] = search.records
        for name, rows in tmp.items():
            path = os.path.join(self.out_dir, name)
            with open(path + ".tmp", "w") as f:
                for row in rows:
                    f.write(json.dumps(row, default=str) + "\n")
            os.replace(path + ".tmp", path)
        summary = {"iterations": search.iteration, "steps": search.total_steps, "nodes": len(tree.nodes),
                   "expansions": len(tree.expansions), "cells": search.archive.n_cells, "archive": len(search.archive),
                   "seconds": search.elapsed, "max_depth": max((n.depth for n in tree.nodes), default=0),
                   "stop_reason": stop_reason}
        with open(os.path.join(self.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)


def read_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
