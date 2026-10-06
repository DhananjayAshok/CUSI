"""The pre-exploration tree search loop (plans/skill_discovery.md §1.4).

    search = TreeSearch(envs={"viridian": env}, encoder=encoder, scorer=scorer, archive=archive,
                        expander=RandomExpander(), stats=stats, selector=EnergySelector(stats=stats, ...),
                        k_steps=10, node_threshold=0.05, prior=None, logger=None)
    reason = search.run(stop=lambda s: "done" if s.iteration >= 20 else None)

One iteration: select a node (selector) -> restore it (env.load_state) -> run the expander for up
to k_steps -> score every step with the scorer against the ONE archive of the search (never
reset; every step is added and counted in its cell) -> a step becomes a child node when its score
passes node_threshold, and the segment end always does (children get a saved state and a VLM
prior) -> the expansion's yield goes to the stats provider, which propagates it.

Passed in, not decided here: envs (one per root scene, keyed by scene; a node's state lives on
its env), the stats provider (q(node), n(node), n_total(), expansion_yield(expansion=...),
record_yield(tree=..., node_id=..., value=...)), the stopping rule (stop(search) -> reason or
None), and an optional logger (on_iteration(search=..., record=...)).
"""
import time
from typing import Any, Callable, Optional
from cusi.utils.log_handling import log_info, log_warn
from cusi.state import StateRecord, frame_for_embedding
from cusi.agents.records import EncodedImage
from cusi.search.selection import EnergySelector
from cusi.search.tree import Step, Tree, compact_info


class TreeSearch:
    def __init__(self, *, envs: dict, encoder, scorer, archive, expander, stats, selector: EnergySelector,
                 k_steps: int, node_threshold: float, prior=None, noop_similarity: float = 0.999,
                 logger=None, parameters: dict[str, Any] = None) -> None:
        self.envs, self.encoder, self.scorer, self.archive = envs, encoder, scorer, archive
        self.expander, self.stats, self.selector, self.prior = expander, stats, selector, prior
        self.k_steps, self.node_threshold, self.noop_similarity = k_steps, node_threshold, noop_similarity
        self.logger, self._parameters = logger, parameters
        self.tree = Tree()
        self.iteration = 0
        self.total_steps = 0
        self.start_time = time.time()
        self.records: list = []

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    # ------------------------------------------------------------ nodes

    def _make_node(self, *, env_key: str, obs: dict, info: dict, embedding, cell: Optional[int],
                   parent: Optional[int], segment: Optional[tuple]):
        env = self.envs[env_key]
        state_id, flags = None, {}
        try:
            state_id = env.save_state()
        except RuntimeError as e:
            flags["unrestorable"] = True
            log_warn(f"save_state failed on {env_key}: {e}", parameters=self._parameters)
        frame = frame_for_embedding(obs=obs, info=info)
        node = self.tree.add_node(parent=parent, env_key=env_key, state_id=state_id, frame=frame,
                                  texts=obs.get("texts", {}), embedding=embedding, cell=cell, segment=segment,
                                  created_at=self.iteration)
        node.flags.update(flags)
        if self.prior is not None:
            p = self.prior(env_description=env.env_description, frame=frame, texts=obs.get("texts", {}))
            node.prior, node.prior_reason, node.prior_raw = p["prior"], p["reason"], p["raw"]
            if not p["parsed"]:
                node.flags["prior_parse_failed"] = True
        return node

    def add_roots(self) -> None:
        for key, env in self.envs.items():
            obs, info = env.reset()
            emb = self.encoder.encode_one(obs=obs, info=info)
            cell = self.archive.cell_of(emb)
            self.archive.add(emb)
            node = self._make_node(env_key=key, obs=obs, info=info, embedding=emb, cell=cell, parent=None,
                                   segment=None)
            log_info(f"root {node.id}: scene {key}, cell {cell}, prior {node.prior}", parameters=self._parameters)

    # ------------------------------------------------------------ one iteration

    def iterate(self) -> Optional[dict]:
        sel = self.selector.select(tree=self.tree)
        if sel is None:
            return None
        self.iteration += 1
        t0 = time.time()
        node = self.tree.nodes[sel.node_id]
        env = self.envs[node.env_key]
        try:
            env.load_state(state_id=node.state_id)
        except RuntimeError as e:
            node.flags["unrestorable"] = True
            log_warn(f"load_state failed for node {node.id}: {e}", parameters=self._parameters)
            return {"iteration": self.iteration, "selected": node.id, "failed": "load_state"}
        t_load = time.time() - t0
        obs, info = env.current_obs, {}
        prev = StateRecord(obs=obs, info=info, embedding=node.embedding)
        exp = self.tree.add_expansion(node_id=node.id, iteration=self.iteration)
        self.expander.start(env=env)
        parent_id, seg_start = node.id, 0
        n_valid = 0
        for t in range(self.k_steps):
            action, meta = self.expander.next_action(env=env, obs=obs, info=info)
            obs2, _r, terminated, truncated, info2 = env.step(action)
            emb = self.encoder.encode_one(obs=obs2, info=info2)
            n_cells = self.archive.n_cells
            cell = self.archive.cell_of(emb)
            nxt = StateRecord(obs=obs2, info=info2, embedding=emb, prev_image=prev.embedding.image)
            value, components = self.scorer.score(prev=prev, action=action, next=nxt, archive=self.archive, add=True)
            changed = self.encoder.similarity(a=prev.embedding, b=emb) < self.noop_similarity
            valid = bool(info2.get("valid", True))
            n_valid += valid
            self.expander.observe(action=action, changed=changed, valid=valid, error=info2.get("error"))
            step = Step(action=action, frame=EncodedImage.of(frame_for_embedding(obs=obs2, info=info2)), texts=obs2.get("texts", {}),
                        info=compact_info(info2), value=float(value), components=dict(components), cell=cell,
                        new_cell=self.archive.n_cells > n_cells, extra={**meta, "changed": changed})
            exp.steps.append(step)
            self.total_steps += 1
            end = t == self.k_steps - 1 or terminated or truncated
            if value >= self.node_threshold or end:
                child = self._make_node(env_key=node.env_key, obs=obs2, info=info2, embedding=emb, cell=cell,
                                        parent=parent_id, segment=(exp.id, seg_start, t))
                step.node_id = child.id
                exp.children.append(child.id)
                parent_id, seg_start = child.id, t + 1
            prev, obs, info = nxt, obs2, info2
            if terminated or truncated:
                break
        exp.yield_value = float(self.stats.expansion_yield(expansion=exp))
        self.stats.record_yield(tree=self.tree, node_id=node.id, value=exp.yield_value)
        node.n_expanded += 1
        exp.stats = {"seconds": time.time() - t0, "load_seconds": t_load, "valid_rate": n_valid / max(1, len(exp.steps)),
                     "new_cells": sum(s.new_cell for s in exp.steps)}
        k = sel.candidates.index(node.id)
        record = {"iteration": self.iteration, "selected": node.id, "energy": sel.energies[k], "prob": sel.probs[k],
                  "candidates": sel.candidates, "energies": sel.energies, "probs": sel.probs, "q": sel.q, "n": sel.n,
                  "p": sel.p, "n_total": sel.n_total, "expansion": exp.id, "steps": len(exp.steps),
                  "children": list(exp.children), "yield": exp.yield_value, **exp.stats,
                  "n_nodes": len(self.tree.nodes), "n_cells": self.archive.n_cells, "archive": len(self.archive)}
        self.records.append(record)
        log_info(f"iter {self.iteration}: node {node.id} (p={sel.probs[k]:.3f}, E={sel.energies[k]:.3f}) -> "
                 f"{len(exp.steps)} steps, {len(exp.children)} children, yield {exp.yield_value:.3f}, "
                 f"{exp.stats['new_cells']} new cells; tree {len(self.tree.nodes)} nodes, "
                 f"{self.archive.n_cells} cells ({exp.stats['seconds']:.1f}s)", parameters=self._parameters)
        if self.logger is not None:
            self.logger.on_iteration(search=self, record=record)
        return record

    def run(self, *, stop: Callable) -> str:
        if not self.tree.nodes:
            self.add_roots()
        while True:
            reason = stop(self)
            if reason is not None:
                return reason
            if self.iterate() is None:
                return "no restorable node"
