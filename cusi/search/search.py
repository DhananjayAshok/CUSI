"""
The tree search loop: select a node, restore it, expand it, score the steps against one never-reset archive.
"""
import time
from typing import Any, Callable, Optional
import numpy as np
from cusi.utils.log_handling import log_info, log_warn
from cusi.state import StateRecord, frame_for_embedding
from cusi.agents.records import EncodedImage
from cusi.search.selection import EnergySelector
from cusi.search.tree import Step, Tree, compact_info

RESTORE_MODES = ("state", "replay")


class TreeSearch:
    def __init__(self, *, envs, encoder, scorer, archive, expander, stats, selector: EnergySelector,
                 k_steps: int, node_threshold: float, prior=None, noop_similarity: float = 0.999,
                 restore: str = "state", replay_similarity: float = 0.95, store=None, logger=None,
                 parameters: dict[str, Any] = None) -> None:
        """envs: env key (scene) -> env, a dict or cusi.search.envs.SceneEnvs. restore: "state" saves and loads a
        state per node; "replay" saves none and rebuilds a node by reset + its root->node actions, accepted when the
        result's similarity to the node is >= replay_similarity. store: a SearchStore that commits the roots and
        every expansion, and resume() reads back."""
        if restore not in RESTORE_MODES:
            raise ValueError(f"restore must be one of {RESTORE_MODES}, got {restore!r}")
        self.envs, self.encoder, self.scorer, self.archive = envs, encoder, scorer, archive
        self.expander, self.stats, self.selector, self.prior = expander, stats, selector, prior
        self.k_steps, self.node_threshold, self.noop_similarity = k_steps, node_threshold, noop_similarity
        self.restore, self.replay_similarity = restore, replay_similarity
        self.store, self.logger, self._parameters = store, logger, parameters
        self.tree = Tree()
        self.iteration = 0
        self.total_steps = 0
        self.start_time = time.time()
        self.records: list = []
        self._at: dict = {}       # env key -> (node id, env) the env is currently in, so its restore can be skipped

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    # ------------------------------------------------------------ nodes

    def _make_node(self, *, env_key: str, obs: dict, info: dict, embedding, value: Optional[float],
                   parent: Optional[int], segment: Optional[tuple]):
        env = self.envs[env_key]
        state_id, flags = None, {}
        challenge = bool(info.get("challenge"))
        if challenge:
            flags["challenge_page"] = True
        elif self.restore == "state":
            try:
                state_id = env.save_state()
            except RuntimeError as e:
                flags["unrestorable"] = True
                log_warn(f"save_state failed on {env_key}: {e}", parameters=self._parameters)
        frame = frame_for_embedding(obs=obs, info=info)
        node = self.tree.add_node(parent=parent, env_key=env_key, state_id=state_id, frame=frame,
                                  texts=obs.get("texts", {}), embedding=embedding, value=value, segment=segment,
                                  created_at=self.iteration, replay=self.restore == "replay" and not challenge)
        node.flags.update(flags)
        if self.prior is not None and not challenge:
            p = self.prior(env_description=env.env_description, frame=frame, texts=obs.get("texts", {}))
            node.prior, node.prior_reason, node.prior_raw = p["prior"], p["reason"], p["raw"]
            if not p["parsed"]:
                node.flags["prior_parse_failed"] = True
        return node

    def add_roots(self) -> None:
        for key, env in self.envs.items():
            obs, info = env.reset()
            emb = self.encoder.encode_one(obs=obs, info=info)
            if not info.get("challenge"):
                self.archive.add(emb)
            node = self._make_node(env_key=key, obs=obs, info=info, embedding=emb, value=None, parent=None,
                                   segment=None)
            self._at[key] = (node.id, env)
            log_info(f"root {node.id}: scene {key}, prior {node.prior}, flags {node.flags}", parameters=self._parameters)
        if self.store is not None:
            self.store.commit(search=self)

    def resume(self) -> bool:
        """Load the store's last commit into this (fresh) search; False if the run has none."""
        if self.store is None or not self.store.load(search=self):
            return False
        log_info(f"resumed {self.store.directory}: {len(self.tree.expansions)} expansions, {len(self.tree.nodes)} "
                 f"nodes, archive {len(self.archive)}", parameters=self._parameters)
        return True

    # ------------------------------------------------------------ restoring a node

    def _unselectable(self, *, node, flag: str, message: str) -> None:
        node.flags[flag] = True
        if self.store is not None:
            self.store.write_flags(node=node)
        log_warn(f"node {node.id}: {message}; marked {flag}", parameters=self._parameters)

    def _replay(self, *, node, env) -> tuple:
        """Reset the scene and redo the root->node actions -> (similarity to the node, failure or None)."""
        obs, info = env.reset()
        path = self.tree.path_steps(node.id)
        for i, step in enumerate(path):
            obs, _r, terminated, truncated, info = env.step(step.action)
            if (terminated or truncated) and i < len(path) - 1:
                return 0.0, f"the episode ended at replay step {i + 1}/{len(path)}"
        sim = float(self.encoder.similarity(a=node.embedding, b=self.encoder.encode_one(obs=obs, info=info)))
        if sim < self.replay_similarity:
            return sim, f"replay reached similarity {sim:.3f} < {self.replay_similarity}"
        return sim, None

    def _restore_node(self, *, node, env) -> Optional[dict]:
        """Put env in node's state -> restore stats, or None if it could not be (node is then flagged)."""
        at = self._at.get(node.env_key)
        if at is not None and at[0] == node.id and at[1] is env:
            return {"restore": "skipped"}
        self._at.pop(node.env_key, None)
        if node.state_id is not None:
            try:
                env.load_state(state_id=node.state_id)
            except RuntimeError as e:
                self._unselectable(node=node, flag="unrestorable", message=f"load_state failed ({e})")
                return None
            return {"restore": "state"}
        sim, failure = self._replay(node=node, env=env)
        if failure is not None:
            self._unselectable(node=node, flag="replay_diverged", message=failure)
            return None
        return {"restore": "replay", "replay_steps": len(self.tree.path_steps(node.id)), "replay_similarity": sim}

    def verify_restores(self, *, limit: Optional[int] = None) -> dict:
        """Restore each selectable node (up to limit) and compare with what it stored: pixel-exact frames and the
        embedding similarity. Nodes that fail to restore are flagged as in a search."""
        exact, sims, failed = 0, [], []
        nodes = [n for n in self.tree.nodes if n.restorable][:limit]
        for node in nodes:
            env = self.envs[node.env_key]
            self._at.clear()
            if self._restore_node(node=node, env=env) is None:
                failed.append(node.id)
                continue
            obs, info = env.current_obs, {"raw_frame": env.raw_frame}
            exact += bool(np.array_equal(frame_for_embedding(obs=obs, info=info), node.frame.numpy()))
            sims.append(float(self.encoder.similarity(a=node.embedding, b=self.encoder.encode_one(obs=obs, info=info))))
        self._at.clear()
        out = {"nodes": len(nodes), "failed": failed, "pixel_exact": exact,
               "similarity_min": min(sims) if sims else None, "similarity_mean": sum(sims) / len(sims) if sims else None}
        log_info(f"verify_restores: {out}", parameters=self._parameters)
        return out

    # ------------------------------------------------------------ one iteration

    def iterate(self) -> Optional[dict]:
        self.selector.reseed(iteration=self.iteration)    # a resumed run draws as an uninterrupted one would
        sel = self.selector.select(tree=self.tree)
        if sel is None:
            return None
        self.iteration += 1
        t0 = time.time()
        node = self.tree.nodes[sel.node_id]
        env = self.envs[node.env_key]
        restored = self._restore_node(node=node, env=env)
        if restored is None:
            return {"iteration": self.iteration, "selected": node.id, "failed": "restore"}
        t_load = time.time() - t0
        obs, info = env.current_obs, {"raw_frame": env.raw_frame}
        prev = StateRecord(obs=obs, info=info, embedding=node.embedding)
        exp = self.tree.add_expansion(node_id=node.id, iteration=self.iteration)
        self.expander.start(env=env, iteration=self.iteration)
        parent_id, seg_start = node.id, 0
        n_valid = n_challenge = 0
        episode_over = False
        for t in range(self.k_steps):
            action, meta = self.expander.next_action(env=env, obs=obs, info=info)
            obs2, _r, terminated, truncated, info2 = env.step(action)
            emb = self.encoder.encode_one(obs=obs2, info=info2)
            nxt = StateRecord(obs=obs2, info=info2, embedding=emb, prev_image=prev.embedding.image)
            challenge = bool(info2.get("challenge"))
            if challenge:      # a bot check is not new content: score 0 and keep it out of the archive
                value, components = 0.0, {}
                n_challenge += 1
            else:
                value, components = self.scorer.score(prev=prev, action=action, next=nxt, archive=self.archive,
                                                      add=True)
            changed = self.encoder.similarity(a=prev.embedding, b=emb) < self.noop_similarity
            valid = bool(info2.get("valid", True))
            n_valid += valid
            self.expander.observe(action=action, changed=changed, valid=valid, error=info2.get("error"))
            step = Step(action=action, frame=EncodedImage.of(frame_for_embedding(obs=obs2, info=info2)),
                        texts=obs2.get("texts", {}), info=compact_info(info2), value=float(value),
                        components=dict(components), extra={**meta, "changed": changed})
            exp.steps.append(step)
            self.total_steps += 1
            episode_over = terminated or truncated
            end = t == self.k_steps - 1 or episode_over
            if (value >= self.node_threshold and not challenge) or end:
                child = self._make_node(env_key=node.env_key, obs=obs2, info=info2, embedding=emb,
                                        value=float(value), parent=parent_id, segment=(exp.id, seg_start, t))
                step.node_id = child.id
                exp.children.append(child.id)
                parent_id, seg_start = child.id, t + 1
            prev, obs, info = nxt, obs2, info2
            if episode_over:
                break
        if episode_over:
            self._at.pop(node.env_key, None)
        else:
            self._at[node.env_key] = (parent_id, env)
        exp.yield_value = float(self.stats.expansion_yield(expansion=exp))
        self.stats.record_yield(tree=self.tree, node_id=node.id, value=exp.yield_value)
        node.n_expanded += 1
        exp.stats = {"seconds": time.time() - t0, "load_seconds": t_load, **restored,
                     "valid_rate": n_valid / max(1, len(exp.steps)), "challenges": n_challenge}
        k = sel.candidates.index(node.id)
        record = {"iteration": self.iteration, "selected": node.id, "energy": sel.energies[k], "prob": sel.probs[k],
                  "candidates": sel.candidates, "energies": sel.energies, "probs": sel.probs, "q": sel.q, "n": sel.n,
                  "p": sel.p, "n_total": sel.n_total, "expansion": exp.id, "steps": len(exp.steps),
                  "children": list(exp.children), "yield": exp.yield_value, **exp.stats,
                  "n_nodes": len(self.tree.nodes), "archive": len(self.archive)}
        self.records.append(record)
        if self.store is not None:
            self.store.commit(search=self, expansion=exp, record=record)
        log_info(f"iter {self.iteration}: node {node.id} (p={sel.probs[k]:.3f}, E={sel.energies[k]:.3f}, "
                 f"restore {restored['restore']} {t_load:.1f}s) -> {len(exp.steps)} steps, {len(exp.children)} "
                 f"children, yield {exp.yield_value:.3f}; tree {len(self.tree.nodes)} nodes, archive "
                 f"{len(self.archive)} ({exp.stats['seconds']:.1f}s)", parameters=self._parameters)
        if self.logger is not None:
            self.logger.on_iteration(search=self, record=record)
        return record

    def run(self, *, stop: Callable) -> str:
        """Iterate until stop(search) returns a reason (or no node is restorable)."""
        if not self.tree.nodes:
            self.add_roots()
        while True:
            reason = stop(self)
            if reason is not None:
                return reason
            if self.iterate() is None:
                return "no restorable node"
