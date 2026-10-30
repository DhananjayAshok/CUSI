"""
The tree search loop: select a node, restore it, expand it, score the steps against one never-reset archive.
"""
import time
from collections import OrderedDict
from typing import Any, Callable, Optional
import numpy as np
from cusi.utils.compute import Tally, gpu_work, part, tally
from cusi.utils.log_handling import log_info, log_warn
from cusi.state import StateRecord, frame_for_embedding
from cusi.agents.records import EncodedImage
from cusi.search.scoring import discovered_set
from cusi.search.selection import EnergySelector
from cusi.search.tree import Step, Tree, boxes_as_dicts, compact_info

RESTORE_MODES = ("state", "replay")


def report_segments(*, tree, describer, env_description: str, exp) -> None:
    """Report every node the expansion made: its parent's screen, then each step from there up to the node,
    described window by window (TransitionDescriber). Steps after the last node are not reported."""
    for cid in exp.children:
        child = tree.nodes[cid]
        if child.flags.get("challenge_page"):
            continue
        _, first, last = child.segment
        report = describer(env_description=env_description, start_frame=tree.nodes[child.parent].frame,
                           steps=exp.steps[first:last + 1])
        for key, value in report.items():       # description, discovered, window_descriptions
            setattr(child, key, value)


class TreeSearch:
    def __init__(self, *, envs, encoder, scorer, archive, expander, stats, selector: EnergySelector,
                 k_steps: int, node_threshold: float, closing_node: bool, archive_nodes_only: bool,
                 features=None, describer=None, decision_scorer=None, score_top_m: int = 8,
                 restore: str = "state", replay_similarity: float = 0.95, state_cache: int = 0, store=None,
                 logger=None, parameters: dict[str, Any] = None) -> None:
        """envs: env key (scene) -> env, a dict or cusi.search.envs.SceneEnvs. features, describer and
        decision_scorer come together (the VLM explorer's search) or not at all: features, a VLMFeatures run on every
        new node; describer, a TransitionDescriber that reports each new node's segment (its parent's screen, then
        every step to it), whose discovered features make up the scene's discovered set; decision_scorer, a VLMScorer
        that, at each decision, re-scores the score_top_m candidates with the highest energy (0: all) against their
        scene's discovered set, the scores being P(n). restore: "state" saves and loads a state per node; "replay" saves none and
        rebuilds a node by reset + its root->node actions, accepted when the result's similarity to the node is
        >= replay_similarity. state_cache (replay only): keep saved states of up to this many nodes, least recently
        used evicted, never written to the run (Android: emulator snapshots); a replay then starts from the node's
        nearest cached ancestor (or the node itself) instead of the scene's reset. closing_node: an expansion's last step always becomes a node; without it, only steps
        reaching node_threshold do, so a fruitless expansion adds none. archive_nodes_only: only states that become
        nodes enter the novelty archive (and text buffer), so novelty is the distance to the nearest node.
        store: a SearchStore that commits the roots and every expansion, and resume() reads back."""
        if restore not in RESTORE_MODES:
            raise ValueError(f"restore must be one of {RESTORE_MODES}, got {restore!r}")
        if state_cache and restore != "replay":
            raise ValueError("state_cache goes with restore='replay'")
        self.envs, self.encoder, self.scorer, self.archive = envs, encoder, scorer, archive
        self.expander, self.stats, self.selector = expander, stats, selector
        # An expander may expand only some nodes (eligible(node, search)): the selector draws among those
        eligible = getattr(expander, "eligible", None)
        self.selector.eligible = None if eligible is None else (lambda node: eligible(node=node, search=self))
        if len({features is None, describer is None, decision_scorer is None}) > 1:
            raise ValueError("features, describer and decision_scorer come together or not at all")
        self.features, self.describer, self.decision_scorer = features, describer, decision_scorer
        self.annotated, self.score_top_m = features is not None, score_top_m
        self.node_scores: dict = {}       # node id -> latest decision score (never scored: 1.0)
        self._n_discovered = None
        self.k_steps, self.node_threshold = k_steps, node_threshold
        self.closing_node, self.archive_nodes_only = closing_node, archive_nodes_only
        self.restore, self.replay_similarity, self.state_cache = restore, replay_similarity, state_cache
        self.store, self.logger, self._parameters = store, logger, parameters
        self.tree = Tree()
        self.iteration = 0
        self.total_steps = 0
        self.records: list = []
        # env key -> (node id, env, its device_epoch) the env is currently in, so its restore can be skipped
        self._at: dict = {}
        self._cache: OrderedDict = OrderedDict()    # node id -> (env, state id), least recently used first
        self.max_expansions: Optional[int] = None    # caps the expansions one iteration may add (several plans)
        self.roots_compute: dict = {}     # scene -> compute of making its root (cusi.utils.compute)

    # ------------------------------------------------------------ nodes

    def _encode(self, *, obs: dict, info: dict):
        with part("embed"), gpu_work("embed", key="one"):
            return self.encoder.encode_one(obs=obs, info=info)

    def _score(self, *, prev, action, nxt):
        with part("score"), gpu_work("score", key="one"):
            return self.scorer.score(prev=prev, action=action, next=nxt, archive=self.archive,
                                     add=not self.archive_nodes_only)

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
                                  created_at=self.iteration, replay=self.restore == "replay" and not challenge,
                                  text_boxes=obs.get("text_boxes"))
        node.flags.update(flags)
        typable_fields = getattr(env, "typable_fields", None)
        if typable_fields is not None:
            node.typable = typable_fields()
        if self.state_cache and not challenge and parent is not None:     # a root is restored by reset()
            self._cache_put(node_id=node.id, env=env)
        if self.annotated and not challenge:
            f = self.features(env_description=env.env_description, frame=frame, text_boxes=obs.get("text_boxes", []))
            if not f.pop("parsed"):
                node.flags["features_parse_failed"] = True
            for key, value in f.items():
                setattr(node, key, value)
        return node

    def add_roots(self) -> None:
        for key, env in self.envs.items():
            with tally() as t:
                with part("env"):
                    obs, info = env.reset()
                emb = self._encode(obs=obs, info=info)
                if not info.get("challenge"):
                    self.archive.add(emb)
                with part("node"):
                    node = self._make_node(env_key=key, obs=obs, info=info, embedding=emb, value=None, parent=None,
                                           segment=None)
            self.roots_compute[key] = t.snapshot()
            self._mark_at(key=key, node_id=node.id, env=env)
            log_info(f"root {node.id}: scene {key}, features {node.features}, flags {node.flags}",
                     parameters=self._parameters)
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

    def _mark_at(self, *, key: str, node_id: int, env) -> None:
        self._at[key] = (node_id, env, getattr(env, "device_epoch", 0))

    def _cache_put(self, *, node_id: int, env) -> None:
        """Save env's state (it is in node_id's) into the cache, evicting the least recently used beyond its size."""
        if node_id in self._cache:
            self._cache.move_to_end(node_id)
            return
        try:
            state_id = env.save_state()
        except RuntimeError as e:
            log_warn(f"state cache: saving node {node_id} failed ({e})", parameters=self._parameters)
            return
        self._cache[node_id] = (env, state_id)
        while len(self._cache) > self.state_cache:
            old_env, old_id = self._cache.popitem(last=False)[1]
            if old_env.has_state(state_id=old_id):
                try:
                    old_env.delete_state(state_id=old_id)
                except RuntimeError as e:
                    log_warn(f"state cache: deleting a state failed ({e})", parameters=self._parameters)

    def _cache_load(self, *, node, env) -> tuple:
        """Load the saved state of node or of its nearest cached ancestor -> (that node's id, or None when there
        was none and the scene was reset; obs; info)."""
        for nid in self.tree.ancestors(node.id):
            entry = self._cache.get(nid)
            if entry is None:
                continue
            cached_env, state_id = entry
            if cached_env is not env or not env.has_state(state_id=state_id):
                del self._cache[nid]      # the env was rebuilt, or restarted its emulator
                continue
            try:
                env.load_state(state_id=state_id)
            except RuntimeError as e:
                log_warn(f"state cache: loading node {nid} failed ({e})", parameters=self._parameters)
                del self._cache[nid]
                env.delete_state(state_id=state_id)
                continue
            self._cache.move_to_end(nid)
            return nid, env.current_obs, {"raw_frame": env.raw_frame}
        return None, *env.reset()

    def _replay(self, *, node, env) -> tuple:
        """Reset the scene, or load the nearest cached state on the way (state_cache), and redo the actions from
        there to node -> (similarity to the node, failure or None, the node replayed from or None, steps replayed).
        A replay the env had to recover from (its emulator hung or died) starts over once, rather than counting as a
        divergence."""
        for attempt in range(2):
            outcome = self._replay_once(node=node, env=env)
            if outcome is not None:
                return outcome
            if attempt == 0:
                log_warn(f"node {node.id}: env recovered during its replay; replaying again",
                         parameters=self._parameters)
        return 0.0, "the env had to recover twice during the replay", None, 0

    def _replay_once(self, *, node, env) -> Optional[tuple]:
        """One _replay attempt; None when the env recovered (its scene was rebuilt) on the way."""
        if self.state_cache:
            start, obs, info = self._cache_load(node=node, env=env)
        else:
            start, (obs, info) = None, env.reset()
        path = self.tree.path_steps(node.id, since=start)
        for i, step in enumerate(path):
            obs, _r, terminated, truncated, info = env.step(step.action)
            if info.get("env_recovered"):
                return None
            if (terminated or truncated) and i < len(path) - 1:
                return 0.0, f"the episode ended at replay step {i + 1}/{len(path)}", start, len(path)
        sim = float(self.encoder.similarity(a=node.embedding, b=self._encode(obs=obs, info=info)))
        if sim < self.replay_similarity and hasattr(env, "observe"):
            # Android: the a11y tree can lag the screen for a moment (no text, or the previous screen's): look again.
            time.sleep(2.0)
            obs, info = env.observe()
            if info.get("env_recovered"):
                return None
            sim = max(sim, float(self.encoder.similarity(a=node.embedding, b=self._encode(obs=obs, info=info))))
        if sim < self.replay_similarity:
            return sim, f"replay reached similarity {sim:.3f} < {self.replay_similarity}", start, len(path)
        return sim, None, start, len(path)

    def _restore_node(self, *, node, env) -> Optional[dict]:
        """Put env in node's state -> restore stats, or None if it could not be (node is then flagged)."""
        at = self._at.get(node.env_key)
        if at is not None and at[0] == node.id and at[1] is env and at[2] == getattr(env, "device_epoch", 0):
            return {"restore": "skipped"}
        self._at.pop(node.env_key, None)
        if node.state_id is not None:
            try:
                env.load_state(state_id=node.state_id)
            except RuntimeError as e:
                self._unselectable(node=node, flag="unrestorable", message=f"load_state failed ({e})")
                return None
            return {"restore": "state"}
        sim, failure, start, n_steps = self._replay(node=node, env=env)
        if failure is not None:
            self._unselectable(node=node, flag="replay_diverged", message=failure)
            return None
        if self.state_cache:
            if start == node.id:
                return {"restore": "cached"}
            self._cache_put(node_id=node.id, env=env)
        return {"restore": "replay", "replay_from": start, "replay_steps": n_steps, "replay_similarity": sim}

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

    # ------------------------------------------------------------ decision scores

    def env_description(self, key: str) -> str:
        """Without touching the scene's env: on Android, indexing a scene that is not live rebuilds its env."""
        if hasattr(self.envs, "description"):
            return self.envs.description(key)
        return self.envs[key].env_description

    def p_of(self, node) -> float:
        """P(n): the cached decision score (1.0 before the first), or the selector's default without scoring."""
        if not self.annotated:
            return self.selector.default_prior
        return self.node_scores.get(node.id, 1.0)

    def discovered(self, *, env_key: str) -> list:
        return discovered_set(nodes=[n for n in self.tree.nodes if n.env_key == env_key])

    def rescore(self, ranked: list) -> list:
        """Score the top candidates (by energy) against their scene's discovered set; updates the cache."""
        top = ranked if self.score_top_m <= 0 else ranked[:self.score_top_m]
        sets, items = {}, []
        for node in top:
            if node.env_key not in sets:
                sets[node.env_key] = self.discovered(env_key=node.env_key)
            items.append({"node_id": node.id, "features": node.features or [], "discovered": sets[node.env_key],
                          "env_description": self.env_description(node.env_key)})
        entries = self.decision_scorer.score(items=items)
        for e in entries:
            self.node_scores[e["node_id"]] = e["score"]
        self._n_discovered = {k: len(v) for k, v in sets.items()}
        return entries

    def replay_scores(self, *, record: dict) -> None:
        """Rebuild the score cache from a saved selection record (resume)."""
        for e in record.get("scores") or []:
            self.node_scores[e["node_id"]] = e["score"]

    # ------------------------------------------------------------ one iteration

    def iterate(self) -> Optional[dict]:
        self.selector.reseed(iteration=self.iteration)    # a resumed run draws as an uninterrupted one would
        self._n_discovered = None
        # The selection itself (rescoring, for the VLM explorer) is counted with the expansion it chose.
        with tally() as compute:
            with part("select"):
                sel = self.selector.select(tree=self.tree, p_of=self.p_of,
                                           rescore=self.rescore if self.annotated else None)
            if sel is None:
                return None
            self.iteration += 1
            node = self.tree.nodes[sel.node_id]
            env = self.envs[node.env_key]
            t0 = time.time()
            with part("restore"):
                restored = self._restore_node(node=node, env=env)
            if restored is None:
                return {"iteration": self.iteration, "selected": node.id, "failed": "restore"}
            t_load = time.time() - t0
            exp = self.tree.add_expansion(node_id=node.id, iteration=self.iteration)
            exp.expander = self.expander.name
            with part("session"):
                self.expander.start(env=env, iteration=self.iteration, tree=self.tree, node=node, expansion_id=exp.id,
                                    search=self)
            aborted = getattr(self.expander, "aborted", None)
            if aborted:
                # A session that could not restore its node (Android replay): no expansion, the node is set aside.
                self.tree.expansions.pop()
                self._at.pop(node.env_key, None)
                self._unselectable(node=node, flag="replay_diverged", message=f"curiosity session aborted: {aborted}")
                return {"iteration": self.iteration, "selected": node.id, "failed": "session"}
            t_session = time.time() - t0 - t_load
            # a curiosity session can commit several of its episodes, each as its own expansion of the node
            n_plans = getattr(self.expander, "n_plans", 1)
            if self.max_expansions is not None:
                n_plans = max(1, min(n_plans, self.max_expansions - len(self.tree.expansions) + 1))
            record = self._materialise(node=node, env=env, exp=exp, sel=sel, restored=restored, t0=t0,
                                       t_load=t_load, t_session=t_session, compute=compute)
        for plan in range(1, n_plans):
            with tally() as compute:
                t0 = time.time()
                with part("restore"):
                    restored = self._restore_node(node=node, env=env)
                if restored is None:
                    break
                exp = self.tree.add_expansion(node_id=node.id, iteration=self.iteration)
                exp.expander = self.expander.name
                self.expander.begin_plan(plan=plan)
                record = self._materialise(node=node, env=env, exp=exp, sel=sel, restored=restored, t0=t0,
                                           t_load=time.time() - t0, t_session=0.0, compute=compute)
        return record

    def _materialise(self, *, node, env, exp, sel, restored: dict, t0: float, t_load: float, t_session: float,
                     compute: Tally) -> dict:
        """Run the expander's k_steps from node (env is in its state) into exp, make its nodes, describe and commit
        it -> the selection record. compute: the expansion's tally; each step also keeps its own
        (step.extra["compute"]) and the expansion its total (exp.stats["compute"])."""
        obs, info = env.current_obs, {"raw_frame": env.raw_frame}
        prev = StateRecord(obs=obs, info=info, embedding=node.embedding)
        exp.expander_info = self.expander.info()
        checks = getattr(self.expander, "after_step", None)
        parent_id, seg_start = node.id, 0
        n_valid = n_challenge = 0
        episode_over = False
        for t in range(self.k_steps):
            with tally() as step_compute:
                with part("decide"):
                    action, meta = self.expander.next_action(env=env, obs=obs, info=info)
                with part("env"):
                    result = env.step(action)
                obs2, _r, terminated, truncated, info2 = result
                if info2.get("env_recovered"):
                    # The emulator hung or died and the scene was rebuilt: obs2 is the scene's start, not where the
                    # action led, so the step is not recorded and the expansion ends with what it reached so far.
                    log_warn(f"expansion {exp.id}: env recovered at step {t + 1} ({info2['env_recovered'][:120]}); "
                             f"ending the expansion", parameters=self._parameters)
                    episode_over = True     # the env is in no node's state: the next restore must not be skipped
                    break
                emb = self._encode(obs=obs2, info=info2)
                nxt = StateRecord(obs=obs2, info=info2, embedding=emb, prev_image=prev.embedding.image)
                challenge = bool(info2.get("challenge"))
                if challenge:      # a bot check is not new content: score 0 and keep it out of the archive
                    value, components = 0.0, {}
                    n_challenge += 1
                else:
                    value, components = self._score(prev=prev, action=action, nxt=nxt)
                changed = bool(info2.get("changed", True))
                n_valid += bool(info2.get("valid", True))
                with part("observe"):     # the VLM explorer's executor memory and done-check
                    self.expander.observe(action=action, result=result)
                with part("record"):
                    step = Step(action=action, frame=EncodedImage.of(frame_for_embedding(obs=obs2, info=info2)),
                                texts=obs2.get("texts", {}), text_boxes=boxes_as_dicts(obs2.get("text_boxes")),
                                info=compact_info(info2), value=float(value),
                                components=dict(components), extra={self.expander.name: meta, "changed": changed})
                exp.steps.append(step)
                self.total_steps += 1
                episode_over = terminated or truncated
                # a materialised episode (curiosity) ends where it ended, or where its replay diverged
                stop = checks is not None and (not checks(obs=obs2, info=info2, embedding=emb) or self.expander.done())
                end = self.closing_node and (t == self.k_steps - 1 or episode_over or stop)
                # no_node: a screen the env could not save as it is (Web: typed, not submitted)
                if (value >= self.node_threshold and not challenge and not info2.get("no_node")) or end:
                    with part("node"):
                        child = self._make_node(env_key=node.env_key, obs=obs2, info=info2, embedding=emb,
                                                value=float(value), parent=parent_id, segment=(exp.id, seg_start, t))
                        step.node_id = child.id
                        if self.archive_nodes_only and not challenge:     # the node's state is now seen
                            with gpu_work("archive"):
                                self.scorer.remember(next=nxt, archive=self.archive)
                    exp.children.append(child.id)
                    parent_id, seg_start = child.id, t + 1
                prev, obs, info = nxt, obs2, info2
            step.extra["compute"] = step_compute.snapshot()
            if episode_over or stop:
                break
        exp.expander_info = self.expander.info()
        if episode_over or not exp.steps or exp.steps[-1].node_id is None:
            # ended, or past its last node (no closing node): the env is in no node's state, so the next restore runs
            self._at.pop(node.env_key, None)
        else:
            self._mark_at(key=node.env_key, node_id=parent_id, env=env)
        t1 = time.time()
        if self.annotated:
            with part("describe"):
                report_segments(tree=self.tree, describer=self.describer, env_description=env.env_description,
                                exp=exp)
        t_describe = time.time() - t1
        exp.yield_value = float(self.stats.expansion_yield(expansion=exp))
        self.stats.record_yield(tree=self.tree, node_id=node.id, value=exp.yield_value)
        node.n_expanded += 1
        exp.stats = {"seconds": time.time() - t0, "load_seconds": t_load, "session_seconds": t_session, **restored,
                     "describe_seconds": t_describe, "valid_rate": n_valid / max(1, len(exp.steps)),
                     "challenges": n_challenge}
        k = sel.candidates.index(node.id)
        record = {"iteration": self.iteration, "selected": node.id, "energy": sel.energies[k], "prob": sel.probs[k],
                  "candidates": sel.candidates, "energies": sel.energies, "probs": sel.probs, "q": sel.q, "n": sel.n,
                  "p": sel.p, "n_total": sel.n_total, "expansion": exp.id, "steps": len(exp.steps),
                  "children": list(exp.children), "yield": exp.yield_value, **exp.stats,
                  "n_nodes": len(self.tree.nodes), "archive": len(self.archive)}
        if self.annotated:
            record["scores"], record["n_discovered"] = sel.scores, self._n_discovered
        exp.stats["compute"] = compute.snapshot()     # after the record: the selection log keeps only the timings
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
                # An expander that limits which nodes it expands (the typing explorer) says why none is left
                return getattr(self.expander, "exhausted_reason", None) or "no restorable node"
