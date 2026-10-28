"""
A finished search run's nodes restored in a fresh env of their scene (practice starts tasks from them). GameBoy and
Web nodes import the state the search exported (states/); Android nodes, which have none, replay the root -> node
actions from the scene's reset and are checked against the node's stored embedding with the run's own encoder, as the
search checks its replays. With state_cache, a replayed node's state is saved on its env and later restores of it (a
node's other tasks and tries) load it, and of its descendants replay only from it, as the search's state cache does.
"""
import json
import os
import threading
import time
import weakref
from collections import OrderedDict
from typing import Optional
import torch
from cusi.search.store import EXPANSIONS_FILE, node_rows, read_rows
from cusi.utils.compute import gpu_work
from cusi.utils.log_handling import log_warn

#: Android keeps emulator snapshots of its recent nodes (~3.6 GB each, cusi.envs.android_world) to replay from.
DEFAULT_STATE_CACHE = {"gameboy": 0, "web": 0, "android": 24}


class RestoreFailed(RuntimeError):
    """A node could not be put back in its env (a missing or failing state, or a diverged replay)."""


class NodeRestorer:
    """Shared by worker threads, each restoring into its own env. state_cache: saved states kept per env (replayed
    nodes only), least recently used evicted."""

    def __init__(self, *, directory: str, parameters: dict, state_cache: int = 0) -> None:
        self.directory, self._parameters, self.state_cache = directory, parameters, state_cache
        self.nodes = node_rows(directory=directory)
        self._actions = {e["id"]: [s["action"] for s in e["steps"]]
                         for e in read_rows(os.path.join(directory, EXPANSIONS_FILE))}
        with open(os.path.join(directory, "config.json")) as f:
            config = json.load(f)
        self.config = config.get("config", config)
        self._encoder = None
        self._imported = weakref.WeakKeyDictionary()     # env -> {node id: its state id on that env}
        self._cache = weakref.WeakKeyDictionary()        # env -> OrderedDict(node id -> state id), LRU first
        self.stats = {"cached": 0, "replay": 0, "replay_from_cache": 0}
        self._lock = threading.Lock()

    def path_actions(self, *, node_id: int, since: Optional[int] = None) -> list:
        """The root -> node actions, oldest first; with since (an ancestor of node_id), only those after it."""
        chain, nid = [], node_id
        while nid is not None and nid != since:
            chain.append(nid)
            nid = self.nodes[nid]["parent"]
        actions = []
        for nid in reversed(chain):
            seg = self.nodes[nid]["segment"]
            if seg is not None:
                exp_id, first, last = seg
                actions += self._actions[exp_id][first:last + 1]
        return actions

    def restore(self, *, env, node_id: int) -> tuple:
        """Put env in the node's state -> (obs, info); raises RestoreFailed."""
        node = self.nodes[node_id]
        if node.get("state"):
            with self._lock:
                imported = self._imported.setdefault(env, {})
            if node_id not in imported:
                path = os.path.join(self.directory, "states", node["state"])
                if not os.path.exists(path):
                    raise RestoreFailed(f"node {node_id}: no state file {path}")
                imported[node_id] = env.import_state(path=path)
            try:
                env.load_state(state_id=imported[node_id])
            except RuntimeError as e:
                raise RestoreFailed(f"node {node_id}: load_state failed ({e})") from e
            return env.current_obs, {"raw_frame": env.raw_frame}
        return self._replay(env=env, node_id=node_id)

    def _cache_of(self, *, env) -> OrderedDict:
        with self._lock:
            return self._cache.setdefault(env, OrderedDict())

    def _cache_put(self, *, env, node_id: int) -> None:
        """Save env's state (it is in node_id's), evicting the least recently used beyond state_cache."""
        cache = self._cache_of(env=env)
        try:
            cache[node_id] = env.save_state()
        except RuntimeError as e:
            log_warn(f"restore state cache: saving node {node_id} failed ({e})", parameters=self._parameters)
            return
        while len(cache) > self.state_cache:
            old_id = cache.popitem(last=False)[1]
            if env.has_state(state_id=old_id):
                try:
                    env.delete_state(state_id=old_id)
                except RuntimeError as e:
                    log_warn(f"restore state cache: deleting a state failed ({e})", parameters=self._parameters)

    def _cache_load(self, *, env, node_id: int) -> tuple:
        """Load the saved state of node_id or of its nearest cached ancestor -> (that node's id, or None when there
        was none and the scene was reset; obs; info)."""
        cache = self._cache_of(env=env)
        nid = node_id
        while nid is not None:
            state_id = cache.get(nid)
            if state_id is not None:
                if not env.has_state(state_id=state_id):
                    del cache[nid]      # the env restarted its emulator
                else:
                    try:
                        env.load_state(state_id=state_id)
                        cache.move_to_end(nid)
                        return nid, env.current_obs, {"raw_frame": env.raw_frame}
                    except RuntimeError as e:
                        log_warn(f"restore state cache: loading node {nid} failed ({e})", parameters=self._parameters)
                        del cache[nid]
                        if env.has_state(state_id=state_id):
                            env.delete_state(state_id=state_id)
            nid = self.nodes[nid]["parent"]
        return None, *env.reset()

    def _replay(self, *, env, node_id: int) -> tuple:
        if self.state_cache:
            start, obs, info = self._cache_load(env=env, node_id=node_id)
            if start == node_id:
                self.stats["cached"] += 1
                return obs, info
        else:
            start, (obs, info) = None, env.reset()
        self.stats["replay_from_cache" if start is not None else "replay"] += 1
        path = self.path_actions(node_id=node_id, since=start)
        for i, action in enumerate(path):
            obs, _r, terminated, truncated, info = env.step(action)
            if (terminated or truncated) and i < len(path) - 1:
                raise RestoreFailed(f"node {node_id}: the episode ended at replay step {i + 1}/{len(path)}")
        threshold = self.config["replay_similarity"]
        sim = self._similarity(node_id=node_id, obs=obs, info=info)
        if sim < threshold and hasattr(env, "observe"):
            # Android: the a11y tree can lag the screen for a moment (no text, or the previous screen's): look again.
            time.sleep(2.0)
            obs, info = env.observe()
            sim = max(sim, self._similarity(node_id=node_id, obs=obs, info=info))
        if sim < threshold:
            raise RestoreFailed(f"node {node_id}: replay reached similarity {sim:.3f} < {threshold}")
        if self.state_cache:
            self._cache_put(env=env, node_id=node_id)
        return obs, info

    def _similarity(self, *, node_id: int, obs: dict, info: dict) -> float:
        encoder = self._get_encoder()
        path, key = self.nodes[node_id]["embedding"].split("#")
        stored = torch.load(os.path.join(self.directory, path), weights_only=False)[int(key)]
        with gpu_work("embed", key="one"):
            current = encoder.encode_one(obs=obs, info=info)
        return float(encoder.similarity(a=stored, b=current))

    def _get_encoder(self):
        with self._lock:
            return self._build_encoder()

    def _build_encoder(self):
        if self._encoder is None:
            from cusi.state import build_encoder
            config = dict(self.config)
            load_path: Optional[str] = config.get("embedder_load_path")
            if load_path and not os.path.isabs(load_path):
                config["embedder_load_path"] = os.path.join(self._parameters["storage_dir"], load_path)
            self._encoder = build_encoder(config=config, env_name=self.config["env"], parameters=self._parameters)
        return self._encoder
