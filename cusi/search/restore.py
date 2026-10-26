"""
A finished search run's nodes restored in a fresh env of their scene (practice starts tasks from them). GameBoy and
Web nodes import the state the search exported (states/); Android nodes, which have none, replay the root -> node
actions from the scene's reset and are checked against the node's stored embedding with the run's own encoder, as the
search checks its replays.
"""
import json
import os
import threading
import time
import weakref
from typing import Optional
import torch
from cusi.search.store import EXPANSIONS_FILE, node_rows, read_rows
from cusi.utils.compute import gpu_work


class RestoreFailed(RuntimeError):
    """A node could not be put back in its env (a missing or failing state, or a diverged replay)."""


class NodeRestorer:
    """Shared by worker threads, each restoring into its own env."""

    def __init__(self, *, directory: str, parameters: dict) -> None:
        self.directory, self._parameters = directory, parameters
        self.nodes = node_rows(directory=directory)
        self._actions = {e["id"]: [s["action"] for s in e["steps"]]
                         for e in read_rows(os.path.join(directory, EXPANSIONS_FILE))}
        with open(os.path.join(directory, "config.json")) as f:
            config = json.load(f)
        self.config = config.get("config", config)
        self._encoder = None
        self._imported = weakref.WeakKeyDictionary()     # env -> {node id: its state id on that env}
        self._lock = threading.Lock()

    def path_actions(self, *, node_id: int) -> list:
        """The root -> node actions, oldest first."""
        chain, nid = [], node_id
        while nid is not None:
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

    def _replay(self, *, env, node_id: int) -> tuple:
        obs, info = env.reset()
        path = self.path_actions(node_id=node_id)
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
