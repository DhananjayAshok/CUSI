"""
--expander curiosity (plans/archive/next_plan.md §E.5): expanding a node trains the policy from it in one curiosity RL
session, starting from the weights the previous session ended with (the first session's policy is fresh; a resumed
search starts fresh again); the expansion is the session's best episode, materialised by replaying its actions from the node through the
search's ordinary step loop (scored against the global archive). GameBoy replays are checked frame by frame and a
mismatch fails the run; Web and Android replays are checked by embedding similarity, and the expansion stops at the
first diverging step (expander_info["materialise_diverged"]). With config.keep_episodes > 1 the session's top
distinct episodes are each committed as their own expansion of the node (n_plans / begin_plan), so more of what a
session explored reaches the tree and the archive.
"""
import json
import os
import tempfile
from typing import Any, Callable, Optional
import numpy as np
import torch
from cusi.utils.log_handling import log_info, log_warn
from cusi.search.curiosity.overlay import local_base
from cusi.search.curiosity.session import CuriositySession, PPOConfig, RestoreFailed
from cusi.state import frame_for_embedding

REWARD_BUFFERS = ("global", "local")
POLICY_DIR = "policies"
CURIOSITY_DIR = "curiosity"


class MaterialiseMismatch(RuntimeError):
    """A deterministic (GameBoy) replay of the best episode did not reproduce its frames."""


class CuriosityExpander:
    name = "curiosity"

    def __init__(self, *, env_name: str, observer, config: PPOConfig, reward_buffer: str = "global",
                 make_copy: Optional[Callable] = None, keep_policies: str = "all", seed: int = 0,
                 device: str = "cpu", replay_similarity: float = 0.95, parameters: dict[str, Any] = None) -> None:
        """observer: the policy's observer (CNNObserver). make_copy(env_key) -> a new env of that
        scene, for parallel copies (config.num_envs > 1; GameBoy)."""
        if reward_buffer not in REWARD_BUFFERS:
            raise ValueError(f"reward_buffer must be one of {REWARD_BUFFERS}, got {reward_buffer!r}")
        if config.num_envs > 1 and make_copy is None:
            raise ValueError("num_envs > 1 needs make_copy")
        self.env_name, self.observer, self.cfg, self.reward_buffer = env_name, observer, config, reward_buffer
        self.make_copy, self.keep_policies, self.seed, self.device = make_copy, keep_policies, seed, device
        self.replay_similarity, self._parameters = replay_similarity, parameters
        self._copies: dict = {}       # env key -> [extra envs]
        self._plan: list = []
        self._extras: list = []
        self._info: dict = {}
        self._t = 0
        self._search = None
        self._best = None
        self._episodes: list = []
        self._policy_state: Optional[dict] = None    # the last session's final weights, the next one's start
        self.n_plans = 0

    # ------------------------------------------------------------ env copies

    def _copies_for(self, env_key: str) -> list:
        need = self.cfg.num_envs - 1
        have = self._copies.setdefault(env_key, [])
        while len(have) < need:
            have.append(self.make_copy(env_key))
        return have[:need]

    def _restorers(self, *, search, node, env) -> tuple:
        """(envs, restore(i) -> obs): copy 0 is the search's env; extra copies import the node's saved state."""
        envs = [env]
        if node.state_id is not None:
            fns = [lambda: (env.load_state(state_id=node.state_id), env.current_obs)[1]]
            if self.cfg.num_envs > 1:
                with tempfile.TemporaryDirectory() as d:
                    name = env.export_state(state_id=node.state_id, directory=d)
                    for copy in self._copies_for(node.env_key):
                        sid = copy.import_state(path=os.path.join(d, name))
                        envs.append(copy)
                        fns.append(lambda c=copy, s=sid: (c.load_state(state_id=s), c.current_obs)[1])
        else:
            if self.cfg.num_envs > 1:
                raise ValueError("parallel copies need saved node states (--restore state)")

            def replay():
                search._at.clear()
                if search._restore_node(node=node, env=env) is None:
                    raise RestoreFailed(f"node {node.id} could not be restored during its curiosity session")
                return env.current_obs
            fns = [replay]
        return envs, (lambda i: fns[i]())

    # ------------------------------------------------------------ expander interface

    def start(self, *, env, iteration: int, tree=None, node=None, expansion_id: Optional[int] = None,
              search=None) -> None:
        if search is None or node is None:
            raise ValueError("the curiosity expander needs the search and the node")
        self._search, self._exp_id, self._t = search, expansion_id, 0
        self.aborted, self.n_plans = None, 0
        envs, restore = self._restorers(search=search, node=node, env=env)
        if self.reward_buffer == "global":
            base = search.archive
        else:
            base = local_base(like=search.archive, embedding=node.embedding, obs=env.current_obs)
        session = CuriositySession(env_name=self.env_name, envs=envs, restore=restore,
                                   encoder=search.encoder, scorer=search.scorer, base_archive=base,
                                   observer=self.observer, config=self.cfg, seed=int(np.random.SeedSequence(
                                       [self.seed, iteration]).generate_state(1)[0]), device=self.device,
                                   init_state=self._policy_state,
                                   log_every=10 if self.env_name == "gameboy" else 1,
                                   parameters=self._parameters)
        result = session.run()
        self._policy_state = result.policy_state
        best = result.best
        if best is None:
            # Possible only when the node could not be restored (Android replay): the search skips this iteration.
            self.aborted = f"no episode finished ({result.stopped})"
            return
        self._episodes = list(result.top) or [best]
        files = self._write_files(search=search, result=result)
        returns = np.asarray(result.returns, dtype=np.float32)
        summary = [{"episodes": [i, min(i + 100, len(returns))], "mean": float(returns[i:i + 100].mean()),
                    "max": float(returns[i:i + 100].max())} for i in range(0, len(returns), 100)]
        self._info = {"reward_buffer": self.reward_buffer, "total_steps": result.total_steps,
                      "episodes": len(returns), "returns_summary": summary, "returns": files.get("returns"),
                      "true_log": files.get("true_log"),
                      "best_episode": best.index, "best_return": best.ret, "updates": result.updates,
                      "policy": files.get("policy"), "materialise": "replay", "seconds": round(result.seconds, 1),
                      "num_envs": len(envs)}
        log_info(f"curiosity session from node {node.id}: {result.total_steps} steps, {len(returns)} episodes, "
                 f"best episode {best.index} return {best.ret:.3f} ({len(best.actions)} actions), "
                 f"{result.seconds:.0f}s", parameters=self._parameters)
        self._info["stopped_early"] = result.stopped
        self.n_plans = len(self._episodes)
        self.begin_plan(plan=0)
        try:
            restore(0)                # back to the node: the search replays the best episode from here
        except RestoreFailed as e:
            self.aborted = f"could not restore the node to materialise the best episode ({e})"

    def begin_plan(self, *, plan: int) -> None:
        """Materialise the session's plan-th episode next (0: the best); the env must be back at the node."""
        ep = self._episodes[plan]
        self._best, self._t = ep, 0
        self._plan, self._extras = list(ep.actions), list(ep.extras)
        self._info.pop("diverged_at", None)
        self._info.update({"plan": plan, "plans": self.n_plans, "episode": ep.index, "episode_return": ep.ret,
                           "materialise_diverged": False})

    def _write_files(self, *, search, result) -> dict:
        store = getattr(search, "store", None)
        if store is None:
            return {}
        out = {}
        os.makedirs(os.path.join(store.directory, CURIOSITY_DIR), exist_ok=True)
        rel = f"{CURIOSITY_DIR}/{self._exp_id}_returns.npy"
        np.save(os.path.join(store.directory, rel), np.asarray(result.returns, dtype=np.float32))
        out["returns"] = rel
        if result.true_log:       # every session step's true state, for coverage (cusi/search/coverage.py)
            rel = f"{CURIOSITY_DIR}/{self._exp_id}_true.jsonl"
            with open(os.path.join(store.directory, rel), "w") as f:
                f.writelines(json.dumps(row) + "\n" for row in result.true_log)
            out["true_log"] = rel
        if self.keep_policies == "all":
            os.makedirs(os.path.join(store.directory, POLICY_DIR), exist_ok=True)
            rel = f"{POLICY_DIR}/{self._exp_id}.pt"
            torch.save(result.policy_state, os.path.join(store.directory, rel))
            out["policy"] = rel
        return out

    def info(self) -> dict:
        return dict(self._info)

    def next_action(self, *, env, obs: dict, info: dict) -> tuple:
        if self._t >= len(self._plan):
            raise MaterialiseMismatch(f"the replay outlived the best episode ({len(self._plan)} actions)")
        action, extra = self._plan[self._t], self._extras[self._t]
        return action, dict(extra)

    def observe(self, *, action: str, result: tuple) -> None:
        pass

    def after_step(self, *, obs: dict, info: dict, embedding) -> bool:
        """Check the replayed step against the session's; False ends the expansion here (diverged)."""
        t = self._t
        self._t += 1
        best = self._best
        if self.env_name == "gameboy":
            frame = np.asarray(frame_for_embedding(obs=obs, info=info), dtype=np.uint8)
            if not np.array_equal(frame, best.frames[t]):
                raise MaterialiseMismatch(f"replaying the best episode diverged at step {t + 1}: GameBoy should be "
                                          "deterministic")
            return True
        sim = self._search.encoder.similarity(a=embedding, b=best.embeddings[t])
        if sim < self.replay_similarity:
            log_warn(f"best-episode replay diverged at step {t + 1} (similarity {sim:.3f}); the expansion stops there",
                     parameters=self._parameters)
            self._info["materialise_diverged"] = True
            self._info["diverged_at"] = t + 1
            return False
        return True

    def done(self) -> bool:
        return self._t >= len(self._plan)

    def close(self) -> None:
        for envs in self._copies.values():
            for e in envs:
                try:
                    e.close()
                except Exception:
                    pass
        self._copies = {}
