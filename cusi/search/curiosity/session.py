"""
One curiosity RL session (plans/next_plan.md §E.4–E.5.1): a fresh policy trained with PPO from one node, every episode
restoring the node, rewards from the stationary archive B0 ∪ the episode's own states. The session returns its best
episode (by its own return) and the final policy.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import numpy as np
import torch
import torch.nn as nn
from cusi.utils.log_handling import log_info, log_warn
from cusi.search.curiosity.candidates import TYPES, candidates_for
from cusi.search.curiosity.overlay import OverlayArchive
from cusi.search.curiosity.policy import TEXT_DIM, FusionPolicy
from cusi.state import StateRecord, frame_for_embedding


@dataclass
class PPOConfig:
    """GameBoyRL's cleanrl ppo_curiosity settings."""
    total_steps: int = 100_000
    max_minutes: float = 60.0
    k_steps: int = 30
    num_envs: int = 1
    rollout: int = 128
    minibatches: int = 4
    epochs: int = 4
    lr: float = 2.5e-4
    anneal_lr: bool = True
    clip: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    invalid_action_penalty: float = 0.1
    max_type_strings: int = 8
    keep_episodes: int = 1          # the session's top episodes (by return, distinct actions) it returns


@dataclass
class Episode:
    index: int
    actions: list = field(default_factory=list)
    frames: list = field(default_factory=list)      # GameBoy: unlabelled frames after each action (exact checks)
    embeddings: list = field(default_factory=list)  # other envs: state embeddings after each action
    rewards: list = field(default_factory=list)
    extras: list = field(default_factory=list)      # per step: logp, value, entropy, n_candidates, candidate
    terminated: bool = False

    @property
    def ret(self) -> float:
        return float(sum(self.rewards))


def keep_top(*, top: list, ep: Episode, k: int) -> None:
    """Insert ep into top (best first, at most k, no two with the same actions); among equal returns the earlier
    episode ranks first."""
    same = next((e for e in top if e.actions == ep.actions), None)
    if same is not None:
        if ep.ret <= same.ret:
            return
        top.remove(same)
    top.append(ep)
    top.sort(key=lambda e: -e.ret)
    del top[k:]


@dataclass
class SessionResult:
    best: Optional[Episode]
    returns: list
    total_steps: int
    updates: int
    seconds: float
    policy_state: dict
    stopped: Optional[str] = None          # why the session ended early, if it did
    top: list = field(default_factory=list)    # the best config.keep_episodes distinct episodes, best first


class RestoreFailed(RuntimeError):
    """The session's start node could not be restored for an episode."""


class CuriositySession:
    def __init__(self, *, env_name: str, scene: Optional[str], envs: list, restore: Callable, encoder, scorer,
                 base_archive, frozen, config: PPOConfig, seed: int, device: str, strings: Optional[dict] = None,
                 log_every: int = 10, parameters: dict[str, Any] = None) -> None:
        """envs: the env copies; restore(i) -> obs puts copy i in the start node's state. base_archive: B0 (read
        only). frozen: FrozenEncoders."""
        self.env_name, self.scene, self.envs, self.restore = env_name, scene, envs, restore
        self.encoder, self.scorer, self.base, self.frozen = encoder, scorer, base_archive, frozen
        self.cfg, self.seed, self.device, self.strings = config, seed, device, strings
        self.log_every, self._parameters = log_every, parameters

    # ------------------------------------------------------------ helpers

    def _observe(self, obs_list: list, info_list: list, prevs: list) -> tuple:
        """(features, candidates, elements) per env, with the previous action's frozen inputs attached."""
        elements_list, cands_list, frames = [], [], []
        for obs, info in zip(obs_list, info_list):
            elements, cands = candidates_for(env_name=self.env_name, obs=obs, scene=self.scene, strings=self.strings,
                                             max_type_strings=self.cfg.max_type_strings)
            elements_list.append(elements)
            cands_list.append(cands)
            frames.append(np.asarray(frame_for_embedding(obs=obs, info=info), dtype=np.uint8))
        feats = self.frozen.observe(frames=frames, elements_list=elements_list, candidates_list=cands_list)
        for f, prev in zip(feats, prevs):
            if prev is None:
                f["p_type"], f["p_target_text"], f["p_arg"] = -1, torch.zeros(TEXT_DIM).half(), \
                    torch.zeros(TEXT_DIM).half()
            else:
                cand, target_text = prev
                f["p_type"] = TYPES.index(cand.type)
                f["p_target_text"], f["p_arg"] = self.frozen.embed_texts(texts=[target_text, cand.argument or ""]).half()
        return feats, cands_list, elements_list

    @staticmethod
    def _info(env, info: Optional[dict]) -> dict:
        return {"raw_frame": env.raw_frame, **(info or {})}

    # ------------------------------------------------------------ the session

    def run(self) -> SessionResult:
        cfg = self.cfg
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        policy = FusionPolicy().to(self.device)
        optim = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
        n = len(self.envs)
        overlays = [OverlayArchive(base=self.base) for _ in range(n)]
        t0 = time.time()
        deadline = t0 + 60.0 * cfg.max_minutes
        steps = updates = n_episodes = 0
        returns, top = [], []

        def start_episode(i):
            nonlocal n_episodes
            obs = self.restore(i)
            info = self._info(self.envs[i], None)
            overlays[i].new_episode()
            ep = Episode(index=n_episodes)
            n_episodes += 1
            rec = StateRecord(obs=obs, info=info, embedding=self.encoder.encode_one(obs=obs, info=info))
            return {"obs": obs, "info": info, "rec": rec, "prev": None, "t": 0, "ep": ep}

        stopped = None
        try:
            live = [start_episode(i) for i in range(n)]
            while steps < cfg.total_steps and time.time() < deadline:
                buf = []                       # per step: list over envs of dicts
                for _ in range(cfg.rollout):
                    if steps >= cfg.total_steps or time.time() >= deadline:
                        break
                    feats, cands, elements = self._observe([s["obs"] for s in live], [s["info"] for s in live],
                                                           [s["prev"] for s in live])
                    with torch.no_grad():
                        logits, values, _ = policy(FusionPolicy.collate(feats=feats, device=self.device))
                        dist = torch.distributions.Categorical(logits=logits)
                        choice = dist.sample()
                        logp, ent = dist.log_prob(choice), dist.entropy()
                    row = []
                    for i, s in enumerate(live):
                        k = int(choice[i])
                        cand = cands[i][k]
                        env = self.envs[i]
                        obs2, _r, terminated, truncated, info2 = env.step(cand.action)
                        info2 = self._info(env, info2)
                        emb = self.encoder.encode_one(obs=obs2, info=info2)
                        nxt = StateRecord(obs=obs2, info=info2, embedding=emb, prev_image=s["rec"].embedding.image)
                        reward, _ = self.scorer.score(prev=s["rec"], action=cand.action, next=nxt, archive=overlays[i],
                                                      add=True)
                        if not info2.get("valid", True):
                            reward -= cfg.invalid_action_penalty
                        s["t"] += 1
                        ep = s["ep"]
                        ep.actions.append(cand.action)
                        if self.env_name == "gameboy":
                            ep.frames.append(np.asarray(frame_for_embedding(obs=obs2, info=info2), dtype=np.uint8).copy())
                        else:
                            ep.embeddings.append(emb)
                        ep.rewards.append(float(reward))
                        ep.extras.append({"session_reward": float(reward), "logp": float(logp[i]),
                                          "value": float(values[i]), "entropy": float(ent[i]),
                                          "n_candidates": len(cands[i]), "candidate": k})
                        ended = bool(terminated) or s["t"] >= cfg.k_steps or bool(truncated)
                        item = {"feat": feats[i], "choice": k, "logp": float(logp[i]), "value": float(values[i]),
                                "reward": float(reward), "done": ended, "terminated": bool(terminated),
                                "trunc_value": 0.0}
                        target_text = elements[i][cand.target].text if cand.target is not None else ""
                        if ended:
                            ep.terminated = bool(terminated)
                            if not terminated:   # truncation: bootstrap from the value of where it stopped
                                f2, _c, _e = self._observe([obs2], [info2], [(cand, target_text)])
                                with torch.no_grad():
                                    item["trunc_value"] = float(policy(FusionPolicy.collate(feats=f2, device=self.device))[1][0])
                            returns.append(ep.ret)
                            keep_top(top=top, ep=ep, k=max(1, cfg.keep_episodes))
                            live[i] = start_episode(i)
                        else:
                            s.update(obs=obs2, info=info2, rec=nxt, prev=(cand, target_text))
                        row.append(item)
                        steps += 1
                    buf.append(row)
                if not buf:
                    break
                # bootstrap values for unfinished episodes
                feats, _c, _e = self._observe([s["obs"] for s in live], [s["info"] for s in live],
                                              [s["prev"] for s in live])
                with torch.no_grad():
                    last_values = policy(FusionPolicy.collate(feats=feats, device=self.device))[1].cpu().numpy()
                frac = 1.0 - max(steps / cfg.total_steps, (time.time() - t0) / (deadline - t0))
                if cfg.anneal_lr:
                    optim.param_groups[0]["lr"] = cfg.lr * max(frac, 0.0)
                stats = self._update(policy=policy, optim=optim, buf=buf, last_values=last_values, rng=rng)
                updates += 1
                if updates % self.log_every == 0 or steps >= cfg.total_steps:
                    recent = returns[-100:]
                    log_info(f"curiosity session: {steps} steps, {len(returns)} episodes, {updates} updates, "
                             f"{time.time() - t0:.0f}s; return mean(last 100) {np.mean(recent) if recent else 0:.3f}, "
                             f"best {top[0].ret if top else 0:.3f}; {stats}", parameters=self._parameters)
        except RestoreFailed as e:
            # The node could not be restored for a new episode (Android replay): end the session with what it has.
            stopped = str(e)
            log_warn(f"curiosity session stopped early after {steps} steps: {e}", parameters=self._parameters)
        policy_state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        return SessionResult(best=top[0] if top else None, top=list(top), returns=returns, total_steps=steps,
                             updates=updates,
                             seconds=time.time() - t0, policy_state=policy_state, stopped=stopped)

    # ------------------------------------------------------------ PPO

    def _update(self, *, policy, optim, buf: list, last_values: np.ndarray, rng) -> dict:
        cfg = self.cfg
        T, n = len(buf), len(buf[0])
        rewards = np.array([[it["reward"] for it in row] for row in buf], dtype=np.float32)
        values = np.array([[it["value"] for it in row] for row in buf], dtype=np.float32)
        dones = np.array([[it["done"] for it in row] for row in buf], dtype=np.float32)
        trunc = np.array([[it["trunc_value"] for it in row] for row in buf], dtype=np.float32)
        adv = np.zeros_like(rewards)
        last = np.zeros(n, dtype=np.float32)
        for t in reversed(range(T)):
            next_value = last_values if t == T - 1 else values[t + 1]
            delta = rewards[t] + cfg.gamma * (next_value * (1 - dones[t]) + trunc[t] * dones[t]) - values[t]
            last = delta + cfg.gamma * cfg.gae_lambda * (1 - dones[t]) * last
            adv[t] = last
        returns = adv + values
        items = [it for row in buf for it in row]
        adv_f, ret_f = adv.reshape(-1), returns.reshape(-1)
        idx = np.arange(len(items))
        mb = max(1, len(items) // cfg.minibatches)
        clipfracs, losses = [], []
        for _ in range(cfg.epochs):
            rng.shuffle(idx)
            for s0 in range(0, len(items), mb):
                sel = idx[s0:s0 + mb]
                batch = FusionPolicy.collate(feats=[items[j]["feat"] for j in sel], device=self.device)
                logits, new_v, _ = policy(batch)
                dist = torch.distributions.Categorical(logits=logits)
                choice = torch.tensor([items[j]["choice"] for j in sel], device=self.device)
                new_logp, ent = dist.log_prob(choice), dist.entropy()
                old_logp = torch.tensor([items[j]["logp"] for j in sel], device=self.device)
                old_v = torch.tensor([items[j]["value"] for j in sel], device=self.device)
                a = torch.tensor(adv_f[sel], device=self.device)
                r = torch.tensor(ret_f[sel], device=self.device)
                if cfg.norm_adv and len(sel) > 1:
                    a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = (new_logp - old_logp).exp()
                pg = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                v_clipped = old_v + (new_v - old_v).clamp(-cfg.clip, cfg.clip)
                v_loss = 0.5 * torch.max((new_v - r) ** 2, (v_clipped - r) ** 2).mean()
                loss = pg - cfg.ent_coef * ent.mean() + cfg.vf_coef * v_loss
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optim.step()
                clipfracs.append(float(((ratio - 1).abs() > cfg.clip).float().mean()))
                losses.append(float(loss))
        return {"loss": round(float(np.mean(losses)), 4), "clipfrac": round(float(np.mean(clipfracs)), 3)}
