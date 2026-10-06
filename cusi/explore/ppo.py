"""
Curiosity-driven per-token PPO for the VLM text policy (after GameBoyRL's ppo_curiosity.py).
"""
import json
import os
import time
from typing import Any, Optional
import numpy as np
import torch
from cusi.utils.log_handling import log_info, log_warn
from cusi.envs.action_vocab import ActionVocab
from cusi.explore.curiosity import get_curiosity_module
from cusi.state import StateRecord, build_encoder
from cusi.state.canvas import frame_for_embedding
from cusi.explore.policy import VLMPolicy
from cusi.explore.prompts import MAX_NEW_TOKENS, PolicyPrompter
from cusi.explore.replay import ReplayWriter
from cusi.utils.paths import explore_replay
from cusi.agents.records import EncodedImage


def gae(*, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, next_value: float, gamma: float,
        lam: float) -> tuple:
    """Advantages and returns; dones[t] = transition t ended its episode (no bootstrap across it)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        next_v = next_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


class CuriosityPPO:
    def __init__(self, *, env_name: str, env, out_dir: str, total_steps: int, num_steps: int,
                 max_episode_steps: int, state_config: dict, curiosity_module: str, policy_model: str,
                 learning_rate: float = 1e-5, value_learning_rate: float = 1e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95, num_minibatches: int = 8, update_epochs: int = 2,
                 clip_coef: float = 0.2, clip_vloss: bool = True, ent_coef: float = 0.0, vf_coef: float = 0.5,
                 kl_coef: float = 0.05, max_grad_norm: float = 0.5, target_kl: Optional[float] = 0.1,
                 anneal_lr: bool = True, region_alpha: float = 0.5, world_model_load_path: Optional[str] = None,
                 invalid_action_penalty: float = 0.1, normalize_curiosity_reward: bool = False,
                 buffer_load_path: Optional[str] = None, buffer_save_path: Optional[str] = None,
                 reply_format: str = "action_only", policy_kwargs: dict = None, seed: int = 1,
                 save_every: int = 5, parameters: dict[str, Any] = None) -> None:
        self._parameters = parameters
        self.env_name, self.env, self.out_dir = env_name, env, out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.total_steps, self.num_steps, self.max_episode_steps = total_steps, num_steps, max_episode_steps
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.num_minibatches, self.update_epochs = num_minibatches, update_epochs
        self.clip_coef, self.clip_vloss = clip_coef, clip_vloss
        self.ent_coef, self.vf_coef, self.kl_coef = ent_coef, vf_coef, kl_coef
        self.max_grad_norm, self.target_kl, self.anneal_lr = max_grad_norm, target_kl, anneal_lr
        self.seed, self.save_every = seed, save_every
        self.lrs = (learning_rate, value_learning_rate)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.encoder = build_encoder(config=state_config, env_name=env_name, parameters=parameters)
        self.curiosity = get_curiosity_module(
            curiosity_module=curiosity_module, env_name=env_name, encoder=self.encoder, region_alpha=region_alpha,
            world_model_load_path=world_model_load_path, buffer_load_path=buffer_load_path,
            save_path=buffer_save_path, invalid_action_penalty=invalid_action_penalty,
            normalize_curiosity_reward=normalize_curiosity_reward, parameters=parameters)
        self.vocab = ActionVocab(env_name=env_name)
        self.vocab.save(directory=out_dir)
        self.replay = ReplayWriter(directory=explore_replay(run_dir=out_dir))
        policy_kwargs = dict(policy_kwargs or {})
        if policy_kwargs.get("max_new_tokens") is None:
            policy_kwargs["max_new_tokens"] = MAX_NEW_TOKENS[reply_format][env_name]
        self.policy = VLMPolicy(model_name=policy_model, parameters=parameters, **policy_kwargs)
        lora = [p for n, p in self.policy.named_parameters() if p.requires_grad and not n.startswith("value_head")]
        self.optimizer = torch.optim.AdamW([{"params": lora, "lr": learning_rate},
                                            {"params": self.policy.value_head.parameters(),
                                             "lr": value_learning_rate}], eps=1e-5)
        self.prompter = PolicyPrompter(env_name=env_name, env=env, parameters=parameters, reply_format=reply_format)
        self.metrics_path = os.path.join(out_dir, "metrics.jsonl")
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(dict(env_name=env_name, total_steps=total_steps, num_steps=num_steps,
                           max_episode_steps=max_episode_steps, learning_rate=learning_rate,
                           value_learning_rate=value_learning_rate, gamma=gamma, gae_lambda=gae_lambda,
                           num_minibatches=num_minibatches, update_epochs=update_epochs, clip_coef=clip_coef,
                           ent_coef=ent_coef, vf_coef=vf_coef, kl_coef=kl_coef, target_kl=target_kl,
                           objective="per_token_ppo", curiosity_module=curiosity_module, region_alpha=region_alpha,
                           world_model_load_path=world_model_load_path, invalid_action_penalty=invalid_action_penalty,
                           normalize_curiosity_reward=normalize_curiosity_reward, buffer_load_path=buffer_load_path,
                           reply_format=reply_format, policy_kwargs=policy_kwargs, seed=seed, policy=policy_model,
                           lora_targets=self.policy.lora_targets, state=self.encoder.config(), **state_config),
                      f, indent=1, default=str)

    # ------------------------------------------------------------------ episodes

    def _start_episode(self) -> None:
        self.obs, self.info = self.env.reset(seed=self.seed + self.episode)
        self.prompter.reset()
        self.curiosity.reset()
        self.ep_step = 0
        self.error = None
        self.ep_reward = 0.0
        frame = frame_for_embedding(obs=self.obs, info=self.info)
        self.record = StateRecord(obs=self.obs, info=self.info,
                                  embedding=self.encoder.encode_one(obs=self.obs, info=self.info))
        self.curiosity.get_reward(prev=None, action=None, next=self.record)    # seeds the archive (scores 0)
        self.replay.add(episode=self.episode, step=0, frame=frame, embedding=self.record.embedding.image,
                        texts=self.obs["texts"], generated=None, parsed_action=None, action_index=None, valid=True,
                        reward_ext=0.0, reward_frame=0.0, reward_text=0.0, reward=0.0, done=False,
                        embedder=self.encoder.image.name)

    def _step(self) -> dict:
        messages, images = self.prompter.build(obs=self.obs, info=self.info, error=self.error)
        out = self.policy.act(messages=messages, images=images, response_prefix=self.prompter.response_prefix)
        obs2, r_ext, terminated, truncated, info2 = self.env.step(out["text"])
        self.ep_step += 1
        done = bool(terminated or truncated or self.ep_step >= self.max_episode_steps)
        frame = frame_for_embedding(obs=obs2, info=info2)
        record = StateRecord(obs=obs2, info=info2, embedding=self.encoder.encode_one(obs=obs2, info=info2),
                             prev_image=self.record.embedding.image)
        # The final transition's reward must be computed before the curiosity module is reset.
        rew = self.curiosity.get_reward(prev=self.record, action=info2.get("parsed_action"), next=record,
                                        valid=info2["valid"], done=done)
        reward = float(r_ext) + rew["total"]
        action_index = self.vocab.encode(parsed_action=info2.get("parsed_action") if info2["valid"] else None)
        self.replay.add(episode=self.episode, step=self.ep_step, frame=frame, embedding=record.embedding.image,
                        texts=obs2["texts"], generated=out["text"], parsed_action=info2.get("parsed_action"),
                        action_index=action_index, valid=info2["valid"], reward_ext=float(r_ext),
                        reward_frame=rew["frame"], reward_text=rew["region"], reward=reward, done=done,
                        embedder=self.encoder.image.name, reward_penalty=rew["penalty"])
        self.prompter.observe(generated=out["text"], obs_before=self.obs, info_before=self.info, obs_after=obs2,
                              info_after=info2)
        self.ep_reward += reward
        step_record = {"messages": messages, "images": [EncodedImage.of(i) for i in images],
                       "action_ids": out["action_ids"], "token_logprobs": out["token_logprobs"],
                       "ref_token_logprobs": out["ref_token_logprobs"], "value": out["value"], "reward": reward,
                       "done": done, "valid": bool(info2["valid"]), "n_tokens": len(out["action_ids"]),
                       "novelty_frame": rew["frame"], "novelty_region": rew["region"], "penalty": rew["penalty"],
                       "novelty": rew["novelty"]}
        self.error = None if info2["valid"] else info2.get("error")
        if done:
            log_info(f"[{self.env_name}] episode {self.episode}: {self.ep_step} steps, return {self.ep_reward:.3f}",
                     parameters=self._parameters)
            self.episode_returns.append(self.ep_reward)
            self.curiosity.save()
            self.episode += 1
            self._start_episode()
        else:
            self.obs, self.info, self.record = obs2, info2, record
        return step_record

    # ------------------------------------------------------------------ update

    def _update(self, *, batch: list, advantages: np.ndarray, returns: np.ndarray) -> dict:
        n = len(batch)
        mb_size = max(1, n // self.num_minibatches)
        params = self.policy.trainable_parameters()
        stats = {"pg_loss": [], "v_loss": [], "entropy": [], "kl_ref": [], "approx_kl": [], "clipfrac": []}
        stop = False
        for epoch in range(self.update_epochs):
            order = np.random.permutation(n)
            for start in range(0, n, mb_size):
                mb = order[start:start + mb_size]
                mb_adv = advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8) if len(mb) > 1 else mb_adv
                self.optimizer.zero_grad(set_to_none=True)
                approx_kls = []
                for j, i in enumerate(mb):
                    rec = batch[i]
                    ev = self.policy.evaluate(messages=rec["messages"], images=rec["images"],
                                              action_ids=rec["action_ids"], response_prefix=self.prompter.response_prefix)
                    new = ev["token_logprob"]
                    old = rec["token_logprobs"].to(new.device)
                    logratio = new - old                                   # per action token
                    ratio = logratio.exp()
                    adv = float(mb_adv[j])
                    pg = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - self.clip_coef,
                                                                    1 + self.clip_coef)).mean()
                    v = ev["value"]
                    if self.clip_vloss:
                        v_clipped = rec["value"] + torch.clamp(v - rec["value"], -self.clip_coef, self.clip_coef)
                        v_loss = 0.5 * torch.max((v - returns[i]) ** 2, (v_clipped - returns[i]) ** 2)
                    else:
                        v_loss = 0.5 * (v - returns[i]) ** 2
                    kl_ref = (new - rec["ref_token_logprobs"].to(new.device)).mean()
                    loss = pg - self.ent_coef * ev["entropy"] + self.vf_coef * v_loss + self.kl_coef * kl_ref
                    (loss / len(mb)).backward()
                    with torch.no_grad():
                        approx_kls.append(float(((ratio - 1) - logratio).mean()))
                        stats["clipfrac"].append(float(((ratio - 1).abs() > self.clip_coef).float().mean()))
                        stats["pg_loss"].append(float(pg))
                        stats["v_loss"].append(float(v_loss))
                        stats["entropy"].append(float(ev["entropy"]))
                        stats["kl_ref"].append(float(kl_ref))
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                self.optimizer.step()
                stats["approx_kl"] += approx_kls
                if self.target_kl is not None and np.mean(approx_kls) > self.target_kl:
                    stop = True
                    break
            if stop:
                break
        out = {k: float(np.mean(v)) if v else None for k, v in stats.items()}
        out["early_stop"] = stop
        return out

    # ------------------------------------------------------------------ main loop

    def train(self) -> None:
        self.episode = 0
        self.episode_returns: list = []
        self._start_episode()
        n_iterations = max(1, self.total_steps // self.num_steps)
        global_step = 0
        t0 = time.time()
        for iteration in range(1, n_iterations + 1):
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / n_iterations
                for group, lr in zip(self.optimizer.param_groups, self.lrs):
                    group["lr"] = frac * lr
            t_roll = time.time()
            batch = [self._step() for _ in range(self.num_steps)]
            global_step += self.num_steps
            t_roll = time.time() - t_roll
            if batch[-1]["done"]:
                next_value = 0.0
            else:
                messages, images = self.prompter.build(obs=self.obs, info=self.info, error=self.error)
                next_value = self.policy.value(messages=messages, images=images,
                                               response_prefix=self.prompter.response_prefix)
            rewards = np.array([b["reward"] for b in batch], dtype=np.float32)
            values = np.array([b["value"] for b in batch], dtype=np.float32)
            dones = np.array([b["done"] for b in batch], dtype=np.float32)
            advantages, returns = gae(rewards=rewards, values=values, dones=dones, next_value=next_value,
                                      gamma=self.gamma, lam=self.gae_lambda)
            t_upd = time.time()
            stats = self._update(batch=batch, advantages=advantages, returns=returns)
            t_upd = time.time() - t_upd
            var_y = np.var(returns)
            metrics = {"iteration": iteration, "global_step": global_step, "episodes": self.episode,
                       "reward_mean": float(rewards.mean()),
                       "novelty_mean": float(np.mean([b["novelty"] for b in batch])),
                       "novelty_frame_mean": float(np.mean([b["novelty_frame"] for b in batch])),
                       "novelty_region_mean": float(np.mean([b["novelty_region"] for b in batch])),
                       "penalty_mean": float(np.mean([b["penalty"] for b in batch])),
                       "valid_rate": float(np.mean([b["valid"] for b in batch])),
                       "action_tokens_mean": float(np.mean([b["n_tokens"] for b in batch])),
                       "recent_episode_return": float(np.mean(self.episode_returns[-5:])) if self.episode_returns else None,
                       "explained_variance": None if var_y == 0 else float(1 - np.var(returns - values) / var_y),
                       "rollout_seconds": t_roll, "update_seconds": t_upd,
                       "steps_per_second": global_step / (time.time() - t0), **stats}
            with open(self.metrics_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
            log_info(f"[{self.env_name}] iter {iteration}/{n_iterations}: " + json.dumps(
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}),
                parameters=self._parameters)
            if iteration % self.save_every == 0 or iteration == n_iterations:
                self.policy.save(directory=os.path.join(self.out_dir, "policy"))
                self.replay.flush()
        self.replay.close()
        self.curiosity.save()
        log_info(f"[{self.env_name}] PPO done: {global_step} steps, {self.episode} episodes -> {self.out_dir}",
                 parameters=self._parameters)
