"""World model with GameBoyRL's architecture (plan decision 11, Part 2.7), in the frozen
encoder's embedding space.

    [emb(frame_{t-1}), emb(frame_t)] -> Linear(2*D -> 512)
    action index -> nn.Embedding(n_actions, 512)
    concat -> Linear(1024 -> 512) -> ReLU -> LayerNorm(512) -> Linear(512 -> D) -> L2-normalise
    loss: MSE to emb(frame_{t+1})

(cleanrl_utils/port_gameboy_worlds/curiosity.py WorldModel, FRAME_STACK = 2). Trained on the
replay buffers: for every transition t (step >= 1) the inputs are the embeddings of the frames
at steps t-2 and t-1 (the first transition repeats its frame) and the action index of
transition t; the target is the embedding at step t. action_space.json (from the run that
produced the replay) pins the index meaning and is copied next to world_model.pt.

Embeddings: the replay's stored ones, or (curiosity_plan §3.7) re-embedded from the stored frames
with any cusi_state image embedder (`embedder=`). world_model_meta.json records which embedder (name
and its metadata), so the `world_model` scorer can check it is used with the same one.
"""
import json
import os
import shutil
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from cusi_utils.log_handling import log_info
from cusi_explore.action_vocab import ACTION_SPACE_FILENAME, ActionVocab
from cusi_explore.replay import iter_replay, load_episodes

FRAME_STACK = 2
META_FILENAME = "world_model_meta.json"


def read_meta(*, directory: str) -> dict:
    path = os.path.join(directory, META_FILENAME)
    if not os.path.exists(path):
        raise ValueError(f"{directory} has no {META_FILENAME} (trained before cusi_state); retrain it")
    with open(path) as f:
        return json.load(f)


class WorldModel(nn.Module):
    def __init__(self, *, emb_dim: int, n_actions: int, hidden_dim: int = 512, normalized_observations: bool = True):
        super().__init__()
        self.emb_dim, self.n_actions, self.hidden_dim = emb_dim, n_actions, hidden_dim
        self.normalized_observations = normalized_observations
        self.action_embedder = nn.Embedding(n_actions, hidden_dim)
        self.observation_encoder = nn.Linear(emb_dim * FRAME_STACK, hidden_dim)
        self.model = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
                                   nn.Linear(hidden_dim, emb_dim))

    def forward(self, *, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """obs (B, FRAME_STACK*D) = [emb(t-1), emb(t)]; actions (B,) long. Returns (B, D)."""
        x = torch.cat([self.observation_encoder(obs), self.action_embedder(actions.long())], dim=-1)
        pred = self.model(x)
        return nn.functional.normalize(pred, dim=-1) if self.normalized_observations else pred

    def save(self, *, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "emb_dim": self.emb_dim, "n_actions": self.n_actions,
                    "hidden_dim": self.hidden_dim}, os.path.join(directory, "world_model.pt"))

    @classmethod
    def load(cls, *, directory: str, device: str = "cpu") -> "WorldModel":
        ckpt = torch.load(os.path.join(directory, "world_model.pt"), map_location=device)
        model = cls(emb_dim=ckpt["emb_dim"], n_actions=ckpt["n_actions"], hidden_dim=ckpt["hidden_dim"])
        model.load_state_dict(ckpt["state_dict"])
        spec = ActionVocab.load_spec(directory=directory)
        if spec["n_actions"] != ckpt["n_actions"]:
            raise ValueError(f"{directory}: world model has {ckpt['n_actions']} actions, action_space.json"
                             f" {spec['n_actions']}")
        return model.to(device)


def transitions_from_replay(*, replay_dirs: list, embedder=None) -> dict:
    """{"obs": (N, 2D), "action": (N,), "next": (N, D), "episode": (N,), "frame_changed": (N,)} float32/int64.
    embedder: a cusi_state ImageEmbedder to re-embed the frames with (default: the stored embeddings)."""
    obs, act, nxt, epi, changed = [], [], [], [], []
    episode_id = 0
    for directory in replay_dirs:
        for ep in load_episodes(directory=directory, load_frames=embedder is not None):
            if embedder is None:
                embs = [np.asarray(s["embedding"], dtype=np.float32) for s in ep]
            else:
                embs = list(embedder.embed(frames=[s["frame"] for s in ep]).numpy())
            for t in range(1, len(ep)):
                prev = embs[t - 2] if t >= 2 else embs[t - 1]
                obs.append(np.concatenate([prev, embs[t - 1]]))
                act.append(ep[t]["action_index"] if ep[t]["action_index"] is not None else -1)
                nxt.append(embs[t])
                epi.append(episode_id)
                changed.append(float(np.abs(embs[t] - embs[t - 1]).max() > 1e-3))
            episode_id += 1
    return {"obs": np.stack(obs), "action": np.array(act, dtype=np.int64), "next": np.stack(nxt),
            "episode": np.array(epi), "frame_changed": np.array(changed, dtype=np.float32)}


def evaluate(*, model: WorldModel, data: dict, idx: np.ndarray, device: str) -> dict:
    """MSE and cosine of predictions vs the "nothing changes" baseline (copy the current frame)."""
    with torch.no_grad():
        obs = torch.tensor(data["obs"][idx], device=device)
        act = torch.tensor(data["action"][idx], device=device)
        nxt = torch.tensor(data["next"][idx], device=device)
        pred = model(obs=obs, actions=act)
        D = nxt.shape[1]
        copy = nn.functional.normalize(obs[:, D:], dim=-1)
        nxt_n = nn.functional.normalize(nxt, dim=-1)
        changed = torch.tensor(data["frame_changed"][idx], device=device) > 0
        out = {"mse": float(((pred - nxt) ** 2).mean()), "copy_mse": float(((copy - nxt_n) ** 2).mean()),
               "cos": float((pred * nxt_n).sum(-1).mean()), "copy_cos": float((copy * nxt_n).sum(-1).mean()),
               "n": int(len(idx)), "n_changed": int(changed.sum())}
        if changed.any():
            out["cos_changed"] = float((pred[changed] * nxt_n[changed]).sum(-1).mean())
            out["copy_cos_changed"] = float((copy[changed] * nxt_n[changed]).sum(-1).mean())
        return out


def stored_embedder_name(*, replay_dirs: list) -> str:
    for directory in replay_dirs:
        for row in iter_replay(directory=directory, load_frames=False):
            return row.get("embedder") or "siglip"     # pre-migration replays stored SigLIP 2 vectors
    raise ValueError(f"No replay rows in {replay_dirs}")


def train_world_model(*, replay_dirs: list, out_dir: str, action_space_dir: str, epochs: int = 50,
                      batch_size: int = 256, lr: float = 1e-3, val_frac: float = 0.1, seed: int = 0,
                      embedder=None, device: str = None, parameters: dict[str, Any] = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = ActionVocab.load_spec(directory=action_space_dir)
    data = transitions_from_replay(replay_dirs=replay_dirs, embedder=embedder)
    meta = {"image_embedder": embedder.name, "embedder_meta": embedder.meta()} if embedder is not None else \
        {"image_embedder": stored_embedder_name(replay_dirs=replay_dirs), "embedder_meta": None}
    meta.update({"env": spec["env"], "emb_dim": int(data["next"].shape[1])})
    valid = data["action"] >= 0
    for k in data:
        data[k] = data[k][valid]
    n = len(data["action"])
    rng = np.random.default_rng(seed)
    # split by episode, so validation transitions come from unseen episodes
    episodes = np.unique(data["episode"])
    val_eps = set(rng.choice(episodes, size=max(1, int(len(episodes) * val_frac)), replace=False).tolist()) \
        if len(episodes) > 1 else set()
    is_val = np.array([e in val_eps for e in data["episode"]])
    train_idx, val_idx = np.where(~is_val)[0], np.where(is_val)[0]
    torch.manual_seed(seed)
    model = WorldModel(emb_dim=data["next"].shape[1], n_actions=spec["n_actions"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    obs_t = torch.tensor(data["obs"], device=device)
    act_t = torch.tensor(data["action"], device=device)
    nxt_t = torch.tensor(data["next"], device=device)
    history = []
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(perm), batch_size):
            b = torch.tensor(perm[start:start + batch_size], device=device)
            loss = ((model(obs=obs_t[b], actions=act_t[b]) - nxt_t[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        model.eval()
        entry = {"epoch": epoch, "train_mse": float(np.mean(losses)) if losses else None}
        if len(val_idx):
            entry["val"] = evaluate(model=model, data=data, idx=val_idx, device=device)
        history.append(entry)
        if epoch % 10 == 0 or epoch == epochs - 1:
            log_info(f"world model epoch {epoch}: {json.dumps(entry)}", parameters=parameters)
    model.save(directory=out_dir)
    with open(os.path.join(out_dir, META_FILENAME), "w") as f:
        json.dump(meta, f, indent=1)
    shutil.copy(os.path.join(action_space_dir, ACTION_SPACE_FILENAME), os.path.join(out_dir, ACTION_SPACE_FILENAME))
    summary = {"n_transitions": int(n), "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
               "action_counts": {spec["names"][a]: int(c) for a, c in zip(*np.unique(data["action"], return_counts=True))},
               "final": history[-1], "replay_dirs": replay_dirs}
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump({"summary": summary, "history": history}, f, indent=1)
    log_info(f"world model -> {out_dir}: {json.dumps(summary['final'])}", parameters=parameters)
    return summary
