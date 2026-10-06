"""
World-model curiosity: 1 - cos(predicted, real next embedding), and GameBoyRL's world model itself (inference only).
"""
import json
import os
import torch
import torch.nn as nn
from cusi.envs.action_vocab import ActionVocab
from cusi.state.scorers.base import Scorer

FRAME_STACK = 2
WORLD_MODEL_META_FILENAME = "world_model_meta.json"


def read_meta(*, directory: str) -> dict:
    """The env, image embedder and embedding size the model was trained with."""
    path = os.path.join(directory, WORLD_MODEL_META_FILENAME)
    if not os.path.exists(path):
        raise ValueError(f"{directory} has no {WORLD_MODEL_META_FILENAME} (trained before cusi.state); retrain it")
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


class WorldModelScorer(Scorer):
    """Needs a trained model; steps the env rejected score 0 (the model never saw them in training)."""

    name = "world_model"

    def __init__(self, *, load_path: str, env_name: str, encoder, device: str = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        meta = read_meta(directory=load_path)
        if encoder is not None:
            image = encoder.image
            if meta["image_embedder"] != image.name or (meta.get("embedder_meta") not in (None, image.meta())):
                raise ValueError(f"{load_path}: world model trained on {meta['image_embedder']} "
                                 f"({meta.get('embedder_meta')}), this run embeds with {image.name} ({image.meta()})")
        if meta["env"] != env_name:
            raise ValueError(f"{load_path}: world model is for env {meta['env']}, not {env_name}")
        self.vocab = ActionVocab(env_name=env_name)
        spec = ActionVocab.load_spec(directory=load_path)
        if spec["names"] != self.vocab.names:
            raise ValueError(f"{load_path}: action_space.json differs from the current {env_name} vocabulary")
        self.model = WorldModel.load(directory=load_path, device=self.device).eval()

    @torch.no_grad()
    def score(self, *, prev, action, next, archive, add=True, seed_frame=False):
        if prev is None or prev.embedding is None:
            return 0.0, {"world_model": 0.0}
        cur = prev.embedding.image
        before = prev.prev_image if prev.prev_image is not None else cur
        obs = torch.cat([before, cur]).unsqueeze(0).to(self.device)
        if not (next.info or {}).get("valid", True):
            return 0.0, {"world_model": 0.0}
        parsed = action if isinstance(action, dict) else (next.info or {}).get("parsed_action")
        index = self.vocab.encode(parsed_action=parsed)
        pred = self.model(obs=obs, actions=torch.tensor([index], device=self.device))[0]
        real = torch.nn.functional.normalize(next.embedding.image.to(self.device), dim=0)
        value = float(1.0 - (pred * real).sum())
        return value, {"world_model": value}
