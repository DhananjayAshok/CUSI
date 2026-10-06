"""`world_model`: 1 - cos(predicted next embedding, real next embedding) (GameBoyRL's WorldModel
curiosity), and the world model itself.

The model has GameBoyRL's architecture (plan decision 11, Part 2.7; cleanrl_utils/
port_gameboy_worlds/curiosity.py WorldModel, FRAME_STACK = 2), in the frozen encoder's embedding
space:

    [emb(frame_{t-1}), emb(frame_t)] -> Linear(2*D -> 512)
    action index -> nn.Embedding(n_actions, 512)
    concat -> Linear(1024 -> 512) -> ReLU -> LayerNorm(512) -> Linear(512 -> D) -> L2-normalise

The action index comes from cusi.envs.action_vocab. This module holds the model and its loading
(inference); training on replay buffers is cusi.explore.world_model, as the cnn image embedder's
architecture is in cusi.state.encoders and its training in cusi.explore.

The scorer loads a trained model from --world_model_load_path, which is required: a randomly
initialised world model gives a meaningless score. On load it is checked to have been trained with
the same image embedder (world_model_meta.json) and the env's action vocabulary (action_space.json).
It is not trained here, and archive resets don't affect it (the archive is not used).

The action is the env's canonical parsed_action (info["parsed_action"]). A step the env rejected
(next.info["valid"] False) is a no-op (the screen is unchanged) and scores 0, as the archive
scorers give an unchanged screen: world models trained on random-agent replays never see the
"invalid" index, and their arbitrary prediction for it rewarded invalid actions (0.57 mean vs 0.06
for valid steps in the first GameBoy smoke run). The frame before prev is prev.prev_image
(repeated prev frame if missing, as in training).
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
    """world_model_meta.json: the env, the image embedder (name and metadata) and the embedding size
    the model was trained with."""
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
