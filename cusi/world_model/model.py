"""
The tree world model: [emb(t-1), emb(t)] and an action (type + target box) -> emb(t+1), in an image embedder's
L2-normalised space. As cusi.state.scorers.world_model.WorldModel, with the action embedding the sum of the type's
embedding and an MLP of its box features (cusi.world_model.actions.BOX_DIM).
"""
import json
import os
import torch
import torch.nn as nn
from cusi.world_model.actions import BOX_DIM

FRAME_STACK = 2
WEIGHTS = "world_model.pt"


class BoxActionWorldModel(nn.Module):
    def __init__(self, *, emb_dim: int, action_types: list, hidden_dim: int = 512, residual: bool = True):
        super().__init__()
        self.emb_dim, self.hidden_dim, self.residual = emb_dim, hidden_dim, residual
        self.action_types = list(action_types)
        self.type_index = {t: i for i, t in enumerate(self.action_types)}
        self.type_embedder = nn.Embedding(len(self.action_types), hidden_dim)
        self.box_encoder = nn.Sequential(nn.Linear(BOX_DIM, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.observation_encoder = nn.Linear(emb_dim * FRAME_STACK, hidden_dim)
        self.model = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
                                   nn.Linear(hidden_dim, emb_dim))
        if residual:      # start exactly at the "nothing changes" prediction
            nn.init.zeros_(self.model[-1].weight)
            nn.init.zeros_(self.model[-1].bias)

    def forward(self, *, obs: torch.Tensor, types: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """obs (B, 2D) = [emb(t-1), emb(t)]; types (B,) long; boxes (B, BOX_DIM). Returns (B, D), unit norm."""
        action = self.type_embedder(types.long()) + self.box_encoder(boxes.float())
        pred = self.model(torch.cat([self.observation_encoder(obs), action], dim=-1))
        if self.residual:
            pred = obs[:, -self.emb_dim:] + pred
        return nn.functional.normalize(pred, dim=-1)

    def config(self) -> dict:
        return {"emb_dim": self.emb_dim, "action_types": self.action_types, "hidden_dim": self.hidden_dim,
                "residual": self.residual}

    def save(self, *, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), **self.config()}, os.path.join(directory, WEIGHTS))

    @classmethod
    def load(cls, *, directory: str, device: str = "cpu") -> "BoxActionWorldModel":
        ckpt = torch.load(os.path.join(directory, WEIGHTS), map_location=device)
        model = cls(emb_dim=ckpt["emb_dim"], action_types=ckpt["action_types"], hidden_dim=ckpt["hidden_dim"],
                    residual=ckpt["residual"])
        model.load_state_dict(ckpt["state_dict"])
        return model.to(device)
