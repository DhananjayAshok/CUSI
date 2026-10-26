"""
The curiosity policy: GameBoyRL's curiosity agent (cleanrl ppo_curiosity), four stacked grayscale frames through its
small CNN, trained from scratch, with no text at all, and a pointer head over the legal candidates (a candidate is its
action type, its target's box and its argument's id) with a value head (CNNPolicy). CNNObserver turns observations into
the features kept in the rollout, so PPO updates only run the trained network.
(A fusion policy over frozen MobileNetV3 / MiniLM features was removed: no better, and slower.)
"""
import math
import zlib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from cusi.search.curiosity.candidates import TYPES

D = 128
N_FOURIER = 8
#: CNNPolicy: frames stacked, the grayscale size per env (GameBoy native; Web and Android keep their aspect), and the
#: buckets a candidate's argument (button or app name) is hashed into.
FRAME_STACK = 4
CNN_FRAME_HW = {"gameboy": (144, 160), "web": (144, 256), "android": (288, 128)}
N_ARG_BUCKETS = 64


class CNNObserver:
    """CNNPolicy's features: the stacked grayscale frames and, per candidate, its type, target box and argument id."""

    def __init__(self, *, env_name: str) -> None:
        self.frame_hw = CNN_FRAME_HW[env_name]

    def make_policy(self):
        return CNNPolicy(frame_hw=self.frame_hw)

    def gray(self, frame) -> torch.Tensor:
        h, w = self.frame_hw
        img = Image.fromarray(np.ascontiguousarray(frame[:, :, :3])).convert("L").resize((w, h), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy())

    def observe(self, *, frames: list, elements_list: list, candidates_list: list, hists: list) -> list:
        """hists: per observation, the episode's previous grayscale frames (at most FRAME_STACK - 1; at an episode's
        start the first frame fills the stack, as gym's FrameStack does)."""
        out = []
        for frame, elements, cands, hist in zip(frames, elements_list, candidates_list, hists):
            h, w = frame.shape[:2]
            stack = (list(hist) + [self.gray(frame)])[-FRAME_STACK:]
            stack = [stack[0]] * (FRAME_STACK - len(stack)) + stack
            geom = [[0.0] * 4 if c.target is None else
                    [elements[c.target].box[0] / w, elements[c.target].box[1] / h,
                     elements[c.target].box[2] / w, elements[c.target].box[3] / h] for c in cands]
            out.append({"stack": torch.stack(stack),
                        "c_type": torch.tensor([TYPES.index(c.type) for c in cands], dtype=torch.long),
                        "c_has_target": torch.tensor([c.target is not None for c in cands], dtype=torch.bool),
                        "c_geom": torch.tensor(geom, dtype=torch.float32).reshape(len(cands), 4),
                        "c_arg": torch.tensor([0 if not c.argument else 1 + zlib.crc32(c.argument.encode()) %
                                               (N_ARG_BUCKETS - 1) for c in cands], dtype=torch.long)})
        return out


def fourier(*, geom: torch.Tensor) -> torch.Tensor:
    """(..., 4) normalised x0, y0, x1, y1 -> (..., 6 * 2 * N_FOURIER) with w and h added."""
    g = torch.cat([geom, geom[..., 2:3] - geom[..., 0:1], geom[..., 3:4] - geom[..., 1:2]], dim=-1)
    freqs = (2.0 ** torch.arange(N_FOURIER, device=geom.device)) * math.pi
    x = g.unsqueeze(-1) * freqs
    return torch.cat([x.sin(), x.cos()], dim=-1).flatten(-2)


def mlp(*, i: int, o: int, hidden: int = D) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, hidden), nn.GELU(), nn.Linear(hidden, o))


def layer_init(layer, std: float = math.sqrt(2), bias: float = 0.0):
    """cleanrl's initialisation (orthogonal weights, constant bias)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class CNNPolicy(nn.Module):
    """GameBoyRL's curiosity agent with a pointer head: its CNN (get_gameboy_cnn_chain) over the stacked frames gives
    the state; each candidate is its type, target box and argument id; logits come from the state and the candidate,
    the value from the state alone."""

    def __init__(self, *, frame_hw: tuple) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(FRAME_STACK, 32, kernel_size=16, stride=16)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)), nn.ReLU(),
            nn.Flatten())
        with torch.no_grad():
            flat = self.cnn(torch.zeros(1, FRAME_STACK, *frame_hw)).shape[1]
        self.state = nn.Sequential(layer_init(nn.Linear(flat, D)), nn.ReLU())
        self.c_type = nn.Embedding(len(TYPES), D)
        self.c_arg = nn.Embedding(N_ARG_BUCKETS, D)
        self.geom = mlp(i=6 * 2 * N_FOURIER, o=D)
        self.no_target = nn.Parameter(torch.zeros(D))
        self.logit = nn.Sequential(layer_init(nn.Linear(2 * D, D)), nn.ReLU(), layer_init(nn.Linear(D, 1), std=0.01))
        self.value = layer_init(nn.Linear(D, 1), std=1.0)

    @staticmethod
    def collate(*, feats: list, device: str) -> dict:
        b, n_c = len(feats), max(f["c_type"].shape[0] for f in feats)
        out = {"stack": torch.stack([f["stack"] for f in feats]).float() / 255.0,
               "c_type": torch.zeros(b, n_c, dtype=torch.long), "c_arg": torch.zeros(b, n_c, dtype=torch.long),
               "c_geom": torch.zeros(b, n_c, 4), "c_has_target": torch.zeros(b, n_c, dtype=torch.bool),
               "c_mask": torch.zeros(b, n_c, dtype=torch.bool)}
        for i, f in enumerate(feats):
            c = f["c_type"].shape[0]
            out["c_type"][i, :c], out["c_arg"][i, :c], out["c_geom"][i, :c] = f["c_type"], f["c_arg"], f["c_geom"]
            out["c_has_target"][i, :c], out["c_mask"][i, :c] = f["c_has_target"], True
        return {k: v.to(device) for k, v in out.items()}

    def forward(self, batch: dict) -> tuple:
        """(logits (B, C) with -inf on padding, value (B,), candidate embeddings (B, C, D))."""
        z = self.state(self.cnn(batch["stack"]))
        tgt = torch.where(batch["c_has_target"].unsqueeze(-1), self.geom(fourier(geom=batch["c_geom"])),
                          self.no_target.expand(*batch["c_geom"].shape[:2], D))
        e_c = self.c_type(batch["c_type"]) + self.c_arg(batch["c_arg"]) + tgt
        logits = self.logit(torch.cat([z.unsqueeze(1).expand_as(e_c), e_c], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~batch["c_mask"], float("-inf"))
        return logits, self.value(z).squeeze(-1), e_c
