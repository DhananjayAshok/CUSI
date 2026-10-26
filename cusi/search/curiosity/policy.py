"""
The curiosity policies, each a pointer head over the legal candidates with a value head, paired with the observer that
turns observations into its stored features (kept in the rollout, so PPO updates only run the trained network):

- "fusion" (plans/archive/next_plan.md §E.3): frozen pretrained encoders (a MobileNetV3-Small trunk for the frame, MiniLM
  for strings; FrozenEncoders) feed a small trained fusion transformer (FusionPolicy).
- "cnn": GameBoyRL's curiosity agent (cleanrl ppo_curiosity): four stacked grayscale frames through its small CNN,
  trained from scratch, with no text at all; a candidate is its action type, its target's box and its argument's id
  (PixelObserver, PixelPolicy).
"""
import math
import os
import zlib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from cusi.envs.text_boxes import ROLES
from cusi.search.curiosity.candidates import TYPES

D = 128
VISION_DIM = 576            # MobileNetV3-Small's last feature map
TEXT_DIM = 384              # MiniLM-L6
GRID = 4
N_FOURIER = 8
VISION_ENCODERS = ("mobilenet_v3_small",)
POLICIES = ("fusion", "cnn")
#: PixelPolicy: frames stacked, the grayscale size per env (GameBoy native; Web and Android keep their aspect), and the
#: buckets a candidate's argument (button or app name) is hashed into.
FRAME_STACK = 4
PIXEL_FRAME_HW = {"gameboy": (144, 160), "web": (144, 256), "android": (288, 128)}
N_ARG_BUCKETS = 64


class FrozenEncoders:
    """The frozen vision trunk and sentence encoder, shared by every session (never trained)."""

    def __init__(self, *, vision_encoder: str, text_encoder: str, device: str, long_side: int = 448) -> None:
        import torchvision
        from transformers import AutoModel, AutoTokenizer
        if vision_encoder not in VISION_ENCODERS:
            raise ValueError(f"--policy_vision_encoder must be one of {VISION_ENCODERS}, got {vision_encoder!r}")
        self.device, self.long_side = device, long_side
        self.vision_encoder, self.text_encoder = vision_encoder, text_encoder
        # Pretrained weights live under HF_HOME (like every other model's), not in ~/.cache.
        torch.hub.set_dir(os.path.join(os.environ["HF_HOME"], "torch_hub"))
        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.trunk = torchvision.models.mobilenet_v3_small(weights=weights).features.to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)
        self.text_model = AutoModel.from_pretrained(text_encoder).to(device).eval()
        for p in list(self.trunk.parameters()) + list(self.text_model.parameters()):
            p.requires_grad_(False)
        self._text_cache: dict = {}

    @torch.no_grad()
    def embed_texts(self, *, texts: list) -> torch.Tensor:
        """(len(texts), 384) unit vectors, cached per string ("" -> zeros)."""
        todo = [t for t in dict.fromkeys(texts) if t and t not in self._text_cache]
        for i in range(0, len(todo), 256):
            chunk = todo[i:i + 256]
            batch = self.tokenizer(chunk, padding=True, truncation=True, max_length=64,
                                   return_tensors="pt").to(self.device)
            hidden = self.text_model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9), dim=-1).cpu()
            for t, v in zip(chunk, pooled):
                self._text_cache[t] = v
        if len(self._text_cache) > 200_000:
            self._text_cache = {t: self._text_cache[t] for t in texts if t in self._text_cache}
        zero = torch.zeros(TEXT_DIM)
        return torch.stack([self._text_cache.get(t, zero) if t else zero for t in texts]) if texts else \
            torch.zeros(0, TEXT_DIM)

    @torch.no_grad()
    def feature_maps(self, *, frames: list) -> list:
        """Each frame (H, W, 3 uint8), resized to long_side keeping its aspect -> (576, h, w) on the CPU."""
        out = []
        for f in frames:
            x = torch.from_numpy(np.ascontiguousarray(f[:, :, :3])).to(self.device).permute(2, 0, 1)[None].float()
            h, w = x.shape[-2:]
            s = self.long_side / max(h, w)
            x = nn.functional.interpolate(x / 255.0, size=(max(32, round(h * s)), max(32, round(w * s))),
                                          mode="bilinear", align_corners=False)
            out.append(self.trunk((x - self.mean) / self.std)[0].cpu())
        return out

    def make_policy(self):
        return FusionPolicy()

    @torch.no_grad()
    def observe(self, *, frames: list, elements_list: list, candidates_list: list, prevs: list, hists: list) -> list:
        """Frozen features of a batch of observations, one dict per observation, with the previous action's frozen
        inputs (prevs: (candidate, its target's text) or None). hists is not used."""
        from torchvision.ops import roi_align
        maps = self.feature_maps(frames=frames)
        out = []
        for frame, fmap, elements, cands in zip(frames, maps, elements_list, candidates_list):
            h, w = frame.shape[:2]
            grid = nn.functional.adaptive_avg_pool2d(fmap[None], GRID)[0].flatten(1).T          # (16, 576)
            if elements:
                boxes = torch.tensor([b.box for b in elements], dtype=torch.float32)
                scale = fmap.shape[-1] / w
                rois = torch.cat([torch.zeros(len(boxes), 1), boxes], dim=1)
                roi = roi_align(fmap[None], rois, output_size=2, spatial_scale=scale, aligned=True).mean(dim=(2, 3))
                geom = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            else:
                roi, geom = torch.zeros(0, VISION_DIM), torch.zeros(0, 4)
            out.append({
                "grid": grid.half(), "roi": roi.half(), "geom": geom,
                "text": self.embed_texts(texts=[b.text for b in elements]).half(),
                "has_text": torch.tensor([bool(b.text) for b in elements], dtype=torch.bool),
                "role": torch.tensor([ROLES.index(b.role) if b.role in ROLES else ROLES.index("text")
                                      for b in elements], dtype=torch.long),
                "c_type": torch.tensor([TYPES.index(c.type) for c in cands], dtype=torch.long),
                "c_target": torch.tensor([-1 if c.target is None else c.target for c in cands], dtype=torch.long),
                "c_arg": self.embed_texts(texts=[c.argument or "" for c in cands]).half(),
            })
        for f, prev in zip(out, prevs):
            if prev is None:
                f["p_type"], f["p_target_text"], f["p_arg"] = -1, torch.zeros(TEXT_DIM).half(), \
                    torch.zeros(TEXT_DIM).half()
            else:
                cand, target_text = prev
                f["p_type"] = TYPES.index(cand.type)
                f["p_target_text"], f["p_arg"] = self.embed_texts(texts=[target_text, cand.argument or ""]).half()
        return out


class PixelObserver:
    """PixelPolicy's features: the stacked grayscale frames and, per candidate, its type, target box and argument id."""

    def __init__(self, *, env_name: str) -> None:
        self.frame_hw = PIXEL_FRAME_HW[env_name]

    def make_policy(self):
        return PixelPolicy(frame_hw=self.frame_hw)

    def gray(self, frame) -> torch.Tensor:
        h, w = self.frame_hw
        img = Image.fromarray(np.ascontiguousarray(frame[:, :, :3])).convert("L").resize((w, h), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy())

    def observe(self, *, frames: list, elements_list: list, candidates_list: list, prevs: list, hists: list) -> list:
        """hists: per observation, the episode's previous grayscale frames (at most FRAME_STACK - 1; at an episode's
        start the first frame fills the stack, as gym's FrameStack does). prevs is not used."""
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


class FusionPolicy(nn.Module):
    """The trained part: element tokens, [STATE] + [PREV_ACTION] + grid tokens, a 2-layer transformer, then a pointer
    head over candidates and a value head."""

    def __init__(self) -> None:
        super().__init__()
        self.grid_proj = nn.Linear(VISION_DIM, D)
        self.roi_proj = nn.Linear(VISION_DIM, D)
        self.text_proj = nn.Linear(TEXT_DIM, D)
        self.no_text = nn.Parameter(torch.zeros(D))
        self.role = nn.Embedding(len(ROLES), D)
        self.geom = mlp(i=6 * 2 * N_FOURIER, o=D)
        self.state = nn.Parameter(torch.zeros(D))
        self.no_prev = nn.Parameter(torch.zeros(D))
        self.prev_type = nn.Embedding(len(TYPES), D)
        self.prev_target = nn.Linear(TEXT_DIM, D)
        self.prev = mlp(i=3 * D, o=D)
        self.pos = nn.Parameter(torch.zeros(GRID * GRID + 2, D))
        layer = nn.TransformerEncoderLayer(d_model=D, nhead=4, dim_feedforward=4 * D, dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.c_type = nn.Embedding(len(TYPES), D)
        self.no_target = nn.Parameter(torch.zeros(D))
        self.arg_proj = nn.Linear(TEXT_DIM, D)
        self.cand = mlp(i=3 * D, o=D)
        self.logit = mlp(i=2 * D, o=1)
        self.value = mlp(i=D, o=1)
        for p in (self.no_text, self.state, self.no_prev, self.pos, self.no_target):
            nn.init.normal_(p, std=0.02)

    @staticmethod
    def collate(*, feats: list, device: str) -> dict:
        """Pad a list of observation features into batch tensors."""
        b = len(feats)
        n_e = max([f["role"].shape[0] for f in feats] + [1])
        n_c = max(f["c_type"].shape[0] for f in feats)
        out = {"grid": torch.stack([f["grid"] for f in feats]).float(),
               "roi": torch.zeros(b, n_e, VISION_DIM), "geom": torch.zeros(b, n_e, 4),
               "text": torch.zeros(b, n_e, TEXT_DIM), "has_text": torch.zeros(b, n_e, dtype=torch.bool),
               "role": torch.zeros(b, n_e, dtype=torch.long), "e_mask": torch.zeros(b, n_e, dtype=torch.bool),
               "c_type": torch.zeros(b, n_c, dtype=torch.long), "c_target": torch.full((b, n_c), -1, dtype=torch.long),
               "c_arg": torch.zeros(b, n_c, TEXT_DIM), "c_mask": torch.zeros(b, n_c, dtype=torch.bool),
               "p_type": torch.tensor([f["p_type"] for f in feats], dtype=torch.long),
               "p_target_text": torch.stack([f["p_target_text"] for f in feats]).float(),
               "p_arg": torch.stack([f["p_arg"] for f in feats]).float()}
        for i, f in enumerate(feats):
            e, c = f["role"].shape[0], f["c_type"].shape[0]
            out["roi"][i, :e], out["geom"][i, :e], out["text"][i, :e] = f["roi"].float(), f["geom"], f["text"].float()
            out["has_text"][i, :e], out["role"][i, :e], out["e_mask"][i, :e] = f["has_text"], f["role"], True
            out["c_type"][i, :c], out["c_target"][i, :c] = f["c_type"], f["c_target"]
            out["c_arg"][i, :c], out["c_mask"][i, :c] = f["c_arg"].float(), True
        return {k: v.to(device) for k, v in out.items()}

    def candidate_embeddings(self, *, batch: dict, h_e: torch.Tensor) -> torch.Tensor:
        target = batch["c_target"].clamp(min=0)
        gathered = torch.gather(h_e, 1, target.unsqueeze(-1).expand(-1, -1, D))
        tgt = torch.where((batch["c_target"] >= 0).unsqueeze(-1), gathered, self.no_target.expand_as(gathered))
        return self.cand(torch.cat([self.c_type(batch["c_type"]), tgt, self.arg_proj(batch["c_arg"])], dim=-1))

    def prev_token(self, *, batch: dict) -> torch.Tensor:
        """[PREV_ACTION]: the previous action's type, its target element's text and its argument (frozen inputs, so
        an update recomputes it exactly); the learned no_prev vector where there was none (p_type -1)."""
        t = batch["p_type"]
        tok = self.prev(torch.cat([self.prev_type(t.clamp(min=0)), self.prev_target(batch["p_target_text"]),
                                   self.arg_proj(batch["p_arg"])], dim=-1))
        return torch.where((t >= 0).unsqueeze(-1), tok, self.no_prev.expand_as(tok))

    def forward(self, batch: dict) -> tuple:
        """(logits (B, C) with -inf on padding, value (B,), candidate embeddings (B, C, D))."""
        bsz = batch["grid"].shape[0]
        text = torch.where(batch["has_text"].unsqueeze(-1), self.text_proj(batch["text"]),
                           self.no_text.expand(bsz, batch["text"].shape[1], D))
        elems = text + self.role(batch["role"]) + self.geom(fourier(geom=batch["geom"])) + self.roi_proj(batch["roi"])
        prev = self.prev_token(batch=batch)
        head = torch.cat([self.state.expand(bsz, 1, D), prev.unsqueeze(1), self.grid_proj(batch["grid"])], dim=1)
        head = head + self.pos
        tokens = torch.cat([head, elems], dim=1)
        pad = torch.cat([torch.zeros(bsz, head.shape[1], dtype=torch.bool, device=tokens.device),
                         ~batch["e_mask"]], dim=1)
        h = self.encoder(tokens, src_key_padding_mask=pad)
        z, h_e = h[:, 0], h[:, head.shape[1]:]
        e_c = self.candidate_embeddings(batch=batch, h_e=h_e)
        logits = self.logit(torch.cat([z.unsqueeze(1).expand_as(e_c), e_c], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~batch["c_mask"], float("-inf"))
        return logits, self.value(z).squeeze(-1), e_c


def layer_init(layer, std: float = math.sqrt(2), bias: float = 0.0):
    """cleanrl's initialisation (orthogonal weights, constant bias)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class PixelPolicy(nn.Module):
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
