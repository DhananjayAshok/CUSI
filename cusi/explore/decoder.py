"""A decoder from frozen-encoder embeddings to pixels (plan decision 17, Part 2.8), trained per
environment on replay-buffer frames at a reduced resolution, so world-model predictions
(embeddings) can be shown as frames, as GameBoyRL's autoencoder decoder does.

    Linear(D -> 512 * h/16 * w/16) -> 4 x [Upsample x2, Conv3x3, GroupNorm, SiLU] -> Conv3x3 -> sigmoid
    loss: L1 + MSE on [0, 1] RGB at OUTPUT_SIZES[env]

    decoder = EmbeddingDecoder.load(directory=...); frames = decoder.decode(embeddings)  # uint8 (N, h, w, 3)
"""
import json
import os
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from cusi.utils.log_handling import log_info
from cusi.explore.replay import iter_replay

# (height, width), divisible by 16, keeping each environment's aspect ratio roughly:
# GameBoy 144x160 (native), Android 2400x1080 -> 256x112, Web 768x1024 -> 96x128.
OUTPUT_SIZES = {"gameboy": (144, 160), "android": (256, 112), "web": (96, 128)}


class EmbeddingDecoder(nn.Module):
    def __init__(self, *, emb_dim: int, out_size: tuple, base_channels: int = 512):
        super().__init__()
        self.emb_dim, self.out_size, self.base_channels = emb_dim, tuple(out_size), base_channels
        h0, w0 = out_size[0] // 16, out_size[1] // 16
        self.h0, self.w0 = h0, w0
        self.fc = nn.Linear(emb_dim, base_channels * h0 * w0)
        chans = [base_channels, 256, 128, 64, 32]
        blocks = []
        for c_in, c_out in zip(chans[:-1], chans[1:]):
            blocks += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c_in, c_out, 3, padding=1),
                       nn.GroupNorm(8, c_out), nn.SiLU()]
        self.net = nn.Sequential(*blocks, nn.Conv2d(chans[-1], 3, 3, padding=1))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        x = self.fc(emb).view(-1, self.base_channels, self.h0, self.w0)
        return torch.sigmoid(self.net(x))          # (B, 3, h, w) in [0, 1]

    @torch.no_grad()
    def decode(self, emb: torch.Tensor) -> np.ndarray:
        self.eval()
        out = self.forward(emb.to(next(self.parameters()).device).float())
        return (out.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)

    def save(self, *, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "emb_dim": self.emb_dim, "out_size": self.out_size,
                    "base_channels": self.base_channels}, os.path.join(directory, "decoder.pt"))

    @classmethod
    def load(cls, *, directory: str, device: str = "cpu") -> "EmbeddingDecoder":
        ckpt = torch.load(os.path.join(directory, "decoder.pt"), map_location=device)
        model = cls(emb_dim=ckpt["emb_dim"], out_size=ckpt["out_size"], base_channels=ckpt["base_channels"])
        model.load_state_dict(ckpt["state_dict"])
        return model.to(device)


def _load_frames(*, replay_dirs: list, out_size: tuple, max_frames: int) -> tuple:
    embs, frames = [], []
    for directory in replay_dirs:
        for row in iter_replay(directory=directory):
            img = row["frame"].pil().resize((out_size[1], out_size[0]))
            frames.append(np.asarray(img, dtype=np.uint8))
            embs.append(np.asarray(row["embedding"], dtype=np.float32))
            if len(frames) >= max_frames:
                return np.stack(embs), np.stack(frames)
    return np.stack(embs), np.stack(frames)


def train_decoder(*, env_name: str, replay_dirs: list, out_dir: str, epochs: int = 30, batch_size: int = 64,
                  lr: float = 3e-4, max_frames: int = 50_000, val_frac: float = 0.05, seed: int = 0,
                  device: str = None, parameters: dict[str, Any] = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_size = OUTPUT_SIZES[env_name]
    embs, frames = _load_frames(replay_dirs=replay_dirs, out_size=out_size, max_frames=max_frames)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(frames))
    n_val = max(1, int(len(frames) * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    torch.manual_seed(seed)
    model = EmbeddingDecoder(emb_dim=embs.shape[1], out_size=out_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    emb_t = torch.tensor(embs)
    img_t = torch.tensor(frames).permute(0, 3, 1, 2)          # uint8 on CPU; batches move to device

    def loss_on(idx):
        e = emb_t[idx].to(device)
        x = img_t[idx].to(device).float() / 255.0
        pred = model(e)
        return (pred - x).abs().mean() + ((pred - x) ** 2).mean(), pred, x

    history = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(order), batch_size):
            loss, _, _ = loss_on(order[start:start + batch_size])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        model.eval()
        with torch.no_grad():
            vl, pred, x = loss_on(val_idx[:256])
            pixel_mae = float((pred - x).abs().mean() * 255)
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": float(vl),
                 "val_pixel_mae": pixel_mae}
        history.append(entry)
        if epoch % 5 == 0 or epoch == epochs - 1:
            log_info(f"decoder [{env_name}] epoch {epoch}: {json.dumps(entry)}", parameters=parameters)
    model.save(directory=out_dir)
    # a side-by-side sample of validation frames (top: real, bottom: decoded)
    from PIL import Image
    sample = val_idx[:8]
    real = frames[sample]
    decoded = model.decode(emb_t[sample])
    grid = np.concatenate([np.concatenate(list(real), axis=1), np.concatenate(list(decoded), axis=1)], axis=0)
    Image.fromarray(grid).save(os.path.join(out_dir, "val_reconstructions.png"))
    summary = {"n_frames": int(len(frames)), "out_size": out_size, "final": history[-1]}
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump({"summary": summary, "history": history}, f, indent=1)
    return summary
