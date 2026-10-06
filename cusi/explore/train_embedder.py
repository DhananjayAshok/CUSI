"""Training the learned image embedders (curiosity_plan §3.2b); checkpoints in the format
cusi.state loads (weights + embedder_meta.json).

    cnn     GameBoyRL's train_observation_encoder.py, on replay frames: the per-patch autoencoder
            (cusi.state.encoders.cnn) trained on reconstruction (MSE in the BatchNorm-normalised
            patch space).
    siglip  observation-reconstruction fine-tuning: SigLIP's pooled vector -> the pixel decoder
            (cusi.explore.decoder.EmbeddingDecoder) -> the env canvas, L1 + MSE, vision tower
            (low learning rate) and decoder trained together. Writes vision_model.pt +
            embedder_meta.json (+ decoder.pt), which SiglipEmbedder(load_path=...) loads.

Frames come from replay directories (any embedder's replay: frames are always stored), put on the
env canvas. Both stop at --epochs or --max_minutes, whichever comes first.
"""
import json
import os
import time
from typing import Any, Optional
import numpy as np
import torch
from cusi.utils.log_handling import log_info
from cusi.explore.replay import iter_replay
from cusi.state.canvas import to_canvas


def load_canvases(*, replay_dirs: list, env_name: str, max_frames: int) -> np.ndarray:
    frames = []
    for directory in replay_dirs:
        for row in iter_replay(directory=directory):
            frames.append(to_canvas(frame=row["frame"], env_name=env_name))
            if len(frames) >= max_frames:
                return np.stack(frames)
    if not frames:
        raise ValueError(f"No frames in {replay_dirs}")
    return np.stack(frames)


def _split(*, n: int, val_frac: float, seed: int) -> tuple:
    perm = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(n * val_frac))
    return perm[n_val:], perm[:n_val]


def _save_grid(*, real: np.ndarray, decoded: np.ndarray, path: str) -> None:
    from PIL import Image
    def rgb(x):
        return np.repeat(x, 3, axis=-1) if x.shape[-1] == 1 else x
    grid = np.concatenate([np.concatenate(list(rgb(real)), axis=1), np.concatenate(list(rgb(decoded)), axis=1)], axis=0)
    Image.fromarray(grid).save(path)


def train_cnn(*, env_name: str, replay_dirs: list, out_dir: str, epochs: int = 20, max_minutes: float = 5.0,
              batch_size: int = 64, lr: float = 1e-3, max_frames: int = 50_000, val_frac: float = 0.05,
              seed: int = 0, device: Optional[str] = None, parameters: dict[str, Any] = None) -> dict:
    from cusi.state.encoders.cnn import CNNEmbedder
    torch.manual_seed(seed)
    emb = CNNEmbedder(env_name=env_name, device=device)
    canv = load_canvases(replay_dirs=replay_dirs, env_name=env_name, max_frames=max_frames)
    train_idx, val_idx = _split(n=len(canv), val_frac=val_frac, seed=seed)
    x_all = torch.from_numpy(canv).permute(0, 3, 1, 2).float()
    from cusi.state.encoders.random_patch import extract_patches
    patches = extract_patches(x=x_all, kernel_size=emb.kernel_size)        # (N, P, C*k*k) on CPU
    opt = torch.optim.Adam(emb.net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    t0, history = time.time(), []
    for epoch in range(epochs):
        emb.net.train()
        losses = []
        order = rng.permutation(train_idx)
        for start in range(0, len(order), batch_size):
            b = patches[order[start:start + batch_size]]
            recon, target = emb.net(b.to(emb.device))
            loss = torch.nn.functional.mse_loss(recon, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        emb.net.eval()
        with torch.no_grad():
            recon, target = emb.net(patches[val_idx[:512]].to(emb.device))
            val = float(torch.nn.functional.mse_loss(recon, target))
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "val_mse": val,
                        "minutes": (time.time() - t0) / 60})
        log_info(f"cnn [{env_name}] epoch {epoch}: {json.dumps(history[-1])}", parameters=parameters)
        if time.time() - t0 > max_minutes * 60:
            break
    emb.save(directory=out_dir)
    sample = val_idx[:6]
    decoded = emb.decode(emb.embed(frames=list(canv[sample])))
    pixel_mae = float(np.abs(decoded.astype(np.float32) - canv[sample].astype(np.float32)).mean())
    _save_grid(real=canv[sample], decoded=decoded, path=os.path.join(out_dir, "val_reconstructions.png"))
    summary = {"embedder": "cnn", "n_frames": int(len(canv)), "output_dim": emb.output_dim, "final": history[-1],
               "val_pixel_mae_decoded": pixel_mae, "replay_dirs": replay_dirs}
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump({"summary": summary, "history": history}, f, indent=1)
    log_info(f"cnn embedder -> {out_dir}: {json.dumps(summary)}", parameters=parameters)
    return summary


def train_siglip(*, env_name: str, replay_dirs: list, out_dir: str, encoder_model: str, steps: int = 20,
                 max_minutes: float = 10.0, batch_size: int = 16, lr: float = 1e-5, decoder_lr: float = 3e-4,
                 max_frames: int = 5_000, val_frac: float = 0.05, seed: int = 0, device: Optional[str] = None,
                 parameters: dict[str, Any] = None) -> dict:
    from transformers import AutoModel
    from cusi.explore.decoder import EmbeddingDecoder
    from cusi.state.encoders.base import write_meta
    from cusi.state.encoders.siglip import VISION_WEIGHTS, SiglipEmbedder
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    # The inference wrapper gives the processor and canvas handling; training uses an fp32 copy.
    wrapper = SiglipEmbedder(env_name=env_name, model_name=encoder_model, device="cpu")
    model = AutoModel.from_pretrained(encoder_model, dtype=torch.float32).to(device)
    model.train()
    for p in model.text_model.parameters():
        p.requires_grad_(False)
    h, w, _ = wrapper.canvas
    decoder = EmbeddingDecoder(emb_dim=wrapper.output_dim, out_size=(h, w)).to(device)
    canv = load_canvases(replay_dirs=replay_dirs, env_name=env_name, max_frames=max_frames)
    train_idx, val_idx = _split(n=len(canv), val_frac=val_frac, seed=seed)
    opt = torch.optim.AdamW([{"params": model.vision_model.parameters(), "lr": lr},
                             {"params": decoder.parameters(), "lr": decoder_lr}])
    rng = np.random.default_rng(seed)

    def batch_loss(idx):
        frames = list(canv[idx])
        inputs = wrapper.processor(images=wrapper.pil_canvases(frames=frames), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = torch.nn.functional.normalize(SiglipEmbedder.pooled(model=model, inputs=inputs).float(), dim=-1)
        target = torch.from_numpy(np.stack([np.repeat(f, 3, axis=-1) if f.shape[-1] == 1 else f for f in frames]))
        target = target.permute(0, 3, 1, 2).float().to(device) / 255.0
        pred = decoder(feats)
        return (pred - target).abs().mean() + ((pred - target) ** 2).mean(), pred, target

    t0, history = time.time(), []
    for step in range(steps):
        loss, _, _ = batch_loss(rng.choice(train_idx, size=min(batch_size, len(train_idx)), replace=False))
        opt.zero_grad()
        loss.backward()
        opt.step()
        entry = {"step": step, "train_loss": float(loss), "minutes": (time.time() - t0) / 60}
        if step % 5 == 0 or step == steps - 1:
            with torch.no_grad():
                vl, pred, target = batch_loss(val_idx[:batch_size])
            entry.update({"val_loss": float(vl), "val_pixel_mae": float((pred - target).abs().mean() * 255)})
            log_info(f"siglip fine-tune [{env_name}] {json.dumps(entry)}", parameters=parameters)
        history.append(entry)
        if time.time() - t0 > max_minutes * 60:
            break
    os.makedirs(out_dir, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.vision_model.state_dict().items()},
               os.path.join(out_dir, VISION_WEIGHTS))
    write_meta(directory=out_dir, meta=wrapper.meta())
    decoder.save(directory=out_dir)
    summary = {"embedder": "siglip", "base_model": encoder_model, "n_frames": int(len(canv)),
               "steps": len(history), "final": history[-1], "replay_dirs": replay_dirs}
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump({"summary": summary, "history": history}, f, indent=1)
    log_info(f"siglip fine-tune -> {out_dir}: {json.dumps(summary)}", parameters=parameters)
    return summary
