"""
SigLIP fine-tuning for the tree world models: the vision tower and an EmbeddingDecoder trained to reconstruct the
env's canvas from the pooled, L2-normalised feature (L1 + MSE on pixels in [0, 1]; as
cusi.explore.train_embedder.train_siglip), on a dataset's train screens (cusi.world_model.dataset), validated on its
held-out screens every eval_every steps with early stopping: training stops after `patience` validations without a
new best val loss, or at max_steps / max_minutes, and the best validation's weights are saved.

Output (out_dir): vision_model.pt, embedder_meta.json (read by cusi.state.encoders.siglip.SiglipEmbedder load_path),
decoder.pt (cusi.explore.decoder.EmbeddingDecoder), val_reconstructions.png, train_summary.json.
"""
import copy
import json
import os
import time
from typing import Any, Optional
import numpy as np
import torch
from cusi.utils.log_handling import log_info
from cusi.agents.frames import read_frame
from cusi.state.canvas import to_canvas


class _Frames(torch.utils.data.Dataset):
    def __init__(self, *, paths: list, env_name: str) -> None:
        self.paths, self.env_name = paths, env_name

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> np.ndarray:
        return to_canvas(frame=read_frame(path=self.paths[i]), env_name=self.env_name)


def _rgb(canvases: np.ndarray) -> np.ndarray:
    return np.repeat(canvases, 3, axis=-1) if canvases.shape[-1] == 1 else canvases


def train_siglip_early_stopping(*, env_name: str, train_paths: list, val_paths: list, out_dir: str,
                                encoder_model: str, max_steps: int, max_minutes: float, batch_size: int, lr: float,
                                decoder_lr: float, eval_every: int, patience: int, max_val_frames: int, workers: int,
                                seed: int, device: Optional[str] = None, parameters: dict[str, Any] = None) -> dict:
    from transformers import AutoModel
    from cusi.explore.decoder import EmbeddingDecoder
    from cusi.state.encoders.base import write_meta
    from cusi.state.encoders.siglip import VISION_WEIGHTS, SiglipEmbedder
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    wrapper = SiglipEmbedder(env_name=env_name, model_name=encoder_model, device="cpu")
    model = AutoModel.from_pretrained(encoder_model, dtype=torch.float32).to(device)
    model.train()
    for p in model.text_model.parameters():
        p.requires_grad_(False)
    h, w, _ = wrapper.canvas
    decoder = EmbeddingDecoder(emb_dim=wrapper.output_dim, out_size=(h, w)).to(device)
    opt = torch.optim.AdamW([{"params": model.vision_model.parameters(), "lr": lr},
                             {"params": decoder.parameters(), "lr": decoder_lr}])
    val_paths = [val_paths[i] for i in sorted(rng.choice(len(val_paths), size=min(max_val_frames, len(val_paths)),
                                                         replace=False))]
    val_canv = np.stack([_Frames(paths=val_paths, env_name=env_name)[i] for i in range(len(val_paths))])
    loader = torch.utils.data.DataLoader(
        _Frames(paths=train_paths, env_name=env_name), batch_size=batch_size, num_workers=workers,
        sampler=torch.utils.data.RandomSampler(range(len(train_paths)), replacement=True,
                                               num_samples=max_steps * batch_size,
                                               generator=torch.Generator().manual_seed(seed)),
        collate_fn=lambda xs: np.stack(xs), persistent_workers=workers > 0)
    log_info(f"siglip fine-tune [{env_name}]: {len(train_paths)} train screens, {len(val_canv)} val screens, "
             f"batch {batch_size}, eval every {eval_every}, patience {patience}", parameters=parameters)

    def batch_loss(frames: np.ndarray):
        inputs = wrapper.processor(images=wrapper.pil_canvases(frames=list(frames)), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = torch.nn.functional.normalize(SiglipEmbedder.pooled(model=model, inputs=inputs).float(), dim=-1)
        target = torch.from_numpy(_rgb(frames)).permute(0, 3, 1, 2).float().to(device) / 255.0
        pred = decoder(feats)
        return (pred - target).abs().mean() + ((pred - target) ** 2).mean(), pred, target

    def validate() -> dict:
        model.eval()
        losses, maes = [], []
        with torch.no_grad():
            for start in range(0, len(val_canv), batch_size):
                vl, pred, target = batch_loss(val_canv[start:start + batch_size])
                n = len(pred)
                losses.append(float(vl) * n)
                maes.append(float((pred - target).abs().mean() * 255) * n)
        model.train()
        return {"val_loss": sum(losses) / len(val_canv), "val_pixel_mae": sum(maes) / len(val_canv)}

    t0, history = time.time(), []
    best = {"val_loss": float("inf")}
    best_state, since_best, stopped = None, 0, "max_steps"
    for step, frames in enumerate(loader):
        loss, _, _ = batch_loss(frames)
        opt.zero_grad()
        loss.backward()
        opt.step()
        entry = {"step": step, "train_loss": float(loss), "minutes": (time.time() - t0) / 60}
        out_of_time = time.time() - t0 > max_minutes * 60
        if step % eval_every == 0 or step == max_steps - 1 or out_of_time:
            entry.update(validate())
            if entry["val_loss"] < best["val_loss"]:
                best, since_best = dict(entry), 0
                best_state = ({k: v.detach().cpu().clone() for k, v in model.vision_model.state_dict().items()},
                              copy.deepcopy(decoder.state_dict()))
            else:
                since_best += 1
            log_info(f"siglip fine-tune [{env_name}] {json.dumps(entry)} best {best['val_loss']:.5f} "
                     f"@ {best['step']}", parameters=parameters)
        history.append(entry)
        if out_of_time:
            stopped = "max_minutes"
            break
        if since_best >= patience:
            stopped = "early_stopping"
            break
    model.vision_model.load_state_dict(best_state[0])
    decoder.load_state_dict(best_state[1])
    os.makedirs(out_dir, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.vision_model.state_dict().items()},
               os.path.join(out_dir, VISION_WEIGHTS))
    write_meta(directory=out_dir, meta=wrapper.meta())
    decoder.save(directory=out_dir)
    model.eval()
    with torch.no_grad():
        _, pred, _ = batch_loss(val_canv[:6])
    from PIL import Image
    decoded = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
    grid = np.concatenate([np.concatenate(list(_rgb(val_canv[:6])), axis=1), np.concatenate(list(decoded), axis=1)])
    Image.fromarray(grid).save(os.path.join(out_dir, "val_reconstructions.png"))
    summary = {"embedder": "siglip", "base_model": encoder_model, "env": env_name, "n_train_screens": len(train_paths),
               "n_val_screens": int(len(val_canv)), "batch_size": batch_size, "lr": lr, "decoder_lr": decoder_lr,
               "eval_every": eval_every, "patience": patience, "seed": seed, "steps": len(history),
               "stopped": stopped, "best": best, "final": history[-1]}
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump({"summary": summary, "history": history}, f, indent=1)
    log_info(f"siglip fine-tune -> {out_dir}: {json.dumps(summary)}", parameters=parameters)
    return summary
