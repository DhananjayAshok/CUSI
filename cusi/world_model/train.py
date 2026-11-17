"""
Training a tree world model (cusi.world_model.model.BoxActionWorldModel) on a dataset (cusi.world_model.dataset) in a
fine-tuned SigLIP's embedding space (cusi.world_model.embedder): every distinct screen is embedded once, then the
model is trained on the train transitions (MSE between the predicted and the real next embedding, both unit norm)
and validated on the deduplicated val transitions every epoch; training stops after `patience` epochs without a new
best val loss (or at max_epochs) and the best epoch's weights are saved.

Val reports: mse, cos (predicted vs real next), the copy baseline's cos (the current screen as the prediction), and
both on the transitions whose screen changes (cos_changed, copy_cos_changed).

Output (out_dir): world_model.pt, wm_meta.json (env, game, the dataset and embedder directories, the base SigLIP
model, action types), train_summary.json.
"""
import copy
import json
import os
import time
from typing import Any, Optional
import numpy as np
import torch
from cusi.utils.log_handling import log_info
from cusi.world_model.actions import action_types
from cusi.world_model.dataset import read_dataset
from cusi.world_model.embedder import _Frames
from cusi.world_model.model import BoxActionWorldModel

WM_META = "wm_meta.json"


@torch.no_grad()
def embed_screens(*, embedder, env_name: str, paths: list, batch_size: int, workers: int) -> torch.Tensor:
    loader = torch.utils.data.DataLoader(_Frames(paths=paths, env_name=env_name), batch_size=batch_size,
                                         num_workers=workers, collate_fn=lambda xs: list(xs))
    return torch.cat([embedder.embed(frames=frames) for frames in loader]).half()


def _tensors(*, rows: list, index: dict, types: dict, device: str) -> dict:
    h = torch.tensor([[index[x] for x in r["hashes"]] for r in rows], dtype=torch.long)
    return {"h": h.to(device), "type": torch.tensor([types[r["type"]] for r in rows], dtype=torch.long).to(device),
            "box": torch.tensor([r["box"] for r in rows], dtype=torch.float32).to(device),
            "changed": (h[:, 1] != h[:, 2]).to(device)}


def train_world_model(*, env_name: str, dataset_dir: str, embedder_dir: str, out_dir: str, max_epochs: int,
                      patience: int, batch_size: int, lr: float, hidden_dim: int, residual: bool, seed: int,
                      embed_batch_size: int, workers: int, device: Optional[str] = None,
                      parameters: dict[str, Any] = None) -> dict:
    from cusi.state.encoders.siglip import SiglipEmbedder
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    data = read_dataset(directory=dataset_dir)
    if data["env"] != env_name:
        raise ValueError(f"{dataset_dir} is a {data['env']} dataset, not {env_name}")
    with open(os.path.join(embedder_dir, "embedder_meta.json")) as f:
        embedder_meta = json.load(f)
    embedder = SiglipEmbedder(env_name=env_name, model_name=embedder_meta["base_model"], load_path=embedder_dir,
                              device=device)
    screens = {r["hash"]: r["path"] for r in data["frames_train"] + data["frames_val"]}
    hashes = sorted(screens)
    index = {x: i for i, x in enumerate(hashes)}
    t0 = time.time()
    emb = embed_screens(embedder=embedder, env_name=env_name, paths=[screens[x] for x in hashes],
                        batch_size=embed_batch_size, workers=workers).to(device)
    log_info(f"world model [{env_name}]: embedded {len(hashes)} screens in {time.time() - t0:.0f} s",
             parameters=parameters)
    del embedder
    names = action_types(env_name=env_name)
    types = {t: i for i, t in enumerate(names)}
    train, val = (_tensors(rows=data[k], index=index, types=types, device=device) for k in ("train", "val"))
    model = BoxActionWorldModel(emb_dim=emb.shape[1], action_types=names, hidden_dim=hidden_dim,
                                residual=residual).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def predict(batch: dict, idx: torch.Tensor) -> tuple:
        h = batch["h"][idx]
        obs = torch.cat([emb[h[:, 0]], emb[h[:, 1]]], dim=-1).float()
        pred = model(obs=obs, types=batch["type"][idx], boxes=batch["box"][idx])
        return pred, emb[h[:, 2]].float(), emb[h[:, 1]].float()

    @torch.no_grad()
    def evaluate(batch: dict) -> dict:
        model.eval()
        n = len(batch["type"])
        sums = {"mse": 0.0, "cos": 0.0, "copy_cos": 0.0, "cos_changed": 0.0, "copy_cos_changed": 0.0}
        n_changed = int(batch["changed"].sum())
        for start in range(0, n, 8192):
            idx = torch.arange(start, min(start + 8192, n), device=device)
            pred, real, cur = predict(batch, idx)
            cos, copy_cos = (pred * real).sum(-1), (cur * real).sum(-1)
            ch = batch["changed"][idx]
            sums["mse"] += float(((pred - real) ** 2).mean(-1).sum())
            sums["cos"] += float(cos.sum())
            sums["copy_cos"] += float(copy_cos.sum())
            sums["cos_changed"] += float(cos[ch].sum())
            sums["copy_cos_changed"] += float(copy_cos[ch].sum())
        model.train()
        out = {k: v / n for k, v in sums.items() if not k.endswith("changed")}
        out.update({k: v / max(1, n_changed) for k, v in sums.items() if k.endswith("changed")})
        return out

    n_train = len(train["type"])
    best, best_state, since_best, history, stopped = {"mse": float("inf")}, None, 0, [], "max_epochs"
    t0 = time.time()
    for epoch in range(max_epochs):
        order = torch.randperm(n_train, generator=gen).to(device)
        losses = []
        for start in range(0, n_train, batch_size):
            pred, real, _ = predict(train, order[start:start + batch_size])
            loss = ((pred - real) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        entry = {"epoch": epoch, "train_mse": float(np.mean(losses)), "minutes": (time.time() - t0) / 60,
                 **{f"val_{k}": v for k, v in evaluate(val).items()}}
        history.append(entry)
        if entry["val_mse"] < best["mse"]:
            best, best_state, since_best = {"mse": entry["val_mse"], **entry}, copy.deepcopy(model.state_dict()), 0
        else:
            since_best += 1
        log_info(f"world model [{env_name}] {json.dumps(entry)}", parameters=parameters)
        if since_best >= patience:
            stopped = "early_stopping"
            break
    model.load_state_dict(best_state)
    model.save(directory=out_dir)
    meta = {"env": env_name, "game": data["game"], "dataset_dir": dataset_dir, "embedder_dir": embedder_dir,
            "base_model": embedder_meta["base_model"], "canvas": embedder_meta["canvas"], **model.config()}
    with open(os.path.join(out_dir, WM_META), "w") as f:
        json.dump(meta, f, indent=1)
    summary = {"env": env_name, "game": data["game"], "n_train": n_train, "n_val": len(val["type"]),
               "n_val_changed": int(val["changed"].sum()), "n_screens": len(hashes), "batch_size": batch_size,
               "lr": lr, "hidden_dim": hidden_dim, "residual": residual, "patience": patience, "seed": seed,
               "epochs": len(history), "stopped": stopped, "best": best,
               "best_train": evaluate(train) if n_train <= 200_000 else None}
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump({"summary": summary, "history": history}, f, indent=1)
    log_info(f"world model -> {out_dir}: {json.dumps(summary)}", parameters=parameters)
    return summary


def read_wm_meta(*, directory: str) -> dict:
    with open(os.path.join(directory, WM_META)) as f:
        return json.load(f)
