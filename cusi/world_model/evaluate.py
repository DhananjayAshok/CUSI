"""
Testing a tree world model on real task-attempt transitions: the steps an agent took in eval episodes
(storage/eval/<env>/<run>/episodes/<task>/, written by cusi.eval.episode), e.g. base Gemma attempts on the test tasks.
None of these screens were in the tree the model was trained on.

A transition is one valid step: the screen before it, the screen before that (the model's second input; the first
step of a run of steps uses its own screen, as the WM executor does) and the screen after it. Consecutive steps whose
screens chain (one step's after is the next one's before) form a sequence, used for open-loop rollouts.

Reports (overall, per action type, and split by whether the action has a target box):
    one step: embedding mse, cos (predicted vs real next), the copy baseline's cos (the current screen as the
        prediction), both on transitions whose screen changes; decoded pixel MAE (0-255 on the model's canvas) of the
        prediction, of the copy baseline, and of the decoded real next embedding (the decoder's floor).
    rollout k = 1..max_k: from a real screen, the model's own predictions are fed back for k steps of the real
        actions and compared with the real screen k steps later (cos, and the copy baseline: the starting screen).

Eval episodes do not record the screen's element boxes, so an indexed Web / Android action (Click [n], ...) gets
NO_BOX here (the model saw its box in training); those transitions are reported apart (box: missing).
"""
import glob
import json
import os
from typing import Any, Optional
import numpy as np
import torch
from cusi.utils.log_handling import log_info
from cusi.world_model.actions import NO_BOX, action_box, action_type, target_index
from cusi.world_model.embedder import _Frames, _rgb
from cusi.world_model.train import embed_screens, read_wm_meta


def _episode_steps(*, episode_dir: str) -> list:
    """The episode's steps in order: dicts with before, after (absolute frame paths) and parsed_action."""
    steps = []
    for leg_file in sorted(glob.glob(os.path.join(episode_dir, "legs", "*.jsonl")),
                           key=lambda p: int(os.path.basename(p).split(".")[0])):
        with open(leg_file) as f:
            for line in f:
                for s in json.loads(line)["steps"]:
                    if s.get("kind") != "step" or not s.get("valid", True):
                        continue
                    if not s.get("frame_before") or not s.get("frame_after"):
                        continue
                    steps.append({"before": os.path.join(episode_dir, s["frame_before"]),
                                  "after": os.path.join(episode_dir, s["frame_after"]),
                                  "parsed": s.get("parsed_action") or {}})
    return steps


def episode_sequences(*, env_name: str, run_dirs: list, game: Optional[str]) -> list:
    """Sequences of chained transitions from the eval runs' episodes (only episodes of `game` when it is set).
    Each transition: prev, cur, next (paths), type, box, box_state (none / given / missing), task."""
    sequences = []
    for run_dir in run_dirs:
        for episode_dir in sorted(glob.glob(os.path.join(run_dir, "episodes", "*"))):
            meta_path = os.path.join(episode_dir, "meta.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            if game is not None and meta.get("game") not in (None, game):
                continue
            seq, prev_after = [], None
            for s in _episode_steps(episode_dir=episode_dir):
                if prev_after != s["before"] and seq:
                    sequences.append(seq)
                    seq = []
                prev_after = s["after"]
                kind = action_type(env_name=env_name, parsed=s["parsed"])
                if kind is None:
                    if seq:
                        sequences.append(seq)
                    seq = []
                    continue
                box = action_box(env_name=env_name, parsed=s["parsed"], text_boxes=None, frame_size=(1, 1))
                indexed = target_index(env_name=env_name, parsed=s["parsed"]) is not None
                seq.append({"prev": seq[-1]["cur"] if seq else s["before"], "cur": s["before"], "next": s["after"],
                            "type": kind, "box": box if not indexed else list(NO_BOX),
                            "box_state": "missing" if indexed else ("given" if box[4] else "none"),
                            "task": f"{os.path.basename(run_dir)}/{os.path.basename(episode_dir)}"})
            if seq:
                sequences.append(seq)
    return sequences


def _mean(x) -> Optional[float]:
    return float(np.mean(x)) if len(x) else None


def _summarise(*, rows: dict, mask: np.ndarray) -> dict:
    changed = mask & rows["changed"]
    out = {"n": int(mask.sum()), "n_changed": int(changed.sum())}
    for k in ("mse", "cos", "copy_cos", "pixel_mae", "copy_pixel_mae", "decoder_floor_pixel_mae"):
        out[k] = _mean(rows[k][mask])
    for k in ("cos", "copy_cos", "pixel_mae", "copy_pixel_mae"):
        out[f"{k}_changed"] = _mean(rows[k][changed])
    return out


@torch.no_grad()
def evaluate_on_episodes(*, wm_dir: str, run_dirs: list, out_path: str, max_k: int, batch_size: int, workers: int,
                         device: Optional[str] = None, parameters: dict[str, Any] = None) -> dict:
    from cusi.explore.decoder import EmbeddingDecoder
    from cusi.state.encoders.siglip import SiglipEmbedder
    from cusi.world_model.model import BoxActionWorldModel
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    meta = read_wm_meta(directory=wm_dir)
    env_name = meta["env"]
    sequences = episode_sequences(env_name=env_name, run_dirs=run_dirs, game=meta.get("game"))
    flat = [t for seq in sequences for t in seq]
    if not flat:
        raise ValueError(f"no transitions in {run_dirs}")
    paths = sorted({p for t in flat for p in (t["prev"], t["cur"], t["next"])})
    index = {p: i for i, p in enumerate(paths)}
    embedder = SiglipEmbedder(env_name=env_name, model_name=meta["base_model"], load_path=meta["embedder_dir"],
                              device=device)
    emb = embed_screens(embedder=embedder, env_name=env_name, paths=paths, batch_size=batch_size,
                        workers=workers).to(device).float()
    del embedder
    decoder = EmbeddingDecoder.load(directory=meta["embedder_dir"], device=device).eval()
    wm = BoxActionWorldModel.load(directory=wm_dir, device=device).eval()
    log_info(f"world model test [{env_name}] {wm_dir}: {len(flat)} transitions in {len(sequences)} sequences, "
             f"{len(paths)} screens", parameters=parameters)

    def step(e_prev, e_cur, kinds, boxes):
        return wm(obs=torch.cat([e_prev, e_cur], dim=-1),
                  types=torch.tensor([wm.type_index[k] for k in kinds], device=device),
                  boxes=torch.tensor(boxes, dtype=torch.float32, device=device))

    # one step, with decoded pixels
    h = torch.tensor([[index[t[k]] for k in ("prev", "cur", "next")] for t in flat], device=device)
    rows = {k: [] for k in ("mse", "cos", "copy_cos", "pixel_mae", "copy_pixel_mae", "decoder_floor_pixel_mae")}
    frames = _Frames(paths=paths, env_name=env_name)
    for start in range(0, len(flat), batch_size):
        part = flat[start:start + batch_size]
        hh = h[start:start + batch_size]
        pred = step(emb[hh[:, 0]], emb[hh[:, 1]], [t["type"] for t in part], [t["box"] for t in part])
        real, cur = emb[hh[:, 2]], emb[hh[:, 1]]
        rows["mse"].append(((pred - real) ** 2).mean(-1).cpu().numpy())
        rows["cos"].append((pred * real).sum(-1).cpu().numpy())
        rows["copy_cos"].append((cur * real).sum(-1).cpu().numpy())
        nxt = np.stack([_rgb(frames[int(i)][None])[0] for i in hh[:, 2].tolist()]).astype(np.float32)
        now = np.stack([_rgb(frames[int(i)][None])[0] for i in hh[:, 1].tolist()]).astype(np.float32)
        mae = lambda a: np.abs(a.astype(np.float32) - nxt).mean(axis=(1, 2, 3))
        rows["pixel_mae"].append(mae(decoder.decode(pred)))
        rows["copy_pixel_mae"].append(mae(now))
        rows["decoder_floor_pixel_mae"].append(mae(decoder.decode(real)))
    rows = {k: np.concatenate(v) for k, v in rows.items()}
    rows["changed"] = np.array([t["cur"] != t["next"] for t in flat])
    all_mask = np.ones(len(flat), dtype=bool)
    one_step = {"all": _summarise(rows=rows, mask=all_mask)}
    one_step["by_type"] = {k: _summarise(rows=rows, mask=np.array([t["type"] == k for t in flat]))
                           for k in sorted({t["type"] for t in flat})}
    one_step["by_box"] = {k: _summarise(rows=rows, mask=np.array([t["box_state"] == k for t in flat]))
                          for k in sorted({t["box_state"] for t in flat})}

    # open-loop rollouts within each sequence
    rollout = {}
    for k in range(1, max_k + 1):
        starts = [(seq, i) for seq in sequences for i in range(len(seq) - k + 1)]
        if not starts:
            break
        cos, copy_cos, changed = [], [], []
        for b in range(0, len(starts), batch_size):
            part = starts[b:b + batch_size]
            e_prev = emb[[index[seq[i]["prev"]] for seq, i in part]]
            e_cur = emb[[index[seq[i]["cur"]] for seq, i in part]]
            start_emb = e_cur
            for j in range(k):
                pred = step(e_prev, e_cur, [seq[i + j]["type"] for seq, i in part],
                            [seq[i + j]["box"] for seq, i in part])
                e_prev, e_cur = e_cur, pred
            real = emb[[index[seq[i + k - 1]["next"]] for seq, i in part]]
            cos.append((e_cur * real).sum(-1).cpu().numpy())
            copy_cos.append((start_emb * real).sum(-1).cpu().numpy())
            changed += [seq[i]["cur"] != seq[i + k - 1]["next"] for seq, i in part]
        cos, copy_cos, changed = np.concatenate(cos), np.concatenate(copy_cos), np.array(changed)
        rollout[str(k)] = {"n": len(cos), "cos": _mean(cos), "copy_cos": _mean(copy_cos),
                           "n_changed": int(changed.sum()), "cos_changed": _mean(cos[changed]),
                           "copy_cos_changed": _mean(copy_cos[changed])}

    result = {"wm_dir": wm_dir, "env": env_name, "game": meta.get("game"), "eval_runs": run_dirs,
              "n_transitions": len(flat), "n_sequences": len(sequences),
              "n_tasks": len({t["task"] for t in flat}), "one_step": one_step, "rollout": rollout}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    log_info(f"world model test -> {out_path}: {json.dumps({'all': one_step['all'], 'rollout': rollout})}",
             parameters=parameters)
    return result
