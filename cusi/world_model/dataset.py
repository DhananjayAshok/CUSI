"""
Train / val transition sets from one or more search trees (run_search.py runs), for the tree world models.

A transition is one committed expansion step: the screen before it (with the one before that, the world model's
2-frame stack), the action (cusi.world_model.actions: type + target box) and the screen after. An expansion's first
step starts from its node's frame. Steps whose action does not parse to a type, and Web bot checks, are left out.

Split: by expansion (all steps of one expansion on one side), a fraction train_frac to train, by a seeded hash of
(run, expansion id). Val dedup: a val transition is dropped when some train transition has a pixel-identical screen
before AND a pixel-identical screen after (the action is not compared; matching only one of the two keeps it).
max_train (optional, for comparing trees at equal size): the train side is cut to that many transitions, whole
expansions in a seeded hash order (the last one truncated), before the val dedup.

Frames for the embedder: one reference per distinct screen (pixel hash), train = the train transitions' screens,
val = the kept val transitions' screens that are not also a train screen.

Output (directory): train.jsonl / val.jsonl (one transition per line: run, expansion, step, prev2, prev, next frame
paths and their pixel hashes, type, box), frames_train.jsonl / frames_val.jsonl ({"hash", "path"}), dataset.json (the arguments and counts).
"""
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Optional
import numpy as np
from cusi.utils.log_handling import log_info
from cusi.search.store import node_rows
from cusi.world_model.actions import action_box, action_type

DATASET_FILE = "dataset.json"


def tree_game(*, directory: str, env_name: str):
    if env_name != "gameboy":
        return None
    with open(os.path.join(directory, "config.json")) as f:
        config = json.load(f)
    return config.get("config", config).get("game") or "pokemon_red"


def _committed_expansions(*, directory: str):
    with open(os.path.join(directory, "archive", "commit.json")) as f:
        n = json.load(f)["n_expansions"]
    with open(os.path.join(directory, "expansions.jsonl")) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            if line.strip():
                yield json.loads(line)


def _is_train(*, run: str, expansion: int, seed: int, train_frac: float) -> bool:
    digest = hashlib.sha1(f"{seed}:{run}:{expansion}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64 < train_frac


def tree_transitions(*, directory: str, env_name: str) -> tuple:
    """(transitions, skipped counts) of one tree; box holds the raw [x0, y0, x1, y1] (normalised later)."""
    nodes = node_rows(directory=directory)
    out, skipped = [], {"no_type": 0, "challenge": 0}
    for exp in _committed_expansions(directory=directory):
        node = nodes[exp["node_id"]]
        frames = [node["frame"]] + [s["frame"] for s in exp["steps"]]
        boxes = [node.get("text_boxes")] + [s.get("text_boxes") for s in exp["steps"]]
        for t, step in enumerate(exp["steps"]):
            info = step.get("info") or {}
            if info.get("challenge"):
                skipped["challenge"] += 1
                continue
            parsed = info.get("parsed_action")
            kind = action_type(env_name=env_name, parsed=parsed)
            if kind is None:
                skipped["no_type"] += 1
                continue
            out.append({"run": directory, "expansion": exp["id"], "step": t,
                        "prev2": os.path.join(directory, frames[max(t - 1, 0)]),
                        "prev": os.path.join(directory, frames[t]), "next": os.path.join(directory, frames[t + 1]),
                        "type": kind, "parsed": parsed, "boxes_before": boxes[t]})
    return out, skipped


def _hash_zip(args) -> dict:
    """path -> (pixel sha1, (width, height)) for the members of one frame zip."""
    import io
    import zipfile
    from PIL import Image
    archive, members = args
    out = {}
    with zipfile.ZipFile(archive) as z:
        for m in members:
            img = Image.open(io.BytesIO(z.read(m))).convert("RGB")
            x = np.asarray(img, dtype=np.uint8)
            out[f"{archive}/{m}"] = (hashlib.sha1(np.array(x.shape).tobytes() + x.tobytes()).hexdigest(), img.size)
    return out


def pixel_hashes(*, paths: set, workers: int) -> dict:
    by_zip = {}
    for p in paths:
        archive, member = p.rsplit("/", 1)
        by_zip.setdefault(archive, []).append(member)
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(_hash_zip, sorted(by_zip.items()), chunksize=4):
            out.update(part)
    return out


def _cap_train(*, train: list, max_train: int, seed: int) -> list:
    """At most max_train transitions of `train`: whole expansions in a seeded hash order, the last one truncated."""
    groups = {}
    for r in train:
        groups.setdefault((r["run"], r["expansion"]), []).append(r)
    key = lambda g: hashlib.sha1(f"cap:{seed}:{g[0]}:{g[1]}".encode()).digest()
    out = []
    for g in sorted(groups, key=key):
        out += groups[g][:max_train - len(out)]
        if len(out) >= max_train:
            break
    return out


def build_dataset(*, env_name: str, run_dirs: list, out_dir: str, train_frac: float, seed: int, workers: int,
                  max_train: Optional[int] = None, parameters: dict[str, Any] = None) -> dict:
    games = {tree_game(directory=d, env_name=env_name) for d in run_dirs}
    if len(games) > 1:
        raise ValueError(f"trees from different games: {sorted(map(str, games))}")
    transitions, skipped = [], {}
    for d in run_dirs:
        rows, sk = tree_transitions(directory=d, env_name=env_name)
        log_info(f"world model data [{env_name}] {d}: {len(rows)} transitions, skipped {sk}", parameters=parameters)
        transitions += rows
        skipped[d] = sk
    paths = {p for r in transitions for p in (r["prev2"], r["prev"], r["next"])}
    log_info(f"world model data [{env_name}]: hashing {len(paths)} frames", parameters=parameters)
    hashes = pixel_hashes(paths=paths, workers=workers)
    box_stats = {"indexed": 0, "found": 0}
    for r in transitions:
        parsed, boxes = r.pop("parsed"), r.pop("boxes_before")
        r["box"] = action_box(env_name=env_name, parsed=parsed, text_boxes=boxes, frame_size=hashes[r["prev"]][1])
        r["hashes"] = [hashes[r[k]][0] for k in ("prev2", "prev", "next")]
        if parsed.get("index") is not None or isinstance(parsed.get("target"), int):
            box_stats["indexed"] += 1
            box_stats["found"] += int(r["box"][-1] > 0)
    train, val = [], []
    for r in transitions:
        (train if _is_train(run=r["run"], expansion=r["expansion"], seed=seed, train_frac=train_frac) else val).append(r)
    n_train_uncapped = len(train)
    if max_train is not None and len(train) > max_train:
        train = _cap_train(train=train, max_train=max_train, seed=seed)
    pair = lambda r: (hashes[r["prev"]][0], hashes[r["next"]][0])
    train_pairs = {pair(r) for r in train}
    kept = [r for r in val if pair(r) not in train_pairs]
    train_frames, val_frames = {}, {}
    for r in train:
        for p in (r["prev2"], r["prev"], r["next"]):
            train_frames.setdefault(hashes[p][0], p)
    for r in kept:
        for p in (r["prev2"], r["prev"], r["next"]):
            h = hashes[p][0]
            if h not in train_frames:
                val_frames.setdefault(h, p)
    os.makedirs(out_dir, exist_ok=True)
    for name, rows in (("train", train), ("val", kept)):
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    for name, frames in (("frames_train", train_frames), ("frames_val", val_frames)):
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
            for h, p in frames.items():
                f.write(json.dumps({"hash": h, "path": p}) + "\n")
    counts = lambda rows: {t: sum(r["type"] == t for r in rows) for t in sorted({r["type"] for r in rows})}
    summary = {"env": env_name, "game": games.pop(), "run_dirs": run_dirs, "train_frac": train_frac, "seed": seed,
               "split": "by expansion", "val_dedup": "dropped if a train transition has the same before and after pixels",
               "n_transitions": len(transitions), "max_train": max_train, "n_train_uncapped": n_train_uncapped,
               "n_train": len(train), "n_val_before_dedup": len(val),
               "n_val": len(kept), "n_val_dropped": len(val) - len(kept), "n_train_frames": len(train_frames),
               "n_val_frames": len(val_frames), "n_frames": len(paths),
               "n_distinct_screens": len({h for h, _ in hashes.values()}), "skipped": skipped, "boxes": box_stats,
               "train_types": counts(train), "val_types": counts(kept)}
    with open(os.path.join(out_dir, DATASET_FILE), "w") as f:
        json.dump(summary, f, indent=1)
    log_info(f"world model data [{env_name}] -> {out_dir}: {json.dumps({k: v for k, v in summary.items() if k not in ('skipped', 'run_dirs')})}",
             parameters=parameters)
    return summary


def read_dataset(*, directory: str) -> dict:
    """dataset.json plus the train / val transitions and frame lists."""
    with open(os.path.join(directory, DATASET_FILE)) as f:
        out = json.load(f)
    for name in ("train", "val", "frames_train", "frames_val"):
        with open(os.path.join(directory, f"{name}.jsonl")) as f:
            out[name] = [json.loads(line) for line in f if line.strip()]
    return out
