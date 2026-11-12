"""
Screen coverage: compare the screens (frames the agent saw) of search runs of one env, rather than their states
(cusi.search.state_coverage) or tasks (cusi.eval.task_coverage / task_diversity).

Every frame a run saw is embedded: each scene's start frame, then every expansion step's frame in the order the steps
were taken (restores are not exploration and saved no frame; curiosity sessions record states only, no frames). Two
frames show the same screen when their distance, 1 - cosine of their image embeddings (the archive's novelty, image
only), is at most tau. A run's distinct screens: a greedy pass over its frames in the order seen, a frame being a new
screen when no screen kept so far is within tau (overall, and per scene on that scene's frames alone). Runs are
compared in groups (cusi.search.groups; --runs makes each run its own group), never counting a screen twice within a
run: each run is reduced to its distinct screens, a group's distinct screens are a greedy pass at tau over its runs'
pooled distinct screens, and a group's screen is matched by another group when any frame of any of that group's runs
is within tau. Reported at tau / 2, tau and 2 tau, per scene, with each run's own count and the group's mean and sd.

Embedders: random_patch (the search's own, image only; CPU) or siglip (--encoder_model, which alone defaults to
google/siglip2-base-patch16-224: the user's call). tau per env and embedder: configs/screen_coverage.yaml, set by
calibrate.

Stages:
- embed: <run>/screen_coverage/<embedder key>/ (parts/*.npz, one per block of expansions, resumable; a run that grew
  is embedded from where it stopped).
- compare: storage/search/<env>/_compare/screen_coverage/<hash>/ (meta_<hash>.yaml, summary.json, summary.md,
  sheets of each group's unmatched screens beside their nearest frame among the other groups).
- distinct: one or more runs' distinct screens alone (printed, and written to the run's embedding dir as
  distinct_<tau>[_at<N>].json).
- calibrate: --stage sample writes same-scene frame pairs across distance bands, with both embedders' distances, as
  sheets for labelling (storage/tmp/screen_calibration/<env>/); --stage fit reads labels.jsonl there and prints the tau
  that best separates "same" from "different" per embedder.

    python -m cusi.search.screen_coverage embed --env web --run sat_random_s0 --embedder random_patch
    python -m cusi.search.screen_coverage compare --env web --runs sat_random_s0,sat_vlm_w_typing_s0 --embedder siglip
    python -m cusi.search.screen_coverage compare --env web --groups "random=sat_random_s0,sat_random_s1;vlm=..."
"""
import hashlib
import io
import json
import os
import random
import zipfile
from typing import Optional
import click
import numpy as np
import torch
from PIL import Image, ImageDraw
from cusi.search.groups import fmt_mean_sd, mean_sd, parse_groups
from cusi.search.store import EXPANSIONS_FILE, node_rows, read_rows
from cusi.utils.hash_handling import hash_meta_dict, write_meta
from cusi.utils.log_handling import log_info, log_warn
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.paths import search_root, search_run

ENV_NAMES = ("gameboy", "web", "android")
EMBEDDERS = ("random_patch", "siglip")
# The one model default allowed here (user's call 2026-11-11): --embedder siglip without --encoder_model.
DEFAULT_SIGLIP = "google/siglip2-base-patch16-224"
BLOCK = 200                 # expansions per embedding part
COMPARE_CHUNK = 4096        # query rows per matmul
DISTINCT_CHUNK = 1024       # frames checked against the kept screens at once
SHEET_SAMPLES = 24          # unmatched frames per contact sheet
DIR = "screen_coverage"


def embedder_key(*, embedder: str, encoder_model: Optional[str]) -> str:
    """random_patch, or the SigLIP model's short name with only [a-z0-9_] (also its configs/screen_coverage.yaml key)."""
    if embedder == "random_patch":
        return "random_patch"
    name = (encoder_model or DEFAULT_SIGLIP).rstrip("/").split("/")[-1].lower()
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def build_embedder(*, env_name: str, embedder: str, encoder_model: Optional[str]):
    if embedder == "random_patch":
        from cusi.state.encoders.random_patch import RandomPatchEmbedder
        return RandomPatchEmbedder(env_name=env_name)
    from cusi.state.encoders.siglip import SiglipEmbedder
    return SiglipEmbedder(env_name=env_name, model_name=encoder_model or DEFAULT_SIGLIP)


# --------------------------------------------------------------------------- frames

def run_frames(*, directory: str):
    """Blocks of (expansion index or -1 for the scene start frames, [(frame ref, scene, env step), ...]) in the order
    seen; env step counts the run's env steps (start frames are step 0)."""
    nodes = node_rows(directory=directory)
    yield -1, [(n["frame"], n["env_key"], 0) for n in nodes.values() if n["parent"] is None and n.get("frame")]
    steps = 0
    for i, exp in enumerate(read_rows(os.path.join(directory, EXPANSIONS_FILE))):
        scene = nodes[exp["node_id"]]["env_key"]
        rows = []
        for step in exp["steps"]:
            steps += 1
            if step.get("frame"):
                rows.append((step["frame"], scene, steps))
        yield i, rows


class FrameReader:
    """Frame bytes by reference ("frames/<k>.zip/<member>"), keeping the last zip open."""

    def __init__(self, *, directory: str) -> None:
        self.directory, self._name, self._zip = directory, None, None

    def read(self, ref: str) -> bytes:
        archive, member = ref.rsplit("/", 1)
        if archive != self._name:
            if self._zip is not None:
                self._zip.close()
            self._zip, self._name = zipfile.ZipFile(os.path.join(self.directory, archive)), archive
        return self._zip.read(member)

    def image(self, ref: str) -> Image.Image:
        return Image.open(io.BytesIO(self.read(ref))).convert("RGB")


# --------------------------------------------------------------------------- embed

def cache_dir(*, directory: str, key: str) -> str:
    return os.path.join(directory, DIR, key)


def embed_run(*, env_name: str, directory: str, embedder, key: str, parameters: dict) -> int:
    """Embeds the run's frames not yet embedded (whole blocks of BLOCK expansions; the last, possibly partial, block
    is redone when the run has grown). Returns the number of frames embedded now."""
    parts = os.path.join(cache_dir(directory=directory, key=key), "parts")
    os.makedirs(parts, exist_ok=True)
    done = {f for f in os.listdir(parts) if f.endswith(".npz")}
    reader, todo, done_now = FrameReader(directory=directory), {}, 0
    blocks = {}
    for i, rows in run_frames(directory=directory):
        b = -1 if i < 0 else i // BLOCK
        blocks.setdefault(b, []).extend(rows)
    last = max(blocks)
    for b, rows in sorted(blocks.items()):
        name = "roots.npz" if b < 0 else f"block_{b:05d}.npz"
        path = os.path.join(parts, name)
        if name in done:
            n_saved = int(np.load(path)["n"])
            if n_saved == len(rows):
                continue
        # one embedding per distinct frame in the block
        keys = [hashlib.sha1(reader.read(ref)).hexdigest() for ref, _, _ in rows]
        first = {}
        for k, (ref, _, _) in zip(keys, rows):
            first.setdefault(k, ref)
        unique = list(first)
        vecs = {}
        for s in range(0, len(unique), 256):
            batch = unique[s:s + 256]
            emb = embedder.embed(frames=[reader.image(first[k]) for k in batch])
            for k, v in zip(batch, emb.float().cpu().numpy()):
                vecs[k] = v
        emb = np.stack([vecs[k] for k in keys]).astype(np.float16) if rows else np.zeros((0, 1), np.float16)
        np.savez(path, emb=emb, ref=np.array([r[0] for r in rows]), scene=np.array([r[1] for r in rows]),
                 step=np.array([r[2] for r in rows], dtype=np.int64), key=np.array(keys), n=len(rows),
                 complete=b != last)
        done_now += len(rows)
        log_info(f"screen_coverage embed {os.path.basename(directory)} {key}: {name}, {len(rows)} frames "
                 f"({len(unique)} distinct)", parameters=parameters)
    with open(os.path.join(cache_dir(directory=directory, key=key), "meta.json"), "w") as f:
        json.dump({"env": env_name, "embedder": key, "blocks": len(blocks), "block_expansions": BLOCK}, f, indent=1)
    return done_now


def load_frames(*, directory: str, key: str) -> dict:
    """{"emb" (N, D) float16, "ref", "scene", "step", "key"} for every embedded frame, in the order seen."""
    parts = os.path.join(cache_dir(directory=directory, key=key), "parts")
    if not os.path.isdir(parts):
        raise click.UsageError(f"{directory} has no {key} screen embeddings: run the embed stage first")
    names = (["roots.npz"] if os.path.exists(os.path.join(parts, "roots.npz")) else []) + \
        sorted(f for f in os.listdir(parts) if f.startswith("block_"))
    loaded = [np.load(os.path.join(parts, n)) for n in names]
    loaded = [p for p in loaded if int(p["n"]) > 0]
    return {k: np.concatenate([p[k] for p in loaded]) for k in ("emb", "ref", "scene", "step", "key")}


# --------------------------------------------------------------------------- compare

def nearest_distance(*, query: np.ndarray, base: np.ndarray, device: str) -> np.ndarray:
    """For each query row, 1 - max cosine to any base row (unit vectors; float16 storage, float32 maths)."""
    out = np.empty(len(query), dtype=np.float32)
    b = torch.from_numpy(base.astype(np.float32)).to(device)
    for s in range(0, len(query), COMPARE_CHUNK):
        q = torch.from_numpy(query[s:s + COMPARE_CHUNK].astype(np.float32)).to(device)
        out[s:s + len(q)] = (1.0 - (q @ b.T).max(dim=1).values).clamp(min=0.0).cpu().numpy()
    return out


def distinct_screens(*, emb: np.ndarray, keys: np.ndarray, tau: float, device: str) -> np.ndarray:
    """Greedy distinct screens of frames in the order seen: a frame is a new screen when no screen kept so far is
    within tau. Exact, in chunks: a chunk's frames are first checked against the kept screens, then the rest among
    themselves in order. Returns the kept frames' indices."""
    _, first = np.unique(keys, return_index=True)     # identical frames: only the first can be new
    order = np.sort(first)
    kept_idx, kept = [], torch.zeros((0, emb.shape[1]), device=device)
    for s in range(0, len(order), DISTINCT_CHUNK):
        idx = order[s:s + DISTINCT_CHUNK]
        q = torch.from_numpy(emb[idx].astype(np.float32)).to(device)
        new = torch.ones(len(idx), dtype=torch.bool, device=device)
        if len(kept):
            new = (1.0 - (q @ kept.T).max(dim=1).values) > tau
        cand = torch.nonzero(new).flatten().tolist()
        if not cand:
            continue
        c = q[cand]
        near = ((1.0 - c @ c.T) <= tau).cpu().numpy()
        chosen = []
        for i in range(len(cand)):
            if not any(near[i, j] for j in chosen):
                chosen.append(i)
        kept = torch.cat([kept, c[chosen]])
        kept_idx += [int(idx[cand[i]]) for i in chosen]
    return np.array(kept_idx, dtype=np.int64)


def distinct_counts(*, data: dict, taus: dict, device: str) -> dict:
    """{label: {"overall": n, "by_scene": {scene: n}}} of a run's distinct screens at each tau."""
    out = {}
    for label, t in taus.items():
        row = {"overall": int(len(distinct_screens(emb=data["emb"], keys=data["key"], tau=t, device=device))),
               "by_scene": {}}
        for s in np.unique(data["scene"]):
            m = data["scene"] == s
            row["by_scene"][str(s)] = int(len(distinct_screens(emb=data["emb"][m], keys=data["key"][m], tau=t,
                                                                device=device)))
        out[label] = row
    return out


def contact_sheet(*, images: list, captions: list, path: str, cell: int = 220, cols: int = 6) -> None:
    """Images in a grid of cell x cell tiles (aspect kept), each captioned underneath."""
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 28)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (img, cap) in enumerate(zip(images, captions)):
        img = img.copy()
        img.thumbnail((cell - 6, cell - 6))
        x, y = (i % cols) * cell, (i // cols) * (cell + 28)
        sheet.paste(img, (x + 3, y + 3))
        draw.text((x + 3, y + cell), cap[:38], fill="black")
    sheet.save(path)


def _subset(*, data: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in data.items()}


def _concat(*, parts: list) -> dict:
    parts = [p for p in parts if len(p["key"])]
    if not parts:
        return {"emb": np.zeros((0, 1), np.float16), "ref": np.array([]), "scene": np.array([]),
                "step": np.array([], np.int64), "key": np.array([]), "run": np.array([])}
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def run_representatives(*, data: dict, run: str, tau: float, device: str) -> tuple:
    """A run reduced to its distinct screens at tau: (overall representatives, {scene: that scene's representatives,
    from its frames alone}), each as frame arrays with a "run" column."""
    data = {**data, "run": np.array([run] * len(data["key"]))}
    overall = _subset(data=data, idx=distinct_screens(emb=data["emb"], keys=data["key"], tau=tau, device=device))
    by_scene = {}
    for s in np.unique(data["scene"]):
        m = np.where(data["scene"] == s)[0]
        sub = _subset(data=data, idx=m)
        by_scene[str(s)] = _subset(data=sub, idx=distinct_screens(emb=sub["emb"], keys=sub["key"], tau=tau,
                                                                  device=device))
    return overall, by_scene


def group_representatives(*, reps: list, tau: float, device: str) -> dict:
    """The distinct screens of a group: a greedy pass at tau over its runs' representatives, pooled run after run
    (each in the order seen), so a screen two runs share counts once."""
    pooled = _concat(parts=reps)
    if not len(pooled["key"]):
        return pooled
    return _subset(data=pooled, idx=distinct_screens(emb=pooled["emb"], keys=pooled["key"], tau=tau, device=device))


def exact_frames(*, data: dict) -> dict:
    """One frame per distinct image (the base other groups' screens are matched against)."""
    _, first = np.unique(data["key"], return_index=True)
    return _subset(data=data, idx=np.sort(first))


def compare_groups(*, env_name: str, groups: dict, key: str, tau: float, at_steps: Optional[int], parameters: dict,
                   sheets: bool = True) -> dict:
    """Groups of runs (cusi.search.groups) by their distinct screens: each run reduced to its distinct screens, a
    group's distinct screens the greedy union of its runs', and a group's screen matched by another group when any
    frame of any of that group's runs is within tau. At tau / 2, tau and 2 tau; per scene; with sheets of each
    group's screens no other group matched (at tau), each beside its nearest frame among the other groups."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = [r for members in groups.values() for r in members]
    data = {}
    for run in runs:
        d = load_frames(directory=search_run(parameters=parameters, env=env_name, run=run), key=key)
        if at_steps is not None:
            d = _subset(data=d, idx=np.where(d["step"] <= at_steps)[0])
        data[run] = d
    taus = {"tau/2": tau / 2, "tau": tau, "2tau": 2 * tau}
    meta = {"env": env_name, "groups": groups, "embedder": key, "tau": tau, "at_steps": at_steps}
    out_dir = os.path.join(search_root(parameters=parameters, env=env_name), "_compare", DIR, hash_meta_dict(meta))
    os.makedirs(out_dir, exist_ok=True)
    write_meta(out_dir, dict(meta), parameters)
    base = {g: exact_frames(data=_concat(parts=[{**data[r], "run": np.array([r] * len(data[r]["key"]))}
                                                for r in members])) for g, members in groups.items()}
    summary = {**meta, "taus": taus, "runs_detail": {r: {"frames": int(len(d["key"])),
                                                         "distinct_frames_exact": int(len(np.unique(d["key"])))}
                                                     for r, d in data.items()}, "by_tau": {}}
    title = f"# Screen coverage: {env_name}, {key}, tau = {tau:g}" + (f", first {at_steps} env steps of each run"
                                                                      if at_steps else "")
    lines = [title, "", "Distinct screens (greedy at the threshold): each run's own, its group's (the union over the "
             "group's runs, a screen two runs share counted once), and the group's screens that no frame of another "
             "group comes within the threshold of.", ""]
    sheet_reps = {}
    for label, t in taus.items():
        rep = {r: run_representatives(data=data[r], run=r, tau=t, device=device) for r in runs}
        gstats = {}
        group_reps = {}
        for g, members in groups.items():
            group_reps[g] = group_representatives(reps=[rep[r][0] for r in members], tau=t, device=device)
            mean, sd = mean_sd(values=[len(rep[r][0]["key"]) for r in members])
            gstats[g] = {"runs": {r: int(len(rep[r][0]["key"])) for r in members}, "run_mean": mean, "run_sd": sd,
                         "distinct": int(len(group_reps[g]["key"]))}
        # each group's screens' distance to the nearest frame of each other group
        near = {g: {h: nearest_distance(query=group_reps[g]["emb"], base=base[h]["emb"], device=device)
                    if len(group_reps[g]["key"]) and len(base[h]["key"]) else np.full(len(group_reps[g]["key"]), 2.0)
                    for h in groups if h != g} for g in groups}
        scenes = sorted({s for r in runs for s in rep[r][1]})
        by_scene = {s: {} for s in scenes}
        for g, members in groups.items():
            others = [h for h in groups if h != g]
            none_of = np.min(np.stack([near[g][h] for h in others]), axis=0) if others else \
                np.full(len(group_reps[g]["key"]), 2.0)
            all_of = np.max(np.stack([near[g][h] for h in others]), axis=0) if others else none_of
            gstats[g]["unique"] = int((none_of > t).sum())
            if len(groups) > 2:
                gstats[g]["matched_by_all"] = int((all_of <= t).sum())
            for h in others:
                gstats[g][f"matched_by_{h}"] = int((near[g][h] <= t).sum())
            for s in scenes:
                scene_reps = group_representatives(reps=[rep[r][1][s] for r in members if s in rep[r][1]], tau=t,
                                                   device=device)
                in_scene = group_reps[g]["scene"] == s
                by_scene[s][g] = {"distinct": int(len(scene_reps["key"])),
                                  "unique": int((none_of[in_scene] > t).sum())}
            if label == "tau":
                sheet_reps[g] = (group_reps[g], none_of)
        summary["by_tau"][label] = {"tau": t, "groups": gstats, "by_scene": by_scene}
        lines += [f"## {label} = {t:.3g}", "",
                  "| group | runs | each run's distinct screens | mean ± sd | group distinct | not matched by other "
                  "group(s) |" + (" matched by all |" if len(groups) > 2 else ""),
                  "|---|---|---|---|---|---|" + ("---|" if len(groups) > 2 else "")]
        for g, s in gstats.items():
            lines.append(f"| {g} | {len(groups[g])} | {', '.join(str(v) for v in s['runs'].values())} | "
                         f"{fmt_mean_sd(m=s['run_mean'], sd=s['run_sd'])} | {s['distinct']} | {s['unique']} |"
                         + (f" {s['matched_by_all']} |" if len(groups) > 2 else ""))
        if len(groups) == 2:
            a, b = list(groups)
            lines += ["", f"{a} screens matched by {b}: {gstats[a]['matched_by_' + b]}; {b} screens matched by {a}: "
                          f"{gstats[b]['matched_by_' + a]}."]
        if label == "tau":
            lines += ["", "| scene | " + " | ".join(f"{g} distinct | {g} unmatched" for g in groups) + " |",
                      "|---|" + "---|---|" * len(groups)]
            for s in scenes:
                lines.append(f"| {s} | " + " | ".join(f"{by_scene[s][g]['distinct']} | {by_scene[s][g]['unique']}"
                                                     if g in by_scene[s] else "0 | 0" for g in groups) + " |")
        lines.append("")
    if sheets:
        readers = {r: FrameReader(directory=search_run(parameters=parameters, env=env_name, run=r)) for r in runs}
        for g, (reps_g, none_of) in sheet_reps.items():
            idx = np.where(none_of > tau)[0]
            pick = sorted(random.Random(0).sample(list(idx), min(SHEET_SAMPLES, len(idx))))
            others = [h for h in groups if h != g]
            if not pick or not others:
                continue
            other = _concat(parts=[base[h] for h in others])
            q = torch.from_numpy(reps_g["emb"][pick].astype(np.float32))
            j = (q @ torch.from_numpy(other["emb"].astype(np.float32)).T).max(dim=1).indices.tolist()
            images, captions = [], []
            for i, jj in zip(pick, j):
                images += [readers[str(reps_g["run"][i])].image(str(reps_g["ref"][i])),
                           readers[str(other["run"][jj])].image(str(other["ref"][jj]))]
                captions += [f"{g} {reps_g['scene'][i]}", f"nearest d={none_of[i]:.3f} {str(other['run'][jj])[-14:]}"]
            contact_sheet(images=images, captions=captions, path=os.path.join(out_dir, f"unmatched_{g}.png"))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log_info(f"screen_coverage compare -> {out_dir}", parameters=parameters)
    return summary


def config_tau(*, parameters: dict, env_name: str, key: str) -> float:
    table = (parameters.get("screen_coverage_tau") or {}).get(env_name) or {}
    if key not in table:
        raise click.UsageError(f"no tau for {env_name} / {key} in configs/screen_coverage.yaml: pass --tau or "
                               f"calibrate first")
    return float(table[key])


# --------------------------------------------------------------------------- calibrate

BANDS = (0.0, 0.005, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 1.01)
PER_BAND = 3


def calibration_dir(*, parameters: dict, env_name: str) -> str:
    return os.path.join(parameters["storage_dir"], "tmp", "screen_calibration", env_name)


def calibrate_sample(*, env_name: str, runs: list, keys: list, parameters: dict, anchors: int = 1500,
                     per_anchor: int = 40) -> str:
    """Same-scene frame pairs spread over distance bands of each embedder (PER_BAND per band per embedder), every
    pair with every embedder's distance, drawn as numbered side-by-side sheets of 8 pairs."""
    rng = random.Random(0)
    pool = []    # (run, i, j) within a run and scene
    data = {run: {k: load_frames(directory=search_run(parameters=parameters, env=env_name, run=run), key=k)
                  for k in keys} for run in runs}
    for run in runs:
        d = data[run][keys[0]]
        _, first = np.unique(d["key"], return_index=True)        # distinct frames only
        by_scene = {}
        for i in first:
            by_scene.setdefault(str(d["scene"][i]), []).append(int(i))
        scenes = [s for s, v in by_scene.items() if len(v) > 1]
        for _ in range(anchors):
            s = rng.choice(scenes)
            i = rng.choice(by_scene[s])
            for j in rng.sample(by_scene[s], min(per_anchor, len(by_scene[s]))):
                if j != i:
                    pool.append((run, i, j))
    dist = {k: np.array([1.0 - float(data[r][k]["emb"][i].astype(np.float32) @ data[r][k]["emb"][j].astype(np.float32))
                         for r, i, j in pool]) for k in keys}
    chosen = []
    for k in keys:
        for lo, hi in zip(BANDS[:-1], BANDS[1:]):
            idx = [n for n in np.where((dist[k] >= lo) & (dist[k] < hi))[0] if n not in chosen]
            chosen += rng.sample(idx, min(PER_BAND, len(idx)))
    rng.shuffle(chosen)
    out = calibration_dir(parameters=parameters, env_name=env_name)
    os.makedirs(out, exist_ok=True)
    pairs = []
    for n, c in enumerate(chosen):
        r, i, j = pool[c]
        d = data[r][keys[0]]
        pairs.append({"pair": n, "run": r, "scene": str(d["scene"][i]), "a": str(d["ref"][i]), "b": str(d["ref"][j]),
                      "distance": {k: round(float(dist[k][c]), 4) for k in keys}})
    with open(os.path.join(out, "pairs.jsonl"), "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    readers = {r: FrameReader(directory=search_run(parameters=parameters, env=env_name, run=r)) for r in runs}
    cell = {"gameboy": 320, "web": 420, "android": 260}[env_name]
    for s in range(0, len(pairs), 8):
        group = pairs[s:s + 8]
        imgs = []
        for p in group:
            a, b = readers[p["run"]].image(p["a"]), readers[p["run"]].image(p["b"])
            for im in (a, b):
                im.thumbnail((cell, cell * 2 if env_name == "android" else cell))
            h = max(a.height, b.height)
            pair = Image.new("RGB", (a.width + b.width + 12, h + 22), "white")
            pair.paste(a, (0, 22))
            pair.paste(b, (a.width + 12, 22))
            ImageDraw.Draw(pair).text((2, 4), f"#{p['pair']}", fill="red")
            imgs.append(pair)
        w = max(i.width for i in imgs) * 2 + 20
        rows = [imgs[k:k + 2] for k in range(0, len(imgs), 2)]
        sheet = Image.new("RGB", (w, sum(max(i.height for i in r) + 10 for r in rows)), (200, 200, 200))
        y = 0
        for r in rows:
            x = 0
            for im in r:
                sheet.paste(im, (x, y))
                x += im.width + 20
            y += max(i.height for i in r) + 10
        sheet.save(os.path.join(out, f"sheet_{s // 8:02d}.png"))
    log_info(f"screen_coverage calibrate: {len(pairs)} pairs -> {out} (label them in labels.jsonl: "
             f'{{"pair": n, "same": true|false}})', parameters=parameters)
    return out


def calibrate_fit(*, env_name: str, parameters: dict) -> dict:
    """Per embedder, the tau (a labelled pair's distance) with the fewest errors when 'same' means distance <= tau;
    ties go to the midpoint of the tied range."""
    out = calibration_dir(parameters=parameters, env_name=env_name)
    pairs = {p["pair"]: p for p in read_rows(os.path.join(out, "pairs.jsonl"))}
    labels = {r["pair"]: bool(r["same"]) for r in read_rows(os.path.join(out, "labels.jsonl"))}
    fits = {}
    for k in next(iter(pairs.values()))["distance"]:
        pts = sorted((pairs[n]["distance"][k], same) for n, same in labels.items())
        cands = [0.0] + [d for d, _ in pts]
        errs = [sum((d <= t) != same for d, same in pts) for t in cands]
        best = min(errs)
        tied = [t for t, e in zip(cands, errs) if e == best]
        # tau between the last tied candidate and the next labelled distance above it
        hi = next((d for d, _ in pts if d > tied[-1]), tied[-1])
        tau = (tied[0] + hi) / 2 if tied[0] == tied[-1] else (tied[0] + tied[-1]) / 2
        fits[k] = {"tau": round(tau, 4), "errors": best, "labelled": len(pts),
                   "same": sum(s for _, s in pts), "error_pairs": [n for n in labels if
                                                                  (pairs[n]["distance"][k] <= tau) != labels[n]]}
    with open(os.path.join(out, "fit.json"), "w") as f:
        json.dump(fits, f, indent=1)
    log_info(f"screen_coverage calibrate fit {env_name}: {fits}", parameters=parameters)
    return fits


# --------------------------------------------------------------------------- CLI

def _key_opts(f):
    f = click.option("--encoder_model", default=None,
                     help=f"SigLIP model for --embedder siglip (default {DEFAULT_SIGLIP}).")(f)
    return click.option("--embedder", default="siglip", show_default=True, type=click.Choice(EMBEDDERS))(f)


@click.group()
def main():
    pass


@main.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--run", "runs", required=True, multiple=True, help="Run(s) under storage/search/<env>/.")
@_key_opts
def embed(env_name, runs, embedder, encoder_model):
    parameters = load_parameters()
    model = build_embedder(env_name=env_name, embedder=embedder, encoder_model=encoder_model)
    key = embedder_key(embedder=embedder, encoder_model=encoder_model)
    for run in runs:
        n = embed_run(env_name=env_name, directory=search_run(parameters=parameters, env=env_name, run=run),
                      embedder=model, key=key, parameters=parameters)
        log_info(f"screen_coverage embed {run} {key}: {n} frames embedded now", parameters=parameters)


@main.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--runs", default=None, help="Comma-separated runs of this env (2 or more), each its own group.")
@click.option("--groups", default=None, help='Groups of runs: "name=run1,run2;name=run3,run4" (2+ groups).')
@_key_opts
@click.option("--tau", default=None, type=float, help="Default: configs/screen_coverage.yaml for this env/embedder.")
@click.option("--at_steps", default=None, type=int, help="Only each run's first N env steps.")
def compare(env_name, runs, groups, embedder, encoder_model, tau, at_steps):
    parameters = load_parameters()
    groups = parse_groups(groups=groups, runs=runs)
    if len(groups) < 2:
        raise click.UsageError("compare needs two or more runs or groups (one run alone: the distinct command)")
    for run in (r for members in groups.values() for r in members):
        if not os.path.isdir(search_run(parameters=parameters, env=env_name, run=run)):
            raise click.UsageError(f"{run} is not a {env_name} run (runs compared must all be of --env {env_name})")
    key = embedder_key(embedder=embedder, encoder_model=encoder_model)
    tau = tau if tau is not None else config_tau(parameters=parameters, env_name=env_name, key=key)
    summary = compare_groups(env_name=env_name, groups=groups, key=key, tau=tau, at_steps=at_steps,
                             parameters=parameters)
    for g, s in summary["by_tau"]["tau"]["groups"].items():
        print(f"{g}: runs {s['runs']}, group distinct screens {s['distinct']}, not matched by other group(s) "
              f"{s['unique']} (tau = {tau:g})")


@main.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--run", "runs", required=True, multiple=True, help="Run(s) under storage/search/<env>/.")
@_key_opts
@click.option("--tau", default=None, type=float, help="Default: configs/screen_coverage.yaml for this env/embedder.")
@click.option("--at_steps", default=None, type=int, help="Only the run's first N env steps.")
def distinct(env_name, runs, embedder, encoder_model, tau, at_steps):
    """Each run's distinct screens at tau / 2, tau and 2 tau (overall and per scene)."""
    parameters = load_parameters()
    key = embedder_key(embedder=embedder, encoder_model=encoder_model)
    tau = tau if tau is not None else config_tau(parameters=parameters, env_name=env_name, key=key)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    taus = {"tau/2": tau / 2, "tau": tau, "2tau": 2 * tau}
    for run in runs:
        directory = search_run(parameters=parameters, env=env_name, run=run)
        data = load_frames(directory=directory, key=key)
        if at_steps is not None:
            keep = data["step"] <= at_steps
            data = {k: v[keep] for k, v in data.items()}
        counts = distinct_counts(data=data, taus=taus, device=device)
        out = {"run": run, "embedder": key, "taus": taus, "at_steps": at_steps, "frames": int(len(data["key"])),
               "distinct_screens": counts}
        name = f"distinct_{tau:g}" + (f"_at{at_steps}" if at_steps is not None else "") + ".json"
        with open(os.path.join(cache_dir(directory=directory, key=key), name), "w") as f:
            json.dump(out, f, indent=1)
        print(f"{run}: {out['frames']} frames; distinct screens " +
              ", ".join(f"{counts[label]['overall']} at {label} = {t:.3g}" for label, t in taus.items()))


@main.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--stage", required=True, type=click.Choice(["sample", "fit"]))
@click.option("--runs", default=None, help="sample: comma-separated runs to draw pairs from.")
@click.option("--keys", default=f"random_patch,{embedder_key(embedder='siglip', encoder_model=None)}",
              help="sample: embedder keys whose distances the pairs carry (each must be embedded).")
def calibrate(env_name, stage, runs, keys):
    parameters = load_parameters()
    if stage == "sample":
        if not runs:
            raise click.UsageError("--stage sample needs --runs")
        calibrate_sample(env_name=env_name, runs=[r.strip() for r in runs.split(",")],
                         keys=[k.strip() for k in keys.split(",")], parameters=parameters)
    else:
        for k, v in calibrate_fit(env_name=env_name, parameters=parameters).items():
            print(f"{env_name} {k}: tau {v['tau']} ({v['errors']} errors of {v['labelled']}, {v['same']} same)")


if __name__ == "__main__":
    main()
