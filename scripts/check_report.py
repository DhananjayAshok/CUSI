"""
Figures and numbers for check_report.md, drawn only from finished artifacts on disk (search trees, the fine-tuned
SigLIP embedders and the world models). Nothing is trained here.
    python scripts/check_report.py trees  --out <dir>     # CPU: tree plots, galleries, GIFs, novelty stats
    python scripts/check_report.py models --out <dir>     # GPU: embedder reconstructions, world-model predictions
Each command writes <out>/<env>/... images and <out>/<command>_stats.json.
"""
import glob
import json
import os
import sys
from typing import Optional
import click
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cusi.utils.parameter_handling import load_parameters
from cusi.agents.frames import FrameRef

ENVS = ("gameboy", "web", "android")
EXPANDERS = ("random", "vlm", "curiosity")
# thumbnail size (width, height) per env, roughly the screen's aspect ratio
THUMB = {"gameboy": (240, 216), "web": (560, 315), "android": (150, 333)}
COLS = {"gameboy": 8, "web": 4, "android": 8}
COLORS = {"random": "tab:blue", "vlm": "tab:orange", "curiosity": "tab:green"}


def _read_jsonl(*, path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _frame(*, run_dir: str, rel: str, env: str) -> Image.Image:
    return FrameRef(path=os.path.join(run_dir, rel)).pil().convert("RGB").resize(THUMB[env])


def _caption(*, img: Image.Image, lines: list, height: int = 14) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + height * len(lines)), "white")
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    for i, line in enumerate(lines):
        d.text((2, img.height + i * height), line[: img.width // 6 + 4], fill="black")
    return out


def _grid(*, tiles: list, cols: int) -> Image.Image:
    w = max(t.width for t in tiles)
    h = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * (w + 4), rows * (h + 4)), "white")
    for i, t in enumerate(tiles):
        out.paste(t, ((i % cols) * (w + 4), (i // cols) * (h + 4)))
    return out


def _short(action) -> str:
    s = str(action).replace("\n", " ")
    if "Action:" in s:
        s = s.split("Action:", 1)[1].strip()
    try:      # Android actions are JSON: keep the type and its main argument
        a = json.loads(s.strip("`").removeprefix("json").strip())
        arg = a.get("index", a.get("text", a.get("app_name", a.get("direction", ""))))
        return f"{a.get('action_type')} {arg}".strip()
    except (ValueError, AttributeError):
        return s


class Tree:
    def __init__(self, *, run_dir: str, env: str):
        self.run_dir, self.env = run_dir, env
        self.nodes = {n["id"]: n for n in _read_jsonl(path=os.path.join(run_dir, "nodes.jsonl")) if "parent" in n}
        self.expansions = _read_jsonl(path=os.path.join(run_dir, "expansions.jsonl"))
        self.summary = json.load(open(os.path.join(run_dir, "summary.json"))) \
            if os.path.exists(os.path.join(run_dir, "summary.json")) else {}
        self.children = {i: [] for i in self.nodes}
        for n in self.nodes.values():
            if n["parent"] is not None:
                self.children[n["parent"]].append(n["id"])
        self.n_expanded = {i: 0 for i in self.nodes}
        for e in self.expansions:
            self.n_expanded[e["node_id"]] += 1

    def path_to(self, *, node_id: int) -> list:
        out = [node_id]
        while self.nodes[out[-1]]["parent"] is not None:
            out.append(self.nodes[out[-1]]["parent"])
        return out[::-1]

    def segment_actions(self, *, node_id: int) -> list:
        seg = self.nodes[node_id]["segment"]
        if seg is None:
            return []
        exp = self.expansions[seg[0]]
        return [_short(s["action"]) for s in exp["steps"][seg[1]: seg[2] + 1]]

    def stats(self) -> dict:
        values = np.array([n["value"] for n in self.nodes.values() if n["value"] is not None])
        yields = np.array([e["yield"] for e in self.expansions])
        steps = [s for e in self.expansions for s in e["steps"]]
        step_values = np.array([s["value"] for s in steps])
        valid = np.array([bool(s["info"].get("valid", True)) for s in steps])
        changed = np.array([bool(s["extra"].get("changed", True)) for s in steps])
        actions = {}
        for s in steps:
            a = _short(s["action"])
            if self.env == "android":
                a = a.split(" ")[0]
            elif self.env == "web":
                a = a.split(" ")[0].split("[")[0]
            actions[a] = actions.get(a, 0) + 1
        flags = {}
        for n in self.nodes.values():
            for k, v in (n.get("flags") or {}).items():
                if v:
                    flags[k] = flags.get(k, 0) + 1
        return {"nodes": len(self.nodes), "roots": sum(n["parent"] is None for n in self.nodes.values()),
                "expansions": len(self.expansions), "steps": len(steps), "archive": self.summary.get("archive"),
                "max_depth": max(n["depth"] for n in self.nodes.values()),
                "node_value_mean": float(values.mean()), "node_value_median": float(np.median(values)),
                "node_value_max": float(values.max()),
                "yield_mean": float(yields.mean()), "yield_median": float(np.median(yields)),
                "yield_zero_frac": float((yields < 0.01).mean()),
                "step_value_mean": float(step_values.mean()), "step_zero_frac": float((step_values < 1e-3).mean()),
                "valid_rate": float(valid.mean()), "changed_rate": float(changed.mean()),
                "actions": dict(sorted(actions.items(), key=lambda kv: -kv[1])[:12]), "flags": flags,
                "stopped": self.summary.get("stopped"), "nodes_expanded_more_than_once":
                    int(sum(v > 1 for v in self.n_expanded.values())),
                "max_times_expanded": int(max(self.n_expanded.values()))}


def plot_tree(*, tree: Tree, path: str, title: str) -> None:
    """Tidy layout: leaves left-to-right in DFS order, parents centred over children; y = depth."""
    x, next_leaf = {}, [0]

    def place(nid):
        kids = sorted(tree.children[nid])
        if not kids:
            x[nid] = next_leaf[0]
            next_leaf[0] += 1
        else:
            for k in kids:
                place(k)
            x[nid] = float(np.mean([x[k] for k in kids]))
    roots = sorted(i for i, n in tree.nodes.items() if n["parent"] is None)
    for r in roots:
        place(r)
        next_leaf[0] += 1
    fig, ax = plt.subplots(figsize=(min(40, 4 + next_leaf[0] * 0.12), 1.5 + 0.35 * tree.stats()["max_depth"]))
    for nid, n in tree.nodes.items():
        if n["parent"] is not None:
            p = n["parent"]
            ax.plot([x[p], x[nid]], [-tree.nodes[p]["depth"], -n["depth"]], color="0.75", lw=0.6, zorder=1)
    ids = list(tree.nodes)
    vals = [tree.nodes[i]["value"] if tree.nodes[i]["value"] is not None else np.nan for i in ids]
    sizes = [12 + 25 * tree.n_expanded[i] for i in ids]
    edge = ["black" if tree.n_expanded[i] else "none" for i in ids]
    sc = ax.scatter([x[i] for i in ids], [-tree.nodes[i]["depth"] for i in ids], c=vals, s=sizes, cmap="viridis",
                    edgecolors=edge, linewidths=0.8, zorder=2, plotnonfinite=True)
    for r in roots:
        ax.annotate(tree.nodes[r]["env_key"], (x[r], 0.25), ha="center", fontsize=7, rotation=30)
    ax.set_ylabel("depth (negated)")
    ax.set_xticks([])
    ax.set_title(f"{title}: colour = node novelty, outline/size = times expanded", fontsize=9)
    fig.colorbar(sc, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def gallery(*, tree: Tree, ids: list, path: str, cols: int) -> None:
    tiles = []
    for i in ids:
        n = tree.nodes[i]
        acts = tree.segment_actions(node_id=i)
        v = "root" if n["value"] is None else f"{n['value']:.3f}"
        tiles.append(_caption(img=_frame(run_dir=tree.run_dir, rel=n["frame"], env=tree.env),
                              lines=[f"#{i} d{n['depth']} {n['env_key']} nov={v}",
                                     ("> " + " | ".join(acts[-3:])) if acts else ""]))
    _grid(tiles=tiles, cols=cols).save(path)


def lineage(*, tree: Tree, node_id: int, path: str) -> None:
    """Frames from the root to node_id, each captioned with the actions of the segment that reached it."""
    ids = tree.path_to(node_id=node_id)
    if len(ids) > 12:
        ids = ids[:3] + ids[-9:]
    tiles = []
    for i in ids:
        n = tree.nodes[i]
        acts = tree.segment_actions(node_id=i)
        v = "root" if n["value"] is None else f"{n['value']:.3f}"
        tiles.append(_caption(img=_frame(run_dir=tree.run_dir, rel=n["frame"], env=tree.env),
                              lines=[f"#{i} d{n['depth']} nov={v}", " | ".join(acts[:4]), " | ".join(acts[4:8])]))
    _grid(tiles=tiles, cols=min(6, len(tiles))).save(path)


def expansion_gif(*, tree: Tree, exp: dict, path: str) -> dict:
    start = tree.nodes[exp["node_id"]]
    items = [(start["frame"], "start (selected node)", None, True)]
    items += [(s["frame"], _short(s["action"]), s["value"], bool(s["info"].get("valid", True))) for s in exp["steps"]]
    frames = []
    for t, (rel, act, val, ok) in enumerate(items):
        img = _frame(run_dir=tree.run_dir, rel=rel, env=tree.env)
        head = f"exp {exp['id']} step {t}/{len(items) - 1}" + ("" if val is None else f" nov={val:.3f}") + \
            ("" if ok else " INVALID")
        frames.append(_caption(img=img, lines=[head, act[:60], act[60:120]]))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=900, loop=0)
    return {"expansion": exp["id"], "node": exp["node_id"], "yield": exp["yield"],
            "actions": [it[1] for it in items[1:]]}


def curiosity_returns(*, tree: Tree, path: str) -> Optional[dict]:
    files = sorted(glob.glob(os.path.join(tree.run_dir, "curiosity", "*_returns.npy")),
                   key=lambda p: int(os.path.basename(p).split("_")[0]))
    if not files:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    out = {}
    for p in files:
        r = np.load(p)
        k = int(os.path.basename(p).split("_")[0])
        w = max(1, len(r) // 50)
        smooth = np.convolve(r, np.ones(w) / w, mode="valid") if len(r) > w else r
        ax.plot(smooth, lw=0.8, alpha=0.7)
        out[k] = {"episodes": int(len(r)), "first_mean": float(r[: max(1, len(r) // 10)].mean()),
                  "last_mean": float(r[-max(1, len(r) // 10):].mean()), "max": float(r.max())}
    ax.set_xlabel("PPO episode within the session")
    ax.set_ylabel("episode novelty return (smoothed)")
    ax.set_title(f"{tree.env} curiosity: PPO return per session ({len(files)} sessions)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return out


@click.group()
def main():
    pass


@main.command()
@click.option("--out", required=True)
@click.option("--run_suffix", default="_v1")
def trees(out, run_suffix):
    parameters = load_parameters()
    stats = {}
    for env in ENVS:
        os.makedirs(os.path.join(out, env), exist_ok=True)
        loaded = {}
        for ex in EXPANDERS:
            run_dir = os.path.join(parameters["storage_dir"], "search", env, f"tree_{ex}{run_suffix}")
            if not os.path.exists(os.path.join(run_dir, "expansions.jsonl")):
                continue
            tree = Tree(run_dir=run_dir, env=env)
            loaded[ex] = tree
            s = tree.stats()
            pre = os.path.join(out, env, ex)
            plot_tree(tree=tree, path=f"{pre}_tree.png", title=f"{env} / {ex}")
            non_root = [i for i, n in tree.nodes.items() if n["value"] is not None]
            by_val = sorted(non_root, key=lambda i: -tree.nodes[i]["value"])
            gallery(tree=tree, ids=by_val[:16], path=f"{pre}_top_nodes.png", cols=COLS[env])
            gallery(tree=tree, ids=by_val[-8:], path=f"{pre}_bottom_nodes.png", cols=COLS[env])
            rng = np.random.default_rng(0)
            sample = sorted(rng.choice(non_root, size=min(32 if env != "web" else 16, len(non_root)), replace=False).tolist())
            gallery(tree=tree, ids=sample, path=f"{pre}_random_nodes.png", cols=COLS[env])
            deepest = max(non_root, key=lambda i: (tree.nodes[i]["depth"], tree.nodes[i]["value"]))
            lineage(tree=tree, node_id=by_val[0], path=f"{pre}_lineage_top.png")
            lineage(tree=tree, node_id=deepest, path=f"{pre}_lineage_deepest.png")
            s["top_node"], s["deepest_node"] = by_val[0], deepest
            order = sorted(tree.expansions, key=lambda e: -e["yield"])
            s["gifs"] = {"best": expansion_gif(tree=tree, exp=order[0], path=f"{pre}_exp_best.gif"),
                         "median": expansion_gif(tree=tree, exp=order[len(order) // 2], path=f"{pre}_exp_median.gif"),
                         "worst": expansion_gif(tree=tree, exp=order[-1], path=f"{pre}_exp_worst.gif")}
            if ex == "curiosity":
                s["ppo_returns"] = curiosity_returns(tree=tree, path=f"{pre}_ppo_returns.png")
            s["run_dir"] = run_dir
            stats[f"{env}/{ex}"] = s
            print(f"{env}/{ex}: {json.dumps({k: v for k, v in s.items() if k not in ('gifs', 'ppo_returns')})}",
                  flush=True)
        if not loaded:
            continue
        # novelty distributions and yield over the run, the three expanders on one plot
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ex, tree in loaded.items():
            sv = [s["value"] for e in tree.expansions for s in e["steps"]]
            nv = [n["value"] for n in tree.nodes.values() if n["value"] is not None]
            bins = np.linspace(0, max(1e-6, np.percentile(sv, 99.5)), 40)
            axes[0].hist(np.clip(sv, 0, bins[-1]), bins=bins, histtype="step", color=COLORS[ex], label=ex, density=True)
            axes[1].hist(np.clip(nv, 0, bins[-1]), bins=bins, histtype="step", color=COLORS[ex], label=ex, density=True)
            y = [e["yield"] for e in tree.expansions]
            axes[2].plot(range(len(y)), y, ".", color=COLORS[ex], alpha=0.5)
            w = 5
            if len(y) > w:
                axes[2].plot(range(w - 1, len(y)), np.convolve(y, np.ones(w) / w, mode="valid"), color=COLORS[ex],
                             label=ex)
        axes[0].set_title("per-step novelty (all steps)")
        axes[1].set_title("node novelty at creation")
        axes[2].set_title("expansion yield vs expansion index (line: 5-mean)")
        axes[2].set_xlabel("expansion")
        for a in axes:
            a.legend()
        fig.suptitle(env)
        fig.tight_layout()
        fig.savefig(os.path.join(out, env, "novelty_compare.png"), dpi=110)
        plt.close(fig)
    with open(os.path.join(out, "trees_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)

def _upscale(*, x: np.ndarray, env: str) -> Image.Image:
    """A canvas (H x W x 1/3 uint8) as an RGB thumbnail, nearest-neighbour so pixel errors stay visible."""
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    scale = {"gameboy": 2, "web": 2, "android": 2}[env]
    return Image.fromarray(x).resize((x.shape[1] * scale, x.shape[0] * scale), Image.NEAREST)


def _rgb(x: np.ndarray) -> np.ndarray:
    return np.repeat(x, 3, axis=-1) if x.shape[-1] == 1 else x


def embedder_figures(*, env: str, emb_dir: str, out: str, seed: int, device: str) -> dict:
    """Loss curves, reconstructions of the training run's own held-out frames, nearest neighbours."""
    from cusi.explore.decoder import EmbeddingDecoder
    from cusi.explore.train_embedder import _split, load_canvases
    from cusi.state.encoders import build_image_embedder
    record = json.load(open(os.path.join(emb_dir, "train_summary.json")))
    summary, history = record["summary"], record["history"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    steps = [h["step"] for h in history]
    tl = np.array([h["train_loss"] for h in history])
    w = 50
    axes[0].plot(steps, tl, color="0.8", lw=0.5, label="train (per step)")
    axes[0].plot(steps[w - 1:], np.convolve(tl, np.ones(w) / w, mode="valid"), label=f"train ({w}-mean)")
    vh = [h for h in history if "val_loss" in h]
    axes[0].plot([h["step"] for h in vh], [h["val_loss"] for h in vh], label="val (16 frames)", lw=0.8)
    axes[0].set_yscale("log")
    axes[0].set_title(f"{env} SigLIP fine-tune: L1+L2 reconstruction loss")
    axes[0].set_xlabel("step")
    axes[0].legend()
    axes[1].plot([h["step"] for h in vh], [h["val_pixel_mae"] for h in vh], lw=0.8)
    axes[1].set_yscale("log")
    axes[1].set_title("val pixel MAE (0-255)")
    axes[1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "embedder_loss.png"), dpi=110)
    plt.close(fig)

    # the frames and split the fine-tune used (train_siglip: load_canvases(max_frames=5000) then _split(seed))
    canv = load_canvases(replay_dirs=summary["replay_dirs"], env_name=env, max_frames=5000)
    train_idx, val_idx = _split(n=len(canv), val_frac=0.05, seed=seed)
    sources = []
    from cusi.explore.replay import iter_replay
    for d in summary["replay_dirs"]:
        sources += [os.path.basename(d.rstrip("/"))] * sum(1 for _ in iter_replay(directory=d, load_frames=False))
    sources = np.array(sources[: len(canv)])
    emb = build_image_embedder(image_embedder="siglip", env_name=env, encoder_model=summary["base_model"],
                               embedder_load_path=emb_dir, device=device)
    base = build_image_embedder(image_embedder="siglip", env_name=env, encoder_model=summary["base_model"],
                                device=device)
    dec = EmbeddingDecoder.load(directory=emb_dir, device=device)
    z = emb.embed(frames=list(canv))
    z0 = base.embed(frames=list(canv))
    recon = np.concatenate([dec.decode(z[i:i + 64]) for i in range(0, len(z), 64)])
    target = np.stack([_rgb(c) for c in canv])
    mae = np.abs(recon.astype(np.float32) - target.astype(np.float32)).mean(axis=(1, 2, 3))
    # exact duplicates: step 0 of an expansion is the expanded node's frame, which is also another expansion's frame
    flat = canv.reshape(len(canv), -1)
    keys = {}
    for i in train_idx:
        keys.setdefault(flat[i].tobytes(), i)
    val_dup = np.array([flat[i].tobytes() in keys for i in val_idx])
    stats = {"n_frames": int(len(canv)), "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
             "val_exact_duplicate_of_train": int(val_dup.sum()),
             "mae_train": float(mae[train_idx].mean()), "mae_val": float(mae[val_idx].mean()),
             "mae_val_unique": float(mae[val_idx][~val_dup].mean()) if (~val_dup).any() else None,
             "mae_by_source": {s: float(mae[sources == s].mean()) for s in np.unique(sources)},
             "final": summary["final"], "steps": summary["steps"], "base_model": summary["base_model"]}

    order = val_idx[np.argsort(mae[val_idx])]
    picks = {"best": order[:4], "median": order[len(order) // 2 - 2: len(order) // 2 + 2], "worst": order[-4:]}
    for name, ids in picks.items():
        tiles = []
        for i in ids:
            pair = np.concatenate([_rgb(canv[i]), np.full((canv.shape[1], 4, 3), 255, np.uint8), recon[i]], axis=1)
            tiles.append(_caption(img=_upscale(x=pair, env=env), lines=[
                f"val #{i} ({sources[i]}) MAE {mae[i]:.1f}{' dup-of-train' if val_dup[list(val_idx).index(i)] else ''}",
                "left: input canvas | right: decoded"]))
        _grid(tiles=tiles, cols=2 if env == "web" else 4).save(os.path.join(out, f"embedder_recon_val_{name}.png"))
    rng = np.random.default_rng(0)
    tiles = []
    for i in rng.choice(train_idx, size=4, replace=False):
        pair = np.concatenate([_rgb(canv[i]), np.full((canv.shape[1], 4, 3), 255, np.uint8), recon[i]], axis=1)
        tiles.append(_caption(img=_upscale(x=pair, env=env), lines=[f"train #{i} ({sources[i]}) MAE {mae[i]:.1f}"]))
    _grid(tiles=tiles, cols=2 if env == "web" else 4).save(os.path.join(out, "embedder_recon_train.png"))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(mae[train_idx], bins=40, histtype="step", density=True, label="train")
    ax.hist(mae[val_idx], bins=40, histtype="step", density=True, label="val")
    ax.set_xlabel("per-frame pixel MAE (0-255)")
    ax.set_title(f"{env}: reconstruction error per frame")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "embedder_mae_hist.png"), dpi=110)
    plt.close(fig)

    # nearest training frames of held-out queries, fine-tuned vs base SigLIP
    uniq_val = val_idx[~val_dup]
    queries = rng.choice(uniq_val if len(uniq_val) >= 4 else val_idx, size=4, replace=False)
    rows = []
    for q in queries:
        row = [_caption(img=_upscale(x=_rgb(canv[q]), env=env), lines=["query (val)", sources[q]])]
        for label, zz in (("ft", z), ("base", z0)):
            sims = (zz[train_idx] @ zz[q]).numpy()
            for j in np.argsort(-sims)[:3]:
                t = train_idx[j]
                row.append(_caption(img=_upscale(x=_rgb(canv[t]), env=env),
                                    lines=[f"{label} NN cos {sims[j]:.3f}", sources[t]]))
        rows.append(_grid(tiles=row, cols=len(row)))
    _grid(tiles=rows, cols=1).save(os.path.join(out, "embedder_neighbours.png"))
    # how spread the embedding space is: cosine of random pairs vs consecutive frames
    pair_a, pair_b = rng.integers(0, len(canv), 2000), rng.integers(0, len(canv), 2000)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for label, zz in (("fine-tuned", z), ("base", z0)):
        cos_rand = (zz[pair_a] * zz[pair_b]).sum(-1).numpy()
        cos_next = (zz[:-1] * zz[1:]).sum(-1).numpy()
        ax.hist(cos_rand, bins=50, histtype="step", density=True, label=f"{label}: random pairs")
        ax.hist(cos_next, bins=50, histtype="step", density=True, label=f"{label}: consecutive frames", ls="--")
        stats[f"cos_random_pairs_{label}"] = float(cos_rand.mean())
        stats[f"cos_consecutive_{label}"] = float(cos_next.mean())
    ax.set_xlabel("cosine similarity")
    ax.set_title(f"{env}: embedding spread")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "embedder_cos_spread.png"), dpi=110)
    plt.close(fig)
    return stats


def world_model_figures(*, env: str, wm_dir: str, out: str, tag: str, device: str) -> dict:
    """Loss curves; decoded one-step predictions on held-out episodes; an open-loop rollout GIF."""
    import torch
    from cusi.explore.decoder import EmbeddingDecoder
    from cusi.explore.replay import load_episodes
    from cusi.state.encoders import build_image_embedder
    from cusi.state.scorers.world_model import WorldModel, read_meta
    from cusi.envs.action_vocab import ActionVocab
    record = json.load(open(os.path.join(wm_dir, "train_summary.json")))
    summary, history = record["summary"], record["history"]
    ep_x = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].plot(ep_x, [h["train_mse"] for h in history], label="train MSE")
    axes[0].plot(ep_x, [h["val"]["mse"] for h in history], label="held-out MSE")
    axes[0].axhline(history[0]["val"]["copy_mse"], color="k", ls="--", lw=0.8, label="copy baseline (held-out)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].set_title(f"{env} world model ({tag}): MSE on embeddings")
    axes[1].plot(ep_x, [h["val"]["cos"] for h in history], label="held-out cos")
    axes[1].plot(ep_x, [h["val"].get("cos_changed") for h in history], label="held-out cos, frame changed")
    axes[1].axhline(history[0]["val"]["copy_cos"], color="k", ls="--", lw=0.8, label="copy cos")
    axes[1].axhline(history[0]["val"].get("copy_cos_changed", np.nan), color="r", ls=":", lw=0.8,
                    label="copy cos, frame changed")
    axes[1].set_xlabel("epoch")
    axes[1].legend(fontsize=7)
    axes[1].set_title("cosine(pred, real next)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"wm_{tag}_loss.png"), dpi=110)
    plt.close(fig)

    config = read_meta(directory=wm_dir)["embedder_config"]
    embedder = build_image_embedder(env_name=env, device=device, **config)
    dec = EmbeddingDecoder.load(directory=config["embedder_load_path"], device=device)
    wm = WorldModel.load(directory=wm_dir, device=device)
    names = ActionVocab(env_name=env).names
    episodes = [ep for d in summary["replay_dirs"] for ep in load_episodes(directory=d)]
    val_eps = sorted(summary["val_episodes"])
    from cusi.state.canvas import to_canvas
    rows_all = []
    for e in val_eps:
        ep = episodes[e]
        z = embedder.embed(frames=[s["frame"] for s in ep])
        for t in range(1, len(ep)):
            if ep[t]["action_index"] is None:
                continue
            prev = z[t - 2] if t >= 2 else z[t - 1]
            obs = torch.cat([prev, z[t - 1]]).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = wm(obs=obs, actions=torch.tensor([ep[t]["action_index"]], device=device))[0].cpu()
            rows_all.append({"ep": e, "t": t, "action": names[ep[t]["action_index"]], "valid": ep[t]["valid"],
                             "cos": float(pred @ z[t]), "copy_cos": float(z[t - 1] @ z[t]),
                             "pred": pred, "z_next": z[t], "z_cur": z[t - 1],
                             "cur": to_canvas(frame=ep[t - 1]["frame"], env_name=env),
                             "nxt": to_canvas(frame=ep[t]["frame"], env_name=env)})
    # held-out transitions whose (current frame, action, next frame) also occurs, pixel for pixel, in a training episode
    seen = set()
    for e2, ep2 in enumerate(episodes):
        if e2 in val_eps:
            continue
        cv = [to_canvas(frame=s["frame"], env_name=env).tobytes() for s in ep2]
        for t in range(1, len(ep2)):
            seen.add((cv[t - 1], ep2[t]["action_index"], cv[t]))
    for r in rows_all:
        r["dup"] = (r["cur"].tobytes(), names.index(r["action"]), r["nxt"].tobytes()) in seen
    gain = np.array([r["cos"] - r["copy_cos"] for r in rows_all])
    changed = np.array([r["copy_cos"] < 0.999 for r in rows_all])
    stats = {"n_heldout": len(rows_all), "n_changed": int(changed.sum()),
             "cos": float(np.mean([r["cos"] for r in rows_all])),
             "copy_cos": float(np.mean([r["copy_cos"] for r in rows_all])),
             "frac_beats_copy": float((gain > 0).mean()),
             "frac_beats_copy_changed": float((gain[changed] > 0).mean()) if changed.any() else None,
             "final": summary["final"], "n_train": summary["n_train"], "n_val": summary["n_val"],
             "heldout_exact_in_train": int(sum(r["dup"] for r in rows_all))}
    novel = [r for r in rows_all if not r["dup"]]
    if novel:
        stats["cos_unseen"] = float(np.mean([r["cos"] for r in novel]))
        stats["copy_cos_unseen"] = float(np.mean([r["copy_cos"] for r in novel]))
    per_action = {}
    for r, g in zip(rows_all, gain):
        per_action.setdefault(r["action"], []).append((r["cos"], r["copy_cos"]))
    stats["per_action"] = {a: {"n": len(v), "cos": float(np.mean([x[0] for x in v])),
                               "copy_cos": float(np.mean([x[1] for x in v]))} for a, v in per_action.items()}

    def strip(r):
        d = dec.decode(torch.stack([r["z_next"], r["pred"], r["z_cur"]]))
        sep = np.full((r["cur"].shape[0], 4, 3), 255, np.uint8)
        x = np.concatenate([_rgb(r["cur"]), sep, _rgb(r["nxt"]), sep, d[0], sep, d[1], sep, d[2]], axis=1)
        return _caption(img=_upscale(x=x, env=env), lines=[
            f"ep {r['ep']} t {r['t']} action {r['action']}{'' if r['valid'] else ' (invalid)'}"
            f"{' [exact transition in train]' if r['dup'] else ''}: "
            f"cos(pred,next) {r['cos']:.3f} vs copy {r['copy_cos']:.3f}",
            "current | real next | decoded real next | decoded PREDICTED next | decoded current (copy)"])
    order = np.argsort(-gain)
    ch = [i for i in order if changed[i]]
    picks = {"best": ch[:4], "median": ch[len(ch) // 2 - 2: len(ch) // 2 + 2], "worst": ch[-4:]}
    for name, ids in picks.items():
        _grid(tiles=[strip(rows_all[i]) for i in ids], cols=1).save(os.path.join(out, f"wm_{tag}_pred_{name}.png"))
    # open-loop rollout on the held-out episode with the most changed frames
    by_ep = {}
    for r, c in zip(rows_all, changed):
        by_ep[r["ep"]] = by_ep.get(r["ep"], 0) + c
    e = max(by_ep, key=by_ep.get)
    ep = episodes[e]
    z = embedder.embed(frames=[s["frame"] for s in ep])
    hist = [z[0], z[0]]
    frames, roll = [], []
    for t in range(1, len(ep)):
        a = ep[t]["action_index"]
        if a is None:
            break
        with torch.no_grad():
            p1 = wm(obs=torch.cat([z[t - 2] if t >= 2 else z[t - 1], z[t - 1]]).unsqueeze(0).to(device),
                    actions=torch.tensor([a], device=device))[0].cpu()
            po = wm(obs=torch.cat([hist[-2], hist[-1]]).unsqueeze(0).to(device),
                    actions=torch.tensor([a], device=device))[0].cpu()
        hist.append(po)
        d = dec.decode(torch.stack([z[t], p1, po]))
        real = _rgb(to_canvas(frame=ep[t]["frame"], env_name=env))
        sep = np.full((real.shape[0], 4, 3), 255, np.uint8)
        x = np.concatenate([real, sep, d[0], sep, d[1], sep, d[2]], axis=1)
        roll.append({"t": t, "action": names[a], "cos_1step": float(p1 @ z[t]), "cos_openloop": float(po @ z[t])})
        frames.append(_caption(img=_upscale(x=x, env=env), lines=[
            f"held-out ep {e} t {t} action {names[a]}: cos 1-step {roll[-1]['cos_1step']:.3f}, "
            f"open-loop {roll[-1]['cos_openloop']:.3f}",
            "real | decoded real | decoded 1-step pred | decoded open-loop rollout"]))
    frames[0].save(os.path.join(out, f"wm_{tag}_rollout.gif"), save_all=True, append_images=frames[1:],
                   duration=1000, loop=0)
    stats["rollout"] = {"episode": e, "steps": roll}
    return stats


@main.command()
@click.option("--out", required=True)
@click.option("--envs", default=",".join(ENVS))
def models(out, envs):
    import torch
    parameters = load_parameters()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = os.path.join(out, "models_stats.json")
    stats = json.load(open(path)) if os.path.exists(path) else {}
    for env in envs.split(","):
        root = os.path.join(parameters["storage_dir"], "explore", env)
        os.makedirs(os.path.join(out, env), exist_ok=True)
        for emb_dir in sorted(glob.glob(os.path.join(root, "embedder", "siglip_trees*"))):
            if not os.path.exists(os.path.join(emb_dir, "decoder.pt")):
                continue
            s = embedder_figures(env=env, emb_dir=emb_dir, out=os.path.join(out, env),
                                 seed=parameters["random_seed"], device=device)
            stats[f"{env}/embedder/{os.path.basename(emb_dir)}"] = s
            print(f"{env} embedder {os.path.basename(emb_dir)}: {json.dumps(s)}", flush=True)
        for wm_dir in sorted(glob.glob(os.path.join(root, "world_model", "trees*"))):
            if not os.path.exists(os.path.join(wm_dir, "world_model.pt")):
                continue
            tag = os.path.basename(wm_dir)
            s = world_model_figures(env=env, wm_dir=wm_dir, out=os.path.join(out, env), tag=tag, device=device)
            stats[f"{env}/world_model/{tag}"] = s
            print(f"{env} world model {tag}: {json.dumps({k: v for k, v in s.items() if k != 'rollout'})}",
                  flush=True)
        with open(path, "w") as f:
            json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
