"""
World-model / embedder / decoder training from exploration and search replays, one environment per process.

    python run_explore.py --env gameboy <command> [options]
"""
import json
import os
import click
import numpy as np
from cusi.utils.parameter_handling import load_parameters, compute_secondary_parameters
from cusi.utils import paths
from cusi.utils.log_handling import log_info, log_warn
from cusi.agents.specs import ENV_NAMES

loaded_parameters = load_parameters()



@click.group()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--random_seed", default=loaded_parameters["random_seed"])
@click.option("--vllm_port", default=loaded_parameters["vllm_port"], help="Port of the vLLM server.")
@click.pass_context
def main(ctx, env_name, random_seed, vllm_port):
    loaded_parameters["random_seed"] = random_seed
    loaded_parameters["vllm_port"] = vllm_port
    compute_secondary_parameters(loaded_parameters)
    root = paths.explore_root(parameters=loaded_parameters, env=env_name)
    os.makedirs(root, exist_ok=True)
    ctx.obj = {"parameters": loaded_parameters, "env_name": env_name, "root": root}


def _run_dirs(obj, run_names: str) -> list:
    return paths.explore_runs(parameters=obj["parameters"], env=obj["env_name"], run_names=run_names)


@main.command(name="world_model")
@click.option("--run_names", required=True, help="Comma-separated exploration runs whose replay to train on "
              "(debug runs: debug/<name>).")
@click.option("--name", default=None, help="Output name (default: the run names joined).")
@click.option("--epochs", default=50)
@click.option("--image_embedder", default=None, type=click.Choice(["random_patch", "cnn", "siglip"]),
              help="Re-embed the replay frames with this embedder (default: the stored embeddings).")
@click.option("--encoder_model", default=None, help="SigLIP 2 model id (with --image_embedder siglip).")
@click.option("--embedder_load_path", default=None, help="Trained cnn / fine-tuned siglip checkpoint.")
@click.option("--residual", is_flag=True, help="Predict the change from the current embedding.")
@click.pass_obj
def world_model(obj, run_names, name, epochs, image_embedder, encoder_model, embedder_load_path, residual):
    """Train the world model on replay buffers."""
    from cusi.explore.world_model import train_world_model
    from cusi.state import build_image_embedder
    dirs = _run_dirs(obj, run_names)
    out = paths.world_model_dir(parameters=obj["parameters"], env=obj["env_name"],
                                name=name or paths.runs_label(run_names=run_names))
    config = None if image_embedder is None else {"image_embedder": image_embedder, "encoder_model": encoder_model,
                                                  "embedder_load_path": embedder_load_path}
    embedder = None if config is None else build_image_embedder(env_name=obj["env_name"],
                                                                parameters=obj["parameters"], **config)
    train_world_model(replay_dirs=[paths.explore_replay(run_dir=d) for d in dirs], out_dir=out,
                      env_name=obj["env_name"], epochs=epochs, embedder=embedder, embedder_config=config,
                      residual=residual, seed=obj["parameters"]["random_seed"], parameters=obj["parameters"])


@main.command(name="train_embedder")
@click.option("--embedder", required=True, type=click.Choice(["cnn", "siglip"]))
@click.option("--run_names", required=True, help="Comma-separated runs whose replay frames to train on "
              "(debug runs: debug/<name>).")
@click.option("--name", default=None, help="Output: explore/<env>/embedder/<embedder>_<name>.")
@click.option("--encoder_model", default=None, help="SigLIP 2 model id to fine-tune (--embedder siglip).")
@click.option("--epochs", default=20, help="cnn: epochs.")
@click.option("--steps", default=20, help="siglip: optimizer steps.")
@click.option("--max_minutes", default=5.0)
@click.option("--max_frames", default=50_000, help="Training frames (siglip: sampled across the training episodes).")
@click.pass_obj
def train_embedder(obj, embedder, run_names, name, encoder_model, epochs, steps, max_minutes, max_frames):
    """Train a cnn embedder / fine-tune SigLIP on replay frames."""
    from cusi.explore.train_embedder import train_cnn, train_siglip
    dirs = [paths.explore_replay(run_dir=d) for d in _run_dirs(obj, run_names)]
    out = paths.embedder_dir(parameters=obj["parameters"], env=obj["env_name"], embedder=embedder,
                             name=name or paths.runs_label(run_names=run_names))
    if embedder == "cnn":
        train_cnn(env_name=obj["env_name"], replay_dirs=dirs, out_dir=out, epochs=epochs, max_minutes=max_minutes,
                  max_frames=max_frames, seed=obj["parameters"]["random_seed"], parameters=obj["parameters"])
    else:
        if not encoder_model:
            raise click.UsageError("--embedder siglip needs --encoder_model")
        train_siglip(env_name=obj["env_name"], replay_dirs=dirs, out_dir=out, encoder_model=encoder_model,
                     steps=steps, max_minutes=max_minutes, max_frames=max_frames,
                     seed=obj["parameters"]["random_seed"], parameters=obj["parameters"])


@main.command()
@click.option("--run_names", required=True)
@click.option("--name", default=None)
@click.option("--epochs", default=30)
@click.option("--max_frames", default=50_000)
@click.pass_obj
def decoder(obj, run_names, name, epochs, max_frames):
    """Train the embedding -> pixels decoder on replay frames."""
    from cusi.explore.decoder import train_decoder
    dirs = _run_dirs(obj, run_names)
    out = paths.decoder_dir(parameters=obj["parameters"], env=obj["env_name"],
                            name=name or paths.runs_label(run_names=run_names))
    summary = train_decoder(env_name=obj["env_name"], replay_dirs=[paths.explore_replay(run_dir=d) for d in dirs],
                            out_dir=out, epochs=epochs, max_frames=max_frames, seed=obj["parameters"]["random_seed"],
                            parameters=obj["parameters"])
    log_info(f"decoder -> {out}: {summary}", parameters=obj["parameters"])


@main.command(name="wm_eval")
@click.option("--name", required=True, help="world_model/<name>; its decoder is decoder/<name>, or the fine-tuned "
              "SigLIP's own decoder when the world model was trained on one.")
@click.option("--run_names", default=None, help="Replay to draw examples from (default: the world model's).")
@click.option("--n_examples", default=6)
@click.option("--n_pixel", default=2000, help="Held-out transitions decoded for the pixel errors.")
@click.pass_obj
def wm_eval(obj, name, run_names, n_examples, n_pixel):
    """Render decoded world-model predictions for sampled held-out transitions and score the model: embedding
    cosine / MSE against the "nothing changes" baseline, and pixel errors (mean |a - b| in 0-255 levels) of the
    decoded prediction, of copying the current frame and of the decoder's round trip of the real next frame."""
    import torch
    from PIL import Image
    from cusi.explore.decoder import EmbeddingDecoder
    from cusi.explore.world_model import transitions_from_episodes, evaluate
    from cusi.explore.replay import is_validation_episode, load_episodes
    from cusi.state import build_image_embedder
    from cusi.state.scorers.world_model import WorldModel, read_meta
    wm_dir = paths.world_model_dir(parameters=obj["parameters"], env=obj["env_name"], name=name)
    config = read_meta(directory=wm_dir).get("embedder_config")
    embedder = None if config is None else build_image_embedder(env_name=obj["env_name"],
                                                                parameters=obj["parameters"], **config)
    if config is not None and config["image_embedder"] == "siglip":
        # the reconstruction fine-tune trains a decoder on its own embeddings
        if not config.get("embedder_load_path"):
            raise click.UsageError("wm_eval needs a decoder: a zero-shot SigLIP world model has none "
                                   "(train it on a fine-tuned SigLIP, whose directory holds one)")
        dec_dir = config["embedder_load_path"]
    elif config is not None:
        raise click.UsageError(f"wm_eval has no decoder for a world model re-embedded with {config['image_embedder']}")
    else:
        dec_dir = paths.decoder_dir(parameters=obj["parameters"], env=obj["env_name"], name=name)
    summary = json.load(open(os.path.join(wm_dir, "train_summary.json")))["summary"]
    replay_dirs = summary["replay_dirs"] if run_names is None else \
        [paths.explore_replay(run_dir=d) for d in _run_dirs(obj, run_names)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm, dec = WorldModel.load(directory=wm_dir, device=device), EmbeddingDecoder.load(directory=dec_dir, device=device)
    rng = np.random.default_rng(0)
    # frames for the pictures always; stored embeddings only when the world model was trained on them
    episodes = [ep for d in replay_dirs for ep in load_episodes(directory=d, embed=embedder is None)]
    # Examples and metrics come from the world model's held-out episodes (ids: positions in load order), unless
    # another replay was asked for or the model predates the recorded split. Only those episodes are embedded.
    if run_names is not None:
        log_info(f"wm_eval: every transition of {run_names} (held out only if the world model never trained on it)",
                 parameters=obj["parameters"])
        eval_eps = list(range(len(episodes)))
    elif not summary.get("val_episodes"):
        log_warn("wm_eval: no held-out split for this replay; examples and metrics include training transitions",
                 parameters=obj["parameters"])
        eval_eps = list(range(len(episodes)))
    elif "seed" in summary:
        eval_eps = [e for e, ep in enumerate(episodes)
                    if is_validation_episode(source=ep[0]["source"], episode=ep[0]["episode"], seed=summary["seed"])]
    else:
        eval_eps = sorted(summary["val_episodes"])
    data = transitions_from_episodes(episodes=[episodes[e] for e in eval_eps], episode_ids=eval_eps,
                                     embedder=embedder, seed=summary.get("seed", obj["parameters"]["random_seed"]))
    keep = np.where(data["action"] >= 0)[0]
    metrics = evaluate(model=wm, data=data, idx=keep, device=device)
    metrics["held_out"] = run_names is None and bool(summary.get("val_episodes"))

    size = dec.out_size[1], dec.out_size[0]

    def real(e, t):
        return np.asarray(episodes[e][t]["frame"].pil().convert("RGB").resize(size), dtype=np.uint8)

    def predict(i, actions):
        obs = torch.tensor(data["obs"][i], device=device).unsqueeze(0).repeat(len(actions), 1)
        return wm(obs=obs, actions=torch.tensor(actions, device=device))

    # Pixel errors on decoded frames, as GameBoyRL's world-model executor records them (WorldModelDecision), and
    # PSNR / SSIM / LPIPS (as DreamDojo reports them), each for the prediction, copying the current frame and the
    # decoder's round trip of the real next frame
    import lpips
    from skimage.metrics import structural_similarity
    torch.hub.set_dir(os.path.join(obj["parameters"]["model_dir"], "torch_hub"))   # LPIPS's AlexNet weights
    lpips_net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def scores(a, b):
        """mae, psnr (None when identical), ssim, lpips of two HxWx3 frames in 0-255."""
        mse = float(((a - b) ** 2).mean())
        t = torch.tensor(np.stack([a, b]), device=device).permute(0, 3, 1, 2) / 127.5 - 1
        return {"mae": float(np.abs(a - b).mean()), "psnr": None if mse == 0 else 10 * np.log10(255 ** 2 / mse),
                "ssim": float(structural_similarity(a, b, data_range=255, channel_axis=2)),
                "lpips": float(lpips_net(t[:1], t[1:]).item())}

    def shifted(before, after):
        """Whether the change is a scroll: some vertical or horizontal shift of before (1 to 1/8 of the frame)
        matches after on the overlap with under a quarter of the unshifted error."""
        g0, g1 = before.mean(2), after.mean(2)
        base = np.abs(g0 - g1).mean()
        h, w = g0.shape
        errs = [np.abs(g0[d:] - g1[:h - d]).mean() for d in range(1, h // 8 + 1)]
        errs += [np.abs(g0[:h - d] - g1[d:]).mean() for d in range(1, h // 8 + 1)]
        errs += [np.abs(g0[:, d:] - g1[:, :w - d]).mean() for d in range(1, w // 8 + 1)]
        errs += [np.abs(g0[:, :w - d] - g1[:, d:]).mean() for d in range(1, w // 8 + 1)]
        return bool(min(errs) < 0.25 * base)

    pix = {"changed": [], "unchanged": []}
    shift_rows = []
    spread = []
    with torch.no_grad():
        for i in rng.choice(keep, size=min(n_pixel, len(keep)), replace=False):
            e, t = int(data["episode"][i]), int(data["step"][i])
            before, after = real(e, t - 1).astype(np.float32), real(e, t).astype(np.float32)
            preds = dec.decode(predict(i, list(range(wm.n_actions)))).astype(np.float32)
            recon = dec.decode(torch.tensor(data["next"][i]).unsqueeze(0))[0].astype(np.float32)
            row = {"predicted": scores(preds[data["action"][i]], after), "copy": scores(before, after),
                   "decoder_round_trip": scores(recon, after)}
            changed = bool(np.abs(after - before).max() > 0)
            pix["changed" if changed else "unchanged"].append(row)
            if changed and shifted(before, after):
                shift_rows.append(row)
            spread.append(np.mean([np.abs(a - b).mean() for k, a in enumerate(preds) for b in preds[k + 1:]]))
    pix["shifted"] = shift_rows   # the changed transitions that are scrolls
    for kind, rows in pix.items():
        if rows:
            out = {"n": len(rows)}
            for name in ("mae", "psnr", "ssim", "lpips"):
                out[name] = {}
                for source in ("predicted", "copy", "decoder_round_trip"):
                    vals = [r[source][name] for r in rows if r[source][name] is not None]
                    out[name][source] = float(np.mean(vals)) if vals else None   # None: copy is exact (PSNR inf)
            # mae kept under GameBoyRL's names too
            out.update({source: out["mae"][source] for source in ("predicted", "copy", "decoder_round_trip")})
            out["predicted_beats_copy"] = float(np.mean([r["predicted"]["mae"] < r["copy"]["mae"] for r in rows]))
            metrics[f"pixel_{kind}"] = out
    metrics["pixel_spread_across_actions"] = float(np.mean(spread))

    # Grid: current | real next | decoded real next | predicted(taken) | predicted(other) x2
    shown = [i for i in keep if data["frame_changed"][i] > 0] or list(keep)
    rows = []
    with torch.no_grad():
        for i in rng.choice(shown, size=min(n_examples, len(shown)), replace=False):
            e, t, taken = int(data["episode"][i]), int(data["step"][i]), int(data["action"][i])
            others = [a for a in rng.choice(wm.n_actions, size=3, replace=False).tolist() if a != taken][:2]
            tiles = [real(e, t - 1), real(e, t), dec.decode(torch.tensor(data["next"][i]).unsqueeze(0))[0]]
            tiles += list(dec.decode(predict(i, [taken] + others)))
            rows.append(np.concatenate(tiles, axis=1))
    # another replay's results sit beside the world model's own
    suffix = "" if run_names is None else "_" + paths.runs_label(run_names=run_names)
    out_png = os.path.join(wm_dir, f"predictions{suffix}.png")
    Image.fromarray(np.concatenate(rows, axis=0)).save(out_png)
    with open(os.path.join(wm_dir, f"eval{suffix}.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    log_info(f"wm_eval: {metrics}; columns = current | real next | decoded real next | predicted(taken) |"
             f" predicted(other) x2 -> {out_png}", parameters=obj["parameters"])


@main.command()
@click.option("--run_names", required=True)
@click.pass_obj
def elements(obj, run_names):
    """Element counts per screen and the share of indexed actions >= K."""
    from cusi.envs.action_vocab import ActionVocab
    from cusi.state import element_lines
    from cusi.explore.replay import iter_replay
    vocab = ActionVocab(env_name=obj["env_name"])
    counts, overflow, indexed = [], 0, 0
    for d in _run_dirs(obj, run_names):
        for row in iter_replay(directory=paths.explore_replay(run_dir=d), load_frames=False):
            counts.append(len(element_lines(texts=row["texts"])))
            name = vocab.names[row["action_index"]] if row["action_index"] is not None else ""
            if "[" in name:
                indexed += 1
                overflow += ">=" in name
    if not counts:
        log_warn("elements: no replay rows", parameters=obj["parameters"])
        return
    c = np.array(counts)
    stats = {"screens": int(len(c)), "elements_mean": float(c.mean()), "elements_p50": float(np.median(c)),
             "elements_p95": float(np.percentile(c, 95)), "elements_max": int(c.max()), "K": vocab.k,
             "share_screens_over_K": float((c > (vocab.k or 1e9)).mean()), "indexed_actions": indexed,
             "overflow_actions": overflow}
    log_info(f"elements: {stats}", parameters=obj["parameters"])
    with open(paths.elements_file(parameters=obj["parameters"], env=obj["env_name"], run_names=run_names), "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
