"""Curiosity exploration + world model (plans/plan.md Part 2), one environment per process.

    python run_explore.py --env gameboy ppo --scene viridian --run_name dev --total_steps 1024 \
        --policy_model Qwen/Qwen3.5-0.8B --curiosity_module combinationbuffer \
        --image_embedder random_patch --text_embedder none
    python run_explore.py --env gameboy tasks --run_name dev --model_name <served name>   # needs vLLM
    python run_explore.py --env gameboy world_model --run_names dev [--image_embedder random_patch]
    python run_explore.py --env gameboy train_embedder --embedder cnn --run_names debug/rand_patch
    python run_explore.py --env gameboy decoder --run_names dev
    python run_explore.py --env gameboy wm_eval --name dev
    python run_explore.py --env android elements --run_names dev                   # K check

Outputs: storage_dir/explore/<env>/<run_name>/ (replay/, policy/, metrics.jsonl, action_space.json,
buffer/), storage_dir/explore/<env>/world_model/<name>/, storage_dir/explore/<env>/decoder/<name>/.
Curiosity tasks go to storage_dir/practice/<env>/<model>/curiosity/attempts/, then
`python run_practice.py --env <env> --model_name <served name> --source curiosity all` practises them.
"""
import json
import os
import click
import numpy as np
from cusi.utils.parameter_handling import load_parameters, compute_secondary_parameters
from cusi.utils.log_handling import log_info, log_warn
from cusi.agents.specs import ENV_NAMES, ENV_SPECS
from cusi.state import state_config, state_options

loaded_parameters = load_parameters()

# Per-environment episode length (GameBoy: 30, as GameBoyRL's curiosity runs) and rollout length.
EPISODE_STEPS = {"gameboy": 30, "android": 20, "web": 15}
ROLLOUT_STEPS = {"gameboy": 128, "android": 64, "web": 64}


@click.group()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--random_seed", default=loaded_parameters["random_seed"])
@click.option("--vllm_port", default=loaded_parameters["vllm_port"], help="Port of the vLLM server (tasks only).")
@click.pass_context
def main(ctx, env_name, random_seed, vllm_port):
    loaded_parameters["random_seed"] = random_seed
    loaded_parameters["vllm_port"] = vllm_port
    compute_secondary_parameters(loaded_parameters)
    root = os.path.join(loaded_parameters["storage_dir"], "explore", env_name)
    os.makedirs(root, exist_ok=True)
    ctx.obj = {"parameters": loaded_parameters, "env_name": env_name, "root": root}


def _run_dirs(obj, run_names: str) -> list:
    return [os.path.join(obj["root"], r) for r in run_names.split(",") if r.strip()]


@main.command()
@click.option("--scene", required=True)
@click.option("--run_name", required=True)
@click.option("--policy_model", required=True, help="HF id of the VLM policy (e.g. Qwen/Qwen3.5-0.8B).")
@click.option("--curiosity_module", required=True, type=click.Choice(["embedbuffer", "combinationbuffer", "world_model"]))
@state_options(scorer=False)
@click.option("--region_alpha", default=0.5, type=float, help="Region weight (combinationbuffer).")
@click.option("--world_model_load_path", default=None, help="Trained world model dir (world_model module).")
@click.option("--total_steps", default=1024)
@click.option("--num_steps", default=None, type=int, help="Rollout length per update (default per env).")
@click.option("--max_episode_steps", default=None, type=int, help="Episode truncation (default per env).")
@click.option("--learning_rate", default=1e-5)
@click.option("--target_kl", default=0.1, type=float, help="Stop an update when approx KL exceeds this.")
@click.option("--update_epochs", default=2)
@click.option("--num_minibatches", default=8)
@click.option("--kl_coef", default=0.05)
@click.option("--ent_coef", default=0.0)
@click.option("--invalid_action_penalty", default=0.1, type=float,
              help="Subtracted from the curiosity reward of an action the env rejects (0: off).")
@click.option("--normalize_curiosity_reward", is_flag=True, help="Divide by a running std of the curiosity return.")
@click.option("--reply_format", default="action_only", type=click.Choice(["action_only", "native"]))
@click.option("--temperature", default=0.7)
@click.option("--max_new_tokens", default=None, type=int, help="Default: sized to the reply format and env.")
@click.option("--lora_r", default=128)
@click.option("--lora_alpha", default=256)
@click.option("--buffer_load_path", default=None, help="Curiosity prior (an earlier run's buffer/ dir).")
@click.option("--worker", default=0, help="Android: which emulator (console port 5554 + 2*worker).")
@click.pass_obj
def ppo(obj, scene, run_name, policy_model, curiosity_module, region_alpha, world_model_load_path, total_steps,
        num_steps, max_episode_steps, learning_rate, target_kl, update_epochs, num_minibatches, kl_coef, ent_coef,
        invalid_action_penalty, normalize_curiosity_reward, reply_format, temperature, max_new_tokens, lora_r,
        lora_alpha, buffer_load_path, worker, **state_kwargs):
    """Curiosity PPO with the VLM policy on one scene."""
    from cusi.explore.ppo import CuriosityPPO
    env_name, parameters = obj["env_name"], obj["parameters"]
    config = state_config(state_kwargs)
    spec = ENV_SPECS[env_name]
    env = spec.make_env(scene=spec.scenes[scene], worker=worker, parameters=parameters)
    out_dir = os.path.join(obj["root"], run_name)
    try:
        trainer = CuriosityPPO(
            env_name=env_name, env=env, out_dir=out_dir, total_steps=total_steps, state_config=config,
            curiosity_module=curiosity_module, policy_model=policy_model,
            num_steps=num_steps or ROLLOUT_STEPS[env_name],
            max_episode_steps=max_episode_steps or EPISODE_STEPS[env_name], learning_rate=learning_rate,
            update_epochs=update_epochs, num_minibatches=num_minibatches, kl_coef=kl_coef, ent_coef=ent_coef,
            target_kl=target_kl, region_alpha=region_alpha, world_model_load_path=world_model_load_path,
            invalid_action_penalty=invalid_action_penalty, normalize_curiosity_reward=normalize_curiosity_reward,
            buffer_load_path=buffer_load_path, buffer_save_path=os.path.join(out_dir, "buffer"),
            reply_format=reply_format,
            policy_kwargs={"temperature": temperature, "max_new_tokens": max_new_tokens, "lora_r": lora_r,
                           "lora_alpha": lora_alpha},
            seed=parameters["random_seed"], parameters=parameters)
        with open(os.path.join(out_dir, "scene.txt"), "w") as f:
            f.write(scene)
        trainer.train()
    finally:
        env.close()


@main.command()
@click.option("--run_name", required=True)
@click.option("--scene", default=None, help="Default: the run's scene.txt.")
@click.option("--model_name", required=True, help="Model name the vLLM server serves.")
@click.option("--outlier_threshold", default=2.5)
@click.option("--z_min", default=3.0)
@click.option("--max_groups", default=None, type=int)
@click.option("--max_new_tokens", default=2000)
@click.option("--rescore_text_alpha", default=None, type=float,
              help="Recompute intrinsic rewards with the current curiosity code first (for older replays).")
@click.pass_obj
def tasks(obj, run_name, scene, model_name, outlier_threshold, z_min, max_groups, max_new_tokens, rescore_text_alpha):
    """High-novelty trajectories -> groups -> tasks (the curiosity task source for run_practice.py)."""
    from cusi.explore.trajectories import curiosity_tasks
    from cusi.practice.stages.common import PracticePaths
    from cusi.agents.vlm import AgentVLM
    env_name, parameters = obj["env_name"], obj["parameters"]
    run_dir = os.path.join(obj["root"], run_name)
    scene = scene or open(os.path.join(run_dir, "scene.txt")).read().strip()
    paths = PracticePaths(parameters=parameters, env_name=env_name, model_name=model_name, source="curiosity")
    stats = curiosity_tasks(replay_dir=os.path.join(run_dir, "replay"), scene=scene, env_name=env_name,
                            vlm=AgentVLM(model_name=model_name, parameters=parameters),
                            domain=ENV_SPECS[env_name].domain, out_attempts_dir=paths.attempts_dir,
                            max_new_tokens=max_new_tokens, outlier_threshold=outlier_threshold, z_min=z_min,
                            max_groups=max_groups, rescore_text_alpha=rescore_text_alpha, seed=parameters["random_seed"],
                            parameters=parameters)
    log_info(f"curiosity tasks: {stats} -> {paths.attempts_dir}", parameters=parameters)


@main.command(name="world_model")
@click.option("--run_names", required=True, help="Comma-separated exploration runs whose replay to train on "
              "(debug runs: debug/<name>).")
@click.option("--name", default=None, help="Output name (default: the run names joined).")
@click.option("--epochs", default=50)
@click.option("--image_embedder", default=None, type=click.Choice(["random_patch", "cnn", "siglip"]),
              help="Re-embed the replay frames with this embedder (default: the stored embeddings).")
@click.option("--encoder_model", default=None, help="SigLIP 2 model id (with --image_embedder siglip).")
@click.option("--embedder_load_path", default=None, help="Trained cnn / fine-tuned siglip checkpoint.")
@click.pass_obj
def world_model(obj, run_names, name, epochs, image_embedder, encoder_model, embedder_load_path):
    """Train the world model on replay buffers."""
    from cusi.explore.world_model import train_world_model
    from cusi.state import build_image_embedder
    dirs = _run_dirs(obj, run_names)
    out = os.path.join(obj["root"], "world_model", name or run_names.replace(",", "+").replace("/", "_"))
    embedder = None if image_embedder is None else build_image_embedder(
        image_embedder=image_embedder, env_name=obj["env_name"], encoder_model=encoder_model,
        embedder_load_path=embedder_load_path, parameters=obj["parameters"])
    train_world_model(replay_dirs=[os.path.join(d, "replay") for d in dirs], out_dir=out, action_space_dir=dirs[0],
                      epochs=epochs, embedder=embedder, seed=obj["parameters"]["random_seed"],
                      parameters=obj["parameters"])


@main.command(name="train_embedder")
@click.option("--embedder", required=True, type=click.Choice(["cnn", "siglip"]))
@click.option("--run_names", required=True, help="Comma-separated runs whose replay frames to train on "
              "(debug runs: debug/<name>).")
@click.option("--name", default=None, help="Output: explore/<env>/embedder/<embedder>_<name>.")
@click.option("--encoder_model", default=None, help="SigLIP 2 model id to fine-tune (--embedder siglip).")
@click.option("--epochs", default=20, help="cnn: epochs.")
@click.option("--steps", default=20, help="siglip: optimizer steps.")
@click.option("--max_minutes", default=5.0)
@click.option("--max_frames", default=50_000)
@click.pass_obj
def train_embedder(obj, embedder, run_names, name, encoder_model, epochs, steps, max_minutes, max_frames):
    """Train a cnn embedder / fine-tune SigLIP on replay frames (curiosity_plan §3.2b)."""
    from cusi.explore.train_embedder import train_cnn, train_siglip
    dirs = [os.path.join(d, "replay") for d in _run_dirs(obj, run_names)]
    out = os.path.join(obj["root"], "embedder", f"{embedder}_{name or run_names.replace(',', '+').replace('/', '_')}")
    if embedder == "cnn":
        train_cnn(env_name=obj["env_name"], replay_dirs=dirs, out_dir=out, epochs=epochs, max_minutes=max_minutes,
                  max_frames=max_frames, seed=obj["parameters"]["random_seed"], parameters=obj["parameters"])
    else:
        if not encoder_model:
            raise click.UsageError("--embedder siglip needs --encoder_model")
        train_siglip(env_name=obj["env_name"], replay_dirs=dirs, out_dir=out, encoder_model=encoder_model,
                     steps=steps, max_minutes=max_minutes, max_frames=min(max_frames, 5000),
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
    out = os.path.join(obj["root"], "decoder", name or run_names.replace(",", "+"))
    summary = train_decoder(env_name=obj["env_name"], replay_dirs=[os.path.join(d, "replay") for d in dirs],
                            out_dir=out, epochs=epochs, max_frames=max_frames, seed=obj["parameters"]["random_seed"],
                            parameters=obj["parameters"])
    log_info(f"decoder -> {out}: {summary}", parameters=obj["parameters"])


@main.command(name="wm_eval")
@click.option("--name", required=True, help="world_model/<name> and decoder/<name>.")
@click.option("--run_names", default=None, help="Replay to draw examples from (default: the world model's).")
@click.option("--n_examples", default=6)
@click.pass_obj
def wm_eval(obj, name, run_names, n_examples):
    """Show world-model predictions as decoded frames: for sampled transitions, the current frame,
    the real next frame, and the decoded prediction for the action taken and for a few other
    actions; plus the decoder's round-trip of the real next frame."""
    import torch
    from PIL import Image
    from cusi.explore.decoder import EmbeddingDecoder
    from cusi.explore.world_model import transitions_from_replay
    from cusi.state.scorers.world_model import WorldModel
    from cusi.explore.replay import load_episodes
    root = obj["root"]
    wm_dir, dec_dir = os.path.join(root, "world_model", name), os.path.join(root, "decoder", name)
    summary = json.load(open(os.path.join(wm_dir, "train_summary.json")))["summary"]
    replay_dirs = summary["replay_dirs"] if run_names is None else \
        [os.path.join(d, "replay") for d in _run_dirs(obj, run_names)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm, dec = WorldModel.load(directory=wm_dir, device=device), EmbeddingDecoder.load(directory=dec_dir, device=device)
    rng = np.random.default_rng(0)
    rows = []
    episodes = [ep for d in replay_dirs for ep in load_episodes(directory=d)]
    candidates = [(e, t) for e, ep in enumerate(episodes) for t in range(1, len(ep))
                  if ep[t]["action_index"] is not None and ep[t]["valid"]]
    if not candidates:   # no valid action in the replay: show the no-op transitions instead
        candidates = [(e, t) for e, ep in enumerate(episodes) for t in range(1, len(ep))
                      if ep[t]["action_index"] is not None]
    for e, t in [candidates[i] for i in rng.choice(len(candidates), size=min(n_examples, len(candidates)),
                                                   replace=False)]:
        ep = episodes[e]
        embs = [torch.tensor(np.asarray(s["embedding"], dtype=np.float32)) for s in ep]
        prev = embs[t - 2] if t >= 2 else embs[t - 1]
        obs = torch.cat([prev, embs[t - 1]]).unsqueeze(0).to(device)
        taken = ep[t]["action_index"]
        others = [a for a in rng.choice(wm.n_actions, size=3, replace=False).tolist() if a != taken][:2]
        acts = torch.tensor([taken] + others, device=device)
        preds = wm(obs=obs.repeat(len(acts), 1), actions=acts)
        tiles = [ep[t - 1]["frame"].pil(), ep[t]["frame"].pil()]
        size = dec.out_size[1], dec.out_size[0]
        tiles = [np.asarray(x.resize(size)) for x in tiles]
        tiles.append(dec.decode(embs[t].unsqueeze(0))[0])
        tiles += list(dec.decode(preds))
        rows.append(np.concatenate(tiles, axis=1))
    grid = np.concatenate(rows, axis=0)
    out_png = os.path.join(wm_dir, "predictions.png")
    Image.fromarray(grid).save(out_png)
    data = transitions_from_replay(replay_dirs=replay_dirs)
    from cusi.explore.world_model import evaluate
    keep = np.where(data["action"] >= 0)[0]
    metrics = evaluate(model=wm, data=data, idx=keep, device=device)
    with open(os.path.join(wm_dir, "eval.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    log_info(f"wm_eval: {metrics}; columns = current | real next | decoded real next | predicted(taken) |"
             f" predicted(other) x2 -> {out_png}", parameters=obj["parameters"])


@main.command()
@click.option("--run_names", required=True)
@click.pass_obj
def elements(obj, run_names):
    """Element counts per screen and the share of indexed actions >= K (decision 18's check)."""
    from cusi.envs.action_vocab import ActionVocab
    from cusi.state import element_lines
    from cusi.explore.replay import iter_replay
    vocab = ActionVocab(env_name=obj["env_name"])
    counts, overflow, indexed = [], 0, 0
    for d in _run_dirs(obj, run_names):
        for row in iter_replay(directory=os.path.join(d, "replay"), load_frames=False):
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
    with open(os.path.join(obj["root"], f"elements_{run_names.replace(',', '+')}.json"), "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
