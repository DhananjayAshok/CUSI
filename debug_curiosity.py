"""Rough curiosity debug runs without a policy (curiosity_plan §5), outside the main framework.

    python debug_curiosity.py --env gameboy random --scene viridian --run_name rand --n_steps 3000 \
        --curiosity_module embedbuffer --image_embedder random_patch --text_embedder none
    python debug_curiosity.py --env android human --scene contacts --run_name h1 --actions_file acts.txt \
        --curiosity_module combinationbuffer --image_embedder random_patch --text_embedder none
    python debug_curiosity.py --env gameboy human --scene viridian --run_name h2 --interactive ...

random  env.sample_action() for --n_steps, episodes of --episode_steps (the curiosity archive is
        reset every episode, as in PPO). Doubles as cheap training data for the cnn embedder,
        SigLIP fine-tuning and the world model.
human   the action strings of --actions_file (one per line, # comments), or with --interactive
        actions typed on stdin (empty line or "quit" ends); one episode. Every frame is saved with
        its reward written on it (frames/).

Both log the reward per step and per component to rewards.jsonl, write the replay format
(cusi_explore.replay) and print summary statistics. Output:
storage_dir/explore/<env>/debug/<run_name>/ (replay/, action_space.json, rewards.jsonl,
summary.json, config.json, frames/).
"""
import json
import os
import sys
import click
import numpy as np
from cusi_utils.parameter_handling import load_parameters, compute_secondary_parameters
from cusi_utils.log_handling import log_info
from cusi_practice.envs import ENV_NAMES, ENV_SPECS
from cusi_state import StateRecord, build_encoder, state_config, state_options
from cusi_state.canvas import frame_for_embedding

loaded_parameters = load_parameters()


@click.group()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--random_seed", default=loaded_parameters["random_seed"])
@click.pass_context
def main(ctx, env_name, random_seed):
    loaded_parameters["random_seed"] = random_seed
    compute_secondary_parameters(loaded_parameters)
    ctx.obj = {"parameters": loaded_parameters, "env_name": env_name}


def common_options(f):
    f = state_options(scorer=False)(f)
    for option in reversed([
        click.option("--scene", required=True),
        click.option("--run_name", required=True),
        click.option("--curiosity_module", required=True,
                     type=click.Choice(["embedbuffer", "combinationbuffer", "world_model"])),
        click.option("--region_alpha", default=0.5, type=float),
        click.option("--world_model_load_path", default=None),
        click.option("--buffer_load_path", default=None, help="Curiosity prior (an archive dir)."),
        click.option("--invalid_action_penalty", default=0.1, type=float),
        click.option("--worker", default=0, help="Android: which emulator."),
    ]):
        f = option(f)
    return f


class DebugRun:
    """env + encoder + curiosity module + replay; one step at a time."""

    def __init__(self, *, obj: dict, kwargs: dict, mode: str) -> None:
        from cusi_explore.action_vocab import ActionVocab
        from cusi_explore.curiosity import get_curiosity_module
        from cusi_explore.replay import ReplayWriter
        self.parameters, self.env_name = obj["parameters"], obj["env_name"]
        self.config = state_config(kwargs)
        spec = ENV_SPECS[self.env_name]
        self.out_dir = os.path.join(self.parameters["storage_dir"], "explore", self.env_name, "debug",
                                    kwargs["run_name"])
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "config.json"), "w") as f:
            json.dump({"mode": mode, "env": self.env_name, **self.config, **kwargs}, f, indent=1, default=str)
        with open(os.path.join(self.out_dir, "scene.txt"), "w") as f:
            f.write(kwargs["scene"])
        self.encoder = build_encoder(config=self.config, env_name=self.env_name, parameters=self.parameters)
        self.curiosity = get_curiosity_module(
            curiosity_module=kwargs["curiosity_module"], env_name=self.env_name, encoder=self.encoder,
            region_alpha=self.config["region_alpha"], world_model_load_path=self.config["world_model_load_path"],
            buffer_load_path=kwargs["buffer_load_path"], invalid_action_penalty=kwargs["invalid_action_penalty"],
            parameters=self.parameters)
        self.vocab = ActionVocab(env_name=self.env_name)
        self.vocab.save(directory=self.out_dir)
        self.replay = ReplayWriter(directory=os.path.join(self.out_dir, "replay"))
        self.rewards_file = open(os.path.join(self.out_dir, "rewards.jsonl"), "w")
        self.env = spec.make_env(scene=spec.scenes[kwargs["scene"]], worker=kwargs["worker"],
                                 parameters=self.parameters)
        self.episode = -1
        self.rows: list = []

    def start_episode(self) -> dict:
        self.episode += 1
        obs, info = self.env.reset(seed=self.parameters["random_seed"] + self.episode)
        self.curiosity.reset()
        self.record = StateRecord(obs=obs, info=info, embedding=self.encoder.encode_one(obs=obs, info=info))
        self.curiosity.get_reward(prev=None, action=None, next=self.record)    # seeds the archive (scores 0)
        self.step_in_episode = 0
        self.replay.add(episode=self.episode, step=0, frame=frame_for_embedding(obs=obs, info=info),
                        embedding=self.record.embedding.image, texts=obs["texts"], generated=None,
                        parsed_action=None, action_index=None, valid=True, reward_ext=0.0, reward_frame=0.0,
                        reward_text=0.0, reward=0.0, done=False, embedder=self.encoder.image.name)
        return {"obs": obs, "info": info}

    def step(self, *, action: str, done: bool) -> dict:
        obs, r_ext, terminated, truncated, info = self.env.step(action)
        self.step_in_episode += 1
        done = bool(done or terminated or truncated)
        record = StateRecord(obs=obs, info=info, embedding=self.encoder.encode_one(obs=obs, info=info),
                             prev_image=self.record.embedding.image)
        rew = self.curiosity.get_reward(prev=self.record, action=info.get("parsed_action"), next=record,
                                        valid=info["valid"], done=done)
        self.replay.add(episode=self.episode, step=self.step_in_episode, frame=frame_for_embedding(obs=obs, info=info),
                        embedding=record.embedding.image, texts=obs["texts"], generated=action,
                        parsed_action=info.get("parsed_action"),
                        action_index=self.vocab.encode(parsed_action=info.get("parsed_action") if info["valid"] else None),
                        valid=info["valid"], reward_ext=float(r_ext), reward_frame=rew["frame"],
                        reward_text=rew["region"], reward=float(r_ext) + rew["total"], done=done,
                        embedder=self.encoder.image.name, reward_penalty=rew["penalty"])
        changed = bool((record.embedding.image != self.record.embedding.image).any())
        row = {"episode": self.episode, "step": self.step_in_episode, "action": action.splitlines()[-1][:80],
               "valid": bool(info["valid"]), "embedding_changed": changed,
               **{k: round(v, 5) for k, v in rew.items()}}
        self.rewards_file.write(json.dumps(row) + "\n")
        self.rows.append(row)
        self.record = record
        return {"obs": obs, "info": info, "reward": rew, "row": row}

    def finish(self) -> dict:
        self.replay.close()
        self.rewards_file.close()
        self.env.close()
        rows = self.rows
        summary = {"steps": len(rows), "episodes": self.episode + 1}
        for key in ("total", "frame", "region", "penalty"):
            v = np.array([r[key] for r in rows]) if rows else np.zeros(1)
            summary[key] = {"mean": float(v.mean()), "max": float(v.max()), "p95": float(np.percentile(v, 95)),
                            "share_zero": float((np.abs(v) < 1e-6).mean())}
        unchanged = [r["frame"] for r in rows if not r["embedding_changed"]]
        changed = [r["frame"] for r in rows if r["embedding_changed"]]
        summary["frame_when_embedding_unchanged_mean"] = float(np.mean(unchanged)) if unchanged else None
        summary["frame_when_embedding_changed_mean"] = float(np.mean(changed)) if changed else None
        summary["valid_rate"] = float(np.mean([r["valid"] for r in rows])) if rows else None
        with open(os.path.join(self.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        log_info(f"debug run -> {self.out_dir}: {json.dumps(summary)}", parameters=self.parameters)
        return summary


@main.command()
@common_options
@click.option("--n_steps", default=1000)
@click.option("--episode_steps", default=30, help="Episode length (archive reset between episodes).")
@click.option("--log_every", default=100)
@click.pass_obj
def random(obj, n_steps, episode_steps, log_every, **kwargs):
    """Random actions with any curiosity module and embedder."""
    run = DebugRun(obj=obj, kwargs=kwargs, mode="random")
    run.start_episode()
    for t in range(1, n_steps + 1):
        done = run.step_in_episode + 1 >= episode_steps
        out = run.step(action=run.env.sample_action(), done=done)
        if t % log_every == 0 or t <= 5:
            log_info(f"step {t}: {json.dumps(out['row'])}", parameters=run.parameters)
        if done and t < n_steps:
            run.start_episode()
    run.finish()


def _annotate(*, frame, text: str):
    from PIL import Image, ImageDraw
    img = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    scale = max(1, 480 // img.height)
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 14], fill=(0, 0, 0))
    draw.text((2, 1), text, fill=(255, 255, 0))
    return img


@main.command()
@common_options
@click.option("--actions_file", default=None, help="One action string per line (# comments).")
@click.option("--interactive", is_flag=True, help="Read actions from stdin, one at a time.")
@click.pass_obj
def human(obj, actions_file, interactive, **kwargs):
    """A scripted (or typed) action sequence; the reward after each step, and each frame annotated."""
    if (actions_file is None) == (not interactive):
        raise click.UsageError("Pass exactly one of --actions_file and --interactive")
    run = DebugRun(obj=obj, kwargs=kwargs, mode="human")
    frames_dir = os.path.join(run.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    first = run.start_episode()
    _annotate(frame=frame_for_embedding(obs=first["obs"], info=first["info"]), text="step 0 (reset)").save(
        os.path.join(frames_dir, "step_000.png"))
    if actions_file is not None:
        with open(actions_file) as f:
            actions = [l.rstrip("\n").replace("\\n", "\n") for l in f if l.strip() and not l.startswith("#")]
        if not actions:
            raise click.UsageError(f"{actions_file} has no actions")
    print("Actions:\n" + first["obs"]["actions"][:2000], flush=True)
    t = 0
    while True:
        if actions_file is not None:
            if t >= len(actions):
                break
            action = actions[t]
        else:
            sys.stdout.write("action> ")
            sys.stdout.flush()
            action = sys.stdin.readline().rstrip("\n").replace("\\n", "\n")
            if action.strip() in ("", "quit"):
                break
        t += 1
        out = run.step(action=action, done=False)
        r = out["reward"]
        msg = (f"step {t}: total {r['total']:+.4f} frame {r['frame']:.4f} region {r['region']:.4f} "
               f"penalty {r['penalty']:.2f} valid {out['info']['valid']} err {out['info'].get('error')}")
        print(msg, flush=True)
        _annotate(frame=frame_for_embedding(obs=out["obs"], info=out["info"]), text=msg[:90]).save(
            os.path.join(frames_dir, f"step_{t:03d}.png"))
    run.finish()


if __name__ == "__main__":
    main()
