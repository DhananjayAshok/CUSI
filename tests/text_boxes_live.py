"""
Live check of obs["text_boxes"] on a real env: every scene's start screen plus a few random actions, each saved as an
overlay image (boxes in role colours, with their text) and the boxes printed. Android: no status-bar text (the clock).

    bash scripts/container.sh python -m tests.text_boxes_live --env web
    python -m tests.text_boxes_live --env gameboy
"""
import json
import os
import click
import numpy as np
from cusi.utils.parameter_handling import load_parameters
from cusi.agents.specs import ENV_NAMES, ENV_SPECS
from cusi.envs.text_boxes import overlay, text_boxes_of


@click.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--steps", default=3, show_default=True)
@click.option("--seed", default=0, show_default=True)
def main(env_name, steps, seed):
    parameters = load_parameters()
    spec = ENV_SPECS[env_name]
    out_dir = os.path.join(parameters["tmp_dir"], "text_boxes_check", env_name)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    clock_like = 0
    for scene in spec.scenes:
        env = spec.make_env(scene=spec.scenes[scene], worker=0, parameters=parameters)
        try:
            obs, info = env.reset()
            for t in range(steps + 1):
                if t > 0:
                    obs, _r, _te, _tr, info = env.step(env.sample_action(rng=rng))
                boxes = text_boxes_of(obs=obs)
                frame = env.raw_frame
                scale = 4 if env_name == "gameboy" else 1
                overlay(frame=frame, boxes=boxes, scale=scale).save(os.path.join(out_dir, f"{scene}_{t}.png"))
                with open(os.path.join(out_dir, f"{scene}_{t}.json"), "w") as f:
                    json.dump([b.to_dict() for b in boxes], f, indent=0)
                n_text = sum(bool(b.text) for b in boxes)
                n_inter = sum(b.interactive for b in boxes)
                print(f"== {scene} step {t}: {len(boxes)} boxes ({n_text} with text, {n_inter} interactive), "
                      f"frame {frame.shape}")
                for b in boxes[:60]:
                    print(f"   {b.role:8} {'I' if b.interactive else ' '}{'S' if b.scrollable else ' '} "
                          f"ref={b.ref} {list(b.box)} {b.text[:80]!r}")
                clock_like += sum(1 for b in boxes if b.box[1] < 70 and ":" in b.text and len(b.text) <= 5)
        finally:
            env.close()
    print(f"clock-like boxes in the top 70 px: {clock_like}")
    print(f"overlays in {out_dir}")


if __name__ == "__main__":
    main()
