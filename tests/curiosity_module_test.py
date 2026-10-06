"""
CPU checks of the curiosity wrappers (curiosity_plan §3.3) on synthetic states:

    python tests/curiosity_module_test.py [--world_model_load_path <a GameBoy random_patch world model>]

first_add (the first two frames after a reset into an empty archive score 0), reset to the prior,
the invalid-action penalty, reward normalisation, and (optionally) the world_model scorer loading
a trained model, scoring a rejected action 0 and refusing a different embedder.
"""
import tempfile
import click
import numpy as np
import torch


def check(cond: bool, msg: str) -> None:
    print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        raise SystemExit(1)


@click.command()
@click.option("--world_model_load_path", default=None)
def main(world_model_load_path):
    from cusi.explore.curiosity import get_curiosity_module
    from cusi.state import StateEncoder, StateRecord, build_image_embedder, build_text_embedder
    rng = np.random.default_rng(0)
    enc = StateEncoder(image=build_image_embedder(image_embedder="random_patch", env_name="gameboy"),
                       text=build_text_embedder(text_embedder="none"))
    frames = [np.repeat(rng.integers(0, 255, (144, 160, 1), dtype=np.uint8), 3, axis=2) for _ in range(5)]
    recs = [StateRecord(obs={"frame": f, "texts": {}}, info={"valid": True},
                        embedding=enc.encode_one(obs={"frame": f, "texts": {}}, info={})) for f in frames]
    m = get_curiosity_module(curiosity_module="embedbuffer", env_name="gameboy", encoder=enc)
    m.reset()
    r = [m.get_reward(prev=None, action=None, next=x)["total"] for x in recs[:3]]
    check(r[0] == 0 and r[1] == 0 and r[2] > 0, f"first_add: {np.round(r, 3).tolist()}")
    bad = m.get_reward(prev=recs[2], action=None, next=recs[2], valid=False)
    check(abs(bad["total"] + 0.1) < 1e-6 and bad["penalty"] == 0.1, f"invalid action: {bad['total']:.3f}")
    with tempfile.TemporaryDirectory() as d:
        m.save_path = d
        m.save()
        prior = get_curiosity_module(curiosity_module="embedbuffer", env_name="gameboy", encoder=enc,
                                     buffer_load_path=d)
        prior.reset()
        v = prior.get_reward(prev=None, action=None, next=recs[0])["total"]
        n = len(prior.archive)
        prior.get_reward(prev=None, action=None, next=recs[4])
        prior.reset()
        check(v < 1e-5 and len(prior.archive) == n == 3, f"prior: a seen frame scores {v:.4f}; reset back to {n} states")
    norm = get_curiosity_module(curiosity_module="embedbuffer", env_name="gameboy", encoder=enc,
                                normalize_curiosity_reward=True)
    norm.reset()
    out = [norm.get_reward(prev=None, action=None, next=x) for x in recs]
    check(all(np.isfinite(o["total"]) for o in out) and out[-1]["total"] != out[-1]["novelty"],
          f"normalised: novelty {out[-1]['novelty']:.3f} -> {out[-1]['total']:.3f}")
    if world_model_load_path:
        wm = get_curiosity_module(curiosity_module="world_model", env_name="gameboy", encoder=enc,
                                  world_model_load_path=world_model_load_path)
        nxt = StateRecord(obs={}, info={"valid": True, "parsed_action": {"action_type": "LowLevelAction",
                                                                         "low_level_action": "UP"}},
                          embedding=recs[1].embedding, prev_image=recs[0].embedding.image)
        v = wm.get_reward(prev=recs[0], action=nxt.info["parsed_action"], next=nxt)["novelty"]
        nxt.info = {"valid": False}
        v_bad = wm.get_reward(prev=recs[0], action=None, next=nxt)["novelty"]
        check(0 <= v <= 2 and v_bad == 0, f"world_model: valid step {v:.3f}, rejected step {v_bad}")
        cnn_like = StateEncoder(image=build_image_embedder(image_embedder="random_patch", env_name="android"),
                                text=build_text_embedder(text_embedder="none"))
        try:
            get_curiosity_module(curiosity_module="world_model", env_name="gameboy", encoder=cnn_like,
                                 world_model_load_path=world_model_load_path)
            refused = False
        except ValueError:
            refused = True
        check(refused, "world_model refuses an encoder it was not trained with")
    print("ALL OK")


if __name__ == "__main__":
    main()
