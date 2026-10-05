"""
Regression gate for the cusi_state migration (curiosity_plan §3.2.4): the migrated encoder,
archive and scorers must give the same curiosity rewards as the pre-migration cusi_explore code
on a fixed saved replay. From the CUSI root (CPU is enough):

    python tests/cusi_state_regression_test.py

The fixture (tests/fixtures/cusi_state_regression.json) was written once, on 2026-10-04, by the
pre-migration code (cusi_explore.curiosity.CombinationBuffer / EmbedBuffer and
cusi_explore.encoder.FrozenEncoder, never committed), from the same inputs.

Inputs: the stored (float16) SigLIP embeddings and texts of storage_dir/explore/gameboy/dev_viridian
(frame novelty; cosine / distance / hinge) and storage_dir/explore/web/dev_arxiv (frame + text
novelty, text_alpha 0.5); a synthetic buffer that triggers KMeans compaction; and SigLIP 2
embeddings of four GameBoy frames.
"""
import json
import os
import click
import numpy as np
import torch
from cusi_utils.parameter_handling import load_parameters

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "cusi_state_regression.json")
SIGLIP = "google/siglip2-base-patch16-224"
CASES = [("gameboy", "dev_viridian", "cosine", 0.0, None), ("gameboy", "dev_viridian", "distance", 0.0, 10),
         ("gameboy", "dev_viridian", "hinge", 0.0, 10), ("web", "dev_arxiv", "cosine", 0.5, None)]


def replay_episodes(*, parameters: dict, env_name: str, run: str, max_episodes):
    from cusi_explore.replay import load_episodes
    eps = load_episodes(directory=os.path.join(parameters["storage_dir"], "explore", env_name, run, "replay"),
                        load_frames=False)
    return eps[:max_episodes] if max_episodes else eps


def new_rewards(*, episodes, env_name: str, metric: str, text_alpha: float) -> list:
    from cusi_state import StateEmbedding, StateRecord, build_scorer, build_text_embedder, NoveltyArchive
    from cusi_explore.curiosity import CuriosityModule
    text = build_text_embedder(text_embedder="none")
    scorer = build_scorer(novelty_scorer="combination", region_alpha=text_alpha, env_name=env_name)
    module = CuriosityModule(scorer=scorer, archive=NoveltyArchive(metric=metric, text_embedder=text),
                             invalid_action_penalty=0.0)
    out = []
    for ep in episodes:
        module.reset()
        prev = None
        for row in ep:
            emb = StateEmbedding(image=torch.tensor(np.asarray(row["embedding"], dtype=np.float32)), text=None,
                                 image_name="siglip", text_name="none")
            state = StateRecord(obs={"texts": row["texts"]}, info={"valid": True}, embedding=emb)
            r = module.get_reward(prev=prev, action=None, next=state)
            out.append({"frame": r["frame"], "text": r["region"], "total": r["total"]})
            prev = state
    return out


def synthetic() -> list:
    rng = np.random.default_rng(0)
    data = torch.nn.functional.normalize(torch.tensor(rng.normal(size=(260, 16)), dtype=torch.float32), dim=-1)
    from cusi_state import NoveltyArchive, StateEmbedding, build_text_embedder
    arch = NoveltyArchive(metric="cosine", text_embedder=build_text_embedder(text_embedder="none"), max_size=200)
    for i in range(len(data)):
        arch.add(StateEmbedding(image=data[i], text=None, image_name="x", text_name="none"))
    probe = data[:5] + 0.05
    return [arch.novelty(StateEmbedding(image=probe[i], text=None, image_name="x", text_name="none"))
            for i in range(5)] + [len(arch)]


def gameboy_frames(*, parameters: dict) -> list:
    from cusi_explore.replay import load_episodes
    ep = load_episodes(directory=os.path.join(parameters["storage_dir"], "explore", "gameboy", "dev_viridian",
                                              "replay"))[0]
    return [np.asarray(ep[i]["frame"].pil().convert("RGB")) for i in (0, 5, 10, 20)]


def check(*, cond: bool, msg: str) -> None:
    print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        raise SystemExit(1)


@click.command()
def main():
    parameters = load_parameters()
    with open(FIXTURE) as f:
        fixture = json.load(f)
    for case in fixture["cases"]:
        eps = replay_episodes(parameters=parameters, env_name=case["env"], run=case["run"],
                              max_episodes=case["max_episodes"])
        new = new_rewards(episodes=eps, env_name=case["env"], metric=case["metric"], text_alpha=case["alpha"])
        old = case["rewards"]
        diff = max(abs(a[k] - b[k]) for a, b in zip(old, new) for k in ("frame", "text", "total"))
        check(cond=len(old) == len(new) and diff < 1e-6,
              msg=f"{case['env']}/{case['run']} {case['metric']} alpha={case['alpha']}: {len(new)} steps, "
                  f"max |diff| {diff:.2e}")
    new = synthetic()
    diff = max(abs(a - b) for a, b in zip(fixture["synthetic"], new))
    check(cond=diff < 1e-6, msg=f"KMeans compaction (200 -> 100): {new[-1]} entries, max |diff| {diff:.2e}")
    from cusi_state.encoders import build_image_embedder
    emb = build_image_embedder(image_embedder="siglip", env_name="gameboy", encoder_model=SIGLIP, device="cpu",
                               parameters=parameters)
    got = emb.embed(frames=gameboy_frames(parameters=parameters))
    diff = float((got - torch.tensor(fixture["siglip"])).abs().max())
    check(cond=diff < 1e-4, msg=f"siglip embeddings of 4 GameBoy frames: max |diff| {diff:.2e}")
    print("ALL OK")


if __name__ == "__main__":
    main()
