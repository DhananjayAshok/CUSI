"""
Web live check of the cusi_state text side (curiosity_plan §9: text embedders, option A region
novelty, the unlabelled screenshot), a few random steps on arXiv. From the CUSI root:

    bash scripts/container.sh python tests/web_state_check.py --text_embedder_model sentence-transformers/all-MiniLM-L6-v2
"""
import click
import numpy as np


@click.command()
@click.option("--text_embedder_model", required=True, help="Dense sentence model id.")
@click.option("--n_steps", default=6)
def main(text_embedder_model, n_steps):
    from cusi_envs.webvoyager import WebVoyagerPlayEnv
    from cusi_state import (NoveltyArchive, StateEncoder, StateRecord, build_image_embedder, build_scorer,
                            build_text_embedder, element_lines)
    env = WebVoyagerPlayEnv(url="https://arxiv.org/", mode="free_play", fast_waits=True)
    image = build_image_embedder(image_embedder="random_patch", env_name="web")
    encoders = {name: StateEncoder(image=image, text=build_text_embedder(text_embedder=name,
                                                                         text_embedder_model=text_embedder_model),
                                   w_image=0.5)
                for name in ("overlap", "tfidf", "dense")}
    region = build_scorer(novelty_scorer="region", env_name="web")
    archive = NoveltyArchive()
    obs, info = env.reset(seed=0)
    states = [(obs, info)]
    for _ in range(n_steps):
        obs, _r, _t, _tr, info = env.step(env.sample_action())
        states.append((obs, info))
    ok = True
    for i, (obs, info) in enumerate(states):
        raw = info.get("raw_frame")
        diff = float(np.mean(np.any(raw != obs["frame"], axis=-1))) if raw is not None and raw.shape == obs["frame"].shape else None
        rn = region.score(prev=None, action=None, next=StateRecord(obs=obs, info=info), archive=archive)[0]
        print(f"step {i}: url {info.get('url', '')[:60]!r} valid {info.get('valid')} lines "
              f"{len(element_lines(texts=obs['texts']))} raw_frame {None if raw is None else raw.shape} "
              f"(labels change {diff if diff is None else f'{diff:.1%}'} of pixels) region novelty {rn:.3f}",
              flush=True)
        ok &= raw is not None
    for name, enc in encoders.items():
        embs = enc.encode(obs_list=[s[0] for s in states], info_list=[s[1] for s in states])
        text_sims = [float(enc.text.similarities(query=embs[i].text, reps=[embs[i - 1].text])[0])
                     for i in range(1, len(embs))]
        sims = [enc.similarity(a=embs[i], b=embs[i - 1]) for i in range(1, len(embs))]
        self_sim = enc.similarity(a=embs[0], b=embs[0])
        print(f"{name:8s} text sim to previous step: {np.round(text_sims, 3).tolist()}; combined: "
              f"{np.round(sims, 3).tolist()}; self {self_sim:.3f}", flush=True)
        ok &= abs(self_sim - 1) < 1e-4
    env.close()
    print("ALL OK" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
