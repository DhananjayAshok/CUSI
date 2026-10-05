"""
Quick checks of the generalised cusi_state pieces (curiosity_plan §6 step 2) on synthetic data, CPU:

    python tests/cusi_state_unit_test.py

random_patch == GameBoyRL's PatchProjection on GameBoy; canvases for all envs; overlap / tfidf
similarities; archive cells, copy/restore, save/load; region scorers (text lines, GameBoy OCR).
"""
import os
import sys
import tempfile
import numpy as np
import torch


def check(cond: bool, msg: str) -> None:
    print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        raise SystemExit(1)


def main():
    from cusi_state import (NoveltyArchive, StateEmbedding, StateEncoder, StateRecord, build_image_embedder,
                            build_scorer, build_text_embedder, to_canvas)
    rng = np.random.default_rng(0)
    gb = [np.repeat(rng.integers(0, 255, (144, 160, 1), dtype=np.uint8), 3, axis=2) for _ in range(3)]

    # random_patch vs GameBoyRL
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GameBoyRL", "cleanrl"))
    import importlib.util
    path = os.path.join(sys.path[0], "cleanrl_utils", "port_gameboy_worlds", "embedders.py")
    src = open(path).read().replace("from .utils import FRAME_STACK", "FRAME_STACK = 2")
    mod = type(sys)("gbrl_embedders")
    exec(compile(src, path, "exec"), mod.__dict__)
    ref = mod.PatchProjection(seed=1).embed(np.stack([f[:, :, 0] for f in gb]).astype(np.float32))
    ours = build_image_embedder(image_embedder="random_patch", env_name="gameboy").embed(frames=gb)
    check(ours.shape == (3, 720) and float((ours - ref).abs().max()) < 1e-5,
          f"random_patch on GameBoy == GameBoyRL PatchProjection (max diff {float((ours - ref).abs().max()):.1e})")
    for env, raw in (("android", (2400, 1080, 3)), ("web", (768, 1024, 3))):
        frame = rng.integers(0, 255, raw, dtype=np.uint8)
        emb = build_image_embedder(image_embedder="random_patch", env_name=env)
        v = emb.embed(frames=[frame])
        check(to_canvas(frame=frame, env_name=env).shape == emb.canvas and v.shape[1] == emb.output_dim
              and abs(float(v.norm()) - 1) < 1e-5, f"{env}: canvas {emb.canvas}, random_patch dim {emb.output_dim}")

    # text embedders
    a, b, c = ["Settings", "Wi-Fi", "Bluetooth"], ["Settings", "Wi-Fi", "Display"], ["Contacts", "Add contact"]
    ov = build_text_embedder(text_embedder="overlap")
    s = ov.similarities(query=ov.represent(lines=a), reps=[ov.represent(lines=x) for x in (a, b, c)])
    check(abs(s[0] - 1) < 1e-6 and abs(s[1] - 0.5) < 1e-6 and s[2] == 0, f"overlap similarities {s.round(3).tolist()}")
    tf = build_text_embedder(text_embedder="tfidf")
    reps = [tf.represent(lines=x) for x in (a, b, c)]
    s = tf.similarities(query=reps[0], reps=reps)
    check(abs(s[0] - 1) < 1e-5 and 0 < s[1] < 1 and s[2] == 0, f"tfidf similarities {s.round(3).tolist()}")

    # encoder + archive with cells, text weighting
    enc = StateEncoder(image=build_image_embedder(image_embedder="random_patch", env_name="gameboy"),
                       text=ov, w_image=0.5)
    obs = [{"frame": gb[i % 3], "texts": {"t": "\n".join([a, b, c][i % 3])}} for i in range(6)]
    embs = enc.encode(obs_list=obs, info_list=[{}] * 6)
    arch = NoveltyArchive(metric="cosine", text_embedder=ov, w_image=0.5, cell_threshold=0.9)
    cells = [arch.cell_of(e) for e in embs]
    added = [arch.add(e) for e in embs]
    check(cells == [0, 1, 2, 0, 1, 2] and arch.cell_counts == [2, 2, 2], f"cells {cells}, counts {arch.cell_counts}")
    check(added == [True] * 3 + [False] * 3 and len(arch) == 3, f"dedup: added {added}")
    check(arch.novelty(embs[0]) < 1e-5 and abs(enc.similarity(a=embs[0], b=embs[0]) - 1) < 1e-5, "novelty of a stored state is 0")
    snap = arch.copy()
    arch.add(StateEmbedding(image=torch.nn.functional.normalize(torch.randn(720), dim=0), text=frozenset(),
                            image_name="random_patch", text_name="overlap"))
    arch.restore(snap)
    check(len(arch) == 3, "copy / restore")
    with tempfile.TemporaryDirectory() as d:
        arch.save(directory=d)
        arch2 = NoveltyArchive(metric="cosine", text_embedder=ov, w_image=0.5, cell_threshold=0.9)
        arch2.load_from(directory=d)
        check(len(arch2) == 3 and arch2.cell_counts == [2, 2, 2], "save / load")

    # region scorers
    sc = build_scorer(novelty_scorer="region", env_name="web")
    r1 = sc.score(prev=None, action=None, next=StateRecord(obs=obs[0], info={}), archive=arch)[0]
    r2 = sc.score(prev=None, action=None, next=StateRecord(obs=obs[1], info={}), archive=arch)[0]
    check(r1 == 1.0 and abs(r2 - 1 / 3) < 1e-6, f"text-line region novelty {r1}, {r2:.3f}")
    gbsc = build_scorer(novelty_scorer="region", env_name="gameboy")
    crop = rng.integers(0, 255, (1, 1, 40, 160, 1), dtype=np.uint8)
    info = {"text_regions": {"ocr_regions": {"dialogue": (crop[0],)}}}
    v = [gbsc.score(prev=None, action=None, next=StateRecord(obs={}, info=info), archive=arch)[0] for _ in range(2)]
    v.append(gbsc.score(prev=None, action=None, next=StateRecord(obs={}, info={}), archive=arch)[0])
    check(v == [1.0, 0.0, 0.0], f"GameBoy OCR region novelty: new, repeated, none -> {v}")
    comb = build_scorer(novelty_scorer="combination", env_name="web", region_alpha=0.5)
    value, comp = comb.score(prev=None, action=None, next=StateRecord(obs=obs[2], info={}, embedding=embs[2]),
                             archive=arch)
    check(abs(value - 0.5 * comp["region"] - 0.5 * comp["frame"]) < 1e-6, f"combination {value:.3f} = {comp}")
    print("ALL OK")


if __name__ == "__main__":
    main()
