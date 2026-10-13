"""
CPU checks of the cusi.state embedders, archive and scorers on synthetic data.
    python tests/cusi_state_unit_test.py
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


def boxes(*items):
    """(role, text, (x0, y0, x1, y1)) -> TextBox items."""
    from cusi.envs.text_boxes import TextBox
    return [TextBox(text=t, box=b, role=r, interactive=False, source="dom") for r, t, b in items]


def main():
    from cusi.state import NoveltyArchive, StateEmbedding, StateEncoder, StateRecord, build_image_embedder, \
        build_scorer, build_text_embedder, to_canvas
    from cusi.state.encoders.text import BoxSet, box_set, capped_weights, chamfer, key_cosines, normalise
    from cusi.state.state import mask_boxes
    rng = np.random.default_rng(0)
    gb = [np.repeat(rng.integers(0, 255, (144, 160, 1), dtype=np.uint8), 3, axis=2) for _ in range(3)]

    # random_patch vs GameBoyRL
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GameBoyRL", "cleanrl"))
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

    # trigram cosines: the table in plans/next_plan.md §0.2
    table = [("Wi-Fi", "WI-FI", 1.00), ("Hello! Welcome to the world of POKéMON!", "Hello! Welcome to the world of POK",
              0.91), ("Settings", "Search settings", 0.79), ("Network & internet", "Internet", 0.65),
             ("Sign in", "Sign up", 0.62), ("PIKACHU used THUNDERSHOCK!", "PIKACHU used TACKLE!", 0.59),
             ("Wi-Fi", "Wi-Fi preferences", 0.58), ("OK", "On", 0.33), ("Sign in", "Log in", 0.27)]
    for a, b, want in table:
        got = float(key_cosines(a=[f"body\t{normalise(a)}"], b=[f"body\t{normalise(b)}"])[0, 0])
        check(abs(got - want) < 0.015, f"cos({a!r}, {b!r}) = {got:.2f} (table {want:.2f})")
    check(normalise("HP 23/45") == normalise("HP 22/45") == "hp ##/##", "digits masked in keys")
    check(float(key_cosines(a=["button\tok"], b=["body\tok"])[0, 0]) == 0.0, "different roles: cosine 0")
    w = capped_weights(np.array([100.0, 1, 1, 1, 1, 1]))
    check(abs(w.sum() - 1) < 1e-6 and w.max() <= 0.25 + 1e-6, f"area weights capped at 25%: {w.round(3).tolist()}")

    # chamfer properties
    A = box_set(boxes(("heading", "Settings", (0, 0, 100, 20)), ("body", "Wi-Fi", (0, 30, 50, 40)),
                      ("body", "Bluetooth", (0, 50, 60, 60))))
    B = box_set(boxes(("heading", "Contacts", (0, 0, 100, 20)), ("button", "Add contact", (0, 30, 50, 40))))
    A_roles = box_set(boxes(("button", "Settings", (0, 0, 100, 20)), ("button", "Wi-Fi", (0, 30, 50, 40)),
                            ("button", "Bluetooth", (0, 50, 60, 60))))
    check(abs(chamfer(a=A, b=A) - 1) < 1e-6, "chamfer: identical screens = 1")
    check(chamfer(a=A, b=B) < 0.2, f"chamfer: different screens {chamfer(a=A, b=B):.3f}")
    check(chamfer(a=A, b=A_roles) == 0.0, "chamfer: role mismatch ignored (0)")
    check(chamfer(a=BoxSet((), ()), b=BoxSet((), ())) is None and chamfer(a=A, b=BoxSet((), ())) == 0.0,
          "chamfer: empty vs empty undefined, one empty = 0")

    # masking changes only box pixels
    frame = rng.integers(0, 255, (100, 120, 3), dtype=np.uint8)
    frame[10:40, 10:60] = 200
    frame[20:30, 20:50] = 0                                       # "text" on a 200 background
    masked = mask_boxes(frame=frame, boxes=boxes(("body", "x", (20, 20, 50, 30))))
    outside = np.ones((100, 120), dtype=bool)
    outside[20:30, 20:50] = False
    check(np.array_equal(masked[outside], frame[outside]) and (masked[20:30, 20:50] == 200).all(),
          "masking fills the box with its ring colour and leaves the rest unchanged")

    # encoder + archive: text weighting, dedup on image AND key set, cells
    tri = build_text_embedder(text_embedder="trigram")
    enc = StateEncoder(image=build_image_embedder(image_embedder="random_patch", env_name="gameboy"),
                       text=tri, w_image=0.5, mask_text=False)
    screens = [boxes(("body", "Settings", (0, 0, 80, 8)), ("body", "Wi-Fi", (0, 10, 40, 18))),
               boxes(("body", "Contacts", (0, 0, 80, 8))), []]
    obs = [{"frame": gb[i % 3], "texts": {}, "text_boxes": screens[i % 3]} for i in range(6)]
    obs.append({"frame": gb[0], "texts": {}, "text_boxes": screens[1]})        # same image, other text
    embs = enc.encode(obs_list=obs, info_list=[{}] * 7)
    arch = NoveltyArchive(metric="cosine", text_embedder=tri, w_image=0.5, cell_threshold=0.9)
    cells = [arch.cell_of(e) for e in embs[:6]]
    added = [arch.add(e) for e in embs]
    check(cells == [0, 1, 2, 0, 1, 2] and arch.cell_counts == [2, 2, 2], f"cells {cells}, counts {arch.cell_counts}")
    check(added == [True] * 3 + [False] * 3 + [True] and len(arch) == 4,
          f"dedup needs the same image and key set: added {added}")
    check(arch.novelty(embs[0]) < 1e-5 and abs(enc.similarity(a=embs[0], b=embs[0]) - 1) < 1e-5,
          "novelty of a stored state is 0")
    sims = arch.similarities(embs[2])                      # a text-less screen: image only against text-less states
    check(abs(float(sims[2]) - 1) < 1e-5, "both screens without text: image similarity alone")

    # the gather against a brute-force loop on a random archive
    words = ["alpha", "beta", "gamma", "delta", "settings", "search", "sign in", "sign up", "log in", "home",
             "news", "about", "contact", "help", "#### items", "page ##"]
    roles = ["body", "link", "button", "heading"]
    big = NoveltyArchive(metric="cosine", text_embedder=tri, w_image=0.5)
    for i in range(300):
        n = int(rng.integers(0, 8))
        bx = boxes(*[(roles[rng.integers(4)], " ".join(rng.choice(words, rng.integers(1, 4))),
                      (0, 10 * j, int(rng.integers(5, 200)), 10 * j + 8)) for j in range(n)])
        big.add(StateEmbedding(image=torch.nn.functional.normalize(torch.randn(720), dim=0), text=box_set(bx),
                               image_name="random_patch", text_name="trigram"))
    q = box_set(boxes(("body", "alpha beta", (0, 0, 100, 8)), ("link", "sign in", (0, 10, 30, 18)),
                      ("button", "home", (0, 20, 20, 28))))
    fast = big.text_similarities(q)
    slow = np.array([np.nan if (v := chamfer(a=q, b=big.text_of(i))) is None else v for i in range(len(big))])
    check(np.allclose(fast, slow, atol=1e-5, equal_nan=True), f"padded gather == brute-force chamfer on {len(big)} states")
    empty_fast = big.text_similarities(BoxSet((), ()))
    check(all((np.isnan(e) if len(big.state_keys[i]) == 0 else e == 0) for i, e in enumerate(empty_fast)),
          "empty query: NaN against text-less states, 0 otherwise")
    before = {i: big.text_of(i) for i in range(len(big))}
    big.compact_to = 50
    big.compact()
    check(len(big) == 50 and all(big.text_of(i) in before.values() for i in range(len(big)))
          and len(big.store) == len({k for i in range(len(big)) for k in big.text_of(i).keys}),
          f"compaction keeps whole states' keys and drops unused keys ({len(big.store)} keys)")

    snap = arch.copy()
    arch.add(StateEmbedding(image=torch.nn.functional.normalize(torch.randn(720), dim=0), text=BoxSet((), ()),
                            image_name="random_patch", text_name="trigram"))
    arch.restore(snap)
    check(len(arch) == 4, "copy / restore")
    with tempfile.TemporaryDirectory() as d:
        arch.save(directory=d)
        arch2 = NoveltyArchive(metric="cosine", text_embedder=tri, w_image=0.5, cell_threshold=0.9)
        arch2.load_from(directory=d)
        check(len(arch2) == 4 and arch2.cell_counts == [2, 2, 2] and arch2.texts == arch.texts
              and torch.allclose(arch2.similarities(embs[1]), arch.similarities(embs[1])), "save / load (format 2)")

    # unseen-text region scorer
    sc = build_scorer(novelty_scorer="region", env_name="web")
    rec = lambda bx: StateRecord(obs={"text_boxes": bx}, info={})
    r1 = sc.score(prev=None, action=None, next=rec(screens[0]), archive=arch)[0]
    r2 = sc.score(prev=None, action=None, next=rec(screens[0] + screens[1]), archive=arch)[0]
    r3 = sc.score(prev=None, action=None, next=rec(boxes(("body", "WI-FI", (0, 0, 40, 8)))), archive=arch)[0]
    check(r1 == 1.0 and 0 < r2 < 1 and r3 == 0.0, f"unseen text: new {r1}, half new {r2:.3f}, case variant {r3}")
    line = "Hello! Welcome to the world of POKEMON!"
    stream = [line[:n] for n in (6, 14, 22, 30, len(line))]
    values, prev = [], None
    for part in stream:
        nxt = rec(boxes(("dialogue", part, (5, 100, 155, 140))))
        values.append(sc.score(prev=prev, action=None, next=nxt, archive=arch)[0])
        prev = nxt
    check(values[0] == 1.0 and all(0 < b < a for a, b in zip(values, values[1:])) and sum(values) < 2.5,
          f"streaming dialogue credited once plus its new characters: {[round(v, 2) for v in values]}")
    replay = [sc.score(prev=None, action=None, next=rec(boxes(("dialogue", p, (5, 100, 155, 140)))), archive=arch)[0]
              for p in stream]
    check(replay == [0.0] * len(stream), f"a known line typed again scores 0: {replay}")
    comb = build_scorer(novelty_scorer="combination", env_name="web", region_alpha=0.5)
    value, comp = comb.score(prev=None, action=None, next=StateRecord(obs=obs[2], info={}, embedding=embs[2]),
                             archive=arch)
    check(abs(value - 0.5 * comp["region"] - 0.5 * comp["frame"]) < 1e-6, f"combination {value:.3f} = {comp}")
    print("ALL OK")


if __name__ == "__main__":
    main()
