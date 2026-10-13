"""
Fake-VLM checks of the search's VLM add-ons: sliced describe (plans/next_plan.md §C), node features (§A) and decision
scores (§B).
    python -m tests.search_vlm_test
"""
from tests.search_core_test import check


class Frame:
    def __init__(self, w, h):
        from PIL import Image
        self.img = Image.new("RGB", (w, h))

    def pil(self):
        return self.img


class Step:
    def __init__(self, i, w, h):
        self.action, self.frame, self.info = f"action {i}", Frame(w, h), {"valid": True}


class RecordingVLM:
    """Records calls; window calls answer 'window <first frame>', compose and single calls a Description/Discovered."""

    def __init__(self):
        self.calls = []

    def infer(self, *, texts, images, max_new_tokens, temperature):
        self.calls.append((texts, images))
        if isinstance(texts, list):
            return {"output": [f"window {t.split('are frames ', 1)[1].split('-', 1)[0]}" for t in texts]}
        return {"output": "Description: it went somewhere.\nDiscovered:\n- the map\n- The Map\n- a shop"}


def sliced_describe():
    from cusi.search.describe import PROMPT_RESERVE, TransitionDescriber, tokens_per_frame
    vlm = RecordingVLM()
    d = TransitionDescriber(vlm=vlm, max_images=16, max_model_len=32768)
    out = d(env_description="an Android phone", start_frame=Frame(1080, 2400),
            steps=[Step(i, 1080, 2400) for i in range(25)])
    batched, compose = vlm.calls[0], vlm.calls[1]
    per = tokens_per_frame(width=1080, height=2400)
    budget = 32768 - PROMPT_RESERVE - d.max_new_tokens
    wins = out["window_descriptions"]
    check(len(vlm.calls) == 2 and isinstance(batched[0], list) and len(batched[0]) == len(wins) > 1,
          f"25 Android steps: {len(wins)} windows in one batched call, then one compose call")
    check(all(len(imgs) <= 16 and len(imgs) * per <= budget for imgs in batched[1]),
          f"every window within the image ({max(len(i) for i in batched[1])} <= 16) and token limits")
    spans = [w["frames"] for w in wins]
    check(spans[0][0] == 1 and spans[-1][1] == 26 and all(a[1] == b[0] for a, b in zip(spans, spans[1:])),
          f"windows cover frames 1-26 and overlap by one frame: {spans}")
    check(compose[1] is None and all(f"Frames {a}-{b}: window {a}" in compose[0] for a, b in spans),
          "the compose call is text-only and gets every window's description")
    check(out["description"] == "it went somewhere." and out["discovered"] == ["the map", "a shop"],
          f"description and deduplicated discovered list parsed: {out['discovered']}")
    vlm2 = RecordingVLM()
    out2 = TransitionDescriber(vlm=vlm2, max_images=16, max_model_len=32768)(
        env_description="a GameBoy game", start_frame=Frame(160, 144), steps=[Step(i, 160, 144) for i in range(10)])
    check(len(vlm2.calls) == 1 and len(vlm2.calls[0][1]) == 11 and out2["window_descriptions"] is None,
          "a 10-step GameBoy expansion stays one call with 11 frames")
    vlm3 = RecordingVLM()
    TransitionDescriber(vlm=vlm3, max_images=16, max_model_len=4000)(
        env_description="an Android phone", start_frame=Frame(1080, 2400), steps=[Step(i, 1080, 2400) for i in range(3)])
    sizes = {im.size for imgs in vlm3.calls[0][1] for im in imgs}
    check(all(tokens_per_frame(width=w, height=h) <= 4000 - PROMPT_RESERVE - 300 for w, h in sizes),
          f"frames over the token budget are downscaled to fit: {sizes}")


def features_and_scores():
    from cusi.search.features import parse_features
    from cusi.search.scoring import VLMScorer, discovered_set, parse_verdicts, score_of
    feats, ok = parse_features(text="Sure.\nFeatures:\n- Wi-Fi list\n- wi-fi  list\n* Bluetooth\n2. Display\n\n",
                               max_features=8)
    check(ok and feats == ["Wi-Fi list", "Bluetooth", "Display"], f"features parsed, duplicates dropped: {feats}")
    check(parse_features(text="no idea", max_features=8) == ([], False), "unparsable features -> [] and the flag")
    check(len(parse_features(text="Features:\n" + "\n".join(f"- f{i}" for i in range(20)), max_features=8)[0]) == 8,
          "at most max_features")
    v = parse_verdicts(text="1. yes - seen\n2. Partly — half\n3. no: never", features=["a", "b", "c"])
    check([x["verdict"] for x in v] == ["yes", "partly", "no"] and abs(score_of(verdicts=v, default_score=0.5) - 0.5)
          < 1e-9, "verdict lines parsed; score = (0 + 0.5 + 1) / 3")

    class Exp:
        def __init__(self, d):
            self.discovered = d
    check(discovered_set(expansions=[Exp(["Map", "shop"]), Exp(["map", "Gym"]), Exp(None)]) == ["Map", "shop", "Gym"],
          "discovered set: union with normalised duplicates removed")

    class AllYes:
        def __init__(self):
            self.prompts = []

        def infer(self, *, texts, images, max_new_tokens, temperature):
            self.prompts += texts
            return {"output": ["\n".join(f"{k + 1}. yes - found" for k in range(2)) for _ in texts]}
    vlm = AllYes()
    entries = VLMScorer(vlm=vlm, default_score=0.5).score(items=[
        {"node_id": 3, "features": ["map", "shop"], "discovered": ["map", "shop"], "env_description": "x"},
        {"node_id": 4, "features": [], "discovered": ["map"], "env_description": "x"}])
    check(entries[0]["score"] == 0.0 and entries[1]["score"] == 0.5 and len(vlm.prompts) == 1,
          "all features discovered -> score 0; no features -> default score, without a call")
    check("- map" in vlm.prompts[0] and "1. map" in vlm.prompts[0], "the prompt lists the features and the discovered set")


def search_with_scores():
    """Decision scores in the search: top-M re-scored per decision, P(n) = cached score (1.0 before scoring)."""
    from tests.search_core_test import build
    from tests.search_store_test import HashVLM
    from cusi.search.describe import TransitionDescriber
    from cusi.search.features import VLMFeatures
    from cusi.search.scoring import VLMScorer
    search = build()
    search.features, search.describer = VLMFeatures(vlm=HashVLM()), TransitionDescriber(vlm=HashVLM())
    search.decision_scorer, search.score_top_m = VLMScorer(vlm=HashVLM()), 3
    search.run(stop=lambda s: "done" if s.iteration >= 10 else None)
    recs = search.records
    check(all(1 <= len(r["scores"]) <= 3 for r in recs), "each decision re-scores at most score_top_m candidates")
    last = recs[-1]
    p = dict(zip(last["candidates"], last["p"]))
    scored = {e["node_id"]: e["score"] for e in last["scores"]}
    check(all(abs(p[nid] - s) < 1e-12 for nid, s in scored.items()), "re-scored candidates use their new score as P")
    unscored = [nid for nid in last["candidates"] if not any(nid in {e["node_id"] for e in r["scores"]}
                                                                for r in recs)]
    check(all(p[nid] == 1.0 for nid in unscored), f"{len(unscored)} never-scored candidates count as 1.0")
    check(all(e.discovered is not None for e in search.tree.expansions)
          and all(n.features is not None for n in search.tree.nodes), "expansions carry discovered, nodes features")


def main():
    sliced_describe()
    features_and_scores()
    search_with_scores()
    print("ALL OK")


if __name__ == "__main__":
    main()
