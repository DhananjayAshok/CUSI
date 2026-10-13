"""
Checks SearchStore on the fake corridor env: a run crashes mid-iteration, is resumed (equal to the committed
search, partial writes dropped, states and frames restored), extended, and resumed again. Also the config guard.
    python -m tests.search_store_test <empty scratch dir>
"""
import hashlib
import json
import os
import sys
import click
import numpy as np
import torch
from tests.search_core_test import FakeEnv, check


def _h(text: str) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)


class HashVLM:
    """A deterministic VLM for describe, features and decision-score calls, answering from a hash of the prompt."""

    def infer(self, *, texts, images, max_new_tokens, temperature):
        if isinstance(texts, list):                       # decision scores: one reply per prompt
            out = []
            for t in texts:
                n = t.split("one particular screen:\n", 1)[1].split("\n\n", 1)[0].count("\n") + 1
                out.append("\n".join(f"{k + 1}. {['yes', 'partly', 'no'][(_h(t) + k) % 3]} - because"
                                      for k in range(n)))
            return {"output": out}
        h = _h(texts)
        if "Features:" in texts:
            return {"output": "Features:\n" + "\n".join(f"- feature {(h + k) % 7}" for k in range(1 + h % 3))}
        return {"output": f"Description: {len(images)} frames, {h:08x}.\nDiscovered:\n- feature {h % 7}"}


def build(*, directory, config=None, overwrite=False):
    from cusi.state import build_archive, build_encoder, build_scorer_from_config
    from cusi.search.expanders import RandomExpander
    from cusi.search.search import TreeSearch
    from cusi.search.selection import EnergySelector
    from cusi.search.stats import TreeStats
    from cusi.search.store import SearchStore
    from cusi.search.describe import TransitionDescriber
    from cusi.search.features import VLMFeatures
    from cusi.search.scoring import VLMScorer
    state_config = {"image_embedder": "random_patch", "text_embedder": "trigram", "novelty_scorer": "combination"}
    encoder = build_encoder(config=state_config, env_name="gameboy")
    stats = TreeStats(q_ema=0.3)
    store = SearchStore(directory=directory, config=config or {"k_steps": 5, "max_expansions": 10},
                        overwrite=overwrite)
    return TreeSearch(envs={"a": FakeEnv(0), "b": FakeEnv(1)}, encoder=encoder,
                      scorer=build_scorer_from_config(config=state_config, env_name="gameboy", encoder=encoder),
                      archive=build_archive(encoder=encoder), expander=RandomExpander(), stats=stats,
                      selector=EnergySelector(stats=stats, c=1.0, tau=2.0, seed=0), k_steps=5, node_threshold=0.3,
                      describer=TransitionDescriber(vlm=HashVLM()), features=VLMFeatures(vlm=HashVLM()),
                      decision_scorer=VLMScorer(vlm=HashVLM()), score_top_m=3, store=store)


def snapshot(search) -> dict:
    tree = search.tree
    return {"nodes": [(n.id, n.parent, n.depth, n.env_key, n.value, n.segment, dict(n.flags), n.created_at,
                       n.n_expanded, list(n.yields), n.features, n.text_boxes) for n in tree.nodes],
            "scores": dict(search.node_scores),
            "q": [search.stats.q(n) for n in tree.nodes], "n_total": search.stats.n_total(),
            "expansions": [(e.id, e.node_id, e.iteration, list(e.children), e.yield_value, e.description,
                            e.discovered, e.expander, e.expander_info,
                            [(s.action, s.value, s.node_id, s.info, s.text_boxes) for s in e.steps])
                           for e in tree.expansions],
            "seen_text": list(search.archive.region("unseen_text").store.keys),
            "archive_texts": search.archive.texts,
            "counters": (search.iteration, search.total_steps, len(search.records))}


def until(n):
    return lambda s: "max_expansions" if len(s.tree.expansions) >= n else None


def lines(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def main(root):
    run = os.path.join(root, "run")
    a = build(directory=run, overwrite=True)
    a.run(stop=until(10))
    committed = snapshot(a)
    frames_a = {n.id: n.frame.numpy() for n in a.tree.nodes}
    emb_a = {n.id: n.embedding.image.clone() for n in a.tree.nodes}
    archive_a = a.archive.images.clone()

    # crash after the expansion row is written but before the archive commit
    real_save = a.store._save_archive
    a.store._save_archive = lambda **kw: (_ for _ in ()).throw(RuntimeError("simulated crash"))
    try:
        while a.iterate() is not None and len(a.tree.expansions) == 10:
            pass
    except RuntimeError:
        pass
    a.store._save_archive = real_save
    check(len(lines(os.path.join(run, "expansions.jsonl"))) == 11, "the crashed iteration left an uncommitted row")

    b = build(directory=run)
    check(b.resume(), "resume found a commit")
    check(snapshot(b) == committed, "the resumed tree, stats and counters equal the committed search")
    check(len(lines(os.path.join(run, "expansions.jsonl"))) == 10
          and not os.path.exists(os.path.join(run, "frames", "10.zip"))
          and not os.path.exists(os.path.join(run, "embeddings", "10.pt")), "the uncommitted iteration was removed")
    check(torch.equal(b.archive.images, archive_a), f"archive restored ({len(b.archive)} states)")
    check(all(torch.equal(n.embedding.image, emb_a[n.id]) for n in b.tree.nodes), "node embeddings restored")
    check(all(np.array_equal(n.frame.numpy(), frames_a[n.id]) for n in b.tree.nodes), "node frames restored exactly")
    restored = 0
    for n in b.tree.nodes:
        env = b.envs[n.env_key]
        env.load_state(state_id=n.state_id)
        restored += np.array_equal(env.current_obs["frame"], frames_a[n.id])
    check(restored == len(b.tree.nodes), f"all {restored} node states import and load to their frame")
    rows = lines(os.path.join(run, "expansions.jsonl"))
    check(all(r["record"]["scores"] and r["record"]["n_discovered"] is not None and r["discovered"]
              and all(set(e) == {"node_id", "score", "verdicts", "raw"} for e in r["record"]["scores"])
              for r in rows), "every record saves its decision scores (with verdicts) and n_discovered")
    check(len(b.node_scores) > 0 and b.node_scores == committed["scores"], "the score cache is rebuilt on resume")
    check(all(n.features for n in b.tree.nodes), "node features restored")
    b.selector.reseed(iteration=b.iteration)
    a2 = build(directory=run)
    a2.resume()
    a2.selector.reseed(iteration=a2.iteration)
    check(b.selector.select(tree=b.tree, p_of=b.p_of).node_id == a2.selector.select(tree=a2.tree, p_of=a2.p_of).node_id,
          "two resumes of one commit make the same next selection")

    b.run(stop=until(20))
    exp_rows = lines(os.path.join(run, "expansions.jsonl"))
    node_rows = [r for r in lines(os.path.join(run, "nodes.jsonl")) if "parent" in r]
    check([r["id"] for r in exp_rows] == list(range(20)), "extended to 20 expansions, ids contiguous")
    check([r["id"] for r in node_rows] == list(range(len(b.tree.nodes))), f"{len(node_rows)} node rows, ids contiguous")
    x = build(directory=os.path.join(root, "uninterrupted"), overwrite=True)
    x.run(stop=until(20))
    same = snapshot(x) == snapshot(b) and torch.equal(x.archive.images, b.archive.images)
    if not same:
        sx, sb = snapshot(x), snapshot(b)
        print("differ:", [k for k in sx if sx[k] != sb[k]])
    check(same, "crashed at 10, resumed and extended to 20 == run straight to 20 (same actions, tree, stats, archive)")
    c = build(directory=run, config={"k_steps": 5, "max_expansions": 20})
    c.resume()
    check(snapshot(c) == snapshot(b), "a resume of the extended run equals it")

    try:
        build(directory=run, config={"k_steps": 6, "max_expansions": 20})
        check(False, "a changed k_steps was accepted")
    except click.UsageError:
        check(True, "a changed k_steps is a config violation; a changed max_expansions is not")
    d = build(directory=run, overwrite=True)
    check(not d.resume() and not os.path.exists(os.path.join(run, "expansions.jsonl")), "--overwrite starts fresh")
    print("ALL OK")


if __name__ == "__main__":
    main(sys.argv[1])
