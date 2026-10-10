"""
Checks the cusi.search core on a fake env: search, selection, TreeStats, skipped restores, replay restores
(and divergence), challenge pages, and SceneEnvs.
    python -m tests.search_core_test
"""
import json
import os
import numpy as np


class FakeEnv:
    """A 1-D corridor of 30 positions, each with its own frame; actions L, R, X (no-op). Positions in `challenge`
    report info["challenge"] (a bot-check page)."""
    env_description = "a fake corridor"

    def __init__(self, seed: int, challenge=()) -> None:
        self.seed, self.rng, self.challenge = seed, np.random.default_rng(seed), set(challenge)
        self.frames = [np.random.default_rng(i).integers(0, 255, (144, 160, 3), dtype=np.uint8) for i in range(30)]
        self.pos, self.states, self.current_obs = 0, {}, None
        self.n_loads = self.n_resets = 0
        self.closed = False

    def _obs(self):
        return {"frame": self.frames[self.pos], "texts": {}, "actions": "L, R or X", "goal": ""}

    @property
    def raw_frame(self):
        return self.current_obs["frame"]

    def reset(self, seed=None):
        self.pos = 0
        self.n_resets += 1
        self.current_obs = self._obs()
        return self.current_obs, {"valid": True, "challenge": self.pos in self.challenge}

    def step(self, action):
        action = action.strip().splitlines()[-1] if action.strip() else ""     # a VLM reply ends with the action
        self.pos = max(0, min(29, self.pos + {"L": -1, "R": 1}.get(action, 0)))
        self.current_obs = self._obs()
        return self.current_obs, 0.0, False, False, {"valid": action in ("L", "R", "X"),
                                                     "challenge": self.pos in self.challenge}

    def sample_action(self, *, rng=None):
        return str((self.rng if rng is None else rng).choice(["L", "R", "X"]))

    def save_state(self):
        sid = f"s{len(self.states)}"
        self.states[sid] = self.pos
        return sid

    def load_state(self, *, state_id):
        self.n_loads += 1
        self.pos = self.states[state_id]
        self.current_obs = self._obs()

    def export_state(self, *, state_id, directory):
        name = f"env{self.seed}_{state_id}.json"     # ids repeat across fake envs; real envs use uuids
        with open(os.path.join(directory, name), "w") as f:
            json.dump({"pos": self.states[state_id]}, f)
        return name

    def import_state(self, *, path):
        sid = f"s{len(self.states)}"
        with open(path) as f:
            self.states[sid] = json.load(f)["pos"]
        return sid

    def close(self):
        self.closed = True


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        raise SystemExit(1)


def build(*, envs=None, restore="state", challenge=(), node_threshold=0.3):
    from cusi.state import build_archive, build_encoder, build_scorer_from_config
    from cusi.search.expanders import RandomExpander
    from cusi.search.search import TreeSearch
    from cusi.search.selection import EnergySelector
    from cusi.search.stats import TreeStats
    config = {"image_embedder": "random_patch", "text_embedder": "none", "novelty_scorer": "embedding"}
    encoder = build_encoder(config=config, env_name="gameboy")
    stats = TreeStats(q_ema=0.3)
    envs = envs if envs is not None else {"a": FakeEnv(0, challenge), "b": FakeEnv(1, challenge)}
    return TreeSearch(envs=envs, encoder=encoder,
                      scorer=build_scorer_from_config(config=config, env_name="gameboy", encoder=encoder),
                      archive=build_archive(encoder=encoder), expander=RandomExpander(), stats=stats,
                      selector=EnergySelector(stats=stats, c=1.0, tau=2.0, seed=0), k_steps=5,
                      node_threshold=node_threshold, restore=restore)


def shape(search) -> list:
    """The tree as actions and values, independent of how states were restored."""
    return [(e.node_id, list(e.children), [(s.action, s.value, s.node_id) for s in e.steps])
            for e in search.tree.expansions]


def core():
    from cusi.search.selection import EnergySelector
    search = build()
    archive, stats = search.archive, search.stats
    reason = search.run(stop=lambda s: "done" if s.iteration >= 20 else None)
    tree = search.tree
    check(reason == "done" and search.iteration == 20 and search.total_steps == 100, f"20 iterations, {search.total_steps} steps")
    check(len(tree.roots) == 2 and len(tree.nodes) > 20, f"tree grew to {len(tree.nodes)} nodes")
    check(5 < len(archive) <= 30, f"archive holds {len(archive)} states (30 positions exist; repeats deduplicated)")
    check(stats.n_total() == 20 and sum(n.n_expanded for n in tree.nodes) == 20, "N_total counts the 20 expansions")
    qs = [stats.q(n) for n in tree.nodes]
    check(all(-1e-6 <= q <= 1.0 + 1e-6 for q in qs), f"Q in [0, 1] (range {min(qs):.3f}..{max(qs):.3f})")
    fresh = [n for n in tree.nodes if n.parent is not None and not n.yields]
    check(all(stats.q(n) == n.value for n in fresh), f"{len(fresh)} unexpanded nodes take Q from their creating step")
    root = tree.roots[0]
    ema = None
    for y in root.yields:
        ema = y if ema is None else 0.7 * ema + 0.3 * y
    check(len(root.yields) > 0 and abs(stats.q(root) - ema) < 1e-9,
          f"root Q is the moving average of its subtree's {len(root.yields)} yields")
    exp = search.tree.expansions[0]
    check(abs(exp.yield_value - sum(s.value for s in exp.steps) / len(exp.steps)) < 1e-9, "yield = mean step novelty")
    deep = max(tree.nodes, key=lambda n: n.depth)
    path = tree.path_steps(deep.id)
    env = search.envs[deep.env_key]
    env.reset()
    for s in path:
        env.step(s.action)
    check(np.array_equal(env.current_obs["frame"], search.envs[deep.env_key].frames[env.states[deep.state_id]]),
          f"replaying the root->node path of node {deep.id} (depth {deep.depth}, {len(path)} steps) reaches its saved state")
    picked = {r["selected"] for r in search.records}
    check(len(picked) > 3 and all(abs(sum(r["probs"]) - 1) < 1e-6 for r in search.records),
          f"selection spread over {len(picked)} nodes, probabilities sum to 1")
    roots_only = EnergySelector(stats=stats, roots_only=True, seed=0)
    check(all(roots_only.select(tree=tree).node_id in (0, 1) for _ in range(20)), "roots_only selects only roots")
    return search


def skip_and_replay(reference):
    skipped = sum(r.get("restore") == "skipped" for r in reference.records)
    loads = sum(e.n_loads for e in reference.envs.values())
    check(skipped > 0 and loads == 20 - skipped, f"{skipped} of 20 restores skipped (env already there), {loads} loads")
    no_skip = build()
    real = no_skip._restore_node

    def never_skip(*, node, env):
        no_skip._at.clear()
        return real(node=node, env=env)
    no_skip._restore_node = never_skip
    no_skip.run(stop=lambda s: "done" if s.iteration >= 20 else None)
    check(shape(no_skip) == shape(reference), "skipping restores gives the same tree as always restoring")

    replay = build(restore="replay")
    replay.run(stop=lambda s: "done" if s.iteration >= 20 else None)
    check(shape(replay) == shape(reference), "restore=replay gives the same tree as saved states")
    check(all(not e.states for e in replay.envs.values()) and all(n.replay and n.state_id is None for n in replay.tree.nodes),
          "restore=replay saves no states")
    kinds = {r["restore"] for r in replay.records}
    sims = [r["replay_similarity"] for r in replay.records if r["restore"] == "replay"]
    check(kinds <= {"replay", "skipped"} and sims and min(sims) > 0.999,
          f"{len(sims)} replays, similarity >= {min(sims):.4f}")
    node = max(replay.tree.nodes, key=lambda n: n.depth)
    other = next(n for n in replay.tree.nodes if not np.array_equal(n.frame.numpy(), node.frame.numpy()))
    node.embedding = other.embedding           # as if the env no longer reproduced this node
    replay._at.clear()
    check(replay._restore_node(node=node, env=replay.envs[node.env_key]) is None
          and node.flags.get("replay_diverged") and not node.restorable, "a replay that diverges flags the node unselectable")


def challenges():
    search = build(challenge={1, 2, 3}, node_threshold=0.0)    # every step a node, so challenge steps would be too
    search.run(stop=lambda s: "done" if s.iteration >= 30 else None)
    steps = [s for e in search.tree.expansions for s in e.steps if s.info.get("challenge")]
    check(steps and all(s.value == 0.0 for s in steps), f"{len(steps)} challenge steps scored 0")
    encoder = search.encoder
    stored = [search.archive.max_similarity(encoder.encode_one(obs={"frame": search.envs["a"].frames[p], "texts": {}},
                                                               info={})) for p in (1, 2, 3)]
    check(max(stored) < 0.999, f"challenge frames never enter the archive (max similarity {max(stored):.3f})")
    flagged = [n for n in search.tree.nodes if n.flags.get("challenge_page")]
    check(all(not n.restorable and n.state_id is None for n in flagged), f"{len(flagged)} challenge nodes, unselectable")
    nonend = [s for e in search.tree.expansions for i, s in enumerate(e.steps)
              if s.info.get("challenge") and i < len(e.steps) - 1 and s.node_id is not None]
    check(not nonend, "a challenge step only becomes a node at the end of an expansion")
    picked = {r["selected"] for r in search.records}
    check(not picked & {n.id for n in flagged}, "challenge nodes are never selected")


class FakeVLM:
    """Records every prompt; answers a random corridor action."""

    def __init__(self):
        self.prompts, self.rng = [], np.random.default_rng(0)

    def infer(self, *, texts, images, max_new_tokens, temperature):
        self.prompts.append(texts)
        return {"output": f"I will explore.\n{self.rng.choice(['L', 'R', 'X'])}"}


def section(prompt: str, title: str) -> str:
    """The bullet block under a prompt heading that starts with title."""
    block = prompt.split(title, 1)[1].split("\n\n", 1)[0]
    return block.split("\n", 1)[1] if "\n" in block else ""


class FakeDescriberVLM:
    """Records the images and prompt of each describe call; answers 'transition <n>'."""

    def __init__(self):
        self.calls = []

    def infer(self, *, texts, images, max_new_tokens, temperature):
        self.calls.append((texts, images))
        return {"output": f"transition {len(self.calls) - 1}"}


def vlm_context():
    from cusi.search.context import action_line
    from cusi.search.describe import TransitionDescriber
    from cusi.search.expanders import VLMExplorer
    search = build()
    vlm, describer_vlm = FakeVLM(), FakeDescriberVLM()
    search.expander = VLMExplorer(vlm=vlm)
    search.describer = TransitionDescriber(vlm=describer_vlm)
    search.run(stop=lambda s: "done" if s.iteration >= 20 else None)
    tree, k = search.tree, search.k_steps
    check(all(e.description == f"transition {e.id}" for e in tree.expansions), "every expansion is described")
    images_ok = all(len(images) == len(e.steps) + 1
                    and np.array_equal(np.asarray(images[0]), tree.nodes[e.node_id].frame.numpy())
                    and np.array_equal(np.asarray(images[-1]), e.steps[-1].frame.numpy())
                    for e, (_t, images) in zip(tree.expansions, describer_vlm.calls))
    actions_ok = all(all(f"{i + 1}. {action_line(action=s.action)}" in texts for i, s in enumerate(e.steps))
                     for e, (texts, _i) in zip(tree.expansions, describer_vlm.calls))
    check(images_ok and actions_ok, "the describer sees the start frame, every step's frame, and every action")
    first = {e.id: vlm.prompts[e.id * k] for e in tree.expansions}     # the fake env never ends an episode early
    root_exp = next(e for e in tree.expansions if tree.nodes[e.node_id].parent is None)
    check("(none: it is the start screen)" in first[root_exp.id], "a root's prompt says it is the start screen")
    ok_path = 0
    for e in tree.expansions:
        want = [action_line(action=s.action) for s in tree.path_steps(e.node_id)][-20:]
        got = [line[2:] for line in section(first[e.id], "How the screen").splitlines()
               if line.startswith("- ") and not line.endswith("earlier actions)")]
        ok_path += got == want or (not want and got == [])
    check(ok_path == len(tree.expansions), f"every expansion's prompt lists its node's root->node actions ({ok_path})")
    repeats = [e for e in tree.expansions if any(p.node_id == e.node_id and p.id < e.id for p in tree.expansions)]
    ok_tried = 0
    for e in repeats:
        earlier = [p for p in tree.expansions if p.node_id == e.node_id and p.id < e.id][-5:]
        tried = section(first[e.id], "Earlier explorations")
        lines = [l for l in tried.splitlines() if l.startswith("- ") and not l.endswith("not shown)")]
        ok_tried += len(lines) == len(earlier) and all(
            " -> ".join(action_line(action=s.action) for s in p.steps[:5]) in line
            and line.endswith(f"what happened: {p.description}") for p, line in zip(earlier, lines))
    fresh = [e for e in tree.expansions if e not in repeats]
    check(repeats and ok_tried == len(repeats),
          f"{len(repeats)} repeat expansions list every earlier one from their node, with its description")
    check(all("(none: this is the first exploration from it)" in first[e.id] for e in fresh),
          f"{len(fresh)} first expansions say so")
    print("        sample prompt:\n" + "\n".join("          " + l for l in first[repeats[-1].id].splitlines()[:22]))


def scene_envs():
    from cusi.search.envs import SceneEnvs
    made = []

    def make(*, scene):
        made.append(scene)
        return FakeEnv({"a": 0, "b": 1}[scene])
    envs = SceneEnvs(make=make, scenes=["a", "b"], exclusive=True)
    a = envs["a"]
    b = envs["b"]
    check(a.closed and not b.closed and made == ["a", "b"], "exclusive SceneEnvs closes the other scene")
    a2 = envs["a"]
    check(a2 is not a and b.closed, "and rebuilds it on return")
    search = build(envs=SceneEnvs(make=make, scenes=["a", "b"], exclusive=True), restore="replay")
    search.run(stop=lambda s: "done" if s.iteration >= 20 else None)
    check(search.iteration == 20 and not any(r.get("failed") for r in search.records),
          "an exclusive replay search runs across rebuilt envs (no stale skips)")


def main():
    reference = core()
    skip_and_replay(reference)
    challenges()
    vlm_context()
    scene_envs()
    print("ALL OK")


if __name__ == "__main__":
    main()
