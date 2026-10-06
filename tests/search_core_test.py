"""
The cusi.search core runs with cusi.search.toy made unimportable (the strip-out check of
plans/skill_discovery.md §1.5), on a fake env with saved states and an inline stats provider. CPU:

    python -m tests.search_core_test      (from the repo root)
"""
import importlib.abc
import sys
import numpy as np


class BlockToy(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "cusi.search.toy" or name.startswith("cusi.search.toy."):
            raise ImportError(f"blocked for the strip-out test: {name}")
        return None


class FakeEnv:
    """A 1-D corridor of 30 positions; each position has its own frame. Actions: L, R, X (no-op)."""
    env_description = "a fake corridor"

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.frames = [np.random.default_rng(i).integers(0, 255, (144, 160, 3), dtype=np.uint8) for i in range(30)]
        self.pos, self.states, self.current_obs = 0, {}, None

    def _obs(self):
        return {"frame": self.frames[self.pos], "texts": {}, "actions": "L, R or X", "goal": ""}

    def reset(self, seed=None):
        self.pos = 0
        self.current_obs = self._obs()
        return self.current_obs, {"valid": True}

    def step(self, action):
        self.pos = max(0, min(29, self.pos + {"L": -1, "R": 1}.get(action, 0)))
        self.current_obs = self._obs()
        return self.current_obs, 0.0, False, False, {"valid": action in "LRX"}

    def sample_action(self):
        return str(self.rng.choice(["L", "R", "X"]))

    def save_state(self):
        sid = f"s{len(self.states)}"
        self.states[sid] = self.pos
        return sid

    def load_state(self, *, state_id):
        self.pos = self.states[state_id]
        self.current_obs = self._obs()


class Stats:
    def __init__(self, archive):
        self.archive = archive

    def q(self, node):
        return sum(node.yields) / len(node.yields) if node.yields else 0.0

    def n(self, node):
        return self.archive.cell_counts[node.cell]

    def n_total(self):
        return sum(self.archive.cell_counts)

    def expansion_yield(self, *, expansion):
        return sum(s.new_cell for s in expansion.steps)

    def record_yield(self, *, tree, node_id, value):
        for nid in tree.ancestors(node_id):
            tree.nodes[nid].yields.append(value)


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        raise SystemExit(1)


def main():
    sys.meta_path.insert(0, BlockToy())
    from cusi.state import build_archive, build_encoder, build_scorer_from_config
    from cusi.search.expanders import RandomExpander
    from cusi.search.search import TreeSearch
    from cusi.search.selection import EnergySelector
    try:
        import cusi.search.toy  # noqa: F401
        check(False, "toy import was not blocked")
    except ImportError:
        pass
    config = {"image_embedder": "random_patch", "text_embedder": "none", "novelty_scorer": "embedding"}
    encoder = build_encoder(config=config, env_name="gameboy")
    archive = build_archive(encoder=encoder, cell_threshold=0.999)
    stats = Stats(archive)
    search = TreeSearch(envs={"a": FakeEnv(0), "b": FakeEnv(1)}, encoder=encoder,
                        scorer=build_scorer_from_config(config=config, env_name="gameboy", encoder=encoder),
                        archive=archive, expander=RandomExpander(), stats=stats,
                        selector=EnergySelector(stats=stats, c=1.0, tau=2.0, seed=0), k_steps=5, node_threshold=0.3)
    reason = search.run(stop=lambda s: "done" if s.iteration >= 20 else None)
    tree = search.tree
    check(reason == "done" and search.iteration == 20 and search.total_steps == 100, f"20 iterations, {search.total_steps} steps")
    check(len(tree.roots) == 2 and len(tree.nodes) > 20, f"tree grew to {len(tree.nodes)} nodes")
    check(archive.n_cells <= 30 and archive.n_cells > 5, f"{archive.n_cells} cells (30 positions exist)")
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
    print("ALL OK")


if __name__ == "__main__":
    main()
