"""
Fake-env checks of --expander curiosity (plans/next_plan.md §E.7, item 4): the committed expansion is the session's
best episode; in `global` mode the session rewards equal the committed step values; rejected episodes never touch the
global archive; a repeated session (same seeds) is identical; parallel copies and saved policies work.
Needs the frozen encoders (MobileNetV3-Small, MiniLM) in HF_HOME.
    python -m tests.curiosity_expander_test --policy_text_encoder <MiniLM id> [--store_dir <scratch dir>]
"""
import os
import click
import numpy as np
import torch
from tests.search_core_test import FakeEnv, check

BUTTON_TO_MOVE = {"LEFT": "L", "RIGHT": "R"}


class ButtonEnv(FakeEnv):
    """The fake corridor driven by GameBoy buttons: LEFT / RIGHT move, the rest do nothing."""

    def step(self, action):
        return super().step(BUTTON_TO_MOVE.get(action, "X"))


def build(*, frozen, reward_buffer="global", num_envs=1, total_steps=600, store=None, seed=0, restore="state",
          keep_episodes=1):
    from cusi.state import build_archive, build_encoder, build_scorer_from_config
    from cusi.search.curiosity.expander import CuriosityExpander
    from cusi.search.curiosity.session import PPOConfig
    from cusi.search.search import TreeSearch
    from cusi.search.selection import EnergySelector
    from cusi.search.stats import TreeStats
    config = {"image_embedder": "random_patch", "text_embedder": "trigram", "novelty_scorer": "combination"}
    encoder = build_encoder(config=config, env_name="gameboy")
    stats = TreeStats(q_ema=0.3)
    expander = CuriosityExpander(env_name="gameboy", frozen=frozen, reward_buffer=reward_buffer, seed=seed,
                                 config=PPOConfig(total_steps=total_steps, max_minutes=10, k_steps=8,
                                                  num_envs=num_envs, rollout=32, keep_episodes=keep_episodes),
                                 make_copy=lambda key: ButtonEnv({"a": 0, "b": 1}[key]) if num_envs > 1 else None)
    return TreeSearch(envs={"a": ButtonEnv(0), "b": ButtonEnv(1)}, encoder=encoder,
                      scorer=build_scorer_from_config(config=config, env_name="gameboy", encoder=encoder),
                      archive=build_archive(encoder=encoder), expander=expander, stats=stats,
                      selector=EnergySelector(stats=stats, c=1.0, tau=0.5, seed=seed), k_steps=8,
                      node_threshold=0.05, store=store, restore=restore)


@click.command()
@click.option("--policy_text_encoder", required=True)
@click.option("--store_dir", default=None)
def main(policy_text_encoder, store_dir):
    from cusi.search.curiosity.policy import FrozenEncoders, FusionPolicy
    frozen = FrozenEncoders(vision_encoder="mobilenet_v3_small", text_encoder=policy_text_encoder, device="cpu",
                            long_side=224)
    n_params = sum(p.numel() for p in FusionPolicy().parameters())
    check(n_params < 1_000_000, f"trained part has {n_params:,} parameters (< 1M)")

    s = build(frozen=frozen)
    s.add_roots()
    sizes = []
    real_add = s.archive.add

    def counting_add(e):
        sizes.append(1)
        return real_add(e)
    s.archive.add = counting_add
    s.iterate()
    exp = s.tree.expansions[0]
    info = exp.expander_info
    best = s.expander._best
    check(exp.expander == "curiosity" and info["episodes"] > 10 and info["total_steps"] >= 600,
          f"one session: {info['total_steps']} steps, {info['episodes']} episodes, {info['updates']} updates")
    check([st.action for st in exp.steps] == best.actions and abs(info["best_return"] - best.ret) < 1e-9,
          f"the committed expansion is the best episode (episode {best.index}, return {best.ret:.3f})")
    check(len(sizes) == len(exp.steps), f"the global archive was only added to at commit ({len(sizes)} adds for "
          f"{len(exp.steps)} committed steps; the session ran {info['total_steps']})")
    session_rewards = [st.extra["curiosity"]["session_reward"] for st in exp.steps]
    values = [st.value for st in exp.steps]
    check(np.allclose(session_rewards, values, atol=1e-6),
          f"global mode: session rewards == committed step values {np.round(values, 3).tolist()}")
    check(all(set(st.extra["curiosity"]) == {"session_reward", "logp", "value", "entropy", "n_candidates",
                                               "candidate"} for st in exp.steps)
          and all(st.extra["curiosity"]["n_candidates"] == 7 for st in exp.steps), "Step.extra['curiosity'] fields; 7 buttons")

    s2 = build(frozen=frozen)
    s2.add_roots()
    s2.iterate()
    check([st.action for st in s2.tree.expansions[0].steps] == [st.action for st in exp.steps]
          and s2.tree.expansions[0].expander_info["best_return"] == info["best_return"],
          "a repeated session (same seed and iteration) is identical")

    loc = build(frozen=frozen, reward_buffer="local")
    loc.add_roots()
    loc.iterate()
    le = loc.tree.expansions[0]
    check(le.expander_info["reward_buffer"] == "local" and all("session_reward" in st.extra["curiosity"]
                                                               for st in le.steps),
          f"local mode runs (best return {le.expander_info['best_return']:.3f}; committed yield {le.yield_value:.3f})")

    par = build(frozen=frozen, num_envs=3, total_steps=600)
    par.add_roots()
    par.iterate()
    check(par.tree.expansions[0].expander_info["num_envs"] == 3, "three parallel env copies, each restoring the node")

    # several episodes per session: the top distinct ones, each its own expansion of the node, best first
    from cusi.search.curiosity.session import Episode, keep_top
    top = []
    for i, (acts, r) in enumerate([("ab", 1.0), ("ab", 0.5), ("cd", 2.0), ("ef", 1.0), ("ab", 3.0), ("gh", 0.1)]):
        keep_top(top=top, ep=Episode(index=i, actions=list(acts), rewards=[r]), k=3)
    check([e.index for e in top] == [4, 2, 3], "keep_top: best first, one per action sequence, at most k")
    mk = build(frozen=frozen, keep_episodes=3)
    mk.add_roots()
    adds = []
    real_mk_add = mk.archive.add

    def mk_add(e):
        adds.append(1)
        return real_mk_add(e)
    mk.archive.add = mk_add
    rec = mk.iterate()
    exps = mk.tree.expansions
    infos = [e.expander_info for e in exps]
    check(len(exps) == 3 and [i["plan"] for i in infos] == [0, 1, 2] and len({e.node_id for e in exps}) == 1
          and len({tuple(st.action for st in e.steps) for e in exps}) == 3
          and infos[0]["episode_return"] >= infos[1]["episode_return"] >= infos[2]["episode_return"]
          and rec["expansion"] == exps[-1].id,
          f"keep_episodes 3: three expansions of node {exps[0].node_id}, distinct episodes, returns "
          f"{[round(i['episode_return'], 3) for i in infos]}")
    check(len(adds) == sum(len(e.steps) for e in exps) and len(mk.records) == 3,
          "each committed episode was scored into the global archive, one record per expansion")
    capped = build(frozen=frozen, keep_episodes=3)
    capped.max_expansions = 2
    capped.add_roots()
    capped.iterate()
    check(len(capped.tree.expansions) == 2, "max_expansions caps the episodes one iteration commits")

    # replay restores that fail mid-session (Android) end the session early; a node that cannot be restored at all is
    # set aside without an expansion, and the search goes on
    rp = build(frozen=frozen, restore="replay", total_steps=200)
    rp.add_roots()
    real_restore, calls = rp._restore_node, {"n": 0, "fail_from": 10 ** 9}

    def flaky(*, node, env):
        calls["n"] += 1
        return None if calls["n"] == 5 or calls["n"] >= calls["fail_from"] else real_restore(node=node, env=env)
    rp._restore_node = flaky
    rp.iterate()
    e0 = rp.tree.expansions[0] if rp.tree.expansions else None
    check(e0 is not None and e0.expander_info["stopped_early"] and len(e0.steps) > 0,
          f"a session whose node stops restoring ends early and commits its best episode "
          f"({e0.expander_info['episodes'] if e0 else 0} episodes)")
    calls["fail_from"] = calls["n"] + 2     # the next restore works (the iteration's own), later ones fail
    rec = rp.iterate()
    check(rec["failed"] == "session" and len(rp.tree.expansions) == 1
          and rp.tree.nodes[rec["selected"]].flags.get("replay_diverged"),
          "a node that cannot be restored at all: no expansion, node flagged, no crash")

    if store_dir is not None:
        from cusi.search.store import SearchStore
        store = SearchStore(directory=store_dir, config={"t": 1}, overwrite=True)
        st = build(frozen=frozen, store=store, total_steps=300)
        st.run(stop=lambda x: "done" if len(x.tree.expansions) >= 2 else None)
        e0 = st.tree.expansions[0].expander_info
        policy = torch.load(os.path.join(store_dir, e0["policy"]))
        returns = np.load(os.path.join(store_dir, e0["returns"]))
        check(set(policy) == set(FusionPolicy().state_dict()) and len(returns) == e0["episodes"],
              "the session's policy and returns are saved with the expansion")
        os.makedirs(os.path.join(store_dir, "policies"), exist_ok=True)
        torch.save({}, os.path.join(store_dir, "policies", "7.pt"))       # an uncommitted session's file
        resumed = build(frozen=frozen, store=SearchStore(directory=store_dir, config={"t": 1}), total_steps=300)
        resumed.resume()
        check(not os.path.exists(os.path.join(store_dir, "policies", "7.pt"))
              and resumed.tree.expansions[1].expander_info == st.tree.expansions[1].expander_info,
              "resume drops uncommitted session files and restores expander_info")
    print("ALL OK")


if __name__ == "__main__":
    main()
