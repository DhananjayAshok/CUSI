"""
Pre-exploration tree search over one environment's scenes (plans/tree_search_plan.md). The run directory is
storage/search/<env>/<run_name>; rerunning the same command resumes it, and a larger --max_expansions extends it.

    python run_search.py --env gameboy --run_name <run> --max_expansions 200 \
        --image_embedder random_patch --text_embedder none --novelty_scorer embedding
"""
import json
import os
import time
import click
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.log_handling import log_info
from cusi.utils.paths import search_run, storage_relative
from cusi.utils.run_dir import run_options
from cusi.agents.specs import ENV_NAMES, ENV_SPECS
from cusi.state import build_archive, build_encoder, build_scorer_from_config, state_config, state_options

loaded_parameters = load_parameters()

#: Android runs every scene on one emulator, so only one scene env is live at a time and nodes are rebuilt by replay.
DEFAULT_RESTORE = {"gameboy": "state", "web": "state", "android": "replay"}
PATH_KEYS = ("embedder_load_path", "world_model_load_path")


@click.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--run_name", required=True)
@click.option("--scenes", default=None, help="Comma-separated scenes (default: all of the env's).")
@click.option("--max_expansions", required=True, type=int, help="Total expansions in the run (raise it to extend).")
@click.option("--k_steps", default=10, show_default=True, help="Actions per expansion.")
@click.option("--node_threshold", default=0.1, show_default=True,
              help="A step whose novelty reaches this becomes a node (as does each expansion's last step).")
@click.option("--c", "c_puct", default=1.0, show_default=True, help="Exploration-bonus weight in the energy.")
@click.option("--tau", default=0.1, show_default=True, help="Selection temperature.")
@click.option("--q_ema", default=0.3, show_default=True, help="Weight of the newest yield in Q.")
@click.option("--roots_only", is_flag=True, help="Expand only the scene roots (OS-Genesis / AutoPlay style).")
@click.option("--expander", default="random", show_default=True, type=click.Choice(["random", "vlm"]))
@click.option("--prior/--no_prior", default=False, show_default=True, help="Score each new node with the VLM prior.")
@click.option("--model_name", default=None, help="The VLM (required with --expander vlm or --prior).")
@click.option("--model_backend", default="vllm", show_default=True)
@click.option("--vllm_base_url", default=None)
@click.option("--restore", default=None, type=click.Choice(["state", "replay"]),
              help="How a node is restored (default: replay on Android, saved states elsewhere).")
@click.option("--replay_similarity", default=0.95, show_default=True,
              help="Replay restores below this similarity to the node mark it diverged.")
@click.option("--noop_similarity", default=0.999, show_default=True)
@click.option("--seed", default=loaded_parameters["random_seed"], show_default=True)
@click.option("--verify_restores", default=0, show_default=True,
              help="On resume, first restore up to this many nodes and compare them with what they stored.")
@state_options(scorer=True)
@run_options
def main(env_name, run_name, scenes, max_expansions, k_steps, node_threshold, c_puct, tau, q_ema, roots_only,
         expander, prior, model_name, model_backend, vllm_base_url, restore, replay_similarity, noop_similarity,
         seed, verify_restores, overwrite, ignore_config_violation, **kwargs):
    from cusi.search.envs import SceneEnvs
    from cusi.search.search import TreeSearch
    from cusi.search.selection import EnergySelector
    from cusi.search.stats import TreeStats
    from cusi.search.store import SearchStore
    parameters = loaded_parameters
    spec = ENV_SPECS[env_name]
    scene_names = [s.strip() for s in scenes.split(",")] if scenes else list(spec.scenes)
    unknown = [s for s in scene_names if s not in spec.scenes]
    if unknown:
        raise click.UsageError(f"Unknown scenes {unknown}; {env_name} has {list(spec.scenes)}")
    restore = restore or DEFAULT_RESTORE[env_name]
    if env_name == "android" and restore != "replay":
        raise click.UsageError("Android needs --restore replay: its scenes share one emulator, so a scene's env is "
                               "rebuilt (losing its saved states) whenever another scene is used.")
    if (expander == "vlm" or prior) and not model_name:
        raise click.UsageError("--expander vlm and --prior need --model_name.")
    sconfig = state_config(kwargs)
    config = {"env": env_name, "scenes": scene_names, "max_expansions": max_expansions, "k_steps": k_steps,
              "node_threshold": node_threshold, "c": c_puct, "tau": tau, "q_ema": q_ema, "roots_only": roots_only,
              "expander": expander, "prior": prior, "model_name": model_name if (expander == "vlm" or prior) else None,
              "model_backend": model_backend, "vllm_base_url": vllm_base_url, "restore": restore,
              "replay_similarity": replay_similarity, "noop_similarity": noop_similarity, "seed": seed,
              **{k: (storage_relative(parameters=parameters, path=v) if k in PATH_KEYS and v else v)
                 for k, v in sconfig.items()}}
    directory = search_run(parameters=parameters, env=env_name, run=run_name)
    store = SearchStore(directory=directory, config=config, overwrite=overwrite,
                        ignore_config_violation=ignore_config_violation)

    encoder = build_encoder(config=sconfig, env_name=env_name, parameters=parameters)
    archive = build_archive(encoder=encoder)
    scorer = build_scorer_from_config(config=sconfig, env_name=env_name, encoder=encoder)
    stats = TreeStats(q_ema=q_ema)
    vlm = None
    if expander == "vlm" or prior:
        from cusi.agents.vlm import AgentVLM
        vlm = AgentVLM(model_name=model_name, model_backend=model_backend, vllm_base_url=vllm_base_url,
                       parameters=parameters)
    if expander == "vlm":
        from cusi.search.expanders import VLMExplorer
        explorer = VLMExplorer(vlm=vlm)
    else:
        from cusi.search.expanders import RandomExpander
        explorer = RandomExpander(seed=seed)
    node_prior = None
    if prior:
        from cusi.search.prior import VLMPrior
        node_prior = VLMPrior(vlm=vlm)

    def make(*, scene):
        return spec.make_env(scene=spec.scenes[scene], worker=0, parameters=parameters)
    envs = SceneEnvs(make=make, scenes=scene_names, exclusive=env_name == "android")
    search = TreeSearch(envs=envs, encoder=encoder, scorer=scorer, archive=archive, expander=explorer, stats=stats,
                        selector=EnergySelector(stats=stats, c=c_puct, tau=tau, roots_only=roots_only, seed=seed),
                        k_steps=k_steps, node_threshold=node_threshold, prior=node_prior,
                        noop_similarity=noop_similarity, restore=restore, replay_similarity=replay_similarity,
                        store=store, parameters=parameters)
    t0 = time.time()
    verified = None
    try:
        if search.resume() and verify_restores > 0:
            verified = search.verify_restores(limit=verify_restores)
        reason = search.run(stop=lambda s: "max_expansions" if len(s.tree.expansions) >= max_expansions else None)
    finally:
        envs.close()
    tree = search.tree
    restores = {}
    for r in search.records:
        restores[r.get("restore", "?")] = restores.get(r.get("restore", "?"), 0) + 1
    summary = {"stopped": reason, "expansions": len(tree.expansions), "nodes": len(tree.nodes),
               "selectable": sum(n.restorable for n in tree.nodes), "max_depth": max(n.depth for n in tree.nodes),
               "archive": len(archive), "total_steps": search.total_steps,
               "challenge_nodes": sum(bool(n.flags.get("challenge_page")) for n in tree.nodes),
               "unrestorable": sum(bool(n.flags.get("unrestorable")) for n in tree.nodes),
               "replay_diverged": sum(bool(n.flags.get("replay_diverged")) for n in tree.nodes),
               "restores": restores, "verified_on_resume": verified,
               "seconds_this_launch": round(time.time() - t0, 1)}
    with open(os.path.join(directory, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log_info(f"search {directory}: {summary}", parameters=parameters)


if __name__ == "__main__":
    main()
