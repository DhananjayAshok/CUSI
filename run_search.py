"""
Pre-exploration tree search over one environment's scenes (plans/tree_search_plan.md). The run directory is
storage/search/<env>/<run_name>; rerunning the same command resumes it, and a larger --max_expansions extends it.

    python run_search.py --env gameboy --run_name <run> --max_expansions 200 \
        --image_embedder random_patch --text_embedder trigram --novelty_scorer combination
"""
import json
import os
import time
import click
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.log_handling import log_info, log_warn
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
@click.option("--c", "c_puct", default=2.0, show_default=True, help="Exploration-bonus weight in the energy.")
@click.option("--tau", default=0.05, show_default=True, help="Selection temperature.")
@click.option("--normalise_p/--raw_p", default=True, show_default=True,
              help="Normalise P over the candidates (a PUCT prior) in the exploration bonus.")
@click.option("--q_ema", default=0.3, show_default=True, help="Weight of the newest yield in Q.")
@click.option("--roots_only", is_flag=True, help="Expand only the scene roots (OS-Genesis / AutoPlay style).")
@click.option("--expander", default="random", show_default=True, type=click.Choice(["random", "vlm", "curiosity"]))
@click.option("--reward_buffer", default="global", show_default=True, type=click.Choice(["global", "local"]),
              help="Curiosity: reward against the run's archive (global) or only the start node (local).")
@click.option("--rl_total_steps", default=100_000, show_default=True, help="Curiosity: env steps per session.")
@click.option("--rl_max_minutes", default=60.0, show_default=True, help="Curiosity: wall-clock limit per session.")
@click.option("--rl_num_envs", default=1, show_default=True, help="Curiosity: parallel env copies per session.")
@click.option("--rl_commit_episodes", default=1, show_default=True,
              help="Curiosity: commit the session's top this-many distinct episodes, each as an expansion of the node.")
@click.option("--keep_policies", default="all", show_default=True, type=click.Choice(["all", "none"]))
@click.option("--max_type_strings", default=8, show_default=True, help="Curiosity: typed strings per input field.")
@click.option("--policy_vision_encoder", default=None,
              help="Curiosity: the frozen vision trunk (mobilenet_v3_small); required with --expander curiosity.")
@click.option("--policy_text_encoder", default=None,
              help="Curiosity: the frozen sentence encoder (e.g. a MiniLM-L6 id); required with --expander curiosity.")
@click.option("--features/--no_features", default=False, show_default=True,
              help="Have the VLM list what exploring from each new node could surface.")
@click.option("--max_features", default=8, show_default=True)
@click.option("--describe/--no_describe", default=False, show_default=True,
              help="After each expansion, have the VLM describe its sub-trajectory and list what it discovered.")
@click.option("--vlm_score/--no_vlm_score", default=False, show_default=True,
              help="At each decision, score the top candidates' features against their scene's discovered set; the "
                   "score is P(n). Needs --features and --describe.")
@click.option("--score_top_m", default=8, show_default=True, help="Candidates re-scored per decision (0: all).")
@click.option("--default_score", default=0.5, show_default=True, help="Decision score of a node without features.")
@click.option("--model_name", default=None,
              help="The VLM (required with --expander vlm, --features, --describe or --vlm_score).")
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
def main(env_name, run_name, scenes, max_expansions, k_steps, node_threshold, c_puct, tau, normalise_p, q_ema,
         roots_only,
         expander, reward_buffer, rl_total_steps, rl_max_minutes, rl_num_envs, rl_commit_episodes, keep_policies, max_type_strings,
         policy_vision_encoder, policy_text_encoder, features, max_features, describe, vlm_score, score_top_m, default_score, model_name, model_backend,
         vllm_base_url, restore, replay_similarity,
         noop_similarity, seed, verify_restores, overwrite, ignore_config_violation, **kwargs):
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
    uses_vlm = expander == "vlm" or features or describe or vlm_score
    if uses_vlm and not model_name:
        raise click.UsageError("--expander vlm, --features, --describe and --vlm_score need --model_name.")
    if vlm_score and not (features and describe):
        raise click.UsageError("--vlm_score needs --features and --describe.")
    if expander == "curiosity" and not (policy_vision_encoder and policy_text_encoder):
        raise click.UsageError("--expander curiosity needs --policy_vision_encoder and --policy_text_encoder.")
    if expander == "curiosity" and rl_num_envs > 1 and restore == "replay":
        raise click.UsageError("parallel curiosity copies (--rl_num_envs > 1) need saved node states.")
    sconfig = state_config(kwargs)
    curiosity = {"reward_buffer": reward_buffer, "rl_total_steps": rl_total_steps, "rl_max_minutes": rl_max_minutes,
                 "rl_num_envs": rl_num_envs, "rl_commit_episodes": rl_commit_episodes,
                 "keep_policies": keep_policies, "max_type_strings": max_type_strings,
                 "policy_vision_encoder": policy_vision_encoder,
                 "policy_text_encoder": policy_text_encoder} if expander == "curiosity" else {}
    config = {"env": env_name, "scenes": scene_names, "max_expansions": max_expansions, "k_steps": k_steps,
              "node_threshold": node_threshold, "c": c_puct, "tau": tau, "normalise_p": normalise_p, "q_ema": q_ema, "roots_only": roots_only,
              "expander": expander, "features": features, "max_features": max_features, "describe": describe,
              "vlm_score": vlm_score, "score_top_m": score_top_m, "default_score": default_score,
              "model_name": model_name if uses_vlm else None,
              "model_backend": model_backend, "vllm_base_url": vllm_base_url, "restore": restore,
              "replay_similarity": replay_similarity, "noop_similarity": noop_similarity, "seed": seed, **curiosity,
              **{k: (storage_relative(parameters=parameters, path=v) if k in PATH_KEYS and v else v)
                 for k, v in sconfig.items()}}
    directory = search_run(parameters=parameters, env=env_name, run=run_name)
    store = SearchStore(directory=directory, config=config, overwrite=overwrite,
                        ignore_config_violation=ignore_config_violation)
    from cusi.envs.action_vocab import ActionVocab
    ActionVocab(env_name=env_name).save(directory=directory)   # world-model training reads search runs too

    encoder = build_encoder(config=sconfig, env_name=env_name, parameters=parameters)
    archive = build_archive(encoder=encoder, text_seen_threshold=sconfig["text_seen_threshold"])
    scorer = build_scorer_from_config(config=sconfig, env_name=env_name, encoder=encoder)
    stats = TreeStats(q_ema=q_ema)
    vlm = None
    if uses_vlm:
        from cusi.agents.vlm import AgentVLM
        vlm = AgentVLM(model_name=model_name, model_backend=model_backend, vllm_base_url=vllm_base_url,
                       parameters=parameters)
    def make(*, scene):
        return spec.make_env(scene=spec.scenes[scene], worker=0, parameters=parameters)
    if expander == "vlm":
        from cusi.search.expanders import VLMExplorer
        explorer = VLMExplorer(vlm=vlm)
    elif expander == "curiosity":
        import torch
        from cusi.search.curiosity.expander import CuriosityExpander
        from cusi.search.curiosity.policy import FrozenEncoders
        from cusi.search.curiosity.session import PPOConfig
        device = "cuda" if torch.cuda.is_available() else "cpu"
        frozen = FrozenEncoders(vision_encoder=policy_vision_encoder, text_encoder=policy_text_encoder,
                                device=device, long_side=224 if env_name == "gameboy" else 448)
        explorer = CuriosityExpander(
            env_name=env_name, frozen=frozen, reward_buffer=reward_buffer, keep_policies=keep_policies, seed=seed,
            device=device, replay_similarity=replay_similarity, parameters=parameters,
            config=PPOConfig(total_steps=rl_total_steps, max_minutes=rl_max_minutes, k_steps=k_steps,
                             num_envs=rl_num_envs, max_type_strings=max_type_strings,
                             keep_episodes=rl_commit_episodes),
            make_copy=(lambda key: make(scene=key)) if rl_num_envs > 1 else None)
    else:
        from cusi.search.expanders import RandomExpander
        explorer = RandomExpander(seed=seed)
    node_features = None
    if features:
        from cusi.search.features import VLMFeatures
        node_features = VLMFeatures(vlm=vlm, max_features=max_features)
    describer = None
    if describe:
        from cusi.search.describe import TransitionDescriber
        describer = TransitionDescriber(vlm=vlm, parameters=parameters)
    decision_scorer = None
    if vlm_score:
        from cusi.search.scoring import VLMScorer
        decision_scorer = VLMScorer(vlm=vlm, default_score=default_score)

    envs = SceneEnvs(make=make, scenes=scene_names, exclusive=env_name == "android")
    search = TreeSearch(envs=envs, encoder=encoder, scorer=scorer, archive=archive, expander=explorer, stats=stats,
                        selector=EnergySelector(stats=stats, c=c_puct, tau=tau, roots_only=roots_only, normalise_p=normalise_p,
                                                 seed=seed),
                        k_steps=k_steps, node_threshold=node_threshold, features=node_features,
                        decision_scorer=decision_scorer, score_top_m=score_top_m, describer=describer,
                        noop_similarity=noop_similarity, restore=restore, replay_similarity=replay_similarity,
                        store=store, parameters=parameters)
    search.max_expansions = max_expansions
    t0 = time.time()
    verified = None
    try:
        if search.resume() and verify_restores > 0:
            verified = search.verify_restores(limit=verify_restores)
        reason = search.run(stop=lambda s: "max_expansions" if len(s.tree.expansions) >= max_expansions else None)
    finally:
        envs.close()
        if hasattr(explorer, "close"):
            explorer.close()
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
    from cusi.search.viewer import build_viewer
    try:
        log_info(f"viewer: {build_viewer(run_dir=directory, title=f'{env_name} / {run_name}')}", parameters=parameters)
    except Exception as e:     # the viewer is a convenience: never fail a finished run over it
        log_warn(f"could not build the viewer: {e!r}", parameters=parameters)


if __name__ == "__main__":
    main()
