"""
Pre-exploration tree search over one environment's scenes (plans/archive/tree_search_plan.md). The run directory is
storage/search/<env>/<run_name>; rerunning the same command resumes it, and a larger --max_expansions extends it.

    python run_search.py --env gameboy --run_name <run> --max_expansions 200 \
        --image_embedder random_patch --text_embedder trigram --novelty_scorer combination
"""
import json
import os
import shutil
import time
import click
from cusi.utils.compute import BUDGET_METRICS, budget_value
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.log_handling import log_info, log_warn
from cusi.utils.paths import search_run, storage_relative
from cusi.utils.run_dir import run_options
from cusi.agents.specs import ENV_NAMES, ENV_SPECS, Scene, scene_for
from cusi.search.restore import DEFAULT_STATE_CACHE
from cusi.search.store import committed_expansions, fork_run
from cusi.state import build_archive, build_encoder, build_scorer_from_config, state_config, state_options

loaded_parameters = load_parameters()

#: Android runs every scene on one emulator, so only one scene env is live at a time and nodes are rebuilt by replay.
DEFAULT_RESTORE = {"gameboy": "state", "web": "state", "android": "replay"}
PATH_KEYS = ("embedder_load_path", "world_model_load_path")
#: Per expander: (actions per expansion, node threshold, what one expansion's actions are). Each becomes the flags
#: --<expander>_steps and --<expander>_node_threshold; the run uses its own expander's pair.
EXPANDER_DEFAULTS = {"random": (100, 0.2, "actions per expansion (one walk)"),
                     "vlm": (20, 0.1, "actions per expansion"),
                     "vlm_typing": (20, 0.1, "actions per expansion"),
                     "curiosity": (10, 0.1, "PPO episode length, so the most actions a committed episode adds")}


def expander_options(command):
    for name, (steps, threshold, steps_help) in reversed(EXPANDER_DEFAULTS.items()):
        command = click.option(f"--{name}_node_threshold", default=threshold, show_default=True,
                               help=f"{name}: a step whose novelty reaches this becomes a node.")(command)
        command = click.option(f"--{name}_steps", default=steps, show_default=True,
                               help=f"{name}: {steps_help}.")(command)
    return command


def stop_reason(*, search, max_expansions, saturation_patience, budget_metric=None, budget=None, first: int = 0):
    """first: expansions the run started with (a fork's source tree), which none of the limits count."""
    expansions = search.tree.expansions[first:]
    if len(expansions) >= max_expansions:
        return "max_expansions"
    if budget_metric is not None:
        # The expansions' recorded compute (kept across resumes); the roots' is left out.
        used = sum(budget_value(snapshot=e.stats["compute"], metric=budget_metric)
                   for e in expansions if "compute" in e.stats)
        if used >= budget:
            return f"budget ({budget_metric} {used:.4g} >= {budget:.4g})"
    recent = expansions[-saturation_patience:] if saturation_patience > 0 else []
    if len(recent) == saturation_patience > 0 and not any(e.children for e in recent):
        return "saturated"
    return None


@click.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--run_name", required=True)
@click.option("--scenes", default=None, help="Comma-separated scenes (default: all of the env's).")
@click.option("--max_expansions", required=True, type=int, help="Total expansions in the run (raise it to extend).")
@expander_options
@click.option("--closing_node/--no_closing_node", default=False, show_default=True,
              help="Also make each expansion's last step a node, however novel. Off: a fruitless expansion adds no "
                   "node.")
@click.option("--archive_nodes_only/--archive_all_steps", default=True, show_default=True,
              help="Add only states that become nodes to the novelty archive (novelty = distance to the nearest "
                   "node).")
@click.option("--budget_metric", default=None, type=click.Choice(BUDGET_METRICS),
              help="Stop once the expansions' compute reaches --budget in this metric (cusi.utils.compute).")
@click.option("--budget", default=None, type=float, help="The compute budget, in --budget_metric's unit.")
@click.option("--saturation_patience", default=0, show_default=True,
              help="Stop once this many expansions in a row have added no node (0: never; --max_expansions still caps "
                   "the run).")
@click.option("--c", "c_puct", default=2.0, show_default=True, help="Exploration-bonus weight in the energy.")
@click.option("--tau", default=0.05, show_default=True, help="Selection temperature.")
@click.option("--normalise_p/--raw_p", default=True, show_default=True,
              help="Normalise P over the candidates (a PUCT prior) in the exploration bonus.")
@click.option("--q_ema", default=0.3, show_default=True, help="Weight of the newest yield in Q.")
@click.option("--roots_only", is_flag=True, help="Expand only the scene roots (OS-Genesis / AutoPlay style).")
@click.option("--expander", default="random", show_default=True,
              type=click.Choice(["random", "vlm", "vlm_typing", "curiosity"]),
              help="vlm_typing: the VLM explorer typing only, from nodes with a text field (Web, Android).")
@click.option("--typing_field_cap", default=10, show_default=True,
              help="vlm_typing: typing attempts per text field identity (expanders.field_key) per scene.")
@click.option("--from_run", default=None,
              help="Start this run from another run's tree (of any expander), into this run's own directory "
                   "(store.fork_run: the source is not changed). The run's limits count only its own expansions.")
@click.option("--reward_buffer", default="global", show_default=True, type=click.Choice(["global", "local"]),
              help="Curiosity: reward against the run's archive (global) or only the start node (local).")
@click.option("--rl_total_steps", default=100_000, show_default=True, help="Curiosity: env steps per session.")
@click.option("--rl_max_minutes", default=60.0, show_default=True, help="Curiosity: wall-clock limit per session.")
@click.option("--rl_num_envs", default=1, show_default=True, help="Curiosity: parallel env copies per session.")
@click.option("--rl_commit_episodes", default=1, show_default=True,
              help="Curiosity: commit the session's top this-many distinct episodes, each as an expansion of the node.")
@click.option("--keep_policies", default="all", show_default=True, type=click.Choice(["all", "none"]))
@click.option("--max_features", default=8, show_default=True,
              help="VLM: features listed per new node (what exploring from it could surface).")
@click.option("--score_top_m", default=8, show_default=True, help="VLM: candidates re-scored per decision (0: all).")
@click.option("--default_score", default=0.5, show_default=True, help="VLM: decision score of a node without features.")
@click.option("--model_name", default=None,
              help="VLM: the model of the env's eval executor, which explores by pursuing objectives (screen features "
                   "not yet discovered), and which also lists each new node's features, reports the segment that "
                   "reached it and scores candidates at selection. Only with --expander vlm (other trees are annotated "
                   "afterwards by cusi.search.annotate).")
@click.option("--model_backend", default="vllm", show_default=True)
@click.option("--vllm_base_url", default=None)
@click.option("--restore", default=None, type=click.Choice(["state", "replay"]),
              help="How a node is restored (default: replay on Android, saved states elsewhere).")
@click.option("--replay_similarity", default=0.95, show_default=True,
              help="Replay restores below this similarity to the node mark it diverged.")
@click.option("--state_cache", default=None, type=int,
              help="Replay restores: keep saved states of this many recent nodes and replay from the nearest one "
                   "(default: 64 on Android, 0 elsewhere).")
@click.option("--seed", default=loaded_parameters["random_seed"], show_default=True)
@click.option("--verify_restores", default=0, show_default=True,
              help="On resume, first restore up to this many nodes and compare them with what they stored.")
@state_options(scorer=True)
@run_options
def main(env_name, run_name, scenes, max_expansions, closing_node, archive_nodes_only, budget_metric, budget,
         saturation_patience, c_puct,
         tau, normalise_p, q_ema, roots_only, expander, typing_field_cap, from_run, reward_buffer, rl_total_steps,
         rl_max_minutes, rl_num_envs,
         rl_commit_episodes, keep_policies,
         max_features, score_top_m, default_score, model_name, model_backend, vllm_base_url,
         restore, replay_similarity, state_cache, seed, verify_restores, overwrite, ignore_config_violation,
         **kwargs):
    from cusi.search.envs import SceneEnvs
    from cusi.search.search import TreeSearch
    from cusi.search.selection import EnergySelector
    from cusi.search.stats import TreeStats
    from cusi.search.store import SearchStore
    parameters = loaded_parameters
    spec = ENV_SPECS[env_name]
    directory = search_run(parameters=parameters, env=env_name, run=run_name)
    stored_path = os.path.join(directory, "config.json")
    stored = None
    if os.path.exists(stored_path) and not overwrite:
        with open(stored_path) as f:
            stored = json.load(f)
        stored = stored.get("config", stored)
    if expander == "vlm_typing" and env_name == "gameboy":
        raise click.UsageError("--expander vlm_typing needs an env with typing (web, android).")
    if (expander in ("vlm", "vlm_typing")) != bool(model_name):
        raise click.UsageError("--model_name goes with --expander vlm or vlm_typing, and only with them.")
    source_config, first = None, 0
    if from_run is not None:
        source_dir = search_run(parameters=parameters, env=env_name, run=from_run)
        with open(os.path.join(source_dir, "config.json")) as f:
            source_config = json.load(f)
        source_config = source_config.get("config", source_config)
        if stored is None:
            first = committed_expansions(directory=source_dir)
            if overwrite and os.path.exists(directory):
                shutil.rmtree(directory)
            fork_run(source=source_dir, target=directory)
            overwrite = False          # the fork is this run's start: SearchStore must not clear it
        else:
            first = stored.get("from_expansions", 0)
    if scenes:
        scene_names = [s.strip() for s in scenes.split(",")]
    elif source_config is not None:
        scene_names = list(source_config["scenes"])       # a fork goes on with its source's scenes
    else:
        scene_names = list(spec.scenes)
    unknown = [s for s in scene_names if s not in spec.scenes]
    if unknown:
        raise click.UsageError(f"Unknown scenes {unknown}; {env_name} has {list(spec.scenes)}")
    restore = restore or DEFAULT_RESTORE[env_name]
    if env_name == "android" and restore != "replay":
        raise click.UsageError("Android needs --restore replay: its scenes share one emulator, so a scene's env is "
                               "rebuilt (losing its saved states) whenever another scene is used.")
    state_cache = DEFAULT_STATE_CACHE[env_name] if state_cache is None else state_cache
    if state_cache and restore != "replay":
        raise click.UsageError("--state_cache goes with --restore replay.")
    if (budget_metric is None) != (budget is None):
        raise click.UsageError("--budget_metric and --budget go together.")
    if expander == "curiosity" and rl_num_envs > 1 and restore == "replay":
        raise click.UsageError("parallel curiosity copies (--rl_num_envs > 1) need saved node states.")
    per_expander = {name: (kwargs.pop(f"{name}_steps"), kwargs.pop(f"{name}_node_threshold"))
                    for name in EXPANDER_DEFAULTS}
    k_steps, node_threshold = per_expander[expander]
    sconfig = state_config(kwargs)
    curiosity = {"reward_buffer": reward_buffer, "rl_total_steps": rl_total_steps, "rl_max_minutes": rl_max_minutes,
                 "rl_num_envs": rl_num_envs, "rl_commit_episodes": rl_commit_episodes,
                 "keep_policies": keep_policies} if expander == "curiosity" else {}
    vlm_config = {"model_name": model_name, "model_backend": model_backend, "vllm_base_url": vllm_base_url,
                  "max_features": max_features, "score_top_m": score_top_m,
                  "default_score": default_score} if expander in ("vlm", "vlm_typing") else {}
    typing_config = {"typing_field_cap": typing_field_cap} if expander == "vlm_typing" else {}
    fork_config = {"from_run": from_run, "from_expansions": first} if from_run is not None else {}
    config = {"env": env_name, "scenes": scene_names, "max_expansions": max_expansions, "k_steps": k_steps,
              "node_threshold": node_threshold, "closing_node": closing_node, "archive_nodes_only": archive_nodes_only,
              "saturation_patience": saturation_patience, "budget_metric": budget_metric, "budget": budget, "c": c_puct, "tau": tau, "normalise_p": normalise_p,
              "q_ema": q_ema, "roots_only": roots_only,
              "expander": expander, "restore": restore, "replay_similarity": replay_similarity,
              "state_cache": state_cache,
              "seed": seed, **curiosity, **vlm_config, **typing_config, **fork_config,
              **{k: (storage_relative(parameters=parameters, path=v) if k in PATH_KEYS and v else v)
                 for k, v in sconfig.items()}}
    # Scenes are kept on their website or app (specs.scene_for) in new runs; a resumed run from before keeps none, and
    # a fork keeps its source's.
    if stored is not None:
        scoped = bool(stored.get("scoped", False))
    elif source_config is not None:
        scoped = bool(source_config.get("scoped", False))
    else:
        scoped = True
    if scoped:
        config["scoped"] = True
    store = SearchStore(directory=directory, config=config, overwrite=overwrite,
                        ignore_config_violation=ignore_config_violation)
    from cusi.envs.action_vocab import ActionVocab
    ActionVocab(env_name=env_name).save(directory=directory)   # world-model training reads search runs too

    encoder = build_encoder(config=sconfig, env_name=env_name, parameters=parameters)
    archive = build_archive(encoder=encoder, text_seen_threshold=sconfig["text_seen_threshold"])
    scorer = build_scorer_from_config(config=sconfig, env_name=env_name, encoder=encoder)
    stats = TreeStats(q_ema=q_ema)
    def make(*, scene):
        s = scene_for(scene=spec.scenes[scene], scoped=scoped)
        if expander == "vlm_typing":
            s = Scene(s.name, {**s.kwargs, "action_mode": "typing"})
        return spec.make_env(scene=s, worker=0, parameters=parameters)
    annotators = {}
    if expander in ("vlm", "vlm_typing"):
        from cusi.agents.vlm import AgentVLM
        from cusi.search.describe import TransitionDescriber
        from cusi.search.expanders import ExecutorExpander
        from cusi.search.features import VLMFeatures
        from cusi.search.scoring import VLMScorer
        vlm = AgentVLM(model_name=model_name, model_backend=model_backend, vllm_base_url=vllm_base_url,
                       parameters=parameters)
        if expander == "vlm_typing":
            from cusi.search.expanders import TypingExpander
            explorer = TypingExpander(env_name=env_name, vlm=vlm, seed=seed, field_cap=typing_field_cap,
                                      parameters=parameters)
        else:
            explorer = ExecutorExpander(env_name=env_name, vlm=vlm, seed=seed, parameters=parameters)
        annotators = {"features": VLMFeatures(vlm=vlm, max_features=max_features),
                      "describer": TransitionDescriber(vlm=vlm, parameters=parameters),
                      "decision_scorer": VLMScorer(vlm=vlm, default_score=default_score)}
    elif expander == "curiosity":
        import torch
        from cusi.search.curiosity.expander import CuriosityExpander
        from cusi.search.curiosity.policy import CNNObserver
        from cusi.search.curiosity.session import PPOConfig
        device = "cuda" if torch.cuda.is_available() else "cpu"
        observer = CNNObserver(env_name=env_name)
        explorer = CuriosityExpander(
            env_name=env_name, observer=observer, reward_buffer=reward_buffer, keep_policies=keep_policies, seed=seed,
            device=device, replay_similarity=replay_similarity, parameters=parameters,
            config=PPOConfig(total_steps=rl_total_steps, max_minutes=rl_max_minutes, k_steps=k_steps,
                             num_envs=rl_num_envs,
                             keep_episodes=rl_commit_episodes),
            make_copy=(lambda key: make(scene=key)) if rl_num_envs > 1 else None)
    else:
        from cusi.search.expanders import RandomExpander
        explorer = RandomExpander(seed=seed)

    envs = SceneEnvs(make=make, scenes=scene_names, exclusive=env_name == "android")
    search = TreeSearch(envs=envs, encoder=encoder, scorer=scorer, archive=archive, expander=explorer, stats=stats,
                        selector=EnergySelector(stats=stats, c=c_puct, tau=tau, roots_only=roots_only,
                                                normalise_p=normalise_p, seed=seed),
                        k_steps=k_steps, node_threshold=node_threshold, closing_node=closing_node,
                        archive_nodes_only=archive_nodes_only, **annotators, score_top_m=score_top_m,
                        restore=restore, replay_similarity=replay_similarity, state_cache=state_cache,
                        store=store, parameters=parameters)
    search.max_expansions = max_expansions
    t0 = time.time()
    verified = None
    try:
        if search.resume() and verify_restores > 0:
            verified = search.verify_restores(limit=verify_restores)
        reason = search.run(stop=lambda s: stop_reason(search=s, max_expansions=max_expansions,
                                                       saturation_patience=saturation_patience,
                                                       budget_metric=budget_metric, budget=budget, first=first))
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
    # One row per launch: where it ran, its roots' compute and the process total (expansions keep their own).
    from cusi.utils.compute import TOTAL, run_info
    launch = {"run_info": run_info(), "seconds": summary["seconds_this_launch"], "roots": search.roots_compute,
              "total": TOTAL.snapshot()}
    with open(os.path.join(directory, "compute.jsonl"), "a") as f:
        f.write(json.dumps(launch) + "\n")
    log_info(f"compute this launch: {launch['total']}", parameters=parameters)
    from cusi.search.viewer import build_viewer
    try:
        log_info(f"viewer: {build_viewer(run_dir=directory, title=f'{env_name} / {run_name}')}", parameters=parameters)
    except Exception as e:     # the viewer is a convenience: never fail a finished run over it
        log_warn(f"could not build the viewer: {e!r}", parameters=parameters)


if __name__ == "__main__":
    main()
