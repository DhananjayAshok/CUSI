# TOY (skill_discovery.md §1.5): throwaway first-build scaffolding, not part of the kept code.
# Strip it by deleting cusi_search/toy/, run_preexplore_toy.py, slurm/preexplore_toy.sh,
# scripts/slurm/preexplore_toy.sh, tests/*toy* and the TOY block in skill_discovery.md §1.5.
# Nothing outside those files imports this.
"""Toy entry point for the pre-exploration tree search (skill_discovery.md §1.4, §1.5).

    # GameBoy, random expander, no VLM (CPU is enough):
    python run_preexplore_toy.py --env gameboy --scenes viridian,pewter --run_name toy_random \
        --expander random --k_steps 10 --max_expansions 40 \
        --image_embedder random_patch --text_embedder none --novelty_scorer embedding
    # Web, VLM explorer + VLM prior (vLLM must be serving --model_name; see the preexplore_toy Slurm pair):
    python run_preexplore_toy.py --env web --scenes arxiv --run_name toy_vlm --expander vlm --prior \
        --model_name Qwen/Qwen3-VL-2B-Instruct --k_steps 4 --max_expansions 6 \
        --image_embedder random_patch --text_embedder tfidf --novelty_scorer embedding

Thresholds: --cell_threshold / --node_threshold, or calibrated from a short random run
(cusi_search.toy.calibrate) when not given. Output: storage_dir/preexplore_toy/<env>/<run_name>/
(tree_io format + index.html viewer).
"""
import os
import click
from cusi_utils.parameter_handling import load_parameters, compute_secondary_parameters
from cusi_utils.log_handling import log_info
from cusi_practice.envs import ENV_NAMES
from cusi_state import build_archive, build_encoder, build_scorer_from_config, state_config, state_options
from cusi_search.expanders import RandomExpander, VLMExplorer
from cusi_search.prior import VLMPrior
from cusi_search.search import TreeSearch
from cusi_search.selection import EnergySelector
from cusi_search.toy.calibrate import calibrate
from cusi_search.toy.envs import close_envs, make_envs
from cusi_search.toy.stats_handwaved import HandwavedStats
from cusi_search.toy.stopping import StopRule
from cusi_search.toy.tree_io import TreeWriter
from cusi_search.toy.viewer import write_viewer

loaded_parameters = load_parameters()


@click.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--scenes", required=True, help="Comma-separated scene names (cusi_practice.envs); one root each.")
@click.option("--run_name", required=True)
@click.option("--expander", required=True, type=click.Choice(["random", "vlm"]))
@click.option("--prior/--no_prior", default=False, help="Score every new node with the VLM prior.")
@click.option("--model_name", default=None, help="Served VLM (required with --expander vlm or --prior).")
@click.option("--k_steps", default=10, help="Steps per expansion.")
@click.option("--c", "c_explore", default=1.0, help="Exploration constant c in the energy.")
@click.option("--tau", default=3.0, help="Temperature of the selection distribution (toy default: 1 was near-greedy on GameBoy).")
@click.option("--roots_only", is_flag=True, help="Only ever expand roots (restart-from-scene-start methods).")
@click.option("--cell_threshold", default=None, type=float)
@click.option("--node_threshold", default=None, type=float)
@click.option("--calib_steps", default=30, help="Random steps per root for calibration (when thresholds are not given).")
@click.option("--cell_percentile", default=25.0)
@click.option("--node_percentile", default=90.0)
@click.option("--max_expansions", default=None, type=int)
@click.option("--max_steps", default=None, type=int)
@click.option("--max_seconds", default=None, type=float)
@click.option("--max_nodes", default=None, type=int)
@click.option("--worker", default=0, help="Android: which emulator.")
@click.option("--vllm_port", default=loaded_parameters["vllm_port"])
@click.option("--random_seed", default=loaded_parameters["random_seed"])
@state_options(scorer=True)
def main(env_name, scenes, run_name, expander, prior, model_name, k_steps, c_explore, tau, roots_only,
         cell_threshold, node_threshold, calib_steps, cell_percentile, node_percentile, max_expansions, max_steps,
         max_seconds, max_nodes, worker, vllm_port, random_seed, **kwargs):
    parameters = loaded_parameters
    parameters["vllm_port"], parameters["random_seed"] = vllm_port, random_seed
    compute_secondary_parameters(parameters)
    config = state_config(kwargs)
    if (expander == "vlm" or prior) and not model_name:
        raise click.UsageError("--expander vlm / --prior need --model_name")
    if all(x is None for x in (max_expansions, max_steps, max_seconds, max_nodes)):
        raise click.UsageError("give at least one of --max_expansions / --max_steps / --max_seconds / --max_nodes")
    out_dir = os.path.join(parameters["storage_dir"], "preexplore_toy", env_name, run_name)
    os.makedirs(out_dir, exist_ok=True)
    envs = make_envs(env_name=env_name, scenes=[s for s in scenes.split(",") if s], worker=worker,
                     parameters=parameters)
    try:
        for i, env in enumerate(envs.values()):
            env.reset(seed=random_seed + i)       # seeds sample_action
        encoder = build_encoder(config=config, env_name=env_name, parameters=parameters)
        scorer = build_scorer_from_config(config=config, env_name=env_name, encoder=encoder)
        calibration = None
        if cell_threshold is None or node_threshold is None:
            calibration = calibrate(envs=envs, encoder=encoder, scorer=scorer, steps_per_root=calib_steps,
                                    cell_percentile=cell_percentile, node_percentile=node_percentile)
            log_info(f"calibration: {calibration}", parameters=parameters)
            cell_threshold = calibration["cell_threshold"] if cell_threshold is None else cell_threshold
            node_threshold = calibration["node_threshold"] if node_threshold is None else node_threshold
        archive = build_archive(encoder=encoder, cell_threshold=cell_threshold)
        stats = HandwavedStats(archive=archive)
        vlm = None
        if model_name:
            from cusi_practice.vlm import PracticeVLM
            vlm = PracticeVLM(model_name=model_name, parameters=parameters)
        exp = RandomExpander() if expander == "random" else VLMExplorer(vlm=vlm)
        writer = TreeWriter(out_dir=out_dir, config={
            "env": env_name, "scenes": scenes, "expander": expander, "prior": prior, "model_name": model_name,
            "k_steps": k_steps, "c": c_explore, "tau": tau, "roots_only": roots_only, "cell_threshold": cell_threshold,
            "node_threshold": node_threshold, "calibration": calibration, "seed": random_seed,
            "stop": {"max_expansions": max_expansions, "max_steps": max_steps, "max_seconds": max_seconds,
                     "max_nodes": max_nodes}, "state": encoder.config(), **config})
        search = TreeSearch(envs=envs, encoder=encoder, scorer=scorer, archive=archive, expander=exp, stats=stats,
                            selector=EnergySelector(stats=stats, c=c_explore, tau=tau, roots_only=roots_only,
                                                    seed=random_seed),
                            k_steps=k_steps, node_threshold=node_threshold,
                            prior=VLMPrior(vlm=vlm) if prior else None, logger=writer, parameters=parameters)
        reason = search.run(stop=StopRule(max_expansions=max_expansions, max_steps=max_steps,
                                          max_seconds=max_seconds, max_nodes=max_nodes))
        writer.write(search=search, stop_reason=reason)
        page = write_viewer(run_dir=out_dir)
        extra = f"; prior calls {search.prior.n_calls}, unparsed {search.prior.n_failed}" if search.prior else ""
        log_info(f"search stopped ({reason}): {search.iteration} expansions, {search.total_steps} steps, "
                 f"{len(search.tree.nodes)} nodes, {archive.n_cells} cells{extra} -> {page}", parameters=parameters)
    finally:
        close_envs(envs=envs)


if __name__ == "__main__":
    main()
