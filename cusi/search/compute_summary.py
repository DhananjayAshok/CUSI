"""
The compute a search run (and its annotation) used, from what cusi.utils.compute recorded: per expansion
(exp.stats["compute"]), per step (step.extra["compute"]), per launch (compute.jsonl: roots and process totals) and per
annotated node (annotate_compute).

    python -m cusi.search.compute_summary --env gameboy --run <run>
"""
import json
import os
import click
from cusi.agents.specs import ENV_NAMES
from cusi.search.store import EXPANSIONS_FILE, node_rows, read_rows
from cusi.utils.compute import merge
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.paths import search_run


def summarize(*, directory: str) -> dict:
    expansions = read_rows(os.path.join(directory, EXPANSIONS_FILE))
    search, steps, n_steps, missing = {}, {}, 0, 0
    for e in expansions:
        compute = (e.get("stats") or {}).get("compute")
        if compute is None:
            missing += 1
            continue
        merge(into=search, other=compute)
        for s in e["steps"]:
            if "compute" in (s.get("extra") or {}):
                merge(into=steps, other=s["extra"]["compute"])
                n_steps += 1
    launches = []
    path = os.path.join(directory, "compute.jsonl")
    if os.path.exists(path):
        launches = [json.loads(line) for line in open(path)]
    for launch in launches:
        for root in launch["roots"].values():
            merge(into=search, other=root)
    annotate, n_annotated = {}, 0
    for node in node_rows(directory=directory).values():
        if node.get("annotate_compute"):
            merge(into=annotate, other=node["annotate_compute"])
            n_annotated += 1
    per_step = {k: v / n_steps for k, v in steps.get("wall_s", {}).items()} if n_steps else {}
    return {"expansions": len(expansions), "expansions_without_compute": missing, "steps": n_steps,
            "search": search, "per_step_wall_s": per_step, "annotated_nodes": n_annotated, "annotate": annotate,
            "launches": [{"run_info": la["run_info"], "seconds": la["seconds"]} for la in launches]}


@click.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--run", "run_name", required=True)
def main(env_name, run_name):
    parameters = load_parameters()
    out = summarize(directory=search_run(parameters=parameters, env=env_name, run=run_name))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
