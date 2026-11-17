"""
Cut a search run at a compute budget into a new run: the source's first expansions up to the one at which their
recorded compute reaches the budget (the expansion that crosses it is kept, as run_search's budget stop keeps it),
the roots and the nodes those expansions made. The source is never changed: the target starts as a fork of it
(store.fork_run: logs and archive copied, write-once files hard-linked) and is then truncated to the cut with the
store's own commit truncation, which only drops the target's links.

Node ids and expansion ids are in creation order, so the kept nodes are a prefix of the ids. Kept nodes keep their
later update rows (flags, annotations); an annotated parent's task list drops the edge tasks of children past the
cut. The novelty archive is the source's (it is never rebuilt): it matters only if the cut run is extended as a
search. Neither task_coverage / task_diversity nor the viewer are carried over (fork_run). config.json is the
source's with a "cut" entry.

    python -m cusi.search.cut --env gameboy --run sat_vlm_no_typing_s4 --budget 248400
"""
import json
import os
import click
from cusi.search.store import (ARCHIVE_DIR, COMMIT_FILE, EXPANSIONS_FILE, NODES_FILE, SearchStore, fork_run,
                               read_rows)
from cusi.utils.compute import BUDGET_METRICS, budget_value
from cusi.utils.log_handling import log_info
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.paths import search_run


def cut_point(*, expansions: list, budget_metric: str, budget: float) -> int:
    """How many of the expansions (in order) the cut keeps: up to and including the first at which the running
    total of their compute reaches budget; all of them if it never does."""
    used = 0.0
    for i, e in enumerate(expansions):
        if "compute" in e["stats"]:
            used += budget_value(snapshot=e["stats"]["compute"], metric=budget_metric)
        if used >= budget:
            return i + 1
    return len(expansions)


def cut_run(*, source: str, target: str, budget_metric: str, budget: float, parameters: dict) -> dict:
    expansions = read_rows(os.path.join(source, EXPANSIONS_FILE))
    n_exps = cut_point(expansions=expansions, budget_metric=budget_metric, budget=budget)
    kept = expansions[:n_exps]
    node_rows = [r for r in read_rows(os.path.join(source, NODES_FILE)) if "parent" in r]
    n_roots = sum(r["parent"] is None for r in node_rows)
    n_nodes = max([n_roots] + [c + 1 for e in kept for c in e["children"]])
    # creation order: every kept node was made by a root or a kept expansion, and no later one
    made = {c for e in kept for c in e["children"]} | {r["id"] for r in node_rows if r["parent"] is None}
    if made != set(range(n_nodes)):
        raise click.ClickException(f"{source}: the kept expansions' nodes are not the ids below {n_nodes}")
    with open(os.path.join(source, "config.json")) as f:
        config = json.load(f)

    fork_run(source=source, target=target)
    store = SearchStore(directory=target)          # no config: nothing is checked or written
    store._truncate(n_nodes=n_nodes, n_expansions=n_exps)
    # annotated parents: their task lists lose the edge tasks of children past the cut
    rows = read_rows(os.path.join(target, NODES_FILE))
    n_tasks_dropped = 0
    for r in rows:
        if r.get("tasks"):
            keep = [t for t in r["tasks"] if t.get("source") != "edge" or t["child"] < n_nodes]
            n_tasks_dropped += len(r["tasks"]) - len(keep)
            r["tasks"] = keep
    with open(os.path.join(target, NODES_FILE), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(target, ARCHIVE_DIR, COMMIT_FILE)) as f:
        commit = json.load(f)
    commit.update(n_expansions=n_exps, n_nodes=n_nodes,
                  iteration=kept[-1]["iteration"] if kept else 0,
                  total_steps=sum(len(e["steps"]) for e in kept))
    with open(os.path.join(target, ARCHIVE_DIR, COMMIT_FILE), "w") as f:
        json.dump(commit, f)
    used = sum(budget_value(snapshot=e["stats"]["compute"], metric=budget_metric) for e in kept
               if "compute" in e["stats"])
    cut = {"source": os.path.basename(source.rstrip("/")), "budget_metric": budget_metric, "budget": budget,
           "used": used, "expansions": n_exps, "source_expansions": len(expansions), "nodes": n_nodes,
           "source_nodes": len(node_rows), "edge_tasks_dropped": n_tasks_dropped, "archive": "the source's"}
    inner = config.get("config", config)
    inner["cut"] = cut
    with open(os.path.join(target, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    log_info(f"cut {source} -> {target}: {cut}", parameters=parameters)
    return cut


@click.command()
@click.option("--env", "env_name", required=True)
@click.option("--run", "run_name", required=True, help="The source run (not changed).")
@click.option("--target_run", default=None, help="The cut run's name (default: <run>_cut<budget in hours>h for "
                                                  "wall_s, else <run>_cut).")
@click.option("--budget_metric", default="wall_s", show_default=True, type=click.Choice(BUDGET_METRICS))
@click.option("--budget", required=True, type=float, help="In --budget_metric's unit (wall_s: 248400 is 69 h).")
def main(env_name, run_name, target_run, budget_metric, budget):
    parameters = load_parameters()
    if target_run is None:
        target_run = f"{run_name}_cut{budget / 3600:g}h" if budget_metric == "wall_s" else f"{run_name}_cut"
    cut_run(source=search_run(parameters=parameters, env=env_name, run=run_name),
            target=search_run(parameters=parameters, env=env_name, run=target_run),
            budget_metric=budget_metric, budget=budget, parameters=parameters)


if __name__ == "__main__":
    main()
