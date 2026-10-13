"""
Compare search runs (plans/next_plan.md §E.7, step 5): what each run's archive holds against the env steps and time
it cost, whichever expander produced it.

    python -m cusi.search.compare --env gameboy --runs curiosity_global_v1,random_k30_v1,gemma_vlm_v2
"""
import json
import os
import click
import torch
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.paths import search_run


def run_stats(*, directory: str) -> dict:
    rows = [json.loads(l) for l in open(os.path.join(directory, "expansions.jsonl")) if l.strip()]
    nodes = [json.loads(l) for l in open(os.path.join(directory, "nodes.jsonl")) if '"parent"' in l]
    archive = torch.load(os.path.join(directory, "archive", "archive.pt"), weights_only=False)
    seen_path = os.path.join(directory, "archive", "seen_text.json")
    seen = json.load(open(seen_path))["keys"] if os.path.exists(seen_path) else []
    committed = sum(len(r["steps"]) for r in rows)
    # a session committing several episodes repeats its info on each expansion: count it once (plan 0)
    session = sum(int((r.get("expander_info") or {}).get("total_steps", 0)) for r in rows
                  if (r.get("expander_info") or {}).get("plan", 0) == 0)
    seconds = sum(r["stats"].get("seconds", 0.0) for r in rows)
    texts = {k.split("\t", 1)[1] for k in seen}
    return {"expansions": len(rows), "nodes": len(nodes), "max_depth": max(n["depth"] for n in nodes),
            "archive_states": 0 if archive["images"] is None else int(archive["images"].shape[0]),
            "distinct_text_keys": len(seen), "distinct_texts": len(texts),
            "env_steps": committed + session, "committed_steps": committed, "hours": round(seconds / 3600, 2),
            "mean_yield": round(sum(r["yield"] for r in rows) / max(1, len(rows)), 4)}


@click.command()
@click.option("--env", "env_name", required=True)
@click.option("--runs", required=True, help="Comma-separated search runs.")
def main(env_name, runs):
    parameters = load_parameters()
    for run in [r.strip() for r in runs.split(",") if r.strip()]:
        print(run, json.dumps(run_stats(directory=search_run(parameters=parameters, env=env_name, run=run))))


if __name__ == "__main__":
    main()
