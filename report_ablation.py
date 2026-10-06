"""Tables for the supervisor ablation (agents.md section 7): per env and model, each supervisor arm's
test-set success with a 95% bootstrap CI, the paired comparison against baseline, cost (tokens,
calls, time) and the run timing recorded by scripts/slurm/eval.sh.

    python report_ablation.py --prefix abl [--out storage_dir/eval/ablation_tables.md]

Runs are found as storage_dir/eval/<env>/<prefix>_<model>_<supervisor>[_s<k>]/ (Android shards _s0,
_s1, ... are merged). <model> is the last path part of the HF id, lower-cased (gemma-4-31b-it).
Pairing uses the tasks scored (success not None) in both arms. Web rows from sites that
configs/project_vars.yaml excludes are dropped (so a site excluded after a run leaves its tables).
"""
import glob
import json
import os
import re
from collections import defaultdict
import click
import numpy as np
from cusi_utils.parameter_handling import load_parameters
from cusi_utils.tests import paired_bootstrap

ENVS = ("gameboy", "android", "web")
ARMS = ("baseline", "revision", "subgoal")


def short(model: str) -> str:
    return model.split("/")[-1].lower()


def load_run(*, dirs: list) -> dict:
    """{task_id: row} (latest row per task) and the timing rows, over every shard dir."""
    rows, timing = {}, []
    for d in dirs:
        path = os.path.join(d, "results.jsonl")
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        rows[row["task_id"]] = row
        tpath = os.path.join(d, "timing.jsonl")
        if os.path.exists(tpath):
            with open(tpath) as f:
                timing += [json.loads(l) for l in f if l.strip()]
    return {"rows": rows, "timing": timing}


def bootstrap_ci(values: list, *, n: int = 10000, seed: int = 0) -> tuple:
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(n, len(arr)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def gpu_count(timing_row: dict) -> int:
    gpus = timing_row.get("gpus", "")
    return len([g for g in gpus.split(",") if g.strip()]) if gpus and gpus != "none" else 1


def run_table(*, env: str, model: str, runs: dict, parameters: dict) -> list:
    """Markdown lines for one (env, model)."""
    lines = [f"### {env} / {model}", "",
             "| supervisor | n scored | success | 95% CI | vs baseline (paired n, diff, p) | exec tok/task | "
             "sup tok/task | calls/task | s/task mean | s/task p90 | agent wall h | GPU-h | successes/GPU-h |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    base = runs.get("baseline")
    for arm in ARMS:
        run = runs.get(arm)
        if run is None or not run["rows"]:
            lines.append(f"| {arm} | (not run) |||||||||||| ")
            continue
        rows = run["rows"]
        scored = {k: r for k, r in rows.items() if r["success"] is not None}
        succ = [r["success"] for r in scored.values()]
        rate = float(np.mean(succ)) if succ else float("nan")
        lo, hi = bootstrap_ci(succ)
        paired = "—"
        if arm != "baseline" and base is not None:
            common = sorted(k for k in scored if base["rows"].get(k, {}).get("success") is not None)
            if common:
                a = [scored[k]["success"] for k in common]
                b = [base["rows"][k]["success"] for k in common]
                p = paired_bootstrap(a, b, num_samples=10000, sample_ratio=1.0, parameters=parameters)
                paired = f"{len(common)}, {100 * (np.mean(a) - np.mean(b)):+.1f} pt, p={p:.3f}"
        ex = [r["extra"].get("executor_input_tokens", 0) + r["extra"].get("executor_output_tokens", 0)
              if "executor_input_tokens" in r["extra"] else r["input_tokens"] + r["output_tokens"] for r in rows.values()]
        sup = [r["extra"].get("supervisor_input_tokens", 0) + r["extra"].get("supervisor_output_tokens", 0)
               for r in rows.values()]
        calls = [r["extra"].get("n_executor_calls", 0) + r["extra"].get("n_supervisor_calls", 0) for r in rows.values()]
        secs = sorted(float(r["seconds"] or 0) for r in rows.values())
        agent_s = sum(t.get("agent_seconds", 0) for t in run["timing"])
        gpu_h = sum(t.get("agent_seconds", 0) * gpu_count(t) for t in run["timing"]) / 3600
        succ_per_gpu_h = (sum(succ) / gpu_h) if gpu_h else float("nan")
        lines.append(
            f"| {arm} | {len(scored)} | {100 * rate:.1f}% | [{100 * lo:.1f}, {100 * hi:.1f}] | {paired} | "
            f"{np.mean(ex):,.0f} | {np.mean(sup):,.0f} | {np.mean(calls):.1f} | {np.mean(secs):.0f} | "
            f"{secs[min(len(secs) - 1, int(0.9 * len(secs)))]:.0f} | {agent_s / 3600:.1f} | {gpu_h:.1f} | "
            f"{succ_per_gpu_h:.2f} |")
    return lines + [""]


def timing_table(*, all_runs: dict) -> list:
    lines = ["### Run timing (scripts/slurm/eval.sh timing.jsonl)", "",
             "| env | model | supervisor | jobs | node(s) | GPUs | workers / emulators | vLLM start-up s | agent wall h | "
             "judge wall h | tasks | tasks/h |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (env, model), runs in sorted(all_runs.items()):
        for arm in ARMS:
            run = runs.get(arm)
            if run is None or not run["timing"]:
                continue
            t = run["timing"]
            agent = sum(x.get("agent_seconds", 0) for x in t)
            judge = sum(x.get("judge_seconds", 0) for x in t)
            n = len(run["rows"])
            startup = [x.get("vllm_startup_seconds", 0) for x in t if x.get("vllm_startup_seconds")]
            # Android shards run in parallel inside one job, so a job's wall clock covers all of them.
            per_h = n / (agent / 3600) if agent else float("nan")
            lines.append(f"| {env} | {model} | {arm} | {len(t)} | {', '.join(sorted({x['node'] for x in t}))} | "
                         f"{'; '.join(sorted({x.get('gpu_names', '') for x in t}))} | "
                         f"{', '.join(sorted({str(x.get('n_emulators') if env == 'android' else x.get('workers')) for x in t}))} | "
                         f"{', '.join(str(s) for s in startup)} | {agent / 3600:.2f} | {judge / 3600:.2f} | {n} | "
                         f"{per_h:.1f} |")
    return lines + [""]


def breakdowns(*, env: str, model: str, runs: dict) -> list:
    lines = []
    # Per task, against baseline (pokemon_red's benchmark tasks define no real subgoals, only a
    # placeholder, so there is no partial-progress measure).
    base = (runs.get("baseline") or {"rows": {}})["rows"]
    for arm in ARMS[1:]:
        rows = (runs.get(arm) or {"rows": {}})["rows"]
        common = [k for k in rows if rows[k]["success"] is not None and base.get(k, {}).get("success") is not None]
        if not common:
            continue
        won = sorted(k for k in common if rows[k]["success"] > base[k]["success"])
        lost = sorted(k for k in common if rows[k]["success"] < base[k]["success"])
        name = (lambda k: f"{k} ({rows[k]['task'][:50]})") if env == "gameboy" else (lambda k: k)
        lines += [f"{env} {model} {arm} vs baseline over {len(common)} tasks: solved {len(won)} that baseline "
                  f"did not, lost {len(lost)} that baseline solved.",
                  f"  won: {', '.join(name(k) for k in won) or '-'}",
                  f"  lost: {', '.join(name(k) for k in lost) or '-'}", ""]
    if env == "web":
        sites = defaultdict(dict)
        for arm in ARMS:
            if arm not in runs:
                continue
            per = defaultdict(list)
            for k, r in runs[arm]["rows"].items():
                if r["success"] is not None:
                    per[k.split("--")[0]].append(r["success"])
            for site, v in per.items():
                sites[site][arm] = f"{100 * np.mean(v):.0f}% ({len(v)})"
        lines += [f"Web {model} per site:", "", "| site | " + " | ".join(ARMS) + " |",
                  "|---|" + "---|" * len(ARMS)]
        lines += [f"| {s} | " + " | ".join(sites[s].get(a, "") for a in ARMS) + " |" for s in sorted(sites)] + [""]
    if env == "android":
        lines += [f"Android {model}: per-task success by arm in the episodes; see each run's process_episodes.md.", ""]
    termination = defaultdict(lambda: defaultdict(int))
    for arm in ARMS:
        for r in (runs.get(arm) or {"rows": {}})["rows"].values():
            termination[arm][r.get("termination_reason")] += 1
    lines += [f"{env} {model} episode endings: "
              + "; ".join(f"{a}: " + ", ".join(f"{k} {v}" for k, v in sorted(termination[a].items(), key=str))
                          for a in ARMS if termination[a]), ""]
    return lines


@click.command()
@click.option("--prefix", default="abl", show_default=True)
@click.option("--out", default=None, help="Default: storage_dir/eval/ablation_tables.md")
def main(prefix, out):
    parameters = load_parameters()
    root = os.path.join(parameters["storage_dir"], "eval")
    all_runs = defaultdict(dict)
    for env in ENVS:
        for d in sorted(glob.glob(os.path.join(root, env, f"{prefix}_*"))):
            m = re.match(rf"{re.escape(prefix)}_(.+)_({'|'.join(ARMS)})(?:_s\d+)?$", os.path.basename(d))
            if not m:
                continue
            all_runs[(env, m.group(1))].setdefault(m.group(2), []).append(d)
    lines = [f"# Supervisor ablation tables ({prefix})", ""]
    loaded = {}
    for key, arms in sorted(all_runs.items()):
        loaded[key] = {arm: load_run(dirs=dirs) for arm, dirs in arms.items()}
        if key[0] == "web":
            # Sites excluded since the run (configs/project_vars.yaml) are dropped from the tables.
            from cusi_envs.webvoyager import excluded_sites
            excluded = set(excluded_sites(parameters=parameters))
            for run in loaded[key].values():
                run["rows"] = {k: r for k, r in run["rows"].items() if k.split("--")[0] not in excluded}
        lines += run_table(env=key[0], model=key[1], runs=loaded[key], parameters=parameters)
        lines += breakdowns(env=key[0], model=key[1], runs=loaded[key])
    lines += timing_table(all_runs=loaded)
    out = out or os.path.join(root, "ablation_tables.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
