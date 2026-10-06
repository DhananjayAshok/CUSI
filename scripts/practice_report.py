"""
Markdown report of practice-pipeline outputs per environment: stage counts and example dataset rows.

    python scripts/practice_report.py --envs "gameboy web android" --out results/practice_small.md
"""
import json
import os
import click
import pandas as pd
from cusi.utils import load_parameters, log_info
from cusi.utils.paths import PracticePaths


def _csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame([])


def _short(text: str, n: int = 700) -> str:
    text = str(text)
    return text if len(text) <= n else text[:n] + f" ... [{len(text) - n} more chars]"


def env_section(*, env: str, paths: PracticePaths, n_examples: int) -> list:
    lines = [f"## {env}", "", f"Outputs: `{paths.root}`", ""]
    proposals = []
    if os.path.exists(paths.proposals):
        proposals = [json.loads(l) for l in open(paths.proposals)]
    attempts = _csv(os.path.join(paths.attempts_dir, "results.csv"))
    guidance = json.load(open(paths.guidance)) if os.path.exists(paths.guidance) else None
    practice = _csv(os.path.join(paths.practice_dir, "results.csv"))
    decisions = _csv(os.path.join(paths.practice_dir, "clean_decisions.csv"))
    stats_path = os.path.join(paths.dataset_dir, "stats.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}

    def n(df, col=None):
        if df is None:
            return "—"
        if df.empty:
            return 0
        return int(df[col].astype(bool).sum()) if col else len(df)

    lines += ["| stage | count |", "|---|---|",
              f"| proposed tasks (kept) | {sum(len(p['tasks']) for p in proposals) if proposals else '—'} |",
              f"| attempted | {n(attempts)} |",
              f"| attempts succeeded | {n(attempts, 'success')} |",
              f"| tasks with guidance | {len(guidance) if guidance is not None else '—'} |",
              f"| practice runs | {n(practice)} |",
              f"| practice runs succeeded | {n(practice, 'success')} |",
              f"| calls judged by clean | {n(decisions)} |",
              f"| calls accepted by clean | {n(decisions, 'accept')} |",
              f"| dataset rows (train / validation) | {stats.get('train_rows', '—')} / {stats.get('validation_rows', '—')} |",
              ""]
    if stats:
        lines += [f"Dataset stats: `{json.dumps(stats)}`", ""]
    for p in proposals:
        lines += [f"- scene **{p['scene']}**: " + "; ".join(p["tasks"])]
    lines.append("")
    if attempts is not None and not attempts.empty:
        cols = [c for c in ("group_idx", "task_string", "success", "n_tries", "termination_reason", "n_env_steps")
                if c in attempts.columns]
        lines += ["### Attempts", "", attempts[cols].to_markdown(index=False), ""]
    if practice is not None and not practice.empty:
        cols = [c for c in ("group_idx", "attempt", "success", "used_retry", "termination_reason", "n_env_steps")
                if c in practice.columns]
        lines += ["### Practice", "", practice[cols].to_markdown(index=False), ""]
    train = os.path.join(paths.dataset_dir, "train.jsonl")
    if os.path.exists(train):
        rows = [json.loads(l) for l in open(train)][:n_examples]
        lines += ["### Example dataset rows", ""]
        for row in rows:
            lines += [f"**{row['episode_id']} call {row['call_idx']} ({row['call_type']})**, task: {row['task_string']}",
                      "", "```"]
            for msg in row["messages"]:
                if isinstance(msg["content"], str):
                    lines.append(f"[{msg['role']}] {_short(msg['content'])}")
                else:
                    for part in msg["content"]:
                        lines.append(f"[{msg['role']}] " + (_short(part["text"]) if part["type"] == "text"
                                                            else f"<image {os.path.basename(part['image'])}>"))
            lines += ["--- response", _short(row["response"], 1500), "```", ""]
    elif not (attempts is not None and n(attempts, "success")):
        lines += ["No successful attempts, so nothing was practised.", ""]
    return lines


@click.command()
@click.option("--envs", default="gameboy web android")
@click.option("--out", required=True)
@click.option("--model_name", required=True, help="Model whose practice outputs to report on.")
@click.option("--n_examples", default=3)
@click.option("--storage_dir", default=None, help="Override storage_dir (e.g. the mock test's).")
@click.option("--source", default="zeroshot", help="zeroshot | curiosity")
def main(envs, out, model_name, n_examples, storage_dir, source):
    parameters = load_parameters()
    if storage_dir:
        parameters = dict(parameters, storage_dir=storage_dir)
    lines = ["# Practice pipeline report", "", f"Model: `{model_name}`, task source: {source}", ""]
    for env in envs.split():
        lines += env_section(env=env, paths=PracticePaths(parameters=parameters, env_name=env, model_name=model_name,
                                                          source=source), n_examples=n_examples)
    with open(out, "w") as f:
        f.write("\n".join(lines))
    log_info(f"Report -> {out}", parameters=parameters)


if __name__ == "__main__":
    main()
