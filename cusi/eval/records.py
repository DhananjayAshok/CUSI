"""
Per-task eval records. A success of None means the task failed to run and is excluded from the rate.
"""
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
import click
from typing import Optional


@dataclass
class EvalRow:
    env: str
    task_id: str
    task: str
    success: Optional[float]
    n_steps: int
    n_invalid: int
    termination_reason: Optional[str]
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    answer: Optional[str] = None
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)


#: Config keys that don't change results, so they may differ between launches of one run.
OPERATIONAL_KEYS = {"workers", "vllm_base_url", "supervisor_vllm_base_url", "judge_vllm_base_url", "rerun_failed",
                    "console_port", "grpc_port"}
#: Keys a later launch may fill in once ("none" first): Web runs the agent, then judges in a second call.
FILLED_LATER_KEYS = {"judge_model_name"}


def run_options(command):
    """--overwrite and --ignore_config_violation, for every runner."""
    command = click.option("--ignore_config_violation", is_flag=True,
                           help="Resume a run whose stored config differs (kept as stored).")(command)
    return click.option("--overwrite", is_flag=True, help="Delete the run directory first.")(command)


def config_violations(*, stored: dict, new: dict) -> list:
    out = []
    for key in sorted(set(stored) | set(new)):
        if key in OPERATIONAL_KEYS:
            continue
        old, cur = json.loads(json.dumps(stored.get(key), default=str)), json.loads(json.dumps(new.get(key), default=str))
        if old == cur or (key in FILLED_LATER_KEYS and (old in (None, "none") or cur in (None, "none"))):
            continue
        out.append(f"{key}: {old!r} -> {cur!r}")
    return out


class EvalRun:
    FILE = "results.jsonl"

    def __init__(self, *, directory: str, config: Optional[dict] = None, overwrite: bool = False,
                 ignore_config_violation: bool = False) -> None:
        """A run directory may only be resumed with the config it was started with (see run_options)."""
        self.directory = directory
        if overwrite and os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, self.FILE)
        if config is None:
            return
        config_path = os.path.join(directory, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                stored = json.load(f)
            violations = config_violations(stored=stored, new=config)
            if violations and not ignore_config_violation:
                raise click.UsageError(f"{directory} was started with a different config ({'; '.join(violations)}). "
                                       f"Use a new --run_name, --overwrite, or --ignore_config_violation.")
            if violations:
                print(f"WARNING: resuming {directory} despite config differences ({'; '.join(violations)}); "
                      f"config.json keeps the stored values.")
                merged = {**config, **stored}
            else:
                merged = {**stored, **config}
            for key in FILLED_LATER_KEYS:     # filled once, never reset to "none"
                values = [v for v in (stored.get(key), config.get(key)) if v not in (None, "none")]
                if values:
                    merged[key] = values[-1] if not violations else values[0]
            config = merged
        with open(config_path, "w") as f:
            json.dump(config, f, indent=1, default=str)

    def rows(self) -> list:
        if not os.path.exists(self.path):
            return []
        latest = {}
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    latest[row["task_id"]] = row
        return list(latest.values())

    def done(self, task_id: str, *, rerun_failed: bool = True) -> bool:
        for row in self.rows():
            if row["task_id"] == task_id:
                return not (rerun_failed and row["success"] is None)
        return False

    def write(self, row: EvalRow) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(row), default=str) + "\n")

    def summary(self) -> dict:
        """Success over the scored rows, plus time and token statistics; also writes summary.json."""
        rows = self.rows()
        scored = [r for r in rows if r["success"] is not None]
        out = {"n": len(rows), "n_scored": len(scored), "n_failed_to_run": len(rows) - len(scored),
               "success_rate": sum(r["success"] for r in scored) / len(scored) if scored else None}
        seconds = sorted(float(r.get("seconds") or 0.0) for r in rows)
        if seconds:
            out.update(task_seconds_total=round(sum(seconds), 1), task_seconds_mean=round(sum(seconds) / len(seconds), 1),
                       task_seconds_p90=round(seconds[min(len(seconds) - 1, int(0.9 * len(seconds)))], 1),
                       input_tokens_mean=round(sum(r.get("input_tokens") or 0 for r in rows) / len(rows)),
                       output_tokens_mean=round(sum(r.get("output_tokens") or 0 for r in rows) / len(rows)))
        with open(os.path.join(self.directory, "summary.json"), "w") as f:
            json.dump(out, f, indent=1)
        return out


def call_tokens(report) -> tuple:
    """(input, output) tokens over a LegReport's calls."""
    return (sum(c.input_tokens or 0 for c in report.calls), sum(c.output_tokens or 0 for c in report.calls))
