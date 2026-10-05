"""Per-task eval records, shared by the three runners.

    out = EvalRun(directory=storage_dir/eval/<env>/<run_name>)
    if out.done(task_id): skip
    out.write(EvalRow(...))
    out.summary()     # {"n", "n_scored", "success_rate", "n_failed_to_run"}

A row's success is 1.0 / 0.0, or None when the task failed to run (env crash, recovery,
repeated browser death): such rows are excluded from the success rate, as AndroidWorld's
harness does. Resume: tasks with a row are skipped, unless the row failed to run and
rerun_failed is set.
"""
import json
import os
from dataclasses import asdict, dataclass, field
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


class EvalRun:
    FILE = "results.jsonl"

    def __init__(self, *, directory: str, config: Optional[dict] = None) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, self.FILE)
        if config is not None:
            with open(os.path.join(directory, "config.json"), "w") as f:
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
        rows = self.rows()
        scored = [r for r in rows if r["success"] is not None]
        out = {"n": len(rows), "n_scored": len(scored), "n_failed_to_run": len(rows) - len(scored),
               "success_rate": sum(r["success"] for r in scored) / len(scored) if scored else None}
        with open(os.path.join(self.directory, "summary.json"), "w") as f:
            json.dump(out, f, indent=1)
        return out


def call_tokens(report) -> tuple:
    """(input, output) tokens over a LegReport's calls."""
    return (sum(c.input_tokens or 0 for c in report.calls), sum(c.output_tokens or 0 for c in report.calls))
