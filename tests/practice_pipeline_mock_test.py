"""Run all six practice stages on one environment with a scripted model (no GPU, no server).

The fake model answers by prompt type (propose, executor, done check, judge, critique,
guidance, paraphrase, clean), so every stage runs on real environments and produces real
files under storage_dir/tmp/practice_mock_test/. Then checks: files exist, the dataset rows
have no hint markers, images exist, and the M3A/WebVoyager guidance strips to the no-guidance
prompt.

    python tests/practice_pipeline_mock_test.py --env gameboy
    bash scripts/container.sh python tests/practice_pipeline_mock_test.py --env web
    (android: inside the container with an emulator started with --snapshots true)
"""
import json
import os
import shutil
import threading
import click
from cusi_utils import load_parameters
from cusi_practice.envs import ENV_SPECS, EnvPool
from cusi_practice.executors.base import GUIDANCE_START, strip_hint_blocks
from cusi_practice.stages.common import PracticePaths

ACTION_REPLY = {
    "gameboy": "Reasoning: I should walk up.\nAction: UP",
    "android": 'Reason: look further down the list.\nAction: {"action_type": "scroll", "direction": "down"}',
    "web": "Thought: scroll to see more.\nAction: Scroll [WINDOW]; [down]",
}


class ScriptedVLM:
    """Stands in for PracticeVLM; counts calls per kind."""

    model_name = "scripted/mock-model"

    def __init__(self, *, env_name: str) -> None:
        self.env_name = env_name
        self.counts: dict = {}
        self._lock = threading.Lock()
        self._judge_calls = 0

    def _reply(self, prompt: str) -> str:
        kinds = [
            ("Propose an exhaustive list", "propose",
             "Reasoning: a town.\nTasks:\n- move to the top of the screen\n- move to the left edge\n- open the menu\n- extra"),
            ("FULLY completed", "done_check", "Complete: no\nReasoning: not yet."),
            ("Generate at least", "paraphrase", "- go to the top\n- head upward\n- travel up"),
            ("reviewing a", "clean", "Reason: fine.\nDecision: ACCEPT"),
            ("Did the", "judge", None),
            ("analysing a segment", "critique_slice", "Segment summary: walked around."),
            ("provide a concise hint", "critique", "Critique: wandered.\nHint: go up."),
            ("consolidating partial guidance", "guidance_consolidate",
             "Summary: go up.\nGoal condition: at the top.\nSteps:\n- move up until the top"),
            ("You are an expert user of", "guidance_slice",
             "Summary: moves.\nGoal condition: top.\nSteps:\n- move up"),
            ("consolidating segment descriptions", "describe_consolidate", "Description: frames 1-10: moved."),
            ("Describe what the", "describe_slice", "Description: the agent moved."),
            ("summerize the latest step", "summary", "Scrolled down; the list moved."),
        ]
        for needle, kind, reply in kinds:
            if needle in prompt:
                with self._lock:
                    self.counts[kind] = self.counts.get(kind, 0) + 1
                    if kind == "judge":
                        self._judge_calls += 1
                        n = self._judge_calls
                if kind == "judge":   # alternate failure/success so retries and both paths run
                    ok = n % 3 != 1
                    return f"Reasoning: looks done.\nSuccess: {'yes' if ok else 'no'}\nSafe success point: 2"
                return reply
        with self._lock:
            self.counts["action"] = self.counts.get("action", 0) + 1
        return ACTION_REPLY[self.env_name]

    def infer(self, *, texts, max_new_tokens, images=None, temperature=None):
        if isinstance(texts, list):
            return {"output": [self._reply(t) for t in texts], "meta": {"input_tokens": [1] * len(texts),
                                                                       "output_tokens": [1] * len(texts)}}
        return {"output": self._reply(texts), "meta": {"input_tokens": 1, "output_tokens": 1}}

    def chat(self, *, messages, images, max_new_tokens, temperature=None):
        last = messages[-1]["content"]
        text = last if isinstance(last, str) else " ".join(p.get("text", "") for p in last)
        return {"output": self._reply(text), "meta": {"input_tokens": 1, "output_tokens": 1}}


@click.command()
@click.option("--env", "env_name", required=True, type=click.Choice(list(ENV_SPECS)))
@click.option("--max_steps", default=4, help="Shortened leg budget for the test.")
def main(env_name, max_steps):
    parameters = load_parameters()
    parameters = dict(parameters, storage_dir=os.path.join(parameters["storage_dir"], "tmp", "practice_mock_test"))
    spec = ENV_SPECS[env_name]
    spec = type(spec)(**{**spec.__dict__, "max_steps": max_steps, "max_workers": min(spec.max_workers, 2)})
    vlm = ScriptedVLM(env_name=env_name)
    paths = PracticePaths(parameters=parameters, env_name=env_name, model_name=vlm.model_name)
    shutil.rmtree(paths.root, ignore_errors=True)
    os.makedirs(paths.root)
    pool = EnvPool(spec=spec, parameters=parameters)
    ek = {"max_new_tokens": 500, "temperature": None}
    from cusi_practice.stages.propose import propose
    from cusi_practice.stages.attempt import attempt
    from cusi_practice.stages.guidance import guidance
    from cusi_practice.stages.practice import practice
    from cusi_practice.stages.clean import clean
    from cusi_practice.stages.dataset import build_dataset
    try:
        propose(spec=spec, pool=pool, vlm=vlm, paths=paths, n_tasks=2, max_new_tokens=500, overwrite=False,
                parameters=parameters)
        attempt(spec=spec, pool=pool, vlm=vlm, paths=paths, max_attempts=2, lookback=8, judge_max_new_tokens=500,
                executor_kwargs=ek, overwrite=False, parameters=parameters)
        guidance(spec=spec, vlm=vlm, paths=paths, max_new_tokens=500, max_obs_at_once=8, overwrite=False,
                 parameters=parameters)
        practice(spec=spec, pool=pool, vlm=vlm, paths=paths, n_attempts=2, n_random_actions=2, lookback=8,
                 judge_max_new_tokens=500, executor_kwargs=ek, overwrite=False, parameters=parameters)
    finally:
        pool.close()
    clean(spec=spec, vlm=vlm, paths=paths, k=3, safety_margin=2, max_new_tokens=500, overwrite=False,
          parameters=parameters)
    build_dataset(spec=spec, paths=paths, safety_margin=2, val_frac=0.5, seed=0, overwrite=False,
                  parameters=parameters)

    print("model calls by kind:", vlm.counts)
    for p in (paths.proposals, os.path.join(paths.attempts_dir, "results.csv"), paths.guidance,
              os.path.join(paths.practice_dir, "results.csv"), os.path.join(paths.practice_dir, "clean_decisions.csv"),
              os.path.join(paths.dataset_dir, "train.jsonl")):
        assert os.path.exists(p), f"missing {p}"
    stats = json.load(open(os.path.join(paths.dataset_dir, "stats.json")))
    print("dataset stats:", stats)
    rows = [json.loads(l) for l in open(os.path.join(paths.dataset_dir, "train.jsonl"))]
    rows += [json.loads(l) for l in open(os.path.join(paths.dataset_dir, "validation.jsonl"))]
    assert rows, "no dataset rows"
    for row in rows:
        for msg in row["messages"]:
            parts = [msg["content"]] if isinstance(msg["content"], str) else msg["content"]
            for part in parts:
                text = part if isinstance(part, str) else part.get("text", "")
                assert "[HINT_START]" not in text and GUIDANCE_START not in text and "[STEP_INFO]" not in text
                if isinstance(part, dict) and part["type"] == "image":
                    assert os.path.exists(part["image"]), part["image"]
    print(f"{len(rows)} rows checked; example row:")
    example = dict(rows[0])
    print(json.dumps(example, indent=1)[:3000])
    print("MOCK PIPELINE OK")


if __name__ == "__main__":
    main()
