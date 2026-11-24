# Tree annotation: tasks done along the tree's paths, in the form of the train tasks

Code: `cusi/search/annotate.py` (the command), `cusi/search/path_notes.py` (prompts, parsers),
`cusi/search/task_context.py` (train-task context); Slurm pair `annotate`. Readers: `cusi/eval/task_coverage.py` and
`task_diversity.py` (`proposed_tasks`, `--annotation_model`), practice (`cusi/practice/stages/attempt.py`,
`tree_tasks`), `sync_data.py` (finished check). How the design was reached: plans/decisions.md, 2026-11-17 to 18.

## What a task is

A task is what a path of the tree actually did, as long as it took, named in the form of the benchmark's train tasks
(configs/splits/benchmark_splits.json, split = train only). Every task is feasible by construction and comes with
the path that does it.

## Train-task context (`task_context.py`)

- Flavours from the split's train tasks only: GameBoy by the benchmark CSV's `task_category` per game, Web by site,
  Android by app; written by `python -m cusi.search.task_context build` to
  `configs/annotation/<env>_train_flavours.json`.
- Every call sees a seeded sample of `--n_per_flavour` (3) train tasks per flavour, fixed for every call and tree.
  A non-train id raises; the split file's hash and the sample's ids go in meta.json.
- A tree is annotated only if its game / env has train tasks (GameBoy: Pokemon Red, Link's Awakening).

## Stage 1: transition reports

A VLM-explorer search reports every edge it makes (`description`, `window_descriptions` in its base rows), and
those are used as they are. Every other edge gets one from the annotation model's `TransitionDescriber` (its parent's
screen, then every step to it), written to the model's `transitions.jsonl`. meta.json counts both.

## Stage 2: forward extension (one call per edge per chain)

Every node starts a chain on every branch below it (copied at forks). Each call extends a chain by one edge and sees
the chain's start screen, the screens before and after the edge, the reports so far, the edge's actions and report,
the train sample, and the current task if any.

- No task yet (BUILD): `TOO_SHORT` (keep extending; its Partial line names the best smaller task completed so far,
  or NONE) or `TASK`. A task changes something, answers a question shown on screen, or (GameBoy only, where the
  train flavours have it) reaches a named place or menu; it must be achieved, specific, named from the screens, and
  not accidental typing.
- A task named (EXTEND): `CONTINUE` (no effect on it, or only opening further screens on its way; its end does not
  move), `REVISE` (on the reached screens the path changes something or reads out a specific answer: a fuller task;
  the earlier name stays as a nested task), `NEW` (something unrelated; the chain ends).
- A chain stops at NEW, a leaf, a challenge page, or past `--max_steps` (30) env steps; one that never named a task
  keeps its latest Partial task.

Web and Android (`path_notes.STEP_ENVS`), whose edges run 5-19 steps: the call also reads the edge's screen-by-screen
report and names the action that completed the task, so a task ends at (edge end node, `end_step`) with a `replay`
(the edge's start node and its first `end_step` actions) and `restorable`; a task needs a checkable outcome (no "find
information about X", no navigation tacked onto a done task, no search for an exploration-style label).

## Stage 3: fit gate (one call per candidate)

Given the start screen, the path's reports, the end screen (Web / Android: the last edge's screens up to the
completing action) and the train sample: ACCEPT only if (a) the path completes the task, (b) the end screen shows it
done or shows the answer, (c) it reads like a train task, (d) it is not a one-click wrapper. Web / Android: accepted
paraphrases of one start are merged (`alt_ends`).

## Output

`<run>/annotation/<annotation model short name>/` (models side by side): `transitions.jsonl`, `calls.jsonl`,
`gate_calls.jsonl` (caches: a rerun resumes and rebuilds the same tasks), `candidates.jsonl`, `tasks.jsonl` (accepted),
`rejected.jsonl` (with the failed criteria and a reason), `meta.json`, `summary.json` (written last).

## Running it

`slurm/annotate.sh --env <env> --run <run>[,<run>...] --model <model>`: one vLLM server per job, the listed runs
annotated `--parallel` at a time; `--max_starts` annotates a seeded sample of start nodes (and describes only the
edges they reach). Scoring: `task_coverage --annotation_model <model> --test_split test` and `task_diversity
--annotation_model <model>`, reported as coverage@8 (share of test tasks whose best judge score is >= 8), the mean best
judge score (judge_max@k) and the mean best cosine (top1_cos).
