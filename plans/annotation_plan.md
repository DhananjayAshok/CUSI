# Tree annotation: tasks done along the tree's paths, in the form of the train tasks

Status: implemented (2026-11-17) and the only annotation of a search tree. Code: `cusi/search/annotate.py` (the
command), `cusi/search/path_notes.py` (prompts, parser), `cusi/search/task_context.py` (train-task context); Slurm
pair `annotate`. Readers: `cusi/eval/task_coverage.py` and `task_diversity.py` (`proposed_tasks`), practice
(`cusi/practice/stages/attempt.py`, `tree_tasks`), `sync_data.py` (finished check). Pilot results per prompt version:
plans/decisions.md, 2026-11-17.

## Why

Asking per screen "what tasks could start here" gives imagined tasks (never done, invented values), and asking per
edge "what did these 2-5 steps achieve" gives one-step micro-goals ("open the X page"). Here a task is what a path
of the tree actually did, as long as it took, named in the form of the benchmark's train tasks
(configs/splits/benchmark_splits.json, split = train only), so every task is feasible by construction and comes
with its path to replay.

## Tree shapes (s0 trees)

| Tree | Nodes | Leaves | Max / median depth | Ancestor-descendant pairs | Steps per edge (median) |
|---|---|---|---|---|---|
| Red random | 2,910 | 828 | 51 / 13 | 43,811 | 3 |
| Red VLM no-typing | 1,415 | 334 | 36 / 10 | 16,050 | 2 |
| Web random | 5,854 | 1,002 | 151 / 57 | 335,212 | 4 |
| Web VLM + typing | 1,707 | 293 | 90 / 23 | 51,353 | 2 |
| Android random | 357 | 129 | 19 / 4 | 1,828 | 5 |
| Android VLM + typing | 280 | 75 | 24 / 4 | 1,710 | 3 |

Edges are short, so a task spans several; asking about every ancestor -> descendant pair is out of reach, hence
forward extension.

## Train-task context (`task_context.py`)

- Flavours from the split's train tasks only: GameBoy by the benchmark CSV's `task_category` per game, Web by site,
  Android by app; written once by `python -m cusi.search.task_context build` to
  `configs/annotation/<env>_train_flavours.json`.
- Every call sees a seeded sample of `--n_per_flavour` (3) train tasks per flavour, fixed for every call and tree
  (one prompt prefix; the same context for random and VLM trees). A non-train id raises; the split file's hash and
  the sample's ids go in the annotation meta.

## Stage 1: transition reports

Every edge (a node's segment from its parent) without a `description` gets one from `TransitionDescriber` (its
parent's screen, then every step to it), as the VLM-explorer search writes them live; written to nodes.jsonl as update
rows (`description`, `window_descriptions`). Later calls read these reports as the text of a path.

## Stage 2: forward extension (one call per edge per chain)

Every node starts a chain on every branch below it (copied at forks). Each call extends a chain by one edge `v -> w`
and sees the chain's start screen, `v`'s and `w`'s screens, the reports of the chain so far, the new edge's actions
and report, the train sample, and the current task if any.

- No task yet (BUILD): `TOO_SHORT` (keep extending; its Partial line names the best smaller task completed so far,
  or NONE) or `TASK` (the chain completes it). A task changes something, answers a question shown on the last screen,
  or reaches a specific named place / page / screen; it must be achieved, specific, named from the screens, and not
  accidental typing.
- A task named (EXTEND): `CONTINUE` (no effect on it; the chain goes on, the task's end stays), `REVISE` (a more
  complete or specific version of the same goal; the earlier name stays as a nested task ending where it ended; the
  same name again only moves the end), `NEW` (something else starts; the chain ends, the task at `v`).
- A chain stops at NEW, a leaf, a challenge page, or past `--max_steps` (30) env steps. A chain that never named a
  task keeps its latest Partial task (`"partial": true`), or nothing if it named none.

Output, `<run>/annotation/p<prompt version>/`: `calls.jsonl` (cached verdicts keyed by the chain's node path; a rerun
resumes and rebuilds the same tasks), `tasks.jsonl` (`start`, `task`, `end`, `path`, `n_edges`, `n_steps`,
`revisions`, `nested`, `partial`), `meta.json` (model, settings, sample, split hash; refuses mixed settings),
`summary.json` (counts; written last).

## Cost

About 3-3.6 calls per node measured on the Android s0 trees (the worst case, every chain to the 30-step cap, is 3.3-7.1
per node on the s0 trees). A Slurm job with prefix caching does about 1.4 calls/s alone: a Web random tree is a few
hours, Android and Link's Awakening trees minutes to an hour. Plus stage 1: one call per edge without a report.

## Scoring

`task_coverage --test_split test` scores the tasks against the split's test tasks only (the annotator sees train
tasks, so scoring against all benchmark tasks would count train tasks); `task_diversity` is unchanged. Both read
`tasks.jsonl` by default.

## Fairness for the random vs VLM comparison

Same annotator, train sample and `max_steps` for every tree; the cap is in env steps, so random's slightly longer edges
do not buy longer tasks; report coverage and diversity raw and per start node (random trees are 2-4x larger).
