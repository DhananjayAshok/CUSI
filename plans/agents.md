# Agents plan: GameBoyRL's supervisors on all three envs, and episode artifacts

Written 2026-10-05. **Sections 1–6 built 2026-10-06** (status and differences from the plan in
section 8); the section 7 ablation ran on 2026-10-06, with results in `plans/results.md`. Builds on `plans/eval_plan.md` (evaluation through our envs, done and
shown equivalent to the native harnesses) and the GameBoy executor port (all 9 GameBoyRL arms,
byte-identical requests against native on a mock server).

**Goal:** the GameBoyRL supervisor arms (`baseline`, `revision`, `subgoal`,
`info_subgoal_parametric`, `info_subgoal_retrieval`) wrap any of our executors (GameBoy arms,
M3A, WebVoyager) on any of the three envs, through `run_eval.py <env> --supervisor ...`, with one
episode-artifact format that a debug panel can read.

---

## 1. What GameBoyRL's supervisors are (`GameBoyRL/execution/supervisors/`, ~1,900 lines)

| Arm | Class | What it does |
|---|---|---|
| `baseline` | `DummySupervisor` | one executor leg, whole budget, no hint, no self-termination |
| `revision` | `RevisingSupervisor` | short legs (`max_leg_steps` 5); after a failed leg: critique (slices of frames → consolidate), write a resume hint, retry; one target (the task), cleared only by the env |
| `subgoal` | `SubgoalSupervisor` | a VLM plan of visually checkable steps; each step is a target in the revision loop, judged by a VLM (slice judge → consolidate on the final screen); regression check (did a later failed leg undo an earlier step?); plan-flaw check → replan; the last target is cleared only by the env |
| `info_subgoal_*` | `InfoSubgoalSupervisor` | `subgoal` whose planner also reads an info document: `retrieval` (filtered + distilled from documents built from past trajectories) or `parametric` (written by the model from its priors) |

Success is always the environment's verdict; the VLM judge only advances a plan.
Records: `SupervisorReport.event_log` interleaves the supervisor's own calls
(`SupervisorVLMCallRecord`: stage, prompt, images, response, tokens) with every executor leg's full
`ExecutorReport`; plus `narrative` (one summary per leg), `step_log` (per target: attempts,
termination reasons, verdicts, hints, revisions) and the plan.

Couplings to GameBoy: the current screen (`env.get_info()["core"]["current_frame"]`), the executor
contract (a GameBoyRL executor runs its leg in `__init__` and leaves an `ExecutorReport`), the
prompts ("a game of [GAME]", "the player", "do NOT name buttons"), and `from utils import ...`
(GameBoyRL's `utils` package, which cannot be imported next to WebVoyager's).

---

## 2. Decisions (2026-10-05)

1. **Prompt wording for Android / Web: our choice** (there is no reference):
   - GameBoy keeps GameBoyRL's prompts byte for byte.
   - Android / Web substitute the `Domain` fields practice already uses
     (`cusi/practice/prompts.py`): "a game of [GAME]" → `domain` ("an Android phone" /
     "a web browser"), "player" → `actor` ("agent"), "screen" stays "screen".
   - The planner's "do NOT name buttons or directions" rule becomes: "do NOT name element
     numbers or exact actions (e.g. `click [12]`, `scroll down`); describe the element by its
     visible text, icon or position" (element indices change on every screen, as buttons that
     worked once may be wrong from another screen).
   - Every step keeps a visually checkable termination condition ("... until <what the screen
     shows>").
2. **Budgets: unchanged.** Supervisor total = the native budget (GameBoy 175, Android
   `int(10 × complexity)`, Web 15); `max_leg_steps` 5, `max_attempts_per_target` 3,
   `final_attempt_multiplier` 3, as GameBoyRL. (On Web 15 steps is only three legs; left as is.)
3. **Frames: no downscaling.** Android judge slices send full 2400×1080 screenshots.
4. **Live sites:** a retry continues from the current page (no reset), as GameBoy continues
   from the current emulator state. Accepted.
5. **`info_subgoal_*`: built, not tested.** No info documents exist for any env yet. Retrieval
   errors clearly when its documents are missing; parametric needs none but is not run either.

Further choices made here:
- **Leg endings per env** (GameBoyRL's `allow_self_termination`):
  - GameBoy: its done-check (`Executor._check_task_complete`), as native.
  - Android / Web: the agent's own finishing action. On a non-final target it ends the leg
    without reaching the env (`finish_through_env=False`), meaning "I think this step is done",
    then the supervisor's judge decides. On the final target it goes through the env
    (`finish_through_env=True`), so the benchmark scores it and the episode ends there, as the
    native harness would.
- **Supervisor model:** `--supervisor_model`, default the executor's served model.
- **Hints:** the existing channels, unchanged: GameBoy `[HINT_START]` block, M3A
  `additional_guidelines`, WebVoyager's guidance block in the first message (all wrapped in
  markers, stripped for training rows as now).
- **Web judging with several legs:** auto_eval reads one chat. We write the final leg's chat
  (its first user message keeps the original "Now given a task: ..." text; the hint block sits
  inside markers after it, so auto_eval's task regex is unaffected) and every leg's set-of-mark
  screenshots numbered in episode order (auto_eval takes the last 15).

---

## 3. Design: `cusi/agents/supervisors/`

Ported (not imported), so it runs in the Web process. Every GameBoyRL prompt text is copied with
its placeholders; `tests/supervisor_prompts_test.py` (run in a GameBoy process) asserts that the
GameBoy rendering equals GameBoyRL's for every prompt.

| File | Port of | Notes |
|---|---|---|
| `report.py` | `SupervisorReport`, `SupervisorVLMCallRecord` | `event_log` of `SupervisorCall` and our `LegReport`s; `narrative`, `plan`, `original_plan`, `step_log`; token views as GameBoyRL |
| `base.py` | `Supervisor` | holds env, task, domain, budget, supervisor VLM, an executor factory; `current_frame()` = the env's current observation frame (Android: raw screenshot); `call_executor(task, hint, allow_self_termination, max_steps, final)` builds a fresh executor, runs `executor.run(...)` from the current observation, appends the `LegReport` |
| `prompts.py` | `supervisors/prompts.py` | texts with `[DOMAIN]`, `[ACTOR]` and the env-specific planner rule; `render(prompt, domain)` |
| `_format.py`, `checker.py` | same | action traces and slice summaries over `LegReport` (our `StepRecord.action_label()`, call tags, `Reasoning:` / `Reason:` / `Thought:` lines per env) |
| `dummy.py`, `revising.py`, `subgoal.py`, `info_subgoal.py` | same | logic unchanged |

GameBoy executor default: **`single_visual`** (single action policy, visual history; set
2026-10-05 in `cusi/agents/executors/gameboy.py` `DEFAULT_ARM`, so practice uses it too).

Executor side (small): `Executor.run` gains `self_terminate` (GameBoy: the done-check; M3A /
WebVoyager: the finishing action ends the leg) and keeps `finish_through_env` for the final
target. Each leg gets a fresh executor (fresh memory), as GameBoyRL.

Entry point: `run_eval.py <env> --supervisor {baseline,revision,subgoal,info_subgoal_parametric,
info_subgoal_retrieval} [--supervisor_model ...] [--max_leg_steps 5] [--info_docs ...]`, default
`baseline` (the native-comparable path, unchanged). `slurm/eval.sh` passes it through.

---

## 4. Tests

- **GameBoy, against native:** the mock-server request diff already used for the executors,
  extended so the mock answers the supervisor stages (plan lists, judge verdicts, critiques,
  hints, regression / plan-flaw yes/no). `revision` and `subgoal` × `single_visual` (the default, plus one
  other arm), a few tasks: native GameBoyRL supervisors vs ours, byte-identical requests.
- **Android / Web:** no native reference. Smoke runs on 2–3 tasks each per arm (`revision`,
  `subgoal`) with a served model: every stage runs, hints reach the executor prompt, legs chain
  from the current state, the final verdict comes from the env, artifacts are complete.
- **Not tested:** `info_subgoal_*` (decision 5).

---

## 5. Episode artifacts with supervisors

GameBoyRL archives one gzipped pickle per episode (`benchmark_scripts/common.save_report`:
the `SupervisorReport` with every leg, frame and prompt) and renders it to markdown pages with
frames and video links (`debug_scripts/benchmark.py`, including a paired view of two models on the
same task). Ours keeps the same content and structure, but as JSON + PNG, so any panel can read
it without our classes:

```
storage_dir/eval/<env>/<run>/
  config.json, results.jsonl, summary.json          as now (results rows gain supervisor fields)
  episodes/<task_id>/
    meta.json        task, env, model(s), executor arm, supervisor and its settings, success,
                     termination, budget used, tokens (supervisor / executor), timings, answer,
                     error; supervisor state: plan, original_plan, step_log (per target:
                     text, attempts with termination reasons, verdicts with reasoning, hints,
                     revisions), narrative; env extras (GameBoy emulator steps + subgoals,
                     Web judge verdict + reasoning)
    events.jsonl     the episode in order, one line per event:
                       {"kind": "supervisor", "i", "stage", "prompt", "images", "response", tokens}
                       {"kind": "leg", "i", "leg", "target", "hint", "final", "termination",
                        "n_steps", "calls": "legs/<leg>.jsonl"}
    legs/<leg>.jsonl one line per executor call in that leg (tag, decision, prompt or messages,
                     images, response, tokens, steps with action / parsed action / valid / error /
                     reward / frame_before / frame_after)
    frames/<n>.png   every image any call saw or any step produced, deduplicated across legs
    episode.mp4      step video (below)
  native/            env-native artefacts, unchanged (Web: task<id>/interact_messages.json +
                     screenshots for auto_eval; Android: episodes.jsonl, process_episodes.md)
```

`baseline` is the same format with no supervisor events and one leg. Practice legs could write
it too later.

**Step video (`episode.mp4`), all three envs.** Built by the episode writer from the frames it
already has, after the episode ends (no recording during the run): the initial frame, then the
frame after every env step, in episode order across legs, at a fixed rate (default 1 frame per
step, i.e. 1 fps). Each frame is stamped with a caption bar under the image: leg and target (with
supervisors), step number, the action taken (e.g. `UP`, `{"action_type": "click", "index": 7}`,
`Click [7]`), and `INVALID: <reason>` / the env's error when the step failed. Invalid decisions
(no env step) get a frame too: the unchanged screen with the invalid caption, so gaps are visible.
The frames are the unlabelled screenshots where the env has them (Android, Web); GameBoy frames
are upscaled 3× so the text is legible. Encoded with `imageio` + ffmpeg (H.264); the frames are
padded to even sizes. It shows what the agent saw, not what happened between observations
(animations, loading); GameBoy's native `save_video` (every emulator frame) stays available via
`--save_video`. Real Android / Web recording is out of scope for now.

A panel then has three levels: the run (results table, filters by success / termination /
arm), the episode (plan timeline: targets → attempts → verdicts, hints, with supervisor
calls inline), and the leg (step through frames with prompt, reply, parsed action, error).
GameBoyRL's paired view (two runs, same task, side by side) is worth copying.

---

## 6. Order of work

1. `cusi/agents/supervisors/` port + `tests/supervisor_prompts_test.py`.
2. `Executor.run(self_terminate=...)` and per-env leg endings.
3. `run_eval.py --supervisor` for the three envs; Web's multi-leg auto_eval inputs.
4. Episode writer (`cusi/eval/episode.py`, section 5) with the step video, used by `baseline` too.
5. GameBoy mock parity for `revision` / `subgoal`.
6. Android / Web smoke runs.
7. `slurm/eval.sh` + `scripts/slurm/eval.sh` with `--supervisor`.


---

## 7. Ablation: the best supervisor per config (after sections 1–6 are done)

**Question:** for each config (env × model), which supervisor gives the best test-set success,
and what does each cost in time and tokens?

**Grid** (fixed executor per env: GameBoy `single_visual`, Android M3A, Web WebVoyager):

| Axis | Values |
|---|---|
| env | gameboy, android, web |
| model (executor and supervisor) | `google/gemma-4-26b-a4b-it`, `google/gemma-4-31b-it` |
| supervisor | `baseline`, `revision`, `subgoal` (`info_subgoal_*` excluded: no info documents yet, decision 5) |

18 runs. Everything else at the defaults above (native budgets, leg length 5, temperatures as each
native harness).

**Test sets (full):**
- GameBoy: `pokemon_red`, all 49 benchmark tasks (CUSI's game). The other 21 GameBoyWorlds games
  (~1,000 tasks) are out unless asked for.
- Android: the `android_world` family, 116 tasks × 1 instance, seed 30.
- Web: `WebVoyager_data.jsonl` (642 tasks) minus the sites `plans/web_voyager_plan.md` excludes
  (blocked / captcha / login); that filtered task file must exist first.
- **Web judge fixed** for all Web runs: `google/gemma-4-31b-it` through the unchanged
  `auto_eval.py`, so settings differ only in the agent, not the grader.

**Runs:** `slurm/eval.sh` per (env, model, supervisor); one vLLM server per job on its own port,
fresh, prefix caching off. To fit the time, the GameBoy and Web runners take `--workers N`
(tasks in parallel against the one server; Web ≤ 4 browsers); Android runs one emulator per job
(several jobs in parallel on different emulator ports if nodes allow).

**Timing recorded for future reference** (already per task in `results.jsonl` `seconds`; add):
per run wall-clock, vLLM start-up time, tasks/hour, mean and p90 seconds per task, model calls
and tokens per task (supervisor vs executor), env time vs model time (from call timestamps),
and the hardware (GPU type / count, node). Rough expectations before running, to be replaced by
measurements:

| env | tasks | baseline (est.) | supervisors (est.) |
|---|---|---|---|
| GameBoy | 49 × ≤ 175 steps, ~2 calls/step with visual history | ~10–15 h at 1 worker; ~2–3 h at 6 | 1.5–3× baseline |
| Android | 116 × ≤ 10×complexity steps, 2 calls/step + 2 s pause | ~5–8 h | 1.5–3× |
| Web | ~550–640 × ≤ 15 steps | ~12–15 h at 1 browser; ~3–4 h at 4 | 1.5–3× |

**Analysis (written to `plans/results.md` when done, not here; README points to it):**
- Per config: success rate per supervisor with 95% bootstrap CIs, and paired comparisons against
  `baseline` (`cusi.utils` `paired_bootstrap`, tasks paired); the best supervisor per config.
- Breakdowns: GameBoy subgoals reached; Android per task template and difficulty
  (`process_episodes`); Web per site.
- Cost: time and tokens per task per supervisor; success per GPU-hour.
- Where supervisors help or hurt, read from the episode artifacts (plans that were wrong, hints
  that misled, judges that advanced a step too early).
- Caveat: Web on live sites is noisy run to run (native agreed with itself on 20/24 tasks in the
  parity check), so small Web differences are within noise.

---

## 8. Status (2026-10-06): what was built, and where it differs from the plan

**Built:**

| Plan item | Where |
|---|---|
| Supervisor port | `cusi/agents/supervisors/` (`base`, `dummy`, `revising`, `subgoal`, `info_subgoal`, `checker`, `_format`, `prompts`, `parsing`, `report`); `build_supervisor()` holds GameBoyRL's per-arm settings |
| Prompt test | `tests/supervisor_prompts_test.py` |
| `--supervisor` on all three runners | `cusi/eval/{gameboy,android,web}.py`, with the shared options and worker pool in `cusi/eval/supervision.py` |
| Episode writer and step video | `cusi/eval/episode.py`, used by every arm, `baseline` included |
| Slurm pair | `slurm/eval.sh` + `scripts/slurm/eval.sh`: fresh vLLM on a per-job port, prefix caching off, `timing.jsonl` per run |
| Web task filter | `configs/project_vars.yaml` `web_excluded_sites` + `cusi.envs.webvoyager.filtered_task_file()`; `--task_file cusi` = 429 of 643 tasks (471 before BBC News was excluded) |
| Ablation tables | `report_ablation.py` (removed 2026-10-06; output in `storage_dir/eval/ablation_tables.md`) |

**Tests:**
- **Prompt test.** The GameBoy prompts equal GameBoyRL's, all 9. Android and Web keep every placeholder and have no game wording left.
- **GameBoy mock parity** (`storage/eval/gameboy/mock_sup_v1`, 3 tasks, 30 steps). Native GameBoyRL arm vs ours, with byte-identical requests in all 6 configurations:

  | supervisor | `single_visual` | `sequence_actions` |
  |---|---|---|
  | baseline | 136/136 | 48/48 |
  | revision | 197/197 | 90/90 |
  | subgoal | 279/279 | 144/144 |

  - The mock answers every supervisor stage, and every branch was taken: judge yes/no, regression yes/no, plan-flaw yes/no (replans), done-check endings, truncation.
  - `tests/eval_parity_gameboy.py native --supervisor` runs the native arm.
- **Android/Web smoke runs:** see below.

**Differences from the plan:**
- **No new `Executor.run(self_terminate)`.** The existing flags do the job. `Supervisor.call_executor` passes `allow_done_check=allow_self_termination`, which is the GameBoy done-check. It also passes `finish_through_env=not allow_self_termination`, which for M3A and WebVoyager means a finish on a non-final leg ends the leg, while a finish on the final target or under baseline goes through the env.
  - Executors now record `final_obs`/`final_info`, so the next leg starts from the current screen.
  - Every model call records its `seconds`.
- **Android/Web wording.** Practice's `Domain` is not reused; `cusi.agents.supervisors.prompts.SupervisorDomain` holds the supervisors' own wording:
  - "agent", "on an Android phone" / "in a web browser";
  - each env's action names for the hint;
  - the planner rule against naming element numbers;
  - the reply key the executor states its reasons under (GameBoy `Reasoning`, M3A `Reason`, WebVoyager `Thought`).
- **Action lists.** On Android/Web, an env step the env rejected shows its error in action lists. A finish that ended a leg without reaching the env shows as `FINISH`. GameBoy lists are unchanged.
- **Two loop additions, both for the envs with a finishing action:**
  - A finish that did not reach the env is charged one step of the budget, as the native harnesses count it.
  - `model_error` (WebVoyager's run.py ends a task on a model error) ends the episode.
  - Neither ever happens on GameBoy, so parity holds.
- **GameBoy budgets under a supervisor are GameBoyRL's native ones.** Executor `max_new_tokens` is 8000 under revision/subgoal/info (its `--executor_max_new_tokens`) against 2000 for baseline. Supervisor calls get 5000. Android and Web keep their native executor settings.
- **Supervisor temperature** is unset (the server default), as GameBoyRL.
- **`info_subgoal_*` is built but not run.**
  - GameBoy retrieval loads `--info_docs` with GameBoyRL's loader.
  - GameBoy parametric uses GameBoyRL's generator, cached under `storage_dir/eval/info_docs/`.
  - Android and Web stop with "no info documents exist for <env> yet".
- **Not ported:** the checker arm (`AttemptCheckerSupervisor` plus the `CRITIQUE_*`/`DESCRIBE_*`/`JUDGE_BINARY_*` prompts), which is a GameBoyRL data-generation tool, not a benchmark arm.
- **Per-task records.**
  - `trajectories/<id>.pkl.gz` (GameBoy) now holds the SupervisorReport; the parity `compare` reads both formats.
  - Results rows gain the supervisor counts, tokens split, timings, plan and `n_replans`.
  - `summary.json` gains per-task time and token statistics.
- **Web judging in the ablation.** The judge is the 31B model. A 26B run's job stops its server after the agent runs and serves the judge on the same port (`--judge_model`). `auto_eval` is unchanged.
- **Android parallelism:** `--shard k/n`, one emulator per job, one run directory per shard (`<run>_s<k>`); `report_ablation.py` merges the shards.

---

## 9. Executor memory across legs, 10-step legs (2026-10-06, after the ablation)

The ablation (`plans/results.md`) showed two step costs on Android and Web:
- Legs cut off at 5 steps rarely clear a plan step: 13–19%, against 77–86% for legs the agent ended itself.
- Retries redo work after the executor's memory is reset. A `subgoal` retry re-types text the previous leg already typed 71–84% of the time, against 28–47% within a baseline episode.

Changes:
- **Memory.** Every executor takes `previous_history` (its own type: GameBoyRL history records, M3A step summaries, the WebVoyager chat) and returns `history()` after its leg. `Supervisor.call_executor` passes each leg's history to the next leg and never resets it within an episode, replans included.
  - WebVoyager continues the previous chat with a fresh first message carrying the new task, hint and screenshot.
  - `interact_messages.json` still holds only the final leg's own messages, so `auto_eval` reads the real task.
- **Leg cap.** `max_leg_steps` defaults to 10 (`cusi.agents.supervisors.DEFAULT_MAX_LEG_STEPS`, `--max_leg_steps`).
- **Parity.** Both changes depart from GameBoyRL, which builds every leg with empty memory and uses 5-step legs. GameBoy parity with native no longer holds for supervised arms. Baseline (one leg) is unaffected.

**Check:** a GameBoy mock run (`storage/eval/gameboy/mock_mem_v1`) confirms that each leg's first prompt carries the previous legs' recent actions, and that legs run up to 10 steps.

**Quick test** (queued after the 2026-10-06 maintenance window): `subgoal` with both changes, both models.
- Subsets: 40 Android tasks and 120 Web tasks (seed 0; `storage_dir/eval/mem10_subsets/`), runs `mem10_<model>_subgoal`.
- Comparison: the ablation's runs on the same tasks, plus a Web 31B baseline rerun on the same tasks to measure live-site drift.
