# Final port: one fixed supervisor × executor per env

Goal: replace the experiment scaffolding (feature switches, prototype wrappers, the supervisor zoo) with exactly the
setups that won, and nothing else. Everything in `tmp_*.py`, `--features`, `--supervisor` and the alternative
supervisors is scratch and goes. The only extra case the code must support is running **without a task**
(exploration / practice-style rollouts), for both executor and supervisor.

## Update (2026-10-19): what changed after this plan was written
- **GameBoy ships the subgoal supervisor, not the monitor.** On held-out the monitor tied subgoal (-3.4 [-9.5, +2.0])
  and the user chose subgoal. Wherever the sections below say "monitor" / `MonitorSupervisor`, read
  `cusi/agents/supervisors/subgoal.py` (`SubgoalSupervisor`): GameBoyRL's prompts verbatim, 10-step legs, 3 attempts
  per step, 2 replans, 8 frames per slice. The monitor was never ported.
  - The GameBoy executor keeps GameBoyRL's done-check, as `Executor.done_check` plus `run(self_terminate=...)`,
    set only on the subgoal supervisor's plan-step legs (contrary to "unused executor code removed" below).
  - No task: no plan; one executor leg with the whole budget.
  - Identity: a recorded replay of all 49 held-out subgoal episodes on the real emulator (decisions.md E8.1b).
- **The Web env drops Google itself** (E8.2b), and the **Android env's action menu** describes `input_text` as
  the env runs it (E8.2 fix). The env menus, which search and practice read, now match the agents' prompts.
- `merged_leg` takes the run's task, since a GameBoy plan-step leg's own task is the step.

## 0. What ships

| env | executor | supervisor | evidence |
|---|---|---|---|
| Web | WebVoyager + budget note (F10) + no Google action (F14) + 5 screenshots kept (F08=5) | none | full benchmark +10.0, held-out +9.5 |
| Android | M3A + notes (F27) + Minitap bundle F55–F61 | none | held-out +8.0 [+0.9, +15.8] |
| GameBoy | GameBoyRL executor, unchanged (`single_visual`) | **subgoal** (user, 2026-10-19): GameBoyRL's plan / judge / hint / replan loop, 10-step legs | held-out vs base +10.2 [+0.7, +20.4]; ties the monitor (E7.6) |

Model stays a required argument everywhere (never defaulted); supervisor and executor share the model and server.

## 1. Shape of the code after the port

One entry point per env, used by eval, practice and exploration alike:

```python
agent = make_agent(env_name=..., env=env, vlm=vlm, parameters=parameters)
report = agent.run(task=task_or_None, max_steps=..., obs=obs, info=info)   # -> SupervisorReport-compatible record
```

- `cusi/agents/agent.py` (new, ~80 lines): `make_agent` picks the env's fixed pair. Web/Android: the executor runs one
  leg, its finish goes through the env (what the baseline `DummySupervisor` did). GameBoy: `MonitorSupervisor`.
  The report format stays the same as now (`LegReport` / `SupervisorReport`), so the viewer, records and tables
  keep working.
- `cusi/agents/specs.py`: `EnvSpec.make_executor` → `EnvSpec.make_agent`; practice stages call it.
- Kept as is: `cusi/agents/records.py`, `frames.py`, `vlm.py`, `supervisors/report.py`, `supervisors/checker.py`
  (segment summaries, which the monitor uses).

## 1a. No switches: the best option is the only code path

Principle: the submodules (`GameBoyRL`, `android_world`, `WebVoyager`, `llm-utils`) are frozen libraries and never
edited. Root code imports their pieces and hard-codes the one winning configuration. There are no constructor
switches, no CLI flags and no YAML keys for design choices.

- **Design choices are module constants in the root wrapper.** They are never arguments.
  - `webvoyager.py`: `MAX_ATTACHED_IMGS = 5`; the budget note and Google removal are unconditional code.
  - `m3a.py`: the eight Minitap/notes pieces are plain code in `decide` / `after_step`.
  - `monitor.py`: `LEG_STEPS = 5`, `STUCK_REPEATS = 3`, `FRAMES_PER_SLICE = 8`.
  - `gameboy.py`: the `single_visual` arm, the low-level controller, `HISTORY_K` and
    `EXECUTOR_MAX_NEW_TOKENS = 8000`.
  - Per-env step budgets are constants (`WEB_MAX_STEPS = 15`, `GAMEBOY_MAX_STEPS = 175`, Android: AndroidWorld's
    per-task budget).
- **How submodule behaviour is changed without editing the submodule.** Every place we alter it is one root
  function that patches the text or wraps the call:
  - M3A's action prompt: the F55/F57 description edits, the delete tip, the inserted blocks.
  - WebVoyager's system prompt: Google removal.
  - AndroidWorld typing: our own `type_without_enter` instead of `execute_action` for `input_text`.
  - Each text patch is an anchored replace that **raises at import** if its anchor is missing. A submodule update
    that moves the text then fails loudly instead of silently running the unpatched prompt. This is the class of
    bug found in E6.2.
- **Constructors take only per-run things.**
  - `WebVoyagerExecutor(*, env, vlm, previous_history=None, parameters=None)`; the same for `M3AExecutor` and
    `GameBoyExecutor`.
  - `MonitorSupervisor(*, task, env, vlm, obs, info, max_steps, parameters)`.
  - There is no `arm=`, `notes=`, `show_budget=`, `max_leg_steps=`, `temperature=`, `max_new_tokens=` or
    `refresh_before_decide=`. The current default value is inlined.
- **`make_agent(env_name=…)` is the only factory**, with no options beyond the env and the model client. Practice,
  eval and explore call it the same way.
- **No task is not a switch.** It is a run input (`agent.run(task=None, …)`), so it doesn't add a flag either.

**CLI after the port.** Only run plumbing stays; nothing that changes agent behaviour.

| keep | remove |
|---|---|
| `--model_name`, `--model_backend`, `--vllm_base_url` (model is still required, never defaulted) | `--features`, everything from `feature_options` except `--seed` |
| `--run_name`, `--tasks`, `--workers`, `--overwrite` | `--supervisor`, `--supervisor_model/backend/vllm_base_url`, `--max_leg_steps`, `--info_docs` |
| `--seed` (repeat label only) | GameBoy `--executor`, `--controller_variant`, `--temperature`, `--max_steps` |
| GameBoy `--game`, `--n_tasks` (which benchmark rows, not agent behaviour) | `--ignore_config_violation` (nothing left to violate; see below) |
| Android `--shard`, `--console_port`, `--grpc_port`, `--n_task_combinations`, `--task_random_seed`, `--rerun_failed` | |
| Web `--judge_model_*`, `--skip_judge`, `--task_file` | |

- **Run records:** `config.json` gets a fixed `agent: "final-v1"` stamp (env, executor and supervisor names)
  instead of the switch values, so old experiment runs and new ones can't be mixed up. The resume check compares
  the stamp; there are no per-switch config violations.
- **Slurm scripts:** `scripts/slurm/eval*.sh` drop their `--features` / `--supervisor` pass-through, and
  `eval_pack.sh` arms become `<seed>@<split>`.

## 2. Per env

### Web (`cusi/agents/executors/webvoyager.py`): already clean, just remove the switches
- Make the three winning switches the only behaviour. Drop the `max_attached_imgs` / `show_budget` /
  `allow_google` args.
  - `MAX_ATTACHED_IMGS = 5`.
  - `budget_note` is always appended.
  - `remove_google_action` always runs in `reset_memory`, and a `Google` reply always becomes
    `invalid="pruned action"`.
- Keep the judge fix: `cusi/eval/web.py: write_artefacts` writes the user's task into the saved chat.

### Android (`cusi/agents/executors/m3a.py` + `cusi/envs/android_world.py`): the real porting work
The bundle was confirmed with the prototypes (`tmp_exec_features.py`), so the port must reproduce their prompts
byte for byte. All of it moves into `M3AExecutor` as fixed behaviour:
- **F27 notes:** already clean in m3a.py. The switch is removed and notes are always on.
- **F55 clear_text:**
  - The input_text action description gets the `"clear_text"` field.
  - M3A's delete tip is replaced with the "set clear_text: true" sentence.
  - A prompt string edit only; it uses AndroidWorld's native `clear_text`.
- **F56 read-back:** after a valid `input_text`, a deterministic `[Field check] …` note is appended to that step's
  summary. It parses the editable fields in `obs_after`.
- **F57 no Enter:**
  - The input_text description says it does NOT press Enter.
  - The env types without Enter, through `type_without_enter`.
- **F58 rules:** the `MINITAP_RULES` block is inserted.
- **F59 subgoals:**
  - One planner call on the first decide (`PLANNER_PROMPT`, 2–6 subgoals, JSON).
  - Each action prompt shows the `[x]/[ ]` checklist and asks for a `Completed subgoals:` line; ticks are parsed
    after each reply.
- **F60 resource ids:** element lists carry `resource_id`.
- **F61 device context:** each action prompt gets `Device context: foreground app …; device date and time …`, from
  one adb `date` call.
- **Prompt assembly:** one explicit function in m3a.py instead of the MRO chain. Blocks are inserted before M3A's
  final `Your Answer:` in the confirmed order: notes → Additional rules → Subgoal plan → Device context. The
  F55/F57 description edits are applied to the base prompt.
- **Env:** `element_resource_ids` and `input_presses_enter` stop being hooks.
  - `AndroidPlayEnv` always adds resource ids and never presses Enter on `input_text`.
  - `last_ui_elements` (only used by the dropped F64) is removed.
  - Note: this also changes what search/explore see on Android, which is intended: one action semantics
    everywhere. See open question Q2.

### GameBoy (`cusi/agents/supervisors/monitor.py`, new; executor untouched)
`MonitorSupervisor(Supervisor)`, standalone (no `RevisingSupervisor` inheritance), ~90 lines, reproducing the
confirmed prototype (`tmp_sup_features.ProtoMonitor`):
- **Legs:** legs of `min(5, remaining budget)` steps run the task with the current hint. There is no
  self-termination (`allow_done_check=False`), and the executor's history is carried across legs.
- **Ending:** the run stops on an env ending (terminated / truncated / model_error) or when the budget is spent.
- **Stuck rule:** a leg is stuck if no step changed the screen, or if at least half of its frame hashes have been
  seen 3 or more times in the episode.
- **When stuck:**
  - segment summaries (`checker.summarise_trajectory_segments`, `JUDGE_SLICE_PROMPT`, 8 frames per slice);
  - then one `RESUME_HINT_PROMPT` call with the judgement "the agent looks stuck: <why>", `[INSIGHTS]` = "(nothing
    recorded)", the previous hint block and the action trace.
  - The hint is used until a leg is not stuck, then cleared.
- **Constants:** executor `max_new_tokens` = 8000 (as confirmed; it never binds, max seen 765). Supervisor
  `max_new_tokens` = 5000.
- **Prompts:** only the GameBoy versions of `JUDGE_SLICE_PROMPT` and `RESUME_HINT_PROMPT` move into
  `supervisors/prompts.py`. Everything else in that 523-line file goes.
- **Unused executor code removed:** the GameBoy executor's `done_check` (never used by the monitor) and the
  `arm` / `history_k` variants other than the default.

## 3. Running without a task

There is one constant, `EXPLORE_GOAL`, moved from `cusi/explore/prompts.py` into `cusi/agents/agent.py`.
`agent.run(task=None, …)` means exploration. The rules are the same in every env:
- The prompt's task slot gets `EXPLORE_GOAL` ("Explore: take actions that lead to screens and content you have not
  seen before in this session.").
- **Finishing is disabled:**
  - Web: the `ANSWER` action is removed from the system prompt and rejected if used.
  - Android: `status` is removed from the action list and rejected if used.
  - GameBoy has no finish.
- The run ends only on the step budget or an env ending. The termination reason is `max_steps`; nothing is scored.
- **Task-dependent pieces turn off:** the Android F59 planner and its checklist are skipped. Everything else stays
  on:
  - notes now record what was explored;
  - device context, read-back, budget note, screenshots.
- **Monitor with no task:** same loop and the same stuck rule; the hint prompt's `[TASK]` gets `EXPLORE_GOAL`. The
  stuck rule (revisiting screens) is already an exploration signal, so the monitor needs no other change.
- **Callers that move over:**
  - `cusi/explore/prompts.py PolicyPrompter` drops its own copies of the prompt builders and uses the agents'
    prompt functions with `task=None`.
  - `cusi/search/expanders.py VLMExplorer` and the RL expanders are not executors and are left alone (Q3).

## 4. What gets removed (after the port is verified)
- **Root scratch:** `tmp_exec_features.py`, `tmp_sup_features.py`, `tmp_minitap.py`, `tmp_check_*.py`,
  `tmp_test_features.py`, `tmp_design_*.py`, the other `tmp_*.py` / `tmp_*.sh` / `tmp_*.txt` files.
  `tmp_design_tables.py` is moved to `scripts/` first if we still want the tables.
- **`cusi/eval/features.py`:** removed entirely (`--features`, `--seed` stays as a plain repeat label).
- **`cusi/eval/supervision.py`:** `--supervisor*`, `--max_leg_steps`, `--info_docs`, `with_supervisor_features`
  and `run_episode`'s two build paths are removed. It keeps `run_pool` and the record helpers.
- **Supervisors removed:** `revising.py`, `subgoal.py`, `info_subgoal.py`, `dummy.py`, the `SUPERVISORS`
  registry, and the unused prompts.
- **`tests/executor_design_test.py`:** replaced (§5).
- I can't run `git rm`, so the deletions are listed for you to run (or approve) as one command.

## 5. Verification (real envs, real model, small first)
1. **Prompt identity, no model needed:**
   - On recorded observations from the confirmed held-out runs, build each env's first and a mid-episode action
     prompt with the new code and with the prototypes.
   - Android: bundle prototypes vs clean M3A. Web: clean switches vs the fixed executor. GameBoy: monitor hint
     prompt vs `ProtoMonitor`.
   - They must be byte-identical, except for the device date/time.
   - This is done before the prototypes are deleted.
2. **Live smoke, 4 smoke tasks per env, with a task:** records read by hand.
   - Android: planner call, checklist ticks, Field check notes, resource ids, no-Enter typing, device context.
   - Web: budget note, no Google, 5 images.
   - GameBoy: monitor legs and stuck hints.
3. **Live smoke without a task:** 30 steps per env (GameBoy 100).
   - Checks: no finish is ever accepted, there is no planner call on Android, `EXPLORE_GOAL` is in the prompts,
     and the monitor fires on GameBoy loops.
4. **Parity run (one repeat each, dev split):**
   - The new code must land within the screening noise of the confirmed numbers: Web dev250, Android 57 dev,
     GameBoy 50 dev.
   - Pass rule: inside each env's 1-repeat noise band (Web 4.8, Android 6.2, GameBoy 10.5), written into
     decisions.md before the run.
   - A miss means a porting bug; it is fixed, not explained away.
   - **Result (decisions E8.8): PASS on all three.** GameBoy 38.0%, Web 68.4%, Android 66.1% (56/58 scored; the
     2 unscored tasks failed inside AndroidWorld's own code).
5. **Practice:** `run_practice.py` runs one small stage per env through `make_agent`.

## 6. Order of work
1. decisions.md entry: the frozen setups, the no-task rules and the parity pass rule (before any code).
2. Web (smallest), then GameBoy monitor, then Android (largest), each with the prompt-identity check (§5.1) as soon
   as it builds.
3. `agent.py` / `specs.py` / the eval runners on the single path; practice and explore callers moved over.
4. Smokes (§5.2–5.3), then the parity run (§5.4).
5. Deletions (§4), with your OK; tests rewritten around the final setups and the no-task mode.

## Decisions (user, 2026-10-18)
- **Q1 → delete.** Plain `rm` (not `git rm`) of the scratch and experiment code; the unused prompts are cut.
- **Q2 → global.** `AndroidPlayEnv` always shows resource ids and never presses Enter on `input_text`, for the
  agent, search and explore alike.
  - The old exploration trees and curiosity data are deleted, together with everything derived from them, so
    nothing replays actions recorded under the old typing:
    - curiosity tasks inferred from them;
    - practice proposals built from those;
    - curiosity policy and embedder checkpoints trained on them.
  - I list the exact paths before deleting.
- **Q3 → follow-up after the port:** an `AgentExpander` adapter (the final executor with `task=None`, search's tree
  context as the hint). It replaces `VLMExplorer` only if a small per-env search comparison (new screens per
  expansion, cost) shows it is no worse. The monitor stays out of search.
- **No-task defaults accepted:** finishing is disabled, the GameBoy executor cap stays 8000, the Android planner is
  skipped.
- **Executor split (part of the port):** `Executor.begin(task, hint, obs, info)` / `act(obs, info, error) ->
  Decision(actions=[...])` / `record(decision, action, before, after) -> error`.
  - `run()` is rebuilt from these and alone holds the stopping policy.
  - Why: search must own `env.step`, so it needs "next action" and "update" separately. It also removes
    `GameBoyExecutor`'s duplicated `run()`: its multi-button replies are just a multi-action `Decision`.
  - It is a refactor with no behaviour change; the prompt-identity checks and the parity run cover it.
