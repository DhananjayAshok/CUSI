# Evaluation plan: native harnesses (option 1) vs our envs in test mode (option 2)

Written 2026-10-05. Question: should test-set evaluation go through each benchmark's own
harness, or through `cusi_envs` in `mode="test"`? What is easier, and what does option 2 lose?
Sources: a read of the three native harnesses and of our envs and executors (file:line
references in the notes this plan was written from; the key ones are repeated below).

**Short answer.** Option 1 is less work for correct numbers *today*. Option 2 is more work
(roughly a dozen concrete fixes, all small), but the hard part already exists: our envs
reuse each benchmark's own scene setup, action execution and success check, and
`cusi_practice/executors/` already rebuild the native prompts line for line. The scores can
be made equivalent for the main configurations: GameBoy `baseline`, Android M3A on the
`android_world` suite, Web with WebVoyager's multi-turn agent. What option 2 loses is
mostly side configurations (GameBoy supervisor arms, Android T3A/MiniWoB, Web text-only).
Equivalence is not automatic, so option 2 needs a parity check against the native harness
on small subsets, which means running option 1 once anyway.
**Recommendation: option 2, with a one-off parity check against native.**

---

## 1. What exists today

| | GameBoy | Android | Web |
|---|---|---|---|
| Native entry point | `GameBoyRL/run_benchmark.py` (click) | `android_world/run.py` (absl) | `WebVoyager/run.py` (argparse) + `evaluation/auto_eval.py` |
| Test set | `benchmark/tests/*.csv` of the game, prefix `--n_tasks` | `android_world` family (Android + information-retrieval tasks), seed 30, `n_task_combinations` instances each, sorted by name | `WebVoyager_data.jsonl` (643 tasks, 15 sites); `run.sh` points at a 1-task sample; the planned site-exclusion file is not built |
| Agent | executor + supervisor arm; default `baseline` arm = one `single_none` executor leg | M3A (or T3A): action call + summary call per step | multi-turn chat, last 3 screenshots kept, older ones replaced by stubs |
| Budget | `--max_steps` (175 in the shell wrapper, 200 in `run_benchmark.py`), counted twice: executor decisions and emulator steps | `int(10 * complexity)` agent steps | `--max_iter` (15 in `run.sh`, 5 in `run.py`) |
| Success | test tracker `terminated` | `task.is_successful` if the agent said `status`, else 0 | LLM judge on the final answer + last 15 screenshots |
| Model in | `--executor_vlm_model/--executor_vlm_kind`; vLLM URL only from GameBoyRL's config | `--model_name --model_backend --vllm_base_url` | same as Android |
| LoRA policy | only via vLLM `--lora-modules` (nothing sets this up) | same | same |

Our side: `GameBoyPlayEnv` / `AndroidPlayEnv` / `WebVoyagerPlayEnv` have a `test` mode
built from the same benchmark code (`get_test_environment`; AW's `_instantiate_task` +
`initialize_task` with the same instance seeds; WebVoyager's own `run.py` functions).
`cusi_practice/executors/{gameboy,m3a,webvoyager}.py` reproduce GameBoyRL's
`single_actions`, M3A and WebVoyager prompting. **Nothing runs an eval through them yet.**

---

## 2. Option 1: one wrapper over the native harnesses

`run_eval.py --env X --model_name ... [--tasks/--n_tasks]` + a Slurm pair that serves the
model with vLLM, then calls the native script with matching flags (Web: then `auto_eval.py`),
and collects the three result formats into one table.

Work:
- Flags per harness: GameBoy `--executor_vlm_kind vllm`; its vLLM URL is only in
  `GameBoyRL/configs/project_vars.yaml`, so the server must be on that port or the config
  must change. Web `--max_iter 15 --max_attached_imgs 3 --temperature 1 --fix_box_color
  --headless` and a real task file. Android output/checkpoint dirs under storage.
- A trained LoRA policy: start vLLM with `--lora-modules` (pass-through in `serve_vllm.sh`;
  untested) and select it by name. GameBoyRL names CSVs after the model's basename, so
  adapter names must be unique.
- Read three output formats (GameBoy CSV, AW checkpoint pickles + `process_episodes`,
  Web `scores.json`).

Gains: numbers identical to the published harnesses by construction; every native
configuration stays available. Costs: three code paths that share nothing with training
or practice (a policy trained through our envs is evaluated through different code, so a
train/eval mismatch can't be ruled out); three result formats; models only through a server.

---

## 3. Option 2: one eval loop over our envs in test mode

`run_eval.py` builds `cusi_envs` in `mode="test"`, one per test task, runs an **agent**
through it, and records the env's verdict. An agent is a prompt format + a model:
- **native replicas** (score-comparable with published numbers): the existing executors
  (GameBoyRL `single_*`, M3A, WebVoyager), driven by a served or local model;
- **our own policies** (e.g. the PPO policy with its prompter, loaded locally with its LoRA,
  no server needed).

Most of the machinery exists. What has to change for score equivalence:

### 3.1 Shared (executor base, `cusi_practice/executors/base.py`)
1. `MAX_CONSECUTIVE_INVALID = 4` must be a per-env setting: native GameBoy has it, native
   M3A and WebVoyager do not (they keep going until the budget runs out).
2. A finishing action (`status` / `ANSWER`) must reach `env.step`, so the env computes the
   reward. Today the executor stops before it, and a judge decides.
3. Per-env generation settings: GameBoy `max_new_tokens 2000`, temperature unset; Android
   1000 and temperature 0.0; Web 1000 and temperature 1.0. Practice's current defaults
   (1000, server default) match none of them exactly.
4. A model error ends the task with whatever it has (as native), not an exception.

### 3.2 GameBoy
5. `GameBoyPlayEnv`: pass `wait_ticks=20` (native forces it; ours falls back to 8, which
   changes emulator dynamics and frames: **the biggest silent difference**); require an
   explicit `max_steps` (175) and give the same number to the executor; forward
   `save_video` / `session_name`.
6. Executor: add the native default history mode `single_none` (and fix `history_k=0`,
   which today means *all* history because `steps[-0:]` is the whole list).
7. Unavailable actions (only with non-`low_level` controllers): native counts them as
   normal steps with no error message. Either match that or evaluate `low_level` only.
8. Record native fields: success = terminated, `n_steps` in emulator steps
   (`info["core"]["steps"]`), `n_invalid`, tokens, subgoals reached.

### 3.3 Android
9. Driver: tasks sorted by name × instances, `suite_seed` = 30, `reset_mode="reinit"`,
   one `reset()` per episode; budget `int(10 * complexity)` with invalid-format steps
   counted.
10. Call `task.tear_down` after every task (native does; our env never does), so each task
    starts from the same device state as in the native sequential run.
11. A stall/recovery or exception is recorded as **failed-to-run (NaN, excluded)**, as
    native does, not as reward 0.
12. M3A details: an execution failure is not added to history natively (our executor adds
    it); native rejects only the strict `Reason:/Action:` format (our env also accepts bare
    JSON). Small, but listed so parity runs can explain any gap.
13. Metrics: build episode dicts with AW's keys and call `suite_utils.process_episodes`
    directly (per-task mean, average over tasks, tag × difficulty table).

### 3.4 Web
14. Drive the env with `WebVoyagerExecutor` (multi-turn, clipped to 3 screenshots), never
    single-turn env steps. Env with `fast_waits=False`, `fix_box_color`, headless, 1024×768,
    `max_steps=15`. `fast_waits` is a process-global, so never mix eval and practice envs in
    one process.
15. Browser death: rebuild the env and retry the task, up to 3 times, then write
    `task_error.json`. A non-fatal WebDriver error ends the task (native) instead of raising.
16. Write native artefacts: `task{id}/interact_messages.json` (the executor's clipped chat,
    images as placeholders, ending with the raw final reply) on *every* ending, and the
    set-of-mark `screenshot{N}.png` files. Then run `auto_eval.py` unchanged with the same
    judge model. (Our executor's reports keep unlabelled frames; auto_eval needs the labelled
    ones.)
17. Build the filtered task file from `web_voyager_plan.md` (sites we exclude), used by both
    options.

---

## 4. What option 2 loses

| Lost | Matters? | Way back |
|---|---|---|
| GameBoy supervisor arms (`revision`, `subgoal`, `info_subgoal_*`) | Only if we compare against them | Hybrid: build the env with `GameBoyPlayEnv` and hand `env._env` to GameBoyRL's own supervisor/`run_episode` (untried; GameBoyRL's and WebVoyager's `utils` can't share a process, so GameBoy evals run in their own process) |
| GameBoy `scored` / `sequence` / `visual` executors | Same | Port them (`sequence` needs a per-action terminated check and the abort rule) |
| Android T3A, MiniWoB families, `fixed_task_seed` | Low (M3A on `android_world` is the target) | Add later if needed |
| Web text-only (accessibility-tree) mode | Low | The env has no accessibility tree; would need adding |
| Native artefacts and tools (GameBoyRL `SupervisorReport` pickles / `show_trajectories`, AW per-step `episode_data`) | Inspection only, not scores | Write our own trajectories (we already have replay / LegReport formats) |
| Exact timing (Android: when the next observation is taken; Web: labels removed earlier, one extra screenshot) | Small noise, not bias, on live/real UIs | Parity runs measure it |
| Guaranteed equivalence | Yes: numbers must be comparable to published ones | Parity check (section 6) |

Nothing is lost on how the model is passed: both options go through the same
`cusi_utils.model_factory.build_model`. Option 2 *adds* direct local evaluation of a LoRA
policy without a server.

---

## 5. Which is easier

- **Option 1:** about a day. Three flag sets, one Slurm pair, a results collector. Risk is
  low, except serving LoRA adapters (untested).
- **Option 2:** the 17 items above. Each is small (a flag, a call, a file writer); 2, 10,
  15 and 16 are the largest. Most risk sits in Web (artefacts, retries, error paths) and
  Android (inter-task state). Plus the parity runs.

Option 2 is more work, but it pays back: one loop, one result format, the same env code
as training and practice, any agent (native replica or our policy), and local LoRA
evaluation. It is also where saved states and search nodes could later serve as eval start
states.

---

## 6. Parity check (required before trusting option 2)

Run both paths on the same small subset with the same served model
(`google/gemma-4-26b-a4b-it` or `Qwen/Qwen2.5-VL-7B-Instruct`):
- **GameBoy:** first 20 tasks, `baseline` / `single_none`, `low_level`. Deterministic
  emulator: per-task success should match exactly at temperature 0. Native temperature is
  unset, so compare the success rate, plus per-task agreement at temperature 0.
- **Android:** about 20 tasks × 1 instance. Expect the same success rate within noise;
  compare per-task agreement and explain disagreements with the item 12 list.
- **Web:** about 30 tasks on 2–3 stable sites, same judge. Live sites, so compare rates
  within noise (and repeat native once to see its own run-to-run spread).

Exact per-task match is only possible on GameBoy; for Android and Web the bar is "within
the native harness's own run-to-run noise".

---

## 7. Design (option 2)

- `cusi_eval/`: `agents.py` (agent = executor class + generation settings, or a local
  policy), `runners/{gameboy,android,web}.py` (task lists, env construction, per-env
  rules from section 3, retries, NaN handling), `records.py` (one per-task JSONL row:
  env, task id, success / NaN, steps, invalid steps, tokens, wall time, trajectory dir),
  `metrics.py` (overall + per-site / per-task-template; Android through
  `process_episodes`; Web through `auto_eval.py`).
- `run_eval.py --env {gameboy,android,web} --agent {native,policy} --model_name ...
  [--model_backend vllm] [--policy_lora_path ...] [--tasks ... | --n_tasks N] [--run_name]`.
  Models are always required flags. Output: `storage_dir/eval/<env>/<run_name>/`, with
  resume (skip finished tasks, rerun failed ones).
- `slurm/eval.sh` + `scripts/slurm/eval.sh`: serves `--model_name` with vLLM when
  `--agent native`; Android/Web inside the container; Android starts the emulator.
- `tests/eval_parity_*.py` + a Slurm pair for section 6.

## 8. Order of work

1. Shared executor fixes (3.1).
2. GameBoy runner + fixes (3.2); GameBoy parity check (cheap, deterministic).
3. Android runner + fixes (3.3); parity on about 20 tasks.
4. Web task file (17), runner + artefacts (3.4); parity on about 30 tasks.
5. `--agent policy` (the PPO policy with its LoRA, locally).
6. Optional: hybrid GameBoy supervisor arms.

## 9. Open questions

1. **Which configurations must match published numbers?** GameBoy `baseline` only, or the
   supervisor arms too? Android M3A only? If only the main ones, option 2 loses nothing
   that matters.
2. **Prompt format of our trained policies at eval.** The PPO policy is trained with
   `--reply_format action_only`, while native agents reason first. Do we evaluate our
   policies in their training format (our prompter, not comparable with published agents)
   or require training in `native` format so the native replicas can run them?
3. **Web test set:** which sites to exclude (the `web_voyager_plan.md` list), and which
   judge model.
4. **Budgets:** GameBoy 175 (shell wrapper) or 200 (`run_benchmark.py` default)?
