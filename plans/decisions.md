# Decisions taken while implementing plans/archive/plan.md (autonomous run, started 2026-10-04)

Each entry: what was decided, and why. Numbers continue the plan's decision table where useful.

## Part 0: shared vLLM

- **D0.1 vLLM version.** `uv pip install vllm transformers` with no pins: vLLM 0.30.0 (newest on PyPI on
  2026-10-04), which pins torch 2.13.0; transformers is left unpinned so uv picks the newest it can resolve.
  The exact versions are written to `/nas/eclairnas01/users/ashokd/vllm/state/versions.txt` by `setup.sh`.
- **D0.2 Bind address.** `serve_vllm.sh` binds to `127.0.0.1` unless the caller passes `--host`.
  The server never needs to be reachable from other nodes (vLLM, envs and pipeline share one Slurm job),
  and the tool's safety check refused a script that listens on all interfaces by default.
- **D0.3 HF cache (revised after review).** All weights (Gemma 4, SigLIP 2, Qwen3-VL-2B) live in the
  user's own `$HF_HOME`, which must already be set: `scripts/utils.sh`, `scripts/container.sh` and the
  shared `env.sh` exit if it is not. No CUSI config key, no `cache_dir=` in code; the container binds
  `$HF_HOME`. (First run used a separate `vllm/hf_cache`; its weights were moved into `$HF_HOME/hub`.)
- **D0.4 Online (revised after review).** Every node has internet: no offline switches
  (`HF_HUB_OFFLINE`) and no download script; vLLM and transformers fetch into `$HF_HOME` as needed.
- **D0.5 FlashInfer sampler off.** vLLM 0.30's default FlashInfer top-k/top-p sampler JIT-compiles
  at startup and needs a CUDA toolkit matching its bundled headers. The nodes have none, and the
  pip `nvcc` in the venv fails with "CUDA compiler and CUDA toolkit headers are incompatible".
  `env.sh` sets `VLLM_USE_FLASHINFER_SAMPLER=0` (vLLM's PyTorch sampler). Compile/JIT caches go to
  `vllm/cache/`. Verified: Slurm job 282699 on eclairlg03 (2x A6000, driver 580, CUDA 13): healthy
  in 180 s; text and image requests correct; ~725 output tok/s at batch 32.
- **D0.6 Server settings.** `--max-model-len 32768`, `--limit-mm-per-prompt {"image": 16}` (the
  judge's describe slices send 10 frames), `--gpu-memory-utilization 0.90`, `-tp` = visible GPUs.
  Thinking is left at the chat template's default (off).
- **D0.7 Port, served name, health (revised after review).** `vllm_port` (project_vars.yaml) is the
  only place the port is set; `vLLM_base_url` is derived from it in `compute_secondary_parameters`
  and overridden per run with `--vllm_port` (run_practice.py / run_explore.py). The server always
  gets `--served-model-name` (`--served_model_name`, default `--model`; clients pass it as
  `--model_name`); healthy = `/health` on the port and `/v1/models` lists that name. A port already
  serving another name is an error.
- **D0.9 vLLM scripts in the repo (revised after review).** `setup/vllm_scripts/` (serve, stop,
  env) is the source of truth; `setup/move_vllm_scripts.sh` copies it into `shared_vllm_dir`
  (private_vars.yaml) and warns if that directory has no `.venv` with vLLM. The venv is assumed,
  not created: `env.sh` refuses to serve without it. The old shared `setup.sh` is no longer part of setup.
- **D0.8 No default models (revised after review).** No model is set in config or code: the served
  model (`serve_vllm.sh --model`, clients `--model_name`), the exploration policy and the encoder
  (`run_explore.py ppo --policy_model / --encoder_model`) are required arguments, and each
  `scripts/slurm/*.sh` takes them and passes the same value to server and client.
  The shared scripts locate their install from their own path (no hardcoded root).

## Part 1: practice pipeline

- **D1.1 Layout.** Package `cusi/practice/` (records, vlm, prompts, parsing, judging, envs,
  executors/, stages/) and entry point `run_practice.py` (click group: `--env X <stage>|all`).
  Outputs: `storage_dir/practice/<env>/<model>/` (layout in `cusi/practice/stages/common.py`).
- **D1.2 One process per environment.** GameBoyRL's top-level `utils` package and WebVoyager's
  `utils` module clash in one interpreter, so benchmark modules are imported lazily and each run
  handles one environment.
- **D1.3 GameBoy executor reuses GameBoyRL's own prompt constants** (`STEP_PROMPT`,
  `DONE_CHECK_PROMPT`, `SingleActionPolicy`, `STUCK_HINT`) by importing GameBoyRL; the assembly
  is ported. `tests/practice_gameboy_prompt_test.py` builds GameBoyRL's real `PolicyExecutor`
  and checks byte-identical prompts over 12 combinations (hint / error / history with a
  `[no change]` step): all identical. GameBoyRL's parsers (`parse_list`, `parse_int`,
  `parse_yes_no`) are copied (D1.2 prevents importing them in Web runs).
- **D1.4 Hint/guidance stripping.** GameBoy keeps GameBoyRL's `[HINT_START]`/`[STEP_INFO]` markers.
  M3A (`additional_guidelines`) and WebVoyager (after the task sentence in the first user message)
  use `[GUIDANCE_START]...[GUIDANCE_END]` around the whole block, removed with no whitespace
  eaten, so the stripped prompt is exactly what the native agent builds without guidance. The
  M3A/Web guidance carries a "this is secret, don't mention it" note, as GameBoyRL's hint block does.
- **D1.5 Agent-finishing actions are intercepted by the executor**, not sent to the env (which
  rejects them in free_play): M3A `status`, WebVoyager `ANSWER`. The answer text (WebVoyager
  ANSWER, M3A `answer`) is kept and shown to the judge (`JUDGE_ANSWER_NOTE`).
- **D1.6 Faithfulness details kept from the native agents:** M3A's summary call sees an
  *unlabelled* "before" screenshot and a labelled "after" one (that is what M3A.step actually
  sends: it labels a copy), and its malformed-output/parse-error messages go into its history
  with no env step. WebVoyager: run.py's Thought/Action format check, its exact failure
  message for unknown actions, the failure message as a plain user turn with no screenshot,
  `clip_message_and_obs` with 3 images, and PDFs answered by the model as run.py does.
- **D1.7 Invalid actions count against the budget** for every agent (as GameBoyRL and run.py
  do), and 4 consecutive invalid decisions end the leg (GameBoyRL's limit, now shared).
- **D1.8 Judge frames.** The judge, critique and guidance see the raw screenshot where the env
  has one (Android `info["raw_frame"]`), else the observation frame (WebVoyager's is the
  set-of-mark screenshot: there is no unlabelled one).
- **D1.9 Shared prompts** (`cusi/practice/prompts.py`) are GameBoyRL's with "a game of [GAME]" ->
  `[DOMAIN]` and "player" -> `[ACTOR]`; GameBoy's Domain reproduces the original wording.
  The propose prompt also lists the screen's text elements for Android/Web (none for GameBoy).
- **D1.10 Task selection.** Proposals keep the first 3 parsed tasks per scene (the list is ordered
  by the model; first-k is deterministic). Task strings are not lowercased (GameBoyRL lowercased
  them only because they named directories, which ours do not).
- **D1.11 Step budgets.** GameBoy 50 (GameBoyRL attempts/practice), Android 20 (AndroidWorld's
  10 x complexity for complexity <= 2), Web 15 (run.sh `--max_iter`).
- **D1.12 Scenes.** GameBoy: `location_viridian_city_starting_charmander`,
  `location_pewter_city_starting_charmander` (Pokémon Red training states). Android:
  `ContactsAddContact`, `SystemWifiTurnOff` (free_play). Web: `https://arxiv.org/`,
  `https://github.com/`.
- **D1.13 Records store images PNG-encoded** (`EncodedImage`): Android screenshots are 7.7 MB raw.
  Dataset images are written as PNG (lossless).
- **D1.14 Dataset rows** are JSONL (nested chats don't fit CSV): `messages` (role/content, images as
  paths; single-turn rows put the text before the images, as cusi.utils sends them), `response`,
  `call_type`, `task_string`, `episode_id`, `env`, `scene`. M3A `summary` rows are kept when the
  action call they follow is kept. Finishing calls (M3A `status`, WebVoyager `ANSWER`) count as
  env-accepted, so the agent learns to stop.
- **D1.15 Chrome startup is serialized** in `WebVoyagerPlayEnv` (process-wide lock, 3 retries), and
  the `run.py` loader is locked: concurrent starts failed with "DevToolsActivePort file doesn't
  exist", and a second thread could see a half-imported module.
- **D1.16 Android scenes open their app.** Both AndroidWorld tasks start on the home screen, so
  the two scenes looked identical and drew identical generic proposals ("Open the Gmail app").
  `AndroidPlayEnv(start_app=...)` opens an app with M3A's own `open_app` action as the last step
  of building the scene (before the snapshot), so reset() returns to it. Scenes: Contacts
  (`ContactsAddContact` + Contacts) and Settings (`SystemWifiTurnOff` + Settings).
- **D1.17 Propose-prompt rules for Android/Web.** The first run proposed tasks citing element
  label numbers ("click element [13]"), which change from screen to screen, and GitHub
  sign-up/login tasks. Added: describe targets by what is visible, never by label numbers
  (Android, Web); no logging in, signing up, buying or personal data (Web; WebVoyager's own
  guidelines say not to touch login/sign-in elements). The actor is "user" for Android/Web.
  The superseded first-run outputs were moved, not deleted, to
  `storage_dir/practice/{web,android}/gemma-4-26b-a4b-it_superseded_run282701/`.
- **D1.18 Robustness.** `run_jobs` retries a job that raises once before skipping it (the Android
  mock run lost a practice episode to "Device did not come back after the snapshot load"), and
  `AndroidPlayEnv.reset()` now treats a failed snapshot restore like a stall: restart the
  emulator and rebuild the scene (when it was given an `AndroidEmulator`).

## Part 2: curiosity exploration + world model

- **D2.1 Layout.** Package `cusi/explore/` (encoder, curiosity, action_vocab, replay, policy,
  prompts, ppo, trajectories, world_model, decoder) and entry point `run_explore.py`
  (`ppo`, `tasks`, `world_model`, `decoder`, `wm_eval`, `elements`). Outputs:
  `storage_dir/explore/<env>/<run>/` (replay/, policy/, buffer/, metrics.jsonl, action_space.json).
- **D2.2 PPO runs in its own job on 1 A6000** (plan's open question: share the vLLM node or not).
  The policy (Qwen3-VL-2B bf16 + LoRA, gradient checkpointing) and SigLIP 2 fit on one GPU, and
  PPO never calls vLLM; only `tasks` (curiosity task inference) needs the vLLM server.
- **D2.3 Exploration goal.** Free play has no goal, but the native prompts have a goal slot; it
  holds "Explore: take actions that lead to screens and content you have not seen before in this
  session."
- **D2.4 Policy prompts per environment.** GameBoy: GameBoyRL's STEP_PROMPT with the actions
  history (exactly the practice executor's). Android: M3A's action-selection prompt, but its
  history lines are "Action selected: <action>" without M3A's per-step summary call (one model
  call per step). Web: WebVoyager's system prompt + its first-turn observation message each
  step (no chat history), so each training sample has one screenshot instead of three.
- **D2.5 Policy image size.** The policy's processor caps images at `640*28*28` pixels (about 640
  visual tokens) so PPO minibatches on Android/Web screenshots fit in memory. Decision 8 (no
  downscaling) was about the served model in Part 1; the served model still gets full images.
- **D2.6 Log-probabilities and KL.** The sampled distribution is softmax(logits / T), T = 0.7, and
  the stored and re-computed log-probs use the same T (checked: act() and evaluate() agree
  exactly before an update). Value = linear head on the final hidden state at the last prompt
  token. KL penalty to the reference (adapter disabled) = (logp - logp_ref) / n_action_tokens
  per sample, coefficient 0.05, added to the loss. Entropy bonus 0 by default (a token-entropy
  bonus tends to destabilise text policies). LoRA dropout 0 so the first PPO ratio is exactly 1.
- **D2.7 PPO settings.** gamma 0.99, lambda 0.95, clip 0.1, value clipping, 2 update epochs (the
  original's 4 doubles an already long update), 8 minibatches, LoRA LR 1e-5 (2e-5 already
  clipped half the ratios in a CPU smoke test), value-head LR 1e-4, linear annealing, max grad
  norm 0.5 (LR superseded below). Rollouts: 128 steps (GameBoy), 64 (Android/Web). Episodes: 30 / 20 / 15 steps.
  Revised after the first GPU update (GameBoy, LR 1e-5): 98% valid actions, but clipfrac 0.84 and
  approx KL 0.24 (a ratio over a ~150-token action amplifies small per-token changes). That run
  was stopped after one update (kept as `explore/gameboy/dev_viridian_lr1e-5_aborted/`); the
  defaults are now LR 2e-6 and a target-KL early stop of 0.1, checked after every minibatch.
  The short Android/Web dev runs (jobs 282708/282709) had already started with LR 1e-5, no KL stop.
- **D2.8 Stale-reward fix.** The final transition's intrinsic reward is computed before the
  curiosity buffer resets (the original left `rewards[step]` stale at episode ends).
- **D2.9 Curiosity details kept from GameBoyRL:** the novelty buffer resets to its prior every
  episode (`reset_curiosity_module=True`); the reset frame seeds the buffer and, through the
  original's `first_add` behaviour, the first transition of each episode also scores 0; KMeans compaction at 10,000 entries. Embeddings use
  the raw screenshot on Android (set-of-mark labels are not content). Text novelty = fraction
  of the screen's element lines (index numbers stripped) never seen before; `text_alpha` 0.5,
  forced to 0 for GameBoy [GB-OCR].
- **D2.10 Action vocabulary.** Android: click/long_press/input_text x (64 + overflow), scroll x 4
  directions x (64 + overflow), 4 whole-screen scrolls, back, home, enter, wait, open_app,
  answer, status. Web: click/type x (128 + overflow), scroll up/down x (128 + overflow), window
  scroll up/down, wait, goback, google, answer. Every vocabulary ends with "invalid" (a no-op
  step). `run_explore.py elements` reports element counts per screen and overflow use (K check).
- **D2.11 Curiosity task source.** Outliers: global z-scores only (GameBoyRL has `USING_LOCAL =
  False`), threshold 2.5, backtrace 30 steps; grouping at z >= 3 with cosine >= 0.95 in SigLIP
  space on the stored embeddings; cycle snipping keeps the first visit (with the action that
  reached it) and drops the loop. Tasks: INFER on the last 8 frames, REFINE, DISTILL (no
  pairwise DESCRIBE stage: off by default in GameBoyRL). Written as an attempts dir under
  `practice/<env>/<model>/curiosity/`, and `run_practice.py --source curiosity` runs guidance,
  practice, clean and dataset on it unchanged.
- **D2.12 World model** trained on replay with FRAME_STACK = 2: inputs are the embeddings at steps
  t-2, t-1 (the first transition repeats its frame) and the action of transition t; split by
  episode; eval reports cosine/MSE against the "nothing changes" baseline, also on transitions
  whose frame changed.
- **D2.13 Decoder** (per environment): Linear -> 4 x (upsample, conv, GroupNorm, SiLU) -> conv ->
  sigmoid, L1 + MSE, output 144x160 (GameBoy, native), 256x112 (Android), 96x128 (Web).
  `wm_eval` saves a grid: current | real next | decoded real next | decoded prediction for the
  action taken | two other actions.
- **D2.14 No world-model executor.** The plan's "What we build" lists the world model and the
  decoder that gives a world-model executor its frames, but not the executor itself
  (GameBoyRL's `execution/executors/world_model.py`), so it was not ported.
- **D2.15 Opt-in token-mean ratio.** With the plan's sequence-level ratio, even LR 2e-6 trips the
  0.1 KL stop after about one minibatch step on GameBoy, and the LR 1e-5 Android/Web runs clip
  ~80% of ratios. `--ratio_mode token_mean` (mean per-token log ratio) is available as an
  alternative; the default stays the plan's sequence ratio. A comparison run on Pewter uses it
  (`explore/gameboy/dev_pewter_tokenmean`, LR 1e-5).
- **D2.16 Web element splitting bug (fixed).** WebVoyager joins its element list with tabs on one
  line, so text novelty in the first Web run (`explore/web/dev_arxiv`, job 282708) treated the
  whole list as one "line" (novelty 0 or 1 per screen). Fixed in `element_lines`; later runs
  score individual elements. K check on that run (re-computed after the fix): 25 elements per
  screen on average, p95 92, max 92, so K = 128 is never exceeded; 0 overflow actions.
- **D2.17 Short-run outlier thresholds + rescoring.** GameBoyRL's thresholds (save at z > 2.5,
  group at z >= 3) assume long runs; the 320-step Web dev run had no z >= 3 transition even
  after rescoring. `run_explore.py tasks --rescore_text_alpha A` recomputes a replay's intrinsic
  rewards with the current curiosity code (used for the Web run, recorded before D2.16), and the
  Web curiosity-practice job uses `--z_min 2.5`. Defaults are unchanged.

## Executor design plan (plans/archive/design_plan.md), from 2026-10-13

Feature IDs (Fxx) refer to `plans/archive/executor_lit.md`. Rules are written here before the stage they govern runs.

- **E1.1 Dev splits** (chosen once, seeded `random.Random(20261013)`, `tmp_design_split.py`; files in
  `storage/eval/design/dev_<env>.txt`, full sets in `full_<env>.txt`):
  - Web: 15 tasks per site of the CUSI task file (10 sites) = 150 of 429.
  - Android: 58 of 116 AndroidWorld tasks (instance 0, suite seed 30), simple random.
  - GameBoy: 25 Pokémon Red + 25 Pokémon Crystal = 50 of 99 (ids `<game>_<row>`).
  - No prompt examples exist yet. Any example later drawn from a task must come from a non-dev task, and that task
    is then reported as such in the held-out numbers.
  - Sizes were set larger than the minimum so the split is unlikely to need growing; §1's rule still applies.
- **E1.2 Seeds = repeats.** Sampling is not seeded. Web runs at temperature 1.0 (harness default) and GameBoy at the
  server default (Gemma's generation config), so repeats differ by sampling. Android runs at temperature 0, so its
  repeats differ only by env and batching nondeterminism; its noise floor is measured the same way. `--seed` is a
  label recorded in the run config. Run names: `dz_<arm>_r<seed>` (`r`, because Android shards already use `_s<i>`).
- **E1.3 Harness for arms.** `--features F03,F07=3` (`cusi/eval/features.py`) wraps the env's executor class with
  `tmp_exec_features.wrap(...)`; the arm (env + features) is hashed with `hash_meta_dict` and recorded with
  `write_meta` in each run dir. Baseline = `--features none`, i.e. the unchanged executor.
- **E1.4 Noise floor rule (stage 1).** From the 3 baseline repeats, per env: the floor is the larger of (a) the
  largest pairwise success difference between repeats and (b) 1.96 × the SD of the paired per-task difference
  between two repeats ÷ √n. If the floor is above 5 points on an env:
  - Web: grow the dev split (stratified, same seed stream) up to 250 tasks, leaving ≥ 179 held out.
  - Android and GameBoy: growing would leave too little held out (116 and 99 tasks in all), so the split stays and
    screening uses more repeats instead: the number of repeats r for which floor/√r ≤ ~5 points, capped at 3.
  - Otherwise screening uses 1 repeat.
- **E2.1 Screening rules (stage §5.1)**, fixed before any screening run. Per env, each feature alone vs `base` on the
  dev split, same repeats count r (E1.4), compared task by task (paired):
  - **Promote** if diff ≥ floor/√r and calls/task < 2× base; or if diff > −floor/√r (neutral) and it clearly fixes
    its target bucket (≥ half of the dev failures in that bucket that it touches turn into successes, judged from
    the paired wins/losses and the labelled failures).
  - **Drop** if diff ≤ −floor/√r, or calls/task ≥ 2× base with diff < floor/√r.
  - **Retest** (one more repeat) otherwise, i.e. borderline: |diff| < floor/√r and no clear bucket effect, when
    its prior rank is in the top half; else drop.
  - A smoke run (4 tasks, `dzsmoke_*`) precedes each new prototype; its episode records are read to confirm the
    feature does what it claims before any dev run.
- **E2.2 Open questions from the literature review, decided:**
  - **F33 (re-ask after a bad output):** a re-ask counts as a step, as in the harness (parity: step limits are fixed).
  - **F43 on GameBoy (lookahead with emulator save states):** not allowed. A save state is privileged simulator access
    of the same kind as RAM. Web and Android can't reset either, so the rule is the same for all envs.
  - **F53 (cross-task memory):** not used in screening, since it leaks across test tasks. It may be revisited only in
    §7 (exploration knowledge), sourced from exploration runs and reported on held-out tasks.
  - **GameBoyWorlds `agent_state`:** stays unexposed, since it is RAM-derived. F54 infers the mode from screen text.
- **E2.3 Prototype → catalogue IDs:** F04 screen/page text; F08 past screenshots kept (W); F10 step budget shown;
  F11 forced answer at the budget end; F16 multi-action decisions (G: GameBoyRL `sequence`); F25 history form
  (G: `actions` history instead of `visual`, i.e. no frame-diff call); F26 history depth (G `history_k`);
  F27 notes buffer; F31 no-op note (A); F32 loop warning on repeated screens; F37 action vote; F41 completion
  self-check (W before ANSWER, A before `status complete`).

### What didn't work (executor design plan): kept so it isn't retried
- **Offline fake-env / canned-LM tests** of the prototype features (`tmp_test_features.py`). The user ruled them out:
  test only the real envs and served model, at small scale (`dzsmoke_*` runs). The file is unused (no `rm`).
- **Feature parameters read via `self.KEY`.** Once several feature mixins stack, `self.KEY` resolves to the first
  mixin's key in the MRO, so every feature would read the wrong parameter. Fixed before any run used it: each call
  names its key, and one carried-field mixin per env handles both plan and notes.
- **One Web "budget" switch that also forced an answer.** It mixed two catalogue features, so it was split into F10
  (show the budget) and F11 (force ANSWER on the last iteration) before screening.
- **Smoke jobs queued on A6000 nodes behind the 9 baselines.** They were moved to the RTX A5000 node; dev and test arms
  stay on A6000, so GPU type is constant across compared arms.
- **F24 (Gemma thinking mode) not prototyped yet.** It needs `chat_template_kwargs` through the client and a vLLM
  reasoning parser, and leaked thoughts would break the executors' `Thought:`/`Action:` parsers. Deferred, not
  dropped.
- **Serving Gemma 4 31B on the idle Quadro RTX 8000 nodes (Turing, fp16)** failed (job 283992). Gemma 4's full-attention
  heads are 512 wide, so vLLM forces the Triton attention backend, and its kernel needs 96 KB of shared memory against
  Turing's 64 KB. Usable GPUs stay the RTX A6000 nodes (eclairlg02–05 are reserved for another group) plus the RTX
  A5000 node for smoke runs. `eval.sh` keeps the new `--vllm_extra` pass-through.
- **E2.4 Google CAPTCHA trap (Web).** About 29% of Web episodes, in the earlier ablation and in the dev baselines alike,
  hit a Google CAPTCHA after the `Google` action, and `GoBack` doesn't escape it. The env is the harness default and
  stays unchanged. The executor-side fixes are features and go through screening like any other:
  F12 `GoTo [url]` (escape route) and F14=`google` (drop the Google action from the action space). Wait pruning is
  F14 alone, which equals F14=`wait`.
- **F37 vote keyed on the whole action string** (smoke 283996). ANSWER and Type replies carry free text, so samples
  almost never agreed on them. Fixed before screening: answers vote by action type, typing by target element, and
  Android JSON by action type plus index, app, direction and status.
- **Edited `scripts/slurm/eval.sh` while 9 jobs were running it.** Bash reads scripts as it goes, so after the agent
  phase the running copies hit a syntax error: Web lost its judge step and every job lost its `timing.jsonl` row.
  Recovery: judge-only jobs for the Web runs, and GPU-hours for these runs taken from `sacct` elapsed time.
  **Rule from now on:** never edit a script that running jobs execute; copy it, or wait until they finish.
- **E1.5 Noise floor, applied (rule E1.4).**
  - **GameBoy:** repeats 34 / 32 / 34%. The largest pairwise gap is 2 pt; 1.96·SD/√n of the paired per-task
    difference is 7.9–10.5 pt (8–14% of tasks flip between repeats). Floor = 10.5 pt > 5, so screening uses 3
    repeats (floor/√3 ≈ 6 pt is the promote/drop threshold).
  - **Web:** repeats 52.7 / 48.0 / 54.7%. The largest pairwise gap is 6.7 pt; 1.96·SD/√n is 5.5–6.2 pt. Floor =
    6.7 pt > 5, so the dev split grows to 250 tasks (`dev250_web.txt` = dev + `dev2_web.txt`, 10 more per site
    drawn from non-dev tasks with `random.Random(20261014)`, `tmp_design_split2.py`). The extra 100 run as
    `dzdev2_*` runs, merged with `dz_*` in the tables (`--split dev250 --merge dzdev2`). Expected floor at 250
    tasks: ~4.8 pt, so screening uses 1 repeat; held-out Web = 179 tasks.
  - **Android:** repeats 60.5 / 61.4 / 58.9% (57 of 58 tasks scored). The largest pairwise gap is 1.8 pt;
    1.96·SD/√n is 5.3–6.2 pt (5–7% of tasks flip). Floor = 6.2 pt > 5, so screening uses 2 repeats
    (6.2/√2 ≈ 4.4 pt threshold).
- **E1.6 `OsmAndMarker_0` never runs** (Android): its setup fails with `no such table: map_markers` in the AVD's
  OsmAnd database, in all 3 baseline repeats. It stays in the split but is unscored in every arm, so comparisons
  use 57 Android dev tasks. This is an env bug, not fixed here (harness parity).
- **Race on the shared Web task file** (job 284011, `dz_F12_r0`). `filtered_task_file` rewrote
  `WebVoyager_data.cusi.jsonl` in place on every call, so a judge step that read it while another job was writing
  found no tasks. Fixed in `cusi/envs/webvoyager.py`: write only when the content differs, via a temp file and
  rename. The judge was rerun with `--judge_only`; any other run hit the same way gets the same rerun.
- **Moved small jobs onto 2 RTX A5000s (2 × 24 GB).** gemma-4-31b-it in bf16 doesn't fit, and the Android labeller
  (job 284086) failed at vLLM start-up. On A5000 nodes the model needs all 4 GPUs.
- **E1.7 Web floor at 250 tasks:** baseline 52.8% (52.8 / 50.4 / 55.2%); largest pairwise gap 4.8 pt;
  1.96·SD/√n 4.7–4.8 pt. Floor = 4.8 pt ≤ 5, so 1-repeat screening on `dev250` stands, with promote/drop threshold
  ±4.8 pt.
- **E2.5 Web screening round 1** (dev250, 1 repeat, threshold ±4.8 pt; `tables_dz_dev250.md`):
  - **Promote:** F10 (+7.6, p = 0.008) and F11 (+6.4, p = 0.017); calls per task unchanged.
  - **Drop:** F08=1 (−7.2, p = 0.018; a single past screenshot hurts). Borderline and outside the top half of the
    prior ranking, also dropped: F04 (+2.8), F09 (−1.2), F21 (+0.8).
  - **Retest** (borderline, top-half prior): F41 (−3.6), F32 (+0.8; it was +6.2 on the first 150 tasks), F27 (+0.8),
    F14 (+1.2).
  - **New variant, from the F08=1 result:** F08=5 (5 past screenshots instead of 3).
  - Pending: F12, F14=google, F19, F37, F49.

### What didn't work (continued)
- **Web F08=1** (keep 1 past screenshot instead of 3): −7.2 pt on 250 tasks (p = 0.018), dropped.
- **Web F09** (programmatic change report), **F21** (plan field) and **F04** (page text): no gain on 250 tasks
  (−1.2, +0.8, +2.8), dropped at screening.
- **E2.6 Packed arms.** From now on GameBoy and Web arms run 2 per job on one shared vLLM server
  (`slurm/eval_pack.sh`; smoke-tested in job 284102), roughly doubling throughput. Android stays one arm per node
  (adb).
  - Decoding and harness are unchanged; only server concurrency differs. GPU-hours are split evenly between packed
    arms in the tables.
  - **Check:** a packed GameBoy baseline (repeats r3 and r4 in one job) is compared with the unpacked r0–r2. If it
    differs by more than the floor, the comparisons will use packed baselines.
- **E2.7 Web screening round 2, part 1** (dev250, 1 repeat, ±4.8 pt):
  - **Promote:** F14=google (+5.6, p = 0.049; 49 wins, 27 losses). It removes the `Google` action, the cause of 22 of
    the 25 env_error failures (E2.4). Calls per task are unchanged (9.7 vs 10.0).
  - **Drop:** F37 vote k=3 (+0.8 at 3.0× calls, so the cost rule applies) and F49 query rewrite (+0.0).
    F19 structured output (−1.2) is outside the top half of the prior ranking.
  - Still pending: retests of F41, F32, F27 and F14 (seed 1, packed jobs 284124–284127), F08=5 (284128), and F12 on
    the first 150 tasks (judge rerun 284095; +2.7 on dev2).
  - F29 self-written tips: the smoke run (284104) worked as claimed, with tips written once and shown each step,
    but scored 0/4, where other arms scored 2/4. Smoke runs check function, not effect, so F29 is screened
    (packed job 284155).
- **E2.8 Android failure taxonomy** (50 failed baseline episodes, `failures_android.jsonl`, labelled by the same
  Gemma; one reply didn't parse): wrong_plan 11, early_finish 10, out_of_steps 10, loop_noop 7, screen_misread 5,
  none 6 (likely judge or env mistakes). As a secondary bucket, screen_misread is the most common (12), then
  early_finish 6. "Early finish" means status complete was claimed with the work unfinished: items missed in
  multi-item tasks, or a dialog left unsaved.
- **E2.9 Android screening round 1** (2 repeats, ±4.4 pt; `tables_dz.md`):
  - **Promote:** F27 notes buffer (+7.0, 9 wins and 2 losses, p = 0.065). This fits the early-finish bucket: the
    notes track which items are already done.
  - **Drop:** F31 no-op note (+0.0).
  - **Borderline, top half of the prior ranking, so retest with a 3rd repeat:** F32 loop warning (+3.5; 5 wins,
    1 loss), F21 plan (+2.6), F34 reflect (+2.6).
  - F41 completion self-check (−3.5): it aims at early_finish, but losses ≥ wins (4/5), so drop. On Web it also
    trends negative (−3.6, retest pending).
  - Pending: F10 and F11 (running), F25, F08, F38, F19 and F29 (jobs 284158–284167). F37 is not screened on
    Android: vote triples the calls (Web: 3.0×, +0.8), and the cost rule would drop it unless it gained ≥ 4.4 pt.
- **Web F37 (action vote, k=3)**: +0.8 at 3× the calls; dropped. **Web F49 (query rewrite)**: +0.0, dropped.
- **Android F31 (no-op note)**: +0.0 over 2 repeats, dropped. **Android F41 (completion self-check)**: −3.5,
  dropped. It targets the early-finish bucket, but it won 4 tasks and lost 5, and it added about 4 calls per task.
- **E3.1 Combination rules (stage §5.2)**, fixed before any combination run. Per env, separately:
  - **Repeats:** 2 for Web and Android, 3 for GameBoy. The combination step size is t = floor/√r: Web 3.4 pt,
    Android 4.4 pt, GameBoy 6 pt. The baseline and each single-feature arm are topped up to the same repeats if
    they lack them.
  - **Order:** promoted features, ranked by their screening diff (retest repeats included). The best one is the
    starting design D.
  - **Greedy step:** run D+f for the next feature f. Keep f if D+f beats D, paired on the same tasks and seeds,
    by more than t/2 and its calls stay under 2× base. Otherwise f is left out: the simpler design wins ties.
  - **Same-target pairs** are mutually exclusive alternatives: F10 vs F11 (budget), F31 vs F32 (no-op/loop), and
    F27 vs F25 (carried memory). Each pair's lower-ranked feature is tried as a swap for the kept one after the
    greedy pass, kept only by the same rule.
  - **Supervisor:** the supervisor stays off, the setting of every baseline. Supervisor adaptations are a
    separate step on top of the final executor, only if the lit review's supervisor-side features are prototyped.
- **E3.2 Leave-one-out rules (stage §5.3):** 3 repeats per env on dev. Drop each component c of the final design
  if (D − c) − D > −t/2, using the 3-repeat t = floor/√3 (Web 2.8 pt, Android 3.6 pt, GameBoy 6 pt). Ties go
  to the simpler design. If more than one component qualifies, drop only the one whose removal hurts least, then
  re-check the rest.
- **E2.10 Android F10/F11 (2 repeats):** F11 forced finish +3.5 (5 wins, 3 losses), F10 budget shown +1.8 (3/4).
  Both are borderline and in the top half of the prior ranking, so each gets a 3rd repeat (E2.1).
- **E2.11 Web retests, part 1 (2 repeats, threshold 4.8/√2 = 3.4 pt):** F14 (wait pruning) +1.2 and F32 (loop
  warning) +1.6 are still borderline after their one retest, so both are dropped. F14=google stays promoted (E2.7).
- **Web F14=wait and F32**: +1.2 and +1.6 over 2 repeats on 250 tasks. Neither reached the threshold; dropped.
- **E2.12 Web retests, part 2 (2 repeats, ±3.4 pt):** F27 notes +1.2 (47/43), F41 answer self-check −2.8
  (42/48), so both are dropped. The Web screening survivors so far: F10, F11, F14=google. Still pending: F08=5,
  F12, F29.
- **Web F27 (notes buffer)**: +1.2 over 2 repeats; it helped on Android (+7.0) but not on Web (why
  not is unchecked). **Web F41 (self-check before ANSWER)**: −2.8 over 2 repeats, with more losses than wins
  and about 10% more calls. Dropped on both Web and Android.
- **E2.13 Web F08=5 (keep 5 past screenshots, base keeps 3):** +5.2 (46 wins, 28 losses, p = 0.060). Calls are
  unchanged and tokens per task are +26%, so it is **promoted**. With F08=1 at −7.2, the effect rises with
  history depth, so F08=8 is screened as a further variant; the vLLM limit is 16 images per prompt. If F08=8 also
  passes, the higher-ranked of the two enters the combination.
- **E2.14 Web F29 (self-written site tips):** +2.4 (42/32), p = 0.45, and +11% calls. It is borderline and
  not on the Web shortlist (lit review §4), so it is **dropped** for Web. It stays in screening on Android and
  GameBoy.
- **Web F29 (self-written tips)**: +2.4 on 250 tasks with 1 repeat, under the 4.8 threshold; dropped.
- **E3.3 Web combination, step 0:** F10 alone over 2 repeats is +7.2 (64/28, p < 0.001), so D = F10. The greedy
  order is F14=google (+5.6), then F08=5 (+5.2); F11 (+6.4) is F10's same-target swap. D+F14=google is queued
  (284172, 284173). To save queue time, D+F14=google+F08=5 is queued too (2 repeats). It is used only if F14=google
  is kept; otherwise D+F08=5 is run.
- **E2.15 Android F25 (self-summary in the action call, no separate summary call):** −1.8 (3/4), neutral, at
  half the calls (11.4 vs 23.3 per task) and 44% of the tokens. It meets no promote condition (E2.1) and is 7th on
  the shortlist, so it stays out of the success-driven greedy design. Because it halves the cost, it gets one
  pre-set extra test: final Android design + F25 vs the final design (3 repeats). It is reported as a cheaper
  variant if it is within t/2 of the final design. It never replaces the success-chosen design.
- **E2.16 Packing check (E2.6), GameBoy:** packed baseline repeats r3 and r4 (one shared server, job 284153) score
  38% and 36%, against 34 / 32 / 34% unpacked. The largest pairwise gap is 6 pt, under the 10.5 pt floor, and
  task-level disagreement is 4–14%, the same range as unpacked pairs. Packing stays. The GameBoy baseline now
  pools all 5 repeats: 34.8%, with packed and unpacked both present, so neither condition favours the arms. The
  floor stays 10.5 pt and the promote/drop threshold stays ±6 pt for 3-repeat arms.
- **E2.17 Android F08 (screen before the last action as an extra image):** +3.6 (5/4), p = 0.55, at unchanged
  calls. It is borderline and 9th on the Android shortlist, so it is **dropped** (E2.1). On Web, more history
  helped (F08=5, +5.2). A deeper Android variant (the last 3 screens) would need new prototype code. It is listed
  as a possible variant if Android has budget left after combination, not run now.
- **Transient APK download failure (Android, job 284162 `dz_F38_r0`).** One of 3 emulator shards died at start-up:
  `android_env` downloads the accessibility-forwarder APK from the network for each controller, and the read
  was cut short (`http.client.IncompleteRead`). The other two shards ran (39 of 58 tasks). The run was
  resubmitted with the same name, and finished tasks are skipped (`run.done`). Not fixed in code: the download
  is inside `android_env` (site-packages). If it recurs, cache the APK locally.
- **E2.18 Android F19 (structured output):** +5.3 (8 wins, 4 losses, p = 0.27) over 2 repeats, at 1.07× the
  calls, so it is **promoted** (≥ 4.4 pt). Android promoted so far: F27 (+7.0), F19 (+5.3). On Web the same
  feature was −1.2.
- **E3.4 Android combination, step 1:** D = F27 (+7.0, 2 repeats). Next by rank is F19 (+5.3), so D+F19 runs
  with 2 repeats (t = 4.4 pt; F19 is kept if D+F19 − D > 2.2 pt). The borderline 3rd-repeat retests (F10, F11,
  F32, F21, F34) and F38/F29 join the greedy order only if they are promoted.
- **E3.5 Web combination, step 1:** D+F14=google vs D=F10, 2 repeats, 250 tasks: +3.6 (47/32, p = 0.098), above
  t/2 = 1.7 pt, calls unchanged, so it is **kept**. D = F10+F14=google: 63.6% (66.4 / 60.8), +10.8 over base.
  Step 2, D+F08=5, is already queued (284224, 284225).
- **E2.19 GameBoy early drop (added rule):** an arm is dropped before its 3rd repeat if, after 2 repeats, its diff
  is ≤ −floor/√2 (−7.4 pt) **and** McNemar p < 0.05. Its remaining repeat is cancelled to free GPUs. **F30**
  (prior-knowledge game guide) is −12.8 (1 win, 11 losses, p = 0.003), so it is dropped and its r2 is cancelled
  (pack 284143; F04 r2 resubmitted alone). F21 (−7.8, p = 0.27) does not meet the p condition and keeps its 3rd
  repeat.
- **GameBoy F30 (game guide written from priors):** −12.8 over 2 repeats, 1 win and 11 losses: the most harmful
  feature screened so far. Episodes ran longer (133 vs 116 steps per task); why is not checked.
- **E2.20 Web F08=8 (8 past screenshots):** +4.4 (43/29), under the 4.8 threshold and below F08=5 (+5.2), at +24%
  tokens over F08=5. **Not promoted**; F08=5 is the history depth. The effect rises from 1 to 5 screenshots and
  plateaus or falls after 5.
- **Web F08=8**: no better than F08=5 (+4.4 vs +5.2) and costlier, not promoted.
- **E3.6 Web combination, step 2:** D+F08=5 vs D = F10+F14=google: +3.2 (48/34, p = 0.15), above t/2 = 1.7, calls
  unchanged (+21% tokens), so it is **kept**. D = F10+F14=google+F08=5: 66.8% (68.4 / 65.2), +14.0 over base
  (84/20, p < 0.001). The greedy pass is complete; the promoted list is exhausted. Next:
  - **Swap:** F11 for F10 (284239, 284240).
  - **Leave-one-out:** started early with the arm that doesn't depend on the swap outcome, F14=google+F08=5,
    3 repeats.
- **E2.21 Android F29 (self-written app tips):** +2.6 (7/5, p = 0.63), borderline, 8th on the Android shortlist,
  so it is **dropped**. Dropped on Web too (E2.14); GameBoy is pending.
- **E2.22 Android F32 (loop warning), 3 repeats:** +3.2 (5 wins, 1 loss, p = 0.18), under the 3-repeat threshold
  6.2/√3 = 3.6. Bucket check (E2.1): of the 4 tasks with loop_noop baseline failures, it fixed 1
  (SportsTracker), short of half. Its other wins were early_finish/wrong_plan tasks. Borderline after its
  retest, so **dropped**. It is the closest miss so far; per E3.1 it is not retried as a swap, since its pair
  partner F31 was dropped too.
- **E2.23 Android F38 (pre-execution critic):** +3.5 (7/4, p = 0.58) at 1.55× the calls (36.0 vs 23.3 per
  task). It is borderline and not on the Android shortlist, so it is **dropped**. The repeats were far apart
  (70.2 / 57.9%). That completes Android screening apart from the 3rd-repeat retests of F10, F11, F21, F34.
- **E2.24 Android F21 (plan field), 3 repeats:** +4.4 (7/4, p = 0.17), above 6.2/√3 = 3.6, at 1.10× the calls, so
  it is **promoted**. Android greedy order: F27 (+7.0), F19 (+5.3), F21 (+4.4). F21 is added after the
  D+F19 decision. On Web, F21 was +0.8; on GameBoy it is negative so far.
- **E2.25 Android F34 (outcome-classifying reflector), 3 repeats:** +1.8 (5/5, p = 0.58), still borderline after
  its retest, so **dropped**. It was 2nd on the Android shortlist (MobileUse +5.7); with this model and harness,
  the gain did not appear.
- **E2.26 Android F11 (forced finish at the budget end), 3 repeats:** +2.6 (4/3, p = 0.39), under 3.6 and still
  borderline after its retest, so **dropped**. On Web it was +6.4; why the forced finish helps less on Android
  (out_of_steps is 10 of 50 baseline failures) is not checked.
- **E2.27 Android F10 (budget shown), 3 repeats:** +2.0 (4/2, p = 0.55), still borderline after its retest, so
  **dropped**. Android screening is complete. Promoted: F27 (+7.0), F19 (+5.3), F21 (+4.4). Cost variant: F25.
- **E4.1 Full-benchmark confirmation: 1 repeat each (user decision, 2026-10-15).** The plan (§6) said 3 seeds;
  the user approved 1 repeat each of the baseline and the final design per env, on the full benchmark. The report
  keeps the paired per-task difference, its bootstrap CI and McNemar p, on the full set and the held-out part. With
  1 repeat the CIs are wider, roughly the dev noise floors: Web ±4.8, Android ±6.2, GameBoy ±10.5 pt.
- **E3.7 Web same-target swap (E3.1), F11 for F10:** F11+F14=google+F08=5 vs D = F10+F14=google+F08=5, 2 repeats,
  250 tasks: −2.8 (31/43, p = 0.22), so F10 stays and F11 is dropped. Leave-one-out (E3.2, 3 repeats) queued:
  D r2 (284246), D−F14=google = F10+F08=5 r0–r2 (284247–284249), D−F08=5 = F10+F14=google r2 (284250);
  D−F10 = F14=google+F08=5 r0–r2 is running/done (284241–284243). Interim D−F10 after 2 repeats: −7.6.
- **E3.8 Android combination, step 2 (F19):** D+F19 vs D = F27, 2 repeats: −4.4 (3/6, p = 0.23), below t/2 = 3.1,
  so F19 is dropped (alone it was +5.3; with notes it adds nothing, maybe because the notes field already gives
  the reply some structure). Next by rank is F21: D+F21 r0/r1 were queued speculatively and are running
  (284244–284245); interim on 46 tasks: +2.2 (3/1).
  F27 r2 (needed for leave-one-out either way) submitted as 284251.
- **E3.9 Android combination, step 3 (F21):** D+F21 vs D = F27, 2 repeats: +1.8 (6/4, p = 0.51), under t/2 = 3.1,
  so F21 is dropped. The shortlist (F27, F19, F21) is exhausted: **Android D = F27**. Leave-one-out (E3.2) for a
  single component is F27 vs base over 3 repeats (F27 r2 = 284251). The F25 cost variant (E2.15), F27+F25 ×3, is
  submitted (284287–284289); it is reported as the cheaper option if within t/2 = 1.8 of F27 over 3 repeats.
  Note: every feature that helped alone (F19 +5.3, F21 +4.4) adds under t/2 on top of the notes; the gains overlap.
- **E2.28 GameBoy screening, first 6 arms with 3 repeats (E2.1, threshold 10.5/√3 = 6.0):** none promoted.
  - F32 loop warning −0.1 (6/7), F28 objectives log −2.1 (7/11): borderline, 8th and 7th on the shortlist (bottom
    half), so **dropped**.
  - F16 −2.8 (8/11) and F25 −5.5 (4/8; at 0.52× the calls): borderline, not on the shortlist, so **dropped**. F25
    is also outside t/2 = 3.0 of base, so it is not kept as a cost variant.
  - F54 mode detection −3.5 (4/8) and F17 macros −5.5 (4/9): borderline and 2nd/3rd on the shortlist, so each
    gets a 4th repeat (pack 284311; threshold then 10.5/2 = 5.25).
  - Neither the loop features (F32, F28) nor the screen-reading ones (F54) fix their bucket (loop_noop 27 and
    screen_misread 15 failures): their paired wins and losses roughly cancel.
- **E2.29 GameBoy F10 and F26=15, 3 repeats:** F10 budget shown −2.8 (3/6), F26=15 (15-step history) −5.5 (2/8).
  Both borderline (|diff| < 6.0) and neither on the GameBoy shortlist, so both **dropped**.
- **E3.10 Web leave-one-out (E3.2, 3 repeats, drop c if (D−c) − D > −1.4):** D = F10+F14=google+F08=5 is 65.7%
  (68.4 / 65.2 / 63.6%). Removing F10: −7.9 (24/62); removing F14=google: −7.6 (24/63); removing F08=5: −2.7
  (39/53). All three hurt by more than 1.4, so all stay. **Final Web design: F10 + F14=google + F08=5**, 65.7% vs
  base 52.8% on dev250, +12.9 [+9.2, +16.7], 82 wins / 19 losses, p < 0.001, at 0.95× the calls per task.
- **E2.30 GameBoy F31, F21, F19, F07=3 (3 repeats, ±6.0):** F21 plan field −6.8 (4/14) **dropped** (≤ −6.0);
  F31 no-op ban −0.8 (7/6) borderline, 8th on the shortlist, **dropped**; F19 structured output −4.1 (3/8)
  borderline, not shortlisted, **dropped**; F07=3 upscale +2.5 (8/6) borderline, 4th on the shortlist (with F05),
  so a 4th repeat (284321).
- **E5.1 Clean implementation (plan §4), Web and Android winners.** Switches, all off by default:
  `WebVoyagerExecutor(max_attached_imgs=, show_budget=, allow_google=)` for F08, F10, F14=google, and
  `M3AExecutor(notes=)` for F27. `cusi/eval/features.py` routes an arm to the clean switches only when *every*
  feature in it has one (`CLEAN`), else the whole arm runs the prototypes as before, so queued mixed arms (e.g.
  Android F27+F25) are unchanged. Each run's meta records `implementation: clean|prototype`. A scratch check
  (`tmp_check_clean.py`) confirmed the clean code builds byte-identical text to the prototypes on the real
  WebVoyager system prompt and M3A action prompt; `tests/executor_design_test.py` holds the permanent checks. A live
  4-task smoke of each clean design is queued (Web 284328, Android 284329) before the full benchmark uses them.
  `max_attached_imgs=None` falls back to the module global, so the F08 prototype (which sets it) still works.
- **E5.2 §7 interaction check, step 1:** offline, on the GameBoy baseline episodes: does an in-episode novelty-archive
  revisit signal (cusi/state) flag loop failures better than the exact frame-hash detector of F32 (which did
  nothing, −0.1)? Only if it clearly does is an archive-based loop warning (source `ours`) prototyped and screened
  on GameBoy under E2.1, and on top of D for Web/Android under E3.1.
  - **Result (E5.2, 150 GameBoy baseline episodes; `storage/eval/design/s7_loop_signal.md`):** no. In-episode
    archive revisits (random_patch encoder, cosine 0.95–0.995) flag every step the exact hash flags plus
    3.6k–7.8k more of 17.3k, but neither signal separates loop_noop failures from other failures (AUC 0.50–0.58),
    and over the first 40 steps both are at chance even against successes; the whole-episode AUCs (0.81–0.95 vs
    successes) only reflect that failures run to the step budget. Revisiting is normal for this agent: F32's
    warning already fired on 35% of action calls. So no archive-based loop warning is prototyped. Not testable
    offline: the text half of the archive (GameBoy trajectories don't store text_boxes).
### What didn't work (continued, executor design)
- **§7 archive revisit as a loop detector (GameBoy):** see E5.2; denser than the exact hash and no better at
  telling loop failures apart.
- **E3.11 Android leave-one-out (E3.2, 3 repeats):** D = F27 is 65.5% (64.3 / 70.2 / 61.4%); removing F27 (= base,
  60.5%) costs 5.0 > 1.8, so it stays. **Final Android design: F27 (notes)**, +5.0 [+1.2, +9.6] over base on dev,
  8 wins / 2 losses, p = 0.039, at 0.96× the calls. The F25 cost variant (F27+F25 ×3) is still queued.
- **E4.2 Full benchmark submitted (E4.1, 1 repeat each):** Web, 429 tasks in 3 interleaved chunks of 143
  (`full0{0,1,2}_web.txt`), each one pack of base + design on one server (284341–284343; runs
  `dzfull0X_{base,F10+F14-google+F08-5}_r0`). Android, 116 tasks: base 284344, F27 284345 (`dzfull_*_r0`). Both use
  the clean implementation (E5.1); they were queued before the smoke runs finished and are cancelled if a smoke
  fails. GameBoy waits for the end of its screening. Held-out = full minus dev (Web: minus dev250).
- **E5.3 Web clean smoke passed (284328, 4 tasks, 2/4 solved):** meta `implementation: clean`; system prompt has
  no Google action; each observation ends with the [Budget] note (warning in the last 3 iterations); the final
  chat keeps 5 screenshots. The Web full benchmark (284341–284343) stands.
- **E5.4 Android clean smoke passed (284329, 4 tasks, 2/4 solved):** meta `implementation: clean`; every action
  prompt carries the Notes request and, after the first, the model's own notes (7 of 8 calls in
  AudioRecorderRecordAudio_0). The Android full benchmark (284344–284345, running) stands. GameBoy packs 284311 and
  284321 were held for about 1 h so the exclusive Android jobs could get whole nodes.
- **E2.31 GameBoy F37=3 and F38, 3 repeats (±6.0):** F37=3 vote −4.1 (4/8) at 2.16× the calls (491.8 vs 227.9):
  **dropped** (calls ≥ 2× with no gain). F38 pre-execution critic −1.5 (6/8) at 1.73× the calls: borderline, not on
  the GameBoy shortlist, **dropped**. GameBoy now waits on F29 (284156–284157), F04 r2 (284238), and the 4th
  repeats of F54, F17 (284311) and F07=3 (284321).
- **E4.3 Web full benchmark (E4.1, 1 repeat each, clean implementation):** base 56.4% vs design
  (F10+F14=google+F08=5) 66.4% on all 429 tasks: +10.0 [+5.8, +14.2], 66 wins / 23 losses, McNemar p < 0.001.
  **Held-out** (429 − dev250 = 179 tasks): 57.0% vs 66.5%, +9.5 [+2.8, +16.2], 28/11, p = 0.009. So the dev gain
  (+12.9) carries over to unseen tasks, slightly smaller as expected after selection on dev. Cost: 9.8 vs 9.9
  calls per task, 44.1k vs 33.6k tokens (+31%, the 2 extra screenshots), 297 vs 290 s per task; successes per
  GPU-hour 35.3 vs 32.2 (both arms shared each pack's server, so GPU-h is split evenly). Tables:
  `storage/eval/design/tables_dzfull00_{full,heldout}.md`.
- **E4.4 Scheduling:** the queued exclusive Android F25 runs (r1, r2) were reserving whole nodes and keeping 2 free
  A6000s each on eclairlg06/07 idle; they are held until the Android full benchmark frees its nodes. On the freed
  GPUs: GameBoy F07=3 r3 (284321) and the GameBoy **baseline on the full benchmark** (284434, `dzfull_base_r0`),
  which is needed whatever the screening decides (it is the design itself if nothing is promoted).
- **E2.32 GameBoy F04 (text_boxes in the prompt), 3 repeats:** −4.8 (3/8, p = 0.23), borderline (> −6.0) and 1st on
  the GameBoy shortlist, so a 4th repeat under E2.1 (pack below). Promotion would need ≥ +5.25 after 4 repeats,
  so it is very unlikely; the rule is followed as written.
- **E2.33 GameBoy F29 (self-written game tips), 3 repeats:** −2.8 (6/7, p = 0.84), borderline, not on the GameBoy
  shortlist, **dropped** (also dropped on Web and Android). GameBoy screening now waits only on the 4th repeats of
  F54, F17 (284311), F07=3 (284321) and F04 (284435).
- **E2.34 GameBoy F04, 4 repeats:** −3.8 (3/8, p = 0.12), still under +5.25 and a retest has been used, so
  **dropped**. Giving the model the VRAM dialogue/menu text does not help this agent on GameBoy, though the
  literature ranked it first (lmgame, Gemini, BALROG). Episodes with F04 are not shorter (120.8 vs 115.9 steps
  per task). Why it doesn't help is not checked; one untested guess is that the model already reads the text
  from the screenshot.
- **E2.35 GameBoy F54 and F17, 4 repeats (threshold 5.25):** F54 mode detection −3.8 (3/9), F17 macros −5.3 (5/8,
  at 0.87× the calls): both **dropped** (retest used, no gain). F17 is right at −5.25, i.e. it also meets the drop
  line. GameBoy screening now waits only on F07=3 r3 (284321).
- **E2.36 GameBoy F07=3 (3× upscale), 4 repeats:** +1.2 (8/7, p = 0.70; per repeat 36 / 38 / 38 / 32%), under
  5.25, **dropped**.
- **E3.12 GameBoy: no feature promoted, so no combination or leave-one-out; the final GameBoy design is the
  baseline executor (GameBoyRL's, supervisor off).** All 17 screened features were within the noise or harmful
  (best F07=3 +1.2 after 4 repeats; worst F30 −12.8). The full benchmark therefore runs the baseline only
  (`dzfull_base_r0`, 284434) and reports it against the earlier eval's numbers; there is no paired design arm.
  Contrast: in the earlier eval, the supervisors were what helped on GameBoy (31B: revision/subgoal +4.1, n.s.).
- **E4.5 Android full benchmark (E4.1, 1 repeat each, clean implementation):** base 62.2% vs F27 63.0% on 115
  scored tasks: +0.9 [−5.2, +7.0], 8/7, p = 1.0. **Held-out** (58 tasks): 62.9% vs 62.9%, +0.0, 3/3. On the dev
  tasks inside this same run: +1.8 (5/4), against +5.0 over the 3 dev repeats. **The Android gain does not
  replicate**: the dev estimate was mostly selection plus run-to-run noise (one-run floor 6.2 pt). F27 stays the
  reported design (it was chosen by the pre-registered rules and costs nothing: 22.4 vs 23.6 calls, −4% tokens),
  but the write-up states there is no evidence it helps on Android. More repeats on held-out would be needed to
  resolve a ~2 pt effect; not run (E4.1 fixes 1 repeat).
- **E2.37 F12 (Web `GoTo url` action) left unfinished:** its judge rerun (284095) asked for 4 A5000s and sat
  pending for 40 h (the only A5000 node is half used by another user). It was cancelled: F12 targets the same
  CAPTCHA trap as F14=google, which the final Web design already removes, so its result could not change the
  design. Its last partial number is +2.7 on dev2 (100 tasks), unconfirmed.
### What didn't work (continued, executor design)
- **Requesting A5000s for a judge-only job:** 284095 never got the node; judge-only reruns should go to an A6000
  pack instead.
- **E4.6 GameBoy full benchmark (baseline = final design, 1 repeat, 284434):** 31.3% [22.2, 40.4] on 99 tasks;
  dev half 38.0% (cf. 34.8% over 5 dev repeats), held-out 49 tasks 24.5% [12.2, 36.7]. No paired design arm (E3.12).
  The held-out half is harder for the baseline itself (129 vs 116 steps per task).
- **E3.13 Android F25 cost variant (E2.15), 3 repeats:** F27+F25 vs D = F27: −6.4 [−14.0, +0.6], 3/10, p = 0.027,
  outside t/2 = 1.8, so it is **not** reported as a cheaper design. Observation only (not a pre-registered test,
  not run on the full benchmark): against base it is −1.5 (4/5, p = 0.65) at 11.8 vs 23.3 calls and 73k vs 139k
  tokens per task, i.e. close to baseline success at about half the cost. Since F27's own gain did not replicate
  (E4.5), "baseline + F25" may be the more useful Android result; it would need its own held-out test.

### Stage 6: AndroidWorld, ideas from Minitap mobile-use (user request, 2026-10-15)
- **E6.0 Source.** Minitap mobile-use (arXiv 2602.07787, Feb 2026; open-source repo) reports 116/116 on
  AndroidWorld with a multi-agent LangGraph scaffold (Planner, Orchestrator, Contextor, Cortex, Executor,
  Summarizer; frontier models per role). Its own ablation (116 tasks, 1 run): multi-agent −21, post-validation of
  text input −15, sequential execution −12, hybrid perception −11, meta-cognitive rules −9, scratchpad −8, data
  fidelity prompt −3. Model fixed here (gemma-4-31b-it), supervisor off, AndroidWorld budgets unchanged
  (Minitap does not report its budget). Transferable pieces, as new catalogue IDs (source `minitap`):
  - F55 `clear_text` option on input_text (their focus_and_clear_text) instead of manual deletion.
  - F56 post-validation: after input_text, a deterministic note of what the field now contains.
  - F57 input_text without the automatic Enter (their typing tool never presses Enter).
  - F58 Cortex critical rules: history review / never repeat a failed action, complete only on observed
    evidence and verify formatting, data fidelity, form filling.
  - F59 Planner + Orchestrator folded into the executor: one planner call writes subgoals; the action prompt shows
    a checklist the model ticks on observed evidence; finish only when all are ticked.
  - F60 resource ids in the element list (their element targeting is resource-id first).
  - F61 Contextor: foreground app and device date/time in each action prompt.
  - Scratchpad = F27 (already tested); sequential execution = M3A already does one action per step; video: n/a.
  Env hooks needed (off by default, AndroidWorld behaviour unchanged): `element_resource_ids`,
  `input_presses_enter`, `last_ui_elements` on the Android env.
- **E6.1 Rules (fixed before any run).**
  - Smoke: all seven on 4 tasks; the episode records are read to confirm each does what it claims.
  - Screening on the 57 Android dev tasks, each feature alone vs base, 2 repeats (threshold 6.2/√2 = 4.4),
    promote/drop/retest as E2.1. In parallel, the **Minitap bundle** (F55–F61 + F27) vs base, 2 repeats.
  - Combination as E3.1 (2 repeats, keep if > t/2 = 2.2 over the current design), starting from base, promoted
    features by rank; then leave-one-out (3 repeats, E3.2).
  - Final candidate = the better on dev of (greedy design, bundle). **Confirmation on the 58 held-out tasks with 3
    repeats each of base and candidate** (base r0 is the full-benchmark run): improvement is **confirmed** only if
    the pooled paired diff is > 0 with its bootstrap 95% CI excluding 0; otherwise **denied** (or "not shown").
    This replaces E4.1's single repeat for this stage, because the single-repeat Android result (E4.5) could not
    resolve a dev-sized effect.
- **E6.2 Smoke (4 tasks, all of F55–F61; 2/4 solved, as the earlier smokes).** Working as designed: F55 (the model
  set `clear_text: true` on 153 of its input actions), F56 (e.g. "After typing, the text field contains: \"pro
  expense\""; "No text field is on the screen after typing …"), F59 (one planner call per task, 3–5 sensible
  subgoals, ticked via "Completed subgoals:"), F60 (resource ids in the element list). **Two bugs found and
  fixed before screening:** (1) F57's and F58's prompt edits never applied: the prompt-edit features shared one
  `call()`, which through the MRO ran only the first subclass's edit (each now has its own `call`; a check confirms
  all five edits reach the prompt when stacked); during the smoke the env therefore typed without Enter while the
  prompt still said it presses Enter. (2) F61 showed only the weekday: the `date` format was split at its spaces by
  the shell (now underscores). Screening starts with the bundle.
- **E6.3 Second source: Google ARTEMIS (user request, 2026-10-16).** Open-source Android agent (Apache-2.0, Aug
  2026), built on Minitap mobile-use; reports >99% on AndroidWorld (self-reported; the repo has no AndroidWorld
  harness, eval code or app-specific prompts). Gemini 3.x Flash models plus a robotics-ER grounding model; a
  "Flash" single-agent profile and a "Pro" LangGraph graph (planner → perception → operator → pre-execution safety
  net → validator → summarizer, with a final checker), unlimited step budget. Mapping onto what is already tested:
  notes/scratchpad = F27; live plan ledger = F59 (+F21); pixel-identical / dHash stuck hints = F31 + F32; checker
  final review and Outputter (verify before answering) = F41; element-index targeting is M3A's own; resource ids =
  F60; app launch = M3A's open_app. **Not transferable here** (each changes the fixed parts, not the executor):
  Gemini and robotics-ER models (model fixed); unlimited step budget (budgets fixed); Google Vision OCR fused into
  the tree (an external model); ADB shell for the operator and the checker's `content`-URI device probes (they
  bypass the UI and read the very state AndroidWorld's evaluator checks); the pre-execution safety net (it corrects
  coordinate drift, which index targeting does not have); action bursts (several actions per step, which changes
  the step budget's meaning). New catalogue IDs (source `artemis`):
  - F62 `expect`: the action reply adds "Expect: <the exact immediate screen change you expect>"; the next action
    prompt shows it and asks the model first to check the current screen against it (operator Phase 1 state
    verification + "exact immediate screen change you expect as proof of success").
  - F63 `listscan`: ARTEMIS's single-pass list/long-text rule, condensed: one direction with an anchor item,
    one-step recovery if the anchor is lost, the end proven by an end marker or one more scroll with nothing new,
    no item handled twice, stop early if the task needs only one match / N items.
  - F64 `swipe`: ARTEMIS's controlled scroll: the drag covers 70% → 30% of the container (40% of it) in 800 ms,
    instead of AndroidWorld's centre → edge at adb's default speed (which can fling past items). Same x/y centre
    line; prototype patches the env's `_execute` on the instance, the env file is unchanged.
  - Rules: as E6.1. Smoke on the 4 smoke tasks (all three), records read; then each alone vs base on the 57 dev
    tasks, 2 repeats (threshold 4.4); promoted ones join the E6.1 combination pool on equal terms. The bundle of
    E6.1 is not re-run with them; if any is promoted, the final candidate is the greedy design over F55–F64.
- **E6.4 Transfer to GameBoy (user request, 2026-10-16: "taking inspiration from what seems to work in
  androidworld, see if there's anything you can try in GameBoyWorlds").** Nothing Android-specific is confirmed yet
  (F27 did not replicate, E4.5; stage-6 screening is running), so "what works" here means what the two scaffolds
  that solve AndroidWorld rely on. Mapping onto GameBoy (failure taxonomy: loop/no-op 27, screen misread 15, wrong
  plan 4, out of steps 4): scratchpad/notes = F28 (−2.1); planner/subgoals = F21 (−6.8); stuck hints = F31/F32
  (−0.8/−0.1); critic/checker = F38 (−1.5); hybrid perception = F04 VRAM text (−3.8); history review / "never repeat
  a failed action" rules = F31's hard ban; post-validation of text input, resource ids, device context, clear-text,
  no-Enter typing, controlled swipe, list scanning: no GameBoy counterpart (button presses, short menus).
  **The one untested mechanism is F62 (expected change)**, which also targets both main GameBoy buckets (the model
  must notice when a press did not do what it believed) at no extra call. So only F62 is run:
  - Implementation: `GameBoyExpect` in the prototypes, the same text as `AndroidExpect` (EXPECT_SHOWN /
    EXPECT_RULE), the expectation shown before "Available environment actions:", the Expect line asked for after
    "Reasoning:"; GameBoyRL's parser reads only the Action line (checked on the real prompt).
  - Smoke: 1 run on the GameBoy smoke split, records read (Expect lines present and shown on the next step).
  - Screening: F62 vs base on the 50 GameBoy dev tasks, **4 repeats** (r0–r3), paired with base r0–r4 as in the
    earlier GameBoy screening; threshold 10.5/√4 = 5.3 points. Promote if ≥ 5.3, drop if < 5.3 (no retest: 4
    repeats is already the GameBoy retest depth, E3.12).
  - If promoted: held-out confirmation on the 49 GameBoy held-out tasks, 3 repeats each of base and F62 (base r0
    = the full-benchmark run's held-out tasks), confirmed only if the pooled paired diff's bootstrap 95% CI
    excludes 0 (as E6.1). If dropped: GameBoy stays at the baseline (E3.12).
- **E6.5 GameBoy F62 smoke (4 tasks; 1/4 solved).** Every deciding call (387) wrote an Expect line and every one
  after the first showed the previous expectation; the model does compare it ("The last action (A) successfully
  triggered a text box …, which matches the expectation"). GameBoyRL's history-summary calls are untouched. Dev
  screening submitted: F62 r0–r3, two packs of two arms.
- **E6.6 Stage-6 screening results (57 dev tasks, vs base 60.5%, 3 repeats), filled in as runs finish.**
  - Minitap bundle (F27+F55–F61), 2 repeats: 66.1% (67.9 / 64.3), +6.2 [−3.6, +15.8], 11/4, p = 0.26; calls ×1.03.
    Only +0.6 over F27 alone (65.5%). Kept as the bundle candidate for E6.1's final choice.
  - F56 read-back, 2 repeats: +2.6 [−2.3, +8.2], 6/5. Borderline with a top-half prior (Minitap's −15 ablation) →
    retest, 1 more repeat (r2).
  - **Prior ranks for E2.1's retest clause (fixed here, mid-stage, after F56's retest; E6.1 did not state them):**
    by Minitap's own ablation of the component: F59 planner/orchestrator (multi-agent −21) 1st, F56 post-validation
    (−15) 2nd, F60 hybrid perception (−11) 3rd, F58 rules (meta-cognitive −9 + data fidelity −3) 4th; F55, F57,
    F61 not ablated separately (bottom). Top half = top 3. ARTEMIS features (F62–F64) have no ablation: bottom.
  - F58 rules, 2 repeats: 61.4%, +0.9 [−2.3, +5.3], 2/4. Borderline, 4th → **drop**.
  - F59 subgoals, 2 repeats: 64.0%, +3.5 [−1.2, +9.1], 5/3; calls ×1.06. Borderline, 1st → retest (r2).
  - F55 clear_text, 2 repeats: 59.6%, −0.9 [−6.7, +5.3], 4/6. Borderline, bottom rank → **drop**.
  - F57 no automatic Enter, 2 repeats: 63.2%, +2.6 [−2.6, +8.2], 5/3. Borderline, bottom rank → **drop**.
  - F61 device context, 2 repeats: 65.8% (66.7 / 64.3), +5.3 [+0.6, +10.8], 7/2, p = 0.18; calls ×0.99 →
    **promote** (≥ 4.4). The date now shows in full on every prompt ("Sunday 2023-10-15 15:35").
  - F60 resource ids, 2 repeats: 61.4%, +0.9 [−6.7, +7.3], 7/5. Borderline, 3rd (top half) → retest (r2).
  - F56 retest, 3 repeats: 61.7% (59.6 / 65.5 / 59.6), +1.2 [−3.2, +6.1], 5/5 → below 4.4 → **drop**.
  - Queued now for E6.1's final dev comparison (greedy design vs bundle, both at 3 repeats): F61 r2 and bundle r2.
    If the greedy design ends as F61 alone, its 3-repeat dev number is F61's.
  - F62 expectation line (ARTEMIS), 2 repeats: 64.0% (64.9 / 63.2), +3.5 [−2.6, +10.8], 8/5, p = 0.58; calls ×1.01.
    Borderline, bottom rank → **drop**. (Same direction as on GameBoy, +1.2, also dropped.)
  - F63 list-scan rule (ARTEMIS), 2 repeats: 64.0% (64.3 / 61.8), +3.5 [−2.0, +9.4], 6/3, p = 0.55; calls ×1.03.
    Borderline, bottom rank → **drop**.
  - F64 controlled swipe (ARTEMIS), 2 repeats: 61.4% (61.4 / 60.0), +0.9 [−5.8, +7.9], 7/6, p = 1.0; calls ×0.95.
    53 controlled swipes over the 2 repeats, 0 fallbacks (the prints are in the per-shard logs, not the job log).
    Borderline, bottom rank → **drop**. All three ARTEMIS features dropped.
  - F59 retest, 3 repeats: 63.7% (64.9 / 63.2 / 61.8), +3.2 [−1.8, +8.8], 6/3, p = 0.27; calls ×1.06 → below the
    3-repeat threshold (3.6) → **drop**.
  - F60 retest, 3 repeats: 62.9% (61.4 / 61.8 / 64.9), +2.3 [−3.8, +8.5], 8/5, p = 0.48; calls ×1.01 → below 3.6 →
    **drop**.
  - **Screening done: F61 is the only promoted feature (of F55–F64).** The E3.1 combination is therefore trivial:
    base + F61, kept if F61's 3-repeat gain > 2.2; leave-one-out of a one-feature design = the same comparison.
    Final candidate = the better on dev (3 repeats each) of F61 and the bundle; held-out runs wait for that choice
    (no held-out run of the losing candidate).
  - **Dev, 3 repeats each:** F61 66.1% (66.7 / 64.3 / 66.1), +5.6 [+0.6, +11.1], 9/2, p = 0.035; calls ×0.99. Greedy
    design = F61 (+5.6 > 2.2; its leave-one-out is the same comparison). Bundle 67.8% (67.9 / 64.3 / 70.2), +7.3
    [−0.6, +15.2], 11/4, p = 0.029; calls ×1.02. **Final candidate = the bundle** (higher on dev, E6.1).
  - **E6.9 Held-out confirmation (rules as E6.1):** bundle on the 58 held-out tasks, seeds 0–2, vs pooled base
    (dzfull_base_r0 + dzheldout_base_r1, r2; 60.3%). Confirmed only if the pooled paired diff's bootstrap 95% CI
    excludes 0. F61 alone is not run on held-out.
  - **E6.9 result: CONFIRMED.** Bundle on the 58 held-out tasks, 3 repeats: 68.4% (65.8 / 67.2 / 71.6) vs pooled base
    60.3% (62.9 / 58.6 / 59.5): **+8.0 [+0.9, +15.8]**, 11/3, p = 0.006; calls 22.6 vs 21.2 (×1.07), tokens ×1.13.
    The held-out gain matches the dev gain (+7.3). One episode unscored (SimpleSmsSend, r0: AndroidWorld's own SMS
    validator raised "Message could not be sent due to Android/emulator issue"); that task averages its 2 other
    repeats. Counted as a failure instead: +7.5 [+0.3, +15.2], still excludes 0. Caveats: the lower bound is close
    to 0; which components carry the gain is not established (F61 alone was +5.6 on dev, F27 alone +0.0 on
    held-out at 1 repeat); two of the 11 held-out wins are date-relative tasks (calendar next event, tasks due
    next week), consistent with F61's device date, an observation only.
    **Final Android design changes from F27 to the Minitap bundle (F27 + F55–F61).** Its clean implementation
    (off-by-default executor switches, as E4.0) is not done yet; the result above is from the prototypes.
- **E6.7 ARTEMIS smoke (4 tasks, F62+F63+F64; 2/4 solved, as the earlier smokes; second attempt after the port
  failure).** F62: 84 of 87 action calls wrote an Expect line, and each later prompt showed the previous one (the
  3 without are the first steps, which see no expectation; e.g. "Expect: The clipboard list will scroll down,
  potentially revealing the other copied texts"). F63: the list rule is in every action prompt. F64: only 3 scroll
  actions in the smoke, all on short lists that barely move, so a frame comparison could not show the new swipe
  (row-profile shift 0 or identical to earlier smokes); the code path is checked (the env calls `self._execute`,
  which the prototype replaces on the instance; swipe commands checked offline), and F64 now prints each
  controlled swipe so the screening logs show it ran. Screening submitted: F62, F63, F64, 2 repeats each.
- **E6.8 GameBoy F62 screening (50 dev tasks, 4 repeats vs base 5): 36.0% vs 34.8%, +1.2 [−4.8, +6.9], 11/8,
  p = 0.70; calls ×1.02, tokens ×1.11 → drop (< 5.3).** At 35 of 50 tasks it read +7.4 [−1.3, +16.6] (the 15 last
  tasks to finish were the long, mostly failed ones), a reminder that partial tables are not read for decisions.
  GameBoy stays at the baseline executor (E3.12); 18 executor features tried there, none promoted.
### Stage 7: supervisor features (user request, 2026-10-17: "Try all of these features for all of the envs and tell me if any work. You can be dirty about it.")
- **E7.0 Motivation** (plans/archive/results.md): supervisors lost on Android/Web (plans used up the 10–15 step budget on
  intermediate targets; the scored finish only went through on the final target; 0% success when the final target
  was never reached) and helped only on GameBoy (noisy regression check, 52–61% "undone"; strict judge).
- **E7.1 Features** (prototypes in `tmp_sup_features.py`, on GameBoyRL's `subgoal` supervisor unless noted; source
  `ours`, from the E7.0 diagnosis):
  - S1 persistent executor across legs: **already in the current code** (base.py carries `history()` into each leg:
    M3A summaries, WebVoyager chat, GameBoy history), unlike the run analysed in results.md. Not a separate arm;
    the smoke records are checked to confirm leg 2 sees leg 1's history.
  - S2 budget-aware plan: the plan prompt states the agent's total action budget and a step cap
    K = max(1, budget // max_leg_steps); a longer plan is merged down to K steps (the tail joined into the last);
    each leg gets max(max_leg_steps, remaining budget // remaining targets) steps.
  - S3 finish on any leg: every leg runs the real task, with the current plan step (and any resume hint) as the
    hint; the agent's finish goes through the env on every leg.
  - S4 monitor (new arm `monitor`): no plan; the executor runs the real task in 5-step legs (history carried); after
    a leg, if it looks stuck (no screen change in the leg, or at least half of its frames already seen twice in the
    episode), the supervisor writes a resume hint (judge-slice summaries + RESUME_HINT_PROMPT) for the next leg;
    otherwise no supervisor call.
  - S5 regression check off.
  - S6 judge vote: the consolidating judge call sampled 3 times, majority (slice summaries computed once).
  - S7 replan gate: plan-flaw review only after 2 failed attempts at the same step; at most 1 replan per episode.
  - S8 verify before finishing (Android, Web; GameBoy has no finish action): a finish on a scoring leg is held back;
    one supervisor call (task, answer, current screen, action trace) answers "Verified: yes/no"; yes -> the finish is
    sent to the env; no -> the reason becomes the hint for the next leg; at most 1 rejection per episode.
  - S9 knowledge from priors: one call per task writes a short brief of what the model knows about the app / site /
    game relevant to the task; it fills [INSIGHTS] in the plan, hint and plan-flaw prompts.
- **E7.2 Arms** (base executor, no executor features; supervisor and executor share the model and server; existing
  max_leg_steps default 10): `SUP=subgoal` (plain, the reference supervisor), subgoal + each of S2, S3, S5, S6, S7, S8,
  S9, `SUP=monitor` (S4), and the fix bundle subgoal + S2+S3+S5+S6+S7. Passed through `--features` as
  `SUP=<arm>,S..` (stripped before the executor is built), so run names are dz<split>_SUP-<arm>+S.._r<seed>.
- **E7.3 Repeats and splits (dirty, user OK'd):** Android 57 dev tasks × 2 repeats; Web dev250 × 1 repeat (run
  prefix dzdev250, tables merged with dz + dzdev2 for the base); GameBoy 50 dev tasks × 2 repeats. Smoke first: one
  run per env of the bundle + S8 + S9 (Android/Web) and of monitor, on the 4 smoke tasks, records read.
- **E7.4 What "works" means:** (a) **vs the no-supervisor baseline** (the existing base runs, same executor) — the
  real question; (b) vs plain `subgoal` — whether a fix helps the supervisor. Thresholds as the executor screening:
  Android 4.4 (2 reps), Web 4.8 (1 rep), GameBoy 10.5/√2 = 7.4 (2 reps). An arm that beats the baseline by at least
  the threshold gets one more repeat (Android/GameBoy r2, Web r1); if it still holds, held-out confirmation (3 repeats,
  CI excluding 0, as E6.1) is planned and logged then. Everything else is reported as not working. Results go in
  E7.6 as they land, with a tracking table of what worked.
- **E7.5 Smoke (4 smoke tasks per env; subgoal + S2,S3,S5,S6,S7,S8,S9 (no S8 on GameBoy) and monitor (+S8 on
  Android/Web)).** All arms ran end to end on all envs. Solved: Android 2/4 and 2/4 (as earlier smokes); GameBoy
  monitor 2/4, subgoal bundle 0/4; Web 1/4 and 0/4 (these 4 Apple tasks: base 0–2/4 per repeat). Checked in the
  records: S1 (leg 2's first M3A prompt carries steps 1–5; WebVoyager leg 2 starts with the 12-message chat);
  S2 (Android/Web first plans of 1 step at budgets 10–15); S3 (every leg's hint "Work on this part of the task now:
  ..."); S6 (judge calls in threes); S8 (5 accepted, 4 refused, all refusals sensible: an invented release date, an
  answer from "general knowledge", duplicates not checked after one scroll); S9 (one knowledge call per task, brief
  fills [INSIGHTS]); S4 (GameBoy: 61 stuck hints over 4 episodes, about one in two legs, walking revisits screens;
  Android/Web: a few). **Bug fixed before screening:** S2 capped only the first plan, a replan could come back
  longer (a Web plan of 3 steps at cap 1); replans are now merged to the remaining cap. Note: prototype arms on
  GameBoy run the executor with GameBoyRL's supervised token cap (8000 vs 2000 for the baseline), as the earlier
  supervisor runs did. Web smoke: one task lost to a browser renderer timeout (infrastructure).
- **E7.6 Screening findings (as they land).**
  - **Web judge bug (found here; affects every Web subgoal run, including the earlier ones in plans/archive/results.md).**
    WebVoyager's auto_eval reads the task from the saved chat (`interact_messages.json`), and `write_artefacts`
    saves the *final leg's* chat. Under a subgoal plan, the final leg's "task" is the last subgoal, so the judge
    scored about 40% of tasks (92–102 of ~245 per arm) against the subgoal text, not the user's task. Two arms
    (S6, S8) crashed the judge outright because a subgoal ran over several lines and the regex has no DOTALL.
    Arms that pass the original task to every leg are not affected: S2 (1-step plans), S3, monitor, and the bundle
    all had 100% original-task text. **Fix (dirty, post hoc):** `tmp_fix_web_task.py` restores the user's task in
    message 1, backs up the old file as `interact_messages.leg.json` and moves the old scores to `*.legjudge`. Then
    the 6 affected arms (subgoal, +S5, +S6, +S7, +S8, +S9) are re-judged (jobs 284754–6; agent runs kept). The
    proper fix is in `write_artefacts`: always write the user's task. Not applied yet, because running jobs
    import the eval code.
  - **GameBoy token-cap confound checked: no effect.** Supervised arms let the executor generate 8000 tokens, the
    baseline 2000. Executor outputs: base p99 ≈ 200 tokens, max 631; monitor max 613; subgoal max 765. No call is
    near 2000, so the cap never binds.
  - **GameBoy screening (50 dev × 2 repeats, vs base at 5 repeats, 34.8%):**
    - monitor 44.0%: **+9.2** [-0.8, +19.4], W/L 13/5. It is also cheaper than every subgoal arm: 235 calls vs
      332, and 6.6 successes/GPU-h vs base 4.75.
    - Other arms: S3 +7.2, S5 +6.2, S7 +6.2, S6 +3.2, plain subgoal -0.8, S2 -1.8, S9 -2.8, bundle -9.8.
    - Only monitor clears the 7.4 threshold, so it is retested.
    - S3 misses the threshold by 0.2 and has a large split between repeats (48/36), so it gets the same retest
      (dirty).
    - **Retest rule (written before submitting):** run r2 and r3 for monitor and for S3. The decision uses r0–r2
      (3 repeats, threshold 6.0, as E7.4). r3 only tightens the estimate. If monitor still clears the
      threshold, it goes to held-out confirmation (GameBoy test split, 3 repeats, pooled CI excluding 0).
  - **Web re-judge result:** fixing the task text barely moves scores. Old→new: subgoal 44.4→44.0, S5 48.0→46.8,
    S7 44.8→43.6, S9 44.0→41.6. Judging against the subgoal inflated these arms by 0.4–2.4 points, so the earlier
    Web supervisor results in plans/archive/results.md hold up roughly; the conclusion (subgoal hurts on Web) is unchanged.
  - **Slip (process):** rerunning `tmp.sh` resubmitted the 3 re-judge jobs (284757–9), because the file still held
    the sbatch lines and the grep hid the output. I cancelled them before any finished a judge. results.jsonl has
    250 rows and no duplicates for each arm, and scores.json matches the tables. Lesson: overwrite tmp.sh before
    running it, never grep a run whose contents I haven't just written.
  - **Web screening (dev250 × 1 repeat, vs base at 3 repeats, 52.8%; threshold 4.8):**

    | arm | success | vs base | W/L |
    |---|---|---|---|
    | monitor (S4) | 55.6% | +2.8 [-1.5, +6.9] | 40/32 |
    | subgoal + bundle | 53.2% | +0.4 | 43/39 |
    | subgoal + S2 | 52.0% | -0.8 | 40/40 |
    | subgoal + S3 | 48.8% | -4.0 | 37/48 |
    | subgoal + S6 | 47.6% | -5.2 | 35/51 |
    | subgoal + S5 | 46.8% | -6.0 | 34/55 |
    | subgoal (plain) | 44.0% | -8.8 | 33/59 |
    | subgoal + S7 | 43.6% | -9.2 | 33/58 |
    | subgoal + S9 | 41.6% | -11.2 | 24/63 |
    | subgoal + S8 | 40.4% | -12.4 | 25/64 |

    Nothing clears the threshold, so nothing is retested.
    - Fixes that stop the plan from eating the budget (S2, S3, the bundle) bring subgoal back to base, but not
      above it.
    - Monitor is base-level on success at about half the GPU time of plain subgoal.
    - S8 (verifying the finish) hurts. The verifier refused 44 of 117 finishes. Of the 27 tasks plain subgoal
      solves and S8 loses, 23 end at max_steps, so the retry after a refusal ran out of budget. Plain subgoal is
      already budget-starved on Web, and S8 makes that worse.
    - S9 (knowledge brief) hurts the most of the single features.
  - **GameBoy retest result:**
    - **Monitor holds:** r0–r3 = 42 / 46 / 48 / 44. Over r0–r2 it is 45.3%, **+10.5** vs base, above the 6.0
      threshold. With all 4 repeats it is +10.2 [+1.7, +19.1], W/L 14/5, about 6% fewer calls than base and
      6.8 vs 4.75 successes/GPU-h.
    - **S3 dropped:** r0–r3 = 48 / 36 / 32 / 32, so r0–r2 is +3.9. The first repeat was luck.
    - **Held-out confirmation, monitor (rules as E6.4/E6.1, fixed now):** the 49 GameBoy held-out tasks.
      - Base: r0 = `dzfull_base_r0` restricted to held-out tasks, plus new r1 and r2.
      - Monitor: r0–r2.
      - Confirmed only if the pooled paired diff's bootstrap 95% CI excludes 0.
      - 3 packs (jobs 284813–5), 8 workers each.
    - **Result: CONFIRMED.** 49 held-out tasks, 3 repeats each.
      - Monitor 35.4% (28.6 / 36.7 / 40.8) vs base 27.2% (24.5 / 28.6 / 28.6).
      - Pooled paired diff **+8.2 [+0.7, +16.3]**, p = 0.008, W/L 8/2.
      - Same call count as base (249 per task), 4.69 vs 3.33 successes/GPU-h.
      - This is the first GameBoy change that beats the baseline, after 18 executor features failed. It is
        supervisor-side, so the executor-only negative result in `appendix_notes/gameboy_executor_search.tex`
        still stands as written; the appendix could add a pointer to this.
    - **Monitor vs the subgoal supervisor (user, 2026-10-18: "does it beat having subgoal planner as a
      supervisor? That's the real check").**
      - Dev, paired over 50 tasks: monitor 45.0% (4 reps) vs plain subgoal 34.0% (2 reps), **+11.0
        [+2.5, +20.0]**, W/L 16/4.
      - No held-out subgoal run exists. The old `abl_*_subgoal` runs are a different task set (Pokémon Red only,
        older code, 5-step legs), so they are not comparable.
      - **Rule (fixed before running):** plain `SUP=subgoal` (the E7 reference, 10-step legs) r0–r2 on the 49
        held-out tasks, paired with the monitor's held-out r0–r2. The monitor beats subgoal only if the pooled
        paired CI excludes 0.
      - **Result: NOT CONFIRMED. The monitor does not beat subgoal on held-out** (jobs 284943–4, 49 tasks × 3).
        - Subgoal 38.8% (36.7 / 40.8 / 38.8) vs monitor 35.4% (28.6 / 36.7 / 40.8).
        - Monitor − subgoal = **-3.4 [-9.5, +2.0]**, p = 0.36, W/L 5/8. This is inside the subgoal noise floor
          (9.8–13.4 pt).
        - Subgoal vs base (r1–r2 only) = +10.2 [+0.7, +20.4]. On held-out, both supervisors beat base by similar
          margins.
        - Cost: the monitor uses 249 calls/task, 3.7 GPU-h per repeat and 4.69 successes/GPU-h. Subgoal uses 316,
          5.3 and 3.61, so the monitor is ~30% cheaper.
        - Dev favoured the monitor (+11.0 [+2.5, +20.0]), but held-out does not reproduce that. So the dev win
          over subgoal did not hold, and on accuracy the two are a tie.
        - The GameBoy supervisor choice went to the user, who chose **subgoal** (2026-10-19, §E8.0).
  - **Android screening (57 dev × 2 repeats, vs base at 3 repeats, 60.5%; threshold 4.4):**

    | arm | success | vs base | W/L |
    |---|---|---|---|
    | subgoal + bundle (S2,S3,S5,S6,S7) | 64.9% | +4.4 [-0.9, +10.2] | 7/3 |
    | monitor (S4) | 61.4% | +0.9 | 5/4 |
    | subgoal + S2 | 61.4% | +0.9 | 7/5 |
    | subgoal + S3 | 60.5% | +0.0 | 4/5 |
    | subgoal + S6 | 49.1% | -11.4 | 3/13 |
    | subgoal (plain) | 47.4% | -13.2 | 3/14 |
    | subgoal + S5 | 46.5% | -14.0 | 3/14 |
    | subgoal + S7 | 46.5% | -14.0 | 4/16 |
    | subgoal + S8 | 38.6% | -21.9 | 2/19 |
    | subgoal + S9 | 29.8% | -30.7 | 3/23 |

    The same shape as Web. S2 and S3 each repair the budget problem, back to base level. S8 and S9 are strongly
    harmful.
    - S9: the brief adds plausible but wrong app knowledge.
    - S8: refusals plus retries run out of steps.
    - The bundle sits exactly at the threshold, so it gets r2 (decision at 3 repeats vs 3.6). Its comparison to
      the Minitap executor bundle (E6.9) is moot unless it survives.
    - **Retest: holds.** r0–r2 = 64.9 / 64.9 / 66.7, so 65.5%, **+5.0** [-1.2, +11.4], W/L 9/3; threshold 3.6.
      Cost: 32.5 calls vs 23.3, GPU-h ×1.2.
    - **Held-out confirmation (E6.1 rules, fixed now):** bundle r0–r2 on the 58 held-out tasks vs the same
      pooled base as E6.9. Confirmed only if the pooled CI excludes 0. Even if it is confirmed, it is below the
      Minitap executor bundle (held-out +8.0) and cheaper alternatives exist. Stacking the two is a follow-up,
      not part of this stage.
    - **Result: NOT CONFIRMED.** 58 held-out tasks, 3 repeats each.
      - Bundle 62.4% (68.1 / 56.9 / 61.4) vs base 60.3%: **+2.0 [-4.0, +8.3]**, p = 0.54, W/L 8/6.
      - The dev gain (+5.0) was mostly selection on dev noise; the large split between repeats (57–68) says the
        same.
      - Android stays on the Minitap executor bundle (held-out +8.0) with no supervisor.
- **E7.7 What worked: tracking table (screening diff vs the no-supervisor base, points; ✔ = cleared the env
  threshold and its retest, ✘ = did not).**

  | feature | Android | Web | GameBoy | verdict |
  |---|---|---|---|---|
  | plain subgoal (reference) | -13.2 | -8.8 | -0.8 | hurts or is neutral |
  | S1 history carry | built in (already on) | built in | built in | not separable |
  | S2 budget-aware plan | +0.9 ✘ | -0.8 ✘ | -1.8 ✘ | repairs subgoal to base; no gain |
  | S3 legs run the real task | +0.0 ✘ | -4.0 ✘ | +7.2 → +3.9 on retest ✘ | repairs subgoal to base; no gain |
  | S4 monitor (stuck-only hints) | +0.9 ✘ | +2.8 ✘ | **+9.2 → +10.5 at 3 reps ✔; held-out +8.2 [+0.7, +16.3] CONFIRMED vs base; vs subgoal held-out -3.4 [-9.5, +2.0] (tie)** | **beats base on GameBoy; ties subgoal at ~30% less GPU** |
  | S5 no regression check | -14.0 ✘ | -6.0 ✘ | +6.2 ✘ | no |
  | S6 judge voting ×3 | -11.4 ✘ | -5.2 ✘ | +3.2 ✘ | no |
  | S7 replan only after 2 fails | -14.0 ✘ | -9.2 ✘ | +6.2 ✘ | no |
  | S8 verify the finish | -21.9 ✘ | -12.4 ✘ | n/a (no answer) | **harmful** |
  | S9 knowledge brief | -30.7 ✘ | -11.2 ✘ | -2.8 ✘ | **harmful** |
  | bundle S2+S3+S5+S6+S7 | +4.4 → +5.0 at 3 reps ✔; held-out +2.0 [-4.0, +8.3] ✘ | +0.4 ✘ | -9.8 ✘ | no (dev gain did not hold) |

  Pattern:
  - On the short-budget envs (Android, Web), a supervisor that plans subgoals loses, because the plan eats the
    budget. Fixing that (S2/S3) only gets back to base.
  - **The only supervisor that works is the light-touch monitor (S4) on the long-horizon env (GameBoy):**
    confirmed on held-out at +8.2 [+0.7, +16.3]. The Android repair bundle passed dev but failed held-out.
  - Open follow-up, not run: stack the monitor on Android's Minitap executor.
  - **Web judge bug fixed properly (after the queue emptied).** `write_artefacts` in `cusi/eval/web.py` now
    writes the user's task into the saved chat. Live smoke (plain subgoal, 4 smoke tasks, job 284937): every
    episode ran 3–4 legs, and all 4 saved chats carry the original task. The judge ran clean, 0/4 solved (these
    tasks: base 0–2/4).
  - Features that add LLM judgement at the end (S8) or up front (S9) hurt everywhere.
### What didn't work (continued, executor design)
- **Stacked prompt-edit mixins sharing one `call()`:** only the first edit in the MRO applied (E6.2); found by
  reading the smoke records, not by the unit-level prompt check, which tested each class alone.
- **vLLM port colliding with an emulator gRPC port (E6.3 smoke, first attempt):** eval.sh derives the vLLM port as
  8100 + (job id mod 800), and the Android emulators use gRPC ports 8554–8556; one job id mapped onto 8556, the
  third emulator could not bind it and the run failed before any task. Two queued screening jobs (F60 r0/r1) would
  have hit 8554/8555; they were cancelled and resubmitted under new ids (same arm, same seeds). Fix for later, when
  no job is running eval.sh: skip ports 8554–8560 when deriving the vLLM port.
  **Fixed (2026-10-17):** eval.sh and eval_pack.sh now use 8100 + (job id mod 400), i.e. 8100–8499, below the
  emulator gRPC ports.

### Stage 8: final port (user request, 2026-10-18: "I want the code at the root repo to force the best option, and not take in flags to switch stuff.")
Plan: `plans/archive/final_port_plan.md`, including the user's decisions on it.
- **E8.0 Frozen setups.**
  - Web: WebVoyager + F10 + F14=google + F08=5, no supervisor.
  - Android: M3A + F27 + F55–F61, no supervisor.
  - GameBoy: baseline executor + **the subgoal supervisor** (user, 2026-10-19: "1. subgoal", after the held-out tie
    in E7.6).
    - It is the E7 reference exactly as the held-out arm ran (`SUP=subgoal`, no S-features): GameBoyRL's
      plan / judge / regression / resume-hint / plan-flaw loop and prompts verbatim, with insights
      "(nothing recorded)".
    - Settings: 10-step legs, 3 attempts per plan step, 2 replans per episode, 8 frames per judge slice, and
      max(9, budget) attempts at the final step. The plan reviser sees the last 3 attempts. Supervisor calls use
      5000 tokens at the model's default temperature.
    - Plan-step legs (every step but the last) run the step as the executor's task. GameBoyRL's done-check after
      each action may end the leg ("agent_done"). The last step runs the real task, with the step as its hint.
    - The executor's done-check therefore comes back, as a hook that only the subgoal supervisor turns on (a leg
      argument, not a switch).
    - The monitor is not ported; it stays recorded here (E7.6) as a cheaper tie.
  - No design switches, flags or YAML keys. Submodules are untouched; every patch of submodule text is an anchored
    replace that raises at import if its anchor is missing.
- **E8.1 No task (`task=None`).**
  - The prompts get `EXPLORE_GOAL`, finishing is disabled (Web `ANSWER`, Android `status`), and the run ends on
    the budget or an env ending.
  - The Android F59 planner is skipped; everything else stays on.
  - GameBoy with no task: no plan, no judge and no hints. The executor plays one leg with the whole budget
    (there is nothing to plan or judge against), like the skipped Android planner.
- **E8.1b Checks for the subgoal port, rules fixed now (2026-10-19).**
  1. **Recorded replay (prompt identity on real data).** Held-out subgoal r0 episodes (`dzheldout_SUP-subgoal_r0`
     trajectories) are replayed on the real GameBoy emulator by the clean agent. A replay model returns each
     recorded response in order (batched calls take one response per prompt) and compares every call.
     - Pass: on at least 10 episodes, every supervisor and executor prompt equals the recorded prompt byte for
       byte, the attached images are equal, and the call count matches.
     - If the frames diverge, the cause is the emulator, not the code. I report it and check the prompt text
       only up to the first divergent frame.
  2. **Parity (as E8.4).** One dev repeat of the final GameBoy agent must land within 10.5 points of plain
     subgoal's dev mean (34.0%, 2 repeats). A miss counts as a porting bug.
  - **Result 1: PASS.** All 49 held-out r0 episodes, 15,904 recorded calls (`tmp_port_subgoal_replay.py`).
    - **Prompt text: identical on every recorded call of all 49 episodes**, covering every stage: plan, judge
      slices, judge, regression check, plan review, hint, and the executor's action, done-check and frame-diff
      calls.
    - **Pokémon Red (24 tasks): fully identical.** Text, images, call count, termination and reward all match.
    - **Pokémon Crystal (19 of 25): frames diverge from the first frame after an action. This is the emulator.**
      - The frame before the action is equal and the same button goes in, but the frame after differs (another
        palette, the START menu not yet drawn; checked by eye on crystal_007).
      - PyBoy's cartridge RTC reads the host clock (`pyboy/core/cartridge/rtc.py`: `time.time()`). Crystal's
        cartridge has the timer and Red's does not. The recordings ran around 23:00 and the replay around 16:30.
      - Since responses are replayed, this changes only the images sent and, through frame-changed flags, nothing
        in the text.
      - In crystal_006 the drifted game no longer ended where the recording did, so the replay ran past the end
        of the recording (max_invalid vs terminated). All of its 134 recorded calls still matched in text.
    - Re-run on the live tree after the swap (no overlay): the same result, 49/49 text-identical and 24/24 Red
      fully identical. All 113 cusi modules import, and the final agent, practice strip and GameBoy prompt
      tests pass.
    - Parity runs submitted 2026-10-19, each 1 dev repeat, run name `final_<split>_r0`: GameBoy dev 285133,
      Web dev250 285134, Android dev 285135.
    - Side effect to note: **Crystal tasks depend on the wall clock**. Repeats at different times of day are not
      the same episode. This held for every GameBoy run so far, base and every arm alike. It adds noise to
      Crystal comparisons but no bias, so no earlier conclusion changes.
- **E8.2b Web env: no Google (user, 2026-10-19: "2. Yes").** The Web env drops the `Google` action from its menu
  and rejects it, as the agent already did (F14). The agent, search, practice and new world-model data then share
  one action space. The world-model action vocabulary keeps its `google` slot so trained models still load.
  - **Done.** `without_google` and `MSG_NO_GOOGLE` live in `cusi/envs/webvoyager.py`; the executor's prompt uses
    them, and its own pre-env rejection stays, so agent behaviour is unchanged. The test checks both menus.
    A live env check (Apple--0, test and free_play) showed `Google` rejected without any browser action and
    the next action valid.
- **E8.2 Android env global.** Resource ids are always shown and `input_text` never presses Enter, everywhere.
  Old Android exploration trees, curiosity data and everything derived from them are deleted (user OK'd).
  - **Fix (2026-10-19): the env's action menu was left stale.** It still said `input_text` presses Enter and had no
    `clear_text`. Search's VLMExplorer and practice proposals read it (`obs["actions"]`), so they saw the wrong
    action semantics.
    - The F55/F57 action-text edits now live in `cusi/envs/android_world.py` (`describe_actions`). The menu and
      M3A's prompt both use it; the executor keeps only F55's guidance tip.
    - Agent prompts are byte-identical before and after (6 variants: task/no task × guidance × subgoals).
    - `tests/final_agent_test.py` now checks the menu in both modes.
- **E8.3 Executor split.** `begin` / `act` / `record`, with `run()` rebuilt from them; this has no behaviour change.
- **E8.4 Checks, rules fixed now.**
  1. **Prompt identity before any prototype is deleted.** On real recorded observations, the clean code's action
     prompts must equal the confirmed prototypes' prompts byte for byte. The only exception is Android's device
     date/time, which is masked.
  2. **Live smokes**, 4 smoke tasks per env with a task, plus one no-task run per env. Records are read by hand.
  3. **Parity:** 1 repeat on dev with the final code.
     - Web dev250 vs the confirmed design's dev mean, 65.7%.
     - Android 57 dev vs the bundle's dev mean, 67.8%.
     - GameBoy 50 dev vs the chosen GameBoy setup's dev mean.
     - Pass: within the env's 1-repeat noise band (Web 4.8, Android 6.2, GameBoy 10.5). A miss is treated as a
       porting bug to find, not noise.
- **E8.4.1 Prompt identity: how it is checked (the staged code runs via an overlay).**
  - The GameBoy jobs still running import the live supervisor/executor files, which must not be edited. The new
    base / GameBoy executor / supervisors are therefore staged in `tmp_port_stage/`, and `tmp_port_overlay.py`
    makes them shadow the live modules in its own process only.
  - The confirmed implementations are frozen in `tmp_port_ref/`: the executor snapshot and both prototype files.
  - **Monitor, offline, against the confirmed runs' recorded prompts (`tmp_port_monitor_check.py`).** Every
    recorded monitor episode was replayed: held-out r0–r2 and dev r0–r3, 347 episodes, 7,738 legs.
    - All leg budgets and all hints carried into legs agree.
    - Every stuck/not-stuck decision agrees.
    - All 3,335 interventions' judge-slice prompts and hint prompts are **byte-identical**.
  - **Executors, live lockstep (`tmp_port_identity.py`).** The new agent runs a real task with the real model.
    The reference twin gets the same observations and the new code's replies, and every model call (text, images,
    max_new_tokens, temperature) is compared.
    - Android's device date/time is masked.
    - The reference re-observes the screen itself (M3A refresh), so a screen that changes between the two
      observations can differ for reasons outside the code; such cases are read by hand.
    - Smoke tasks: 4 per env, plus a no-task run.
- **E8.5 Implementation decisions taken during the port (none change eval behaviour).**
  - **Sampling settings are executor class constants**, set to the eval values that were confirmed:
    - Web: 1000 tokens, temperature 1.0, model errors end the leg.
    - Android: 1000 tokens, temperature 0.0.
    - GameBoy: 8000 tokens, server-default temperature, 4 invalid decisions in a row end the leg.
  - The invalid-decision limit is now off for Web/Android everywhere, as it was in eval.
  - Practice's `--executor_max_new_tokens` and `--temperature` are removed; practice now samples like eval.
  - **The finish goes to the env iff the env is in test mode.** Eval was test mode with finish-through-env;
    practice was free_play without it, so that is unchanged.
  - **Android: the executor always re-observes before deciding** (eval did; practice did not). Practice's
    Android env is now built with `stabilize_after_action=False`, as eval's.
  - **GameBoy:**
    - The executor is fixed to `single_visual` (one action per decision). Its custom `run()` is gone; three hooks
      on the shared loop reproduce it:
      - an unparseable action counts as an invalid decision;
      - an executed but unavailable action resets the invalid count and shows no error;
      - each executed step is observed by GameBoyRL's history policy.
    - The executor's done-check is removed: no final setup calls it. Practice attempts on GameBoy used it to stop
      early and now run to their budget.
  - **Practice runs the final agent.**
    - The critique hint becomes the agent's standing guidance, shown in every leg. Eval never passes guidance.
    - The judge and dataset get the run's legs merged into one LegReport (`merged_leg`); Web and Android have one
      leg anyway.
  - **Kept as run plumbing:** `--overwrite`, `--ignore_config_violation` (shared with search/explore run dirs),
    `--seed` (a label), and the task and shard selectors. Every run config records `agent: final-v1`.
  - **`cusi/explore/prompts.py` (the small PPO policy, already marked for deletion) keeps its prompts.** It imports
    the shared `EXPLORE_GOAL` and calls M3A's raw prompt; it was never the VLM agent.
  - **Deletions** need `rm`, which this session cannot run. They are collected for the user in
    `tmp_port_delete.sh`. The old Android search trees found: `storage/search/android/{curiosity_android_v2,
    gemma_vlm_v1}` (57 MB; `explore/android` is empty, no models were trained from them).
- **E8.4.1 result: prompt identity holds in every env (live, real model, 4 smoke tasks each).**

  | env | decision-time calls identical | post-step calls identical | decisions identical |
  |---|---|---|---|
  | Web | 53 / 53 | (none) | 53 / 53 |
  | Android | 75 / 75 (incl. 4 planner calls) | 67 / 67 (M3A summaries + read-back) | 71 / 71 |
  | GameBoy (under the monitor, ≤ 60 steps) | 191 / 191 | 191 / 191 (history-policy calls) | 191 / 191 |

  Plus the offline monitor check above, which matched all 3,335 interventions.
  - **Harness fixes, not port bugs:**
    1. The replay queued a batched call's reply list as one reply (GameBoy history policy).
    2. The Android reference first re-read the screen itself. About half the calls then differed only by the
       status bar ("Phone three bars." vs "Phone signal full.") and the clock (15:34 vs 15:35). The reference now
       gets the observation the new code read.
  - **No-task runs, all envs.** No finish was accepted. The explore goal was in every action prompt, there was no
    planner call on Android, and the monitor fired on GameBoy (4 hints in 40 steps).
    - Web: 15 steps, 3 invalid (1 format error).
    - Android: 12 steps, then the env truncated.
    - GameBoy: 40 steps.
  - Smoke outcomes with a task, in line with earlier smokes: Android 2/4, Web 0/4 (these Apple tasks: base 0–2/4),
    GameBoy 2/4 within 60 steps.
- **E8.6 Cleanup (user, 2026-10-19: "delete it [the PPO explore path]. In general do all the cleanup you can do
  right now").**
  - **PPO exploration removed.** It was already marked for deletion in favour of the search expander.
    - `run_explore.py` loses its `ppo` command and the PPO step constants.
    - `cusi/explore/replay.py` loses `ReplayWriter`; only the PPO trainer wrote replays, and the readers stay for
      the existing runs.
    - The scripts for the PPO job (`explore.sh` pair) and the policy-model check (`policy_model_check` pair) lose
      their defaults in `scripts/utils.sh`. Their files, plus `cusi/explore/{ppo,prompts,policy}.py` and
      `tests/policy_model_check.py`, go in the deletion script.
    - The other `run_explore.py` commands (tasks, world_model, train_embedder, decoder, wm_eval, elements) read
      replays and search runs, and stay.
  - **Experiment-era record fields dropped** (staged, as these files are imported by the running GameBoy jobs):
    - `LegEvent.target`, `final` and `self_terminate`, and `SupervisorReport.narrative` and `digest`; the episode
      writer no longer writes them.
    - `supervisor_extra` keeps counts, timing and the monitor's `n_interventions`, instead of the plan/replan/attempt
      keys of the old supervisors.
    - The monitor's state is a plain per-leg log.
    - Old pickled reports still load: pickle restores the extra attributes. The offline monitor check re-run on
      held-out r0 is unchanged: 49 episodes, 519 interventions, all identical.
  - **`scripts/submit_ablation.sh`** (submitted the removed supervisor arms) loses its defaults and goes in the
    deletion script.
  - **Left for the swap:** `cusi/utils/paths.py`'s `info_docs_dir` / `parametric_doc_file` (only `info_subgoal` used
    them), and `eval_pack.sh` arms (feature strings → `<seed>@<split>`). Both are in use by the running jobs.
- **E8.7 Full cleanup (user, 2026-10-19: "Finish the full cleanup").**
  - `tmp_port_delete.sh` listed `supervisors/subgoal.py` among the old supervisors to delete. Since E8.0,
    `subgoal.py` is the live port, so it was taken out of the list. Everything else in the script is unchanged.
  - **Check before deleting.** The repo was imported with the deleted modules blocked:
    - an import hook made the following raise ModuleNotFoundError: `eval.features`,
      `explore.{ppo,prompts,policy}` and `supervisors.{dummy,revising,info_subgoal,checker,_format}`;
    - 113 cusi modules and the 4 entry points (`run_eval`, `run_explore`, `run_practice`, `run_search`) all
      imported, 0 failed;
    - no test, script or slurm file outside the deleted set refers to these modules.
  - **README updated:**
    - the explore row has no `ppo` and no `explore.sh`;
    - the eval row has no `--supervisor`, and points at `eval_pack.sh`;
    - the supervisors row is now the final-agent row;
    - `tests/final_agent_test.py` and the design and port plans are listed.
  - **What didn't work.** `bash tmp_port_delete.sh` was refused for me twice by the permission system as
    irreversible deletion. The user runs it.
  - The script deletes every `tmp_*` file, including the parity watcher and scorer. The parity scorer was copied
    out of the repo first, so scoring after the delete still works.
  - **Done (2026-10-19).** The user approved running the script through `tmp.sh`, and it ran: no `tmp_*` files
    are left. Afterwards all 117 modules and entry points still import, and the three parity jobs kept running.
- **E8.8 Parity result (rule E8.1b / final_port_plan §5.4: one dev repeat per env must land inside the 1-repeat
  noise band, and a miss counts as a porting bug).** Jobs 285133–285135, gemma-4-31b-it, all COMPLETED with exit 0.
  **PASS on all three envs.**

  | env | run | scored | success | reference ± band | verdict |
  |---|---|---|---|---|---|
  | GameBoy | final_dev_r0 | 50/50 | 38.0% | 34.0 ± 10.5 | PASS |
  | Web | final_dev250_r0 | 250/250 | 68.4% | 65.7 ± 4.8 | PASS |
  | Android | final_dev_r0 (3 shards) | 56/58 | 66.1% | 67.8 ± 6.2 | PASS |

  - **Android, 2 unscored tasks:** both errors were raised inside AndroidWorld's own code, not ours.
    - `SimpleSmsSendClipboardContent_0`: the SMS validator raised "Message could not be sent due to
      Android/emulator issue".
    - `OsmAndMarker_0`: scene setup failed with `no such table: map_markers`, because the OsmAnd database
      was missing.
    - Android passes whether these count as failures (37/58 = 63.8%) or as successes (39/58 = 67.2%).
  - **Web:** the agent tried Google 6 times in the log, and the env rejected each one with MSG_NO_GOOGLE as
    intended.
    - Mean task time 219 s; mean tokens per task 41.0k in, 0.8k out.
  - **Wall time:** GameBoy 3.0 h; Web 2.8 h for the agent plus 1.0 h for the judge; Android 2.7 h on
    3 emulators.

- **Android 1 s action wait (job 285649, 2026-10-22): not made the default.** On 57 tasks scored in both runs,
  success fell from 63.2% (2 s) to 57.9% (1 s), −5.3 pt, with 5 tasks lost and 2 gained. That is within the
  ~6 pt noise band, but negative. Each step was only 6% faster (33.7 s → 31.6 s).

## 2026-10-23 VLM exploration variants (results/vlm_exploration_report.md)
- Did not help alone: coverage, think, feedback on GameBoy (still trapped 1-2/3 seeds). Objective alone doubles no-change rate (persists on unreachable goals); all four together fixes it (GameBoy 34-38 nodes, 0/3 trapped).
- Web cutoff 0.5 with objective+onsite looked worse at walk 38 (61% on-site) but at saturation (126 walks) found 138 on-site pages against 50-63 at 0.7, at 80% on-site. Judge cutoffs at saturation, not early.
- Web all four + onsite seed 42 fell into arXiv's donation site (benevity.com); "think" raised invalid actions to 11%.
- Frame-novelty gate (frame < 0.3 => no node) checked offline: removes Android license-scroll nodes but also 3/4 GameBoy dialogue nodes. Not adopted.

## 2026-10-23 VLM exploration runs the eval executors (replaces VLMExplorer)
- Search's VLM expander is `ExecutorExpander` (cusi/search/expanders.py): the env's eval executor, driven per decision
  (begin / act / record). Task = objective `Find and open "<feature>"` (a screen feature the scene has not
  discovered and the walk has not pursued); hint = `explore_hint` (path to the node, earlier walks and what they
  reached, discovered set, Web: stay on site). On finish (M3A status, WebVoyager ANSWER) or GameBoy's done-check
  (every step), the current screen's features give the next objective and the executor is retasked with its memory.
  An invalid decision (or a second finish in a row) sends the env an empty action: it uses a step.
- `EXPLORE_GOAL` and every task-None branch are gone: executors always have a task. `--vlm_variants` removed.
- No-change: every env sets `info["changed"]` (cusi/envs/base.py `screen_changed`: >0.05% of pixels move by >16;
  Android ignores the top 6%, its status bar). GameBoy uses GameBoyRL's screen parsing: an agent-state change
  (free roam / dialogue / menu / battle) counts; in free roam, MoveSteps' four-quadrant `judge_movement` (moved or
  turned); in menus, dialogues and battles, GameBoyRL's `frame_changed`. Tile animations (water, flowers) still make
  some no-ops count as changed (against an exact no-input replay, ~43% of free-roam no-ops and ~65% in menus):
  accepted. Rejected: the exact replay (save/restore the emulator every step) and a tile-map/sprite comparison
  (no misses, but too complex for what it buys).
  The executor base turns consecutive unchanged valid steps into a note for the next decision (GameBoy: error line;
  M3A: step summary; WebVoyager: observation warning). This applies in eval too. Search uses the same flag (the
  embedding `noop_similarity` is gone).
- Gemma-4-31B does not run on Quadro RTX 8000 (Turing): its 512-wide attention heads exceed Triton's shared memory.

## 2026-10-23 Curiosity explorer: no typing
- The curiosity policy's action space has no typing (`type` candidates, `type_strings.json`, `--max_type_strings` and
  the per-scene/generic string lists removed). Random walks never typed, so curiosity and random now share an action
  space; the VLM explorer keeps free-text typing. Web text fields get a click candidate like any element. Android
  `open_app` stays, over a fixed app list (`ANDROID_APPS` in cusi/search/curiosity/candidates.py).
- In the v1 curiosity trees typing was 5-6% of steps but preceded ~10% of nodes (search-results pages), so expect
  somewhat less Web/Android coverage.

## 2026-10-23 True coverage (evaluation only)
- Every env records `info["true"]` (TextActionEnv.true_state) after each reset and step; agents never see it. Saved
  per step, not per node (a node is only a step that crossed the novelty cutoff; coverage counts every state passed
  through): tree steps in expansions.jsonl (INFO_KEYS), curiosity-session steps in curiosity/<exp>_true.jsonl.
  `python -m cusi.search.coverage` computes the measures afterwards (cumulative, per 100 env steps; coverage.json).
- GameBoy: Pokemon RAM (addresses from pret's pokered/pokecrystal symbol files; Crystal's are in WRAM bank 1 and must
  be read from that bank, plain reads glitch to 0 when another bank is mapped): map, x, y, map size, battle, party,
  badges, Pokedex, items, money (Red's is BCD), event flags newly set vs the scene start. ~7 ms/step.
- Web: URL, title, page-skeleton hash (tag paths to depth 8), JavaScript functions newly run (DevTools precise
  coverage, enabled per browser). ~6 ms/step. Measures include on-site route templates.
- Android: foreground activity (gRPC) and the top activity's fragments (dumpsys activity top). ~140 ms/step.
  Google apps' fragment names are obfuscated but still identify screens.
- Not yet: Android settings/app-database diffs, Web network endpoints (need performance logging at driver creation).

## 2026-10-24 Curiosity pixel policies: CNN from scratch vs frozen embedders
- 5 h single-session comparison (gameboy viridian, web arxiv, android settings; one seed): fusion (MobileNetV3 +
  MiniLM + fusion transformer) vs GameBoyRL's CNN from scratch. True coverage: CNN ahead on GameBoy (305 vs 129 tiles,
  3 vs 2 maps; it ran 43% more steps; level at matched steps), tied on Web and Android. Fusion's GameBoy policy
  collapsed (return 11.8 vs 6.0, clipfrac 0) while covering less. Android managed only ~1.8k steps / 15 updates in 5 h
  (~8 s median step: 2 s post-action sleep, ~30 s snapshot restores per episode, Chrome/OsmAnd `am start -W`
  timeouts), too little to judge.
- `--curiosity_policy pixel` renamed `cnn`; new `frozen`: a frozen image embedder (`--policy_vision_encoder`
  mobilenet_v3_small, average-pooled 576-d, or a SigLIP 2 id, pooled 768-d) per frame, the last 4 embeddings through
  a trained 2-layer ReLU tower, then the same pointer head as `cnn`. Running cnn / mobilenet / siglip2-base on
  gameboy (viridian, pewter) and web (arxiv, github), 3 seeds each, 5 h, all on RTX 2080 Ti.
- Result (true coverage, 3 seeds, mean ± sd): no policy is better. Web, at matched steps: on-site routes arxiv cnn
  177±5 / mobilenet 178±7 / siglip 181±13, github 112±7 / 119±9 / 115±3; skeletons, titles, URLs agree. GameBoy tiles:
  viridian 81±18 / 79±10 / 88±16, pewter 117±61 / 112±59 / 88±7; 1-3 maps, no battles or event flags in any run.
  SigLIP runs ~35% fewer GameBoy steps (620k vs 840-960k) at no cost in coverage, because GameBoy coverage plateaus
  within the first ~10-50k steps (90% of final tiles by step 1.4k-220k, median ~15k): the other ~95% of a 5 h
  session adds nothing. The encoder is not the bottleneck; the reward/episode setup is.

## 2026-10-26 Tree -> tasks -> practice -> datasets
- `cusi.search.annotate` is one pass that writes every note on a finished tree (no cap per node): the existing
  features and transition reports, plus `cusi/search/notes.py`: per node a screen description and the tasks that
  could be done from it; per edge the task the transition completed (hindsight, `NO TASK` allowed) and actionable
  knowledge. A node's `tasks` list the tasks starting from it with their source ("node", or "edge" with the child).
- Practice (`run_practice.py --source <annotated run>`): each task once from its node, restored
  (`cusi/search/restore.py`: GameBoy/Web import the exported state, Android replays the path and checks it); the same
  model judges (judge_leg); on a failure a critique hint, shown to the executor only, then retry from the node, up to
  3 tries. One pickle per task with every try's full SupervisorReport (supervisor + executor calls), verdict, judge
  and critique calls. `dataset` converts them: success/ (successful try, cut at the safe success point) and
  failure/ (every failed try), split by task, success train rows repeated with model-written paraphrases.
- Guidance is marked for GameBoy (`Supervisor.MARK_GUIDANCE`) so the dataset cuts only practice's guidance and keeps
  the supervisor's own hints, as the agent sees them at test time (`strip_guidance`).
- Removed: the guidance and perturbed-practice stages, the clean pass (user: not even opt-in), the curiosity task
  source (`run_explore.py tasks`, cusi/explore/trajectories.py) and their Slurm pairs.
- Task prompts (tuned on samples of the random trees, all 3 envs): the first version listed one "Click X" task per
  link (50+ on arXiv's front page) and trivial GameBoy moves ("Move to the left side of the screen"), and knowledge
  cited set-of-mark indices. Now: tasks are goals, not button instructions; take several actions; name concrete
  targets and values; end in a visible state or a readable answer; cover what the screen offers rather than one per
  item. Examples come from other, invented environments (library website, weather app, farming game), since the
  only task lists we have are the benchmarks'. Edge tasks: the goal reached, or NO TASK if nothing or a one-action
  outcome; knowledge never uses numeric labels.
- Few-shot examples are per env (`notes.TASK_EXAMPLES`, node and edge prompts), in the style of each benchmark's
  tasks but still from invented environments. A scan of the test tasks showed the shared examples were off in tone:
  GameBoyWorlds tasks are 2-5 word single goals ("Catch Pidgey", "Read the sign below you"), the farming example a
  two-goal sentence (and compound GameBoy tasks fail most in practice); WebVoyager tasks stack concrete conditions
  and ask for the answer; AndroidWorld tasks name the app, give exact values and an answer format. The style, not
  the content, is borrowed from the test distribution. The random-tree practice runs started before this change.
- Known cost: a GameBoy practice try always runs its full 50 steps (no env success signal in free play; the
  dataset's safe-success cut keeps the data right).

## 2026-10-26: Compute accounting (cusi/utils/compute.py)

- Goal: a fair accounting of every procedure (search per explorer, annotation, practice) in GPU work, time per step
  and LLM calls; env simulation (CPU) is not accounted, only its wall time so the parts add up.
- Thread-local tallies: a search step (step.extra["compute"]), an expansion from its selection on
  (exp.stats["compute"]), each launch (compute.jsonl: run info, roots, process total), an annotated node
  (annotate_compute), a practice try and task (pickle "compute", results.csv columns).
- Recorded: wall seconds by part (select, restore, session, decide, env, embed, score, node, describe; agent, judge,
  critique; observe and record: the VLM explorer's executor update and done-check, and storing the step); local GPU
  work by section (embed, score, archive, policy_forward, ppo_update), timed with
  synchronisation and counted only when it allocated on the GPU, with the device type, and FLOPs from
  FlopCounterMode on the first call per (section, input shape); LLM calls (AgentVLM): latency, tokens, images and
  FLOPs from the served model's cached config and safetensors headers (2N per token, attention over each token's
  context with sliding windows capped, logits per generated token, vision tower per image); NVML on all the job's
  GPUs: energy and busy time (utilisation integrated by a sampler), which covers the vLLM server.
- LLM FLOPs assume no prefix cache (vLLM's prefix caching does less work); NVML busy time and energy are the
  measured side. Containers get --nv whenever the job has GPUs so NVML works inside (search.sh, practice.sh).
- NVML counts only the job's GPUs: scripts/utils.sh exports CUSI_ACCOUNT_GPUS (the job's UUIDs, from
  CUDA_VISIBLE_DEVICES in nvidia-smi order) before scripts narrow CUDA_VISIBLE_DEVICES; without it NVML counts
  every GPU it sees, which only equals the job's under Slurm's cgroup device isolation. run_info records the
  scope.
- run_search.py --budget_metric {wall_s,gpu_s,llm_flops,flops,gpu_busy_s,energy_j} --budget X stops at equal
  compute instead of equal steps (the random trees ran 400-6000 steps, the VLM / curiosity ones 400-600).
- python -m cusi.search.compute_summary --env E --run R sums a run.
- Removed with this (user): every curiosity and VLM search run, the random debug runs, and the trained SigLIP
  embedders and world models built on those trees (storage/explore/<env>/{embedder,world_model}).

## 2026-10-26: Curiosity shelved; fusion policy removed

- Curiosity exploration is shelved (kept in the code, not run): seed variance dominated every hyperparameter, GameBoy
  sessions plateaued within ~15k steps without leaving the start town, the best episodes were menu / dialogue spam
  (only menus that became nodes stop paying: archive_nodes_only, and the episode overlay resets each episode), and
  Android got ~15 PPO updates in 5 h. A best-of-seeds design was sketched (many short seeds per expansion, successive
  halving, episodes re-scored against the tree's node archive, top-k distinct committed; to be compared with
  best-of-N random rollouts under the same selection and budget) but not built.
- The fusion policy (frozen MobileNetV3-Small + MiniLM, a trained fusion transformer) is removed: it tied the CNN on
  Web and Android and was slower on GameBoy; the frozen-embedder heads were no better either. --curiosity_policy,
  --policy_vision_encoder and --policy_text_encoder are gone; the CNN (CNNObserver, CNNPolicy, formerly Pixel*) is the
  only policy, and the curiosity_parity job runs it against GameBoyRL's cleanrl ppo_curiosity (the same CNN).

## 2026-10-26: Task coverage eval (cusi/eval/task_coverage.py)

- Exploration quality as coverage of the benchmark's test tasks by the tasks proposed from an annotated tree. Each
  test task's k=5 nearest proposed tasks by cosine of Qwen3-Embedding-8B vectors (test tasks as queries with an
  instruction, proposals as documents; last-token pooling through transformers, no sentence-transformers), each
  pair scored 1-10 by an LLM judge. Metrics per test task: top1 cos, mean cos@k, judge max and mean@k; aggregates
  overall and per group (Web site, Android app, GameBoy init state) with coverage@8 (best match is the same task up to
  a detail) and coverage@5 (a good part of the way there). A judge sanity check on 20 test tasks: a paraphrase of
  itself (scores 9.85-10) and a random proposal (1.05-2.4).
- Judge rubric, tuned on the three existing random trees by reading samples per score band: the first anchors (7 =
  same goal, other target) let Gemma give 7 to the same kind of action on another site ("Uniqlo locations in Chicago"
  vs "list of Apple Retail Stores") and invent links ("Speak to the aide inside the gym" vs "heal at the Pokémon
  Center" -> 10). Now: judged in order (same website / app / place, same feature, same goal and target); 8-9 only
  for every part of A on another value; a different site / app / place caps at 3; B doing only some of a multi-part
  A, or vaguer than A about its target, caps at 7; "same feature" means the same screen, menu or tool; six examples
  from invented environments. Residual: about 1 in 10 of the 8-9 band is still generous (e.g. "Speak to Pikachu" vs
  "Listen to Squirtle's cry" -> 8).
- Test sets: Web all WebVoyager tasks minus web_excluded_sites (429); GameBoy the 49 pokemon_red tasks; Android one
  goal per template (instance 0, suite seed 30, as cusi.eval.android runs them).
- Output next to the proposals: <run>/task_coverage/<embedder>/ (embeddings cached per text, neighbours) and
  judge_<judge>/ (judgments appended per pair, so reruns and re-annotated trees only judge new pairs).
- Models: required on the CLI; the Slurm pair's defaults are scripts/utils.sh parameters (TASK_COVERAGE_DEFAULTS).
- The embedder runs in float16 on Turing GPUs (no bfloat16), so the embed stage can use the RTX 8000s; the judge
  (Gemma 4) needs Ampere.

## 2026-10-27: Android quick restore (snapshot cache)

- Android searches restored every node by replaying its whole path from the scene's reset (one emulator holds both
  scenes, so per-node states were lost at every scene switch). Replays cost ~15 s a step: 25-70 min per restore at
  depth 124-285 steps, 42-52% of search time, and some diverged. The six first saturation runs (sat_{random,vlm}_s0-2)
  were deleted and restarted after the fix.
- Now (restore stays "replay"; --state_cache N, default 24 on Android, 0 elsewhere): every new non-root node gets an
  emulator snapshot kept in an LRU cache of N; a restore loads the node's snapshot, or its nearest cached ancestor's
  and replays only the rest (from the scene reset only when none is cached), then caches the node. Nothing is written
  to the run: a resumed job or an emulator restart starts with an empty cache and falls back to replay.
- Scene switches no longer close the env: SceneEnvs suspends it (connection closed, snapshots kept) and resumes it
  from its scene snapshot; a scene whose snapshot is gone (the emulator restarted meanwhile) is rebuilt and drops its
  states. The search's restore skip also checks the env's device_epoch (bumped on resume and recovery).
- A snapshot is 3.6 GB (4 GB guest RAM, uncompressed) in the emulator's AVD copy under /tmp: save_state refuses below
  30 GB free there. AndroidPlayEnv save/load failures now raise RuntimeError (the cache drops the entry) instead of
  ending the run.
- Measured (live, 12 expansions, cache 3): save ~4.6 s per node, cached restore ~10 s, replay from an ancestor only
  for the missing segment; the emulator-restart path rebuilt the other scene as intended.

## 2026-10-27: LoRA fine-tuning on practice success rows (cusi/finetune, run_finetune.py)

- Data: the practice dataset's success rows (supervisor and executor calls), one row per model call. Each training
  sequence is the chat template's generation prompt (Gemma 4's ends in an empty thought channel) + response +
  `<turn|>`, loss on the response and `<turn|>` only. Checked against the server: for 18 real rows over all 9 call
  types, vLLM's prompt token ids (sent as the agent sends them) equal ours, image tokens included.
- Test-time match. Single-prompt calls (AgentVLM.infer: every supervisor call, the GameBoy and Android executors) send
  images as JPEG at PIL defaults; chat calls (the Web executor's action) as PNG. The practice dataset saved PNG, so
  training re-encodes as the call sent them (cusi.finetune.data.as_sent); dataset rows now record image_encoding.
  Template lines of every call type compared over all pokemon_red eval prompts vs all training rows: only content
  differs, plus the executor's no-change note, which came after the full_gemma-4-31b-it_r0 baseline (2026-10-19): a
  base-vs-LoRA comparison needs the base model re-evaluated with the current agent.
- First trainer (rejected): bf16 model split with device_map="auto" (one GPU busy at a time), one row per pass:
  0.21 rows/s on 4x A6000 (~59 h per epoch of 45k rows).
- Now FSDP2 (torchrun, one process per GPU), after torchtune / torchtitan: meta-device build, rank 0 reads the
  checkpoint and broadcasts into the shards (buffers in one keyed broadcast: per-buffer broadcasts deadlocked);
  every decoder / encoder layer an FSDP unit, bf16 params, fp32 gradient reduction and LoRA weights; non-reentrant
  activation checkpointing; fused AdamW; sharded DCP checkpoints (resumable), adapter gathered to rank 0 at the end.
- Loss only at response positions (CompletionLoss: the body, then lm_head and the logit softcap on the ~42 of ~830
  positions that carry loss, the aim of Liger's fused linear CE / torchtune's chunked CE): HF's full-position logits
  for an 8k-token micro-batch alone are 3-4 GB, which ran out of memory on an A5000. Checked against HF's own loss (model(..., labels=)) on 16-24 real micro-batches under FSDP: losses equal to <= 4e-8; a LoRA gradient differs by 1.0-1.4% (cosine >= 0.9999), the same as either path run twice (bf16 nondeterminism); 1-4.5 GB less peak memory per GPU at 2k tokens.
- Batching: a step is rows_per_step rows; within it rows are sorted by length and packed into micro-batches of at most
  max_batch_tokens padded tokens (a fixed rows-per-micro-batch put 16k tokens on one GPU and ran out of memory);
  every rank runs as many passes (largest micro-batches split, zero-weight dummy passes as a last resort: none on
  GameBoy). Loss is the token mean over the step (scaled by world / step tokens, as torchtune). Padding overhead
  7% (8 GPUs, 8192 tokens), 19% (4 GPUs, 16384).
- Measured (8x A5000, 32 rows/step, 8192 tokens): 0.7-1.4 rows/s (3.5-6x the first trainer), peak 18.6 GB of 24,
  no data wait; validation loss 0.443 -> 0.310 in 8 steps. Throughput needs rows_per_step near
  GPUs x max_batch_tokens / mean length (~850 tokens), else the per-pass weight all-gathers over PCIe dominate.
- Serving: eval.sh --lora serves the base with the adapter (vLLM --enable-lora); every agent role calls it. On 18
  rows 16 greedy outputs differ between base and adapter (the adapter is applied). 4x A5000 cannot serve Gemma-4-31B
  with LoRA (0.03 GB left for KV cache); 2x A6000 or 8x A5000 can.
- Hyperparameters (rank, alpha, lr, batch, epochs) are left to a sweep.

## 2026-10-28: LoRA memory fixes, the GameBoy sweep, the final run

- Memory on 8x A5000 (24 GB), found by the sweep's out-of-memory crashes:
  - Gemma4VisionPatchEmbedder builds its position embeddings as a one-hot (images x 2520 x 2 x 10240 int64,
    ~394 MiB per image) times the table: 10 GiB for a validation micro-batch. Patched to a row lookup
    (patch_model_classes; bitwise equal on CPU).
  - The CompletionLoss logits are computed in chunks of 256 positions, recomputed in backward (torchtune's chunked
    CE). This alone did not fix the next crash (the same 1.71 GiB allocation in backward).
  - The root FSDP unit (embeddings, tied lm_head) now reshards after forward: ~2.4 GB more room in backward, one
    more all-gather per pass. Together with --max_length 3072 (drops 0.7% of rows) this ran the sweep without a crash.
    The 1.71 GiB allocation matches a 32-head L x L fp32 attention matrix at ~3.7k tokens (an unchunked attention
    path; not checked which kernel is selected).
  - Rule of thumb on A5000: ~10 GB fixed + ~2.5 GB per 1k tokens of micro-batch, so max_batch_tokens 4096 (8192 ran
    out of memory). Peak 17.9 GB at 4096.
- Sweep (gameboy, tree_random_v1, 8000 rows, 125 steps of 64 rows, alpha = 2r, ~2 h each on 8x A5000, ~1.1 rows/s);
  final validation loss (untrained 0.414):

  | rank \ lr | 5e-5  | 1e-4  | 2e-4  |
  |-----------|-------|-------|-------|
  | 16        | 0.271 | 0.277 | 0.293 |
  | 64        | 0.288 | 0.304 | 0.345 |

  Lower lr and the lower rank win throughout; the best is at the grid's edge, so a lower lr was not ruled out.
- Final run: final_r16_lr5e-5 on all 45k success rows (698 steps), evaluated with the adapter (eval run
  lora_final_r16_lr5e-5) against the base model re-evaluated with the current agent (base_gemma-4-31b-it).

## 2026-10-28: Practice restores from a snapshot cache; Web Chrome restarts

- Android practice replayed every task's node from the scene reset for every try (~30-35 min per restore at depth;
  ~33 min per task, ~66 h left for 112 tasks). NodeRestorer now takes state_cache (run_practice.py --state_cache,
  default 24 on Android, 0 elsewhere; DEFAULT_STATE_CACHE moved to cusi/search/restore.py): a replayed node's
  emulator snapshot is kept per env (LRU), the node's other tasks and tries load it, and descendants replay only from
  their nearest cached ancestor. Measured on the live job: first restore of a node 2055 s, then 15-54 s.
- Web searches: a navigation that aborts a WebDriver command can leave the tab detached ("not attached to an active
  page", "inspected target navigated or closed"), and a hung renderer times out every command ("timed out receiving
  message from renderer"); the screenshot / labelling retries then all fail and the search crashed (sat_random_s0
  three times; resumes crashed on the same node). These now restart Chrome on the current URL, as for a dead browser.
  A page that hangs the renderer again after each restart costs ~13 min a restart (one expansion took 61 min with 4).
- Android searches peak at their 64 GB memory limit (all six; likely the snapshot files in page cache, charged to the
  job), and two emulators died with connection refused and failed reconnects. Their pending follow-up jobs were raised
  to 128 GB, taking the memory limit as the cause; wrong, see 2026-10-29.

## 2026-10-29: Task diversity eval; greedy test evals; Red results; Android emulator deaths

- Task diversity (cusi/eval/task_diversity.py, Slurm pair task_diversity.sh): how many distinct functionalities a
  tree's proposed tasks exercise (three near-identical tasks count once, three different ones three times), compared
  within an env only. Tasks are embedded (Qwen3-Embedding-8B, as documents on both sides); each task's 5 nearest by
  cosine are judged 1-10 by Gemma-4-31B for the same functionality whatever the values; while all of a task's judged
  neighbours score >= 8, its next 5 are judged too (up to 40). functions@t = a greedy maximal set of tasks no two of
  which were judged >= t; also the Vendi score of the cosine kernel (no judge) and rarefaction (both at fixed task
  counts, random subsets). Judging every pair was ruled out: ~74M pairs on a 12k-task Web tree (~120 days at the
  judge's ~7 pairs/s on 2x A6000); the neighbour pairs are ~4-5 x n.
- Tried first and rejected: connected groups over pairs >= 8 chained similar tasks (walk north ~ walk to the building
  ~ walk to the edge) into one group of 657 of 823 GameBoy tasks, and the count fell as tasks were added. With the 5
  nearest only, 45% of GameBoy tasks had all 5 judged the same functionality, leaving their other near-duplicates
  counted as distinct (309 -> 268 once deeper neighbours were judged). On Web the deeper rounds added 9k pairs to 48k.
- Judge checks: rewordings 10, the same feature on other values 9 ("Withdraw a healing item" / "Withdraw a POTION"),
  overlapping screens 6, another feature 3; paraphrases 10 on every run, random pairs 1.95-3.6 (one of 20 Web random
  pairs >= 8).
- GameBoy (6 saturation runs), at 200 tasks: functions@8 random 100.5 / 114.5 / 110.3, VLM 114.2 / 101.3 / 90.6;
  Vendi random 14.9, VLM 15.9 (means). No difference beyond seed spread, as with task coverage (coverage@8 random
  37%, VLM 38%). Web random_s1: 12196 tasks, functions@8 6718, 195 of 200 at fixed 200; 2h23 on 2x A6000.
- Test-set evals decode greedily (cusi.eval.supervision.DECODING; the eval client forces temperature 0 on every
  supervisor, executor and judge call; recorded as "decoding" in each eval run's config). Before, the GameBoy
  executor sampled at T=1 (top_k 64, top_p 0.95) and the supervisor at the model's defaults, and the Web executor at
  T=1. Practice and exploration still sample. eval.sh takes --game for GameBoyWorlds games other than pokemon_red.
- GameBoy pokemon_red, 49 tasks, adapter final_r16_lr5e-5 (all 45k success rows, val loss 0.278) vs the base, same
  agent: greedy 19 vs 15 (6 adapter-only, 2 base-only, exact McNemar p = 0.29); sampled 21 vs 16 (7 vs 2, p = 0.18).
  Greedy and sampled runs agree on 43 (adapter) and 44 (base) of 49 tasks: the 49 tasks, not decoding, limit the
  comparison. Gains are on battles, items and moves (an Ultra Ball, a not very effective attack, teaching Toxic,
  catching Pidgey) and long tasks the base ran out of steps on; only 2 of the 7 sampled gains had a near-identical
  practice task (judge >= 8). Losses: "Pick up the item" both times. The Red -> Crystal transfer (50 pokemon_crystal
  tasks, no Crystal practice) is pending: adapter 18/50.
- Android practice (tree_random_v3) with the restore cache: 115 of 239 tasks succeeded, 78% of the 148 whose node
  restored; 91 tasks lost to replays below 0.95 similarity (38%). Dataset: 3260 success train rows.
- Android search emulator deaths (4 in ~2 days over 6 runs, each recovered by the next chained job from the saved
  tree): the emulator stops answering (connection refused / device not found) during a restore or mid-step; a death
  in a restore is recovered by restarting the emulator, one mid-step ends the job. Not memory: the third came at the
  raised 128 GB limit, and every Android job's peak equals its limit, crashing or not (file cache from the 3.6 GB
  snapshots counts toward it; /tmp on the compute nodes is local xfs, not RAM), and no log shows an OOM kill. Cause
  unknown.

## 2026-10-29: Android emulator deaths recovered; emulator logs kept; restore cache 64; snapshot storage measured

- Emulator deaths (each ended a search job when it hit outside a restore): on lg07 the emulator's own log stops at
  the death with no error, netsim logs the emulator's connection dropping the same millisecond, the kernel log
  (covering the time) has no OOM kill or segfault, the emulator's crash database no new report, and no other job of
  ours shared the node. Cause still unknown.
- Now any guarded emulator call that fails with the emulator process gone raises EnvDied (an EnvStalled), so every
  stall handler recovers (restart the emulator, rebuild the scene): reset, step, observe, save/load, and resume on a
  scene switch (a first fix in step alone failed its live test: the kill landed in resume). A recovered step ends
  the expansion unrecorded (its frame is the scene's start, not the action's result) and forces the next restore; a
  replay that hit a recovery starts over once instead of flagging the node. Live test (5 expansions, emulator
  SIGKILLed after iteration 1 and 3): recovered both times (resume, reset), all 5 expansions done.
- scripts/android_emulator.sh: one log per start (emulator.log links the latest), a wrapper appends how the emulator
  ended (exit status or signal), our stops and restarts append a marker first (the log is opened for append: the
  emulator's own writes overwrote markers appended to a truncate-opened log), and stop copies the logs to
  results/logs/emulator/<job>_<node>/. A death with no marker before its end line was not ours.
- Replay divergence is independent of recovery: 287320 had 20 nodes flagged before its first recovery, 287333 33 in
  one job; Android replays diverge throughout.
- Restore cache: at 24 the random searches hit 21-50% (trees of 124-150 nodes) and a miss replayed 60-110 steps
  (12-33 min, up to 734 steps); VLM runs hit 61-67%, misses replaying 6-19 steps. Default now 64 (~230 GB). /tmp is
  426-456 GB on the nodes checked except lg06 (70 GB, saves already refused there at 24): pending Android jobs
  exclude lg06. Correction: /tmp was first checked on the login node only.
- Snapshot storage, measured on real snapshots: ram.bin 1.7-2.1 GB (3.8 GB just after a boot); zstd -1 alone 1.2-3x;
  zstd --patch-from the scene's base snapshot 536-638 MB (8-9 s to encode, 2.5 s to decode, byte-identical); from a
  node saved 24 s earlier, 15 MB. The NAS has 55 TB free (shared, no per-user quota found).

## 2026-10-29: Scenes kept on their website / app; scenes match the test sets; Android and Web exploration rerun

- Web exploration left its two scenes (arxiv, github): 86-95% of every saturation tree was off-site (Google policies,
  EU sites, Apple, Facebook help, sign-in flows), through footers and social links; the env only blocked chrome://
  pages and the VLM hint's "stay on site" was not enforced. Off-site the web never runs out, so no Web run saturated
  (random still 4-6 new nodes per expansion at the end, longest fruitless streak 2 of the 10 needed; VLM 4 of 50).
  Android likewise: 13-26% of random and 46-50% of VLM nodes were in the scene apps (Chrome, Google search, Play
  services, the launcher otherwise; random's navigate_home and taps on the launcher).
- Scenes now follow the test sets. Web: one per WebVoyager test site (10; scope = the site's domain and subdomains,
  path-scoped for Google Flights /travel/flights and Google Maps /maps under www.google.com). The 429 tasks start on
  their site and essentially stay there (subdomains: support.apple.com, blog.arxiv.org, ...; one Hugging Face task
  needs GitHub). Android: one per test app (20; seeded by one of the app's templates, started in the app); scope = the
  app's package, the apps it is paired with in the test set's multi-app templates (10 of 116 use two apps), shared
  system screens, and Settings' screens from other packages; Dialer and Clipper have no scene.
- Enforcement in free_play only (exploration, practice; test evals unscoped): after each step the env checks the URL
  / foreground package; a step that left is undone (Web: browser Back, else reload; Android: Back, else reopen the
  app) and returned invalid ("That action left X for another ..."). Random never picks off-site links (their hrefs,
  read in one call per page) or Android's home. The VLM explorer's hint says "Do not leave {site}" / "{app}".
- Checked: in_scope on 20 hand cases; on the old trees the scope would have kept 4.8% (random) / 13.2% (VLM) of nodes;
  Web random on apple/arxiv/google_maps, 9 expansions: 0 off-site nodes, 0 bounces; all 20 Android scenes build in
  their app (49-109 s) and a Home step is undone in each.
- Runs record "scoped": true; resumed runs without it keep unscoped scenes (specs.scene_for), practice follows its
  source tree's setting.
- All Android and Web exploration (saturation runs, tree_random_v1-v3, pipe_test) and the practice built on them
  were deleted, to rerun with the new scenes (GameBoy kept).

## 2026-10-30: Typing exploration (vlm_typing), forked runs, typable fields; GameBoy scenes = Red test states; reruns

- Every node records its text fields (node.typable, env.typable_fields): Web labelled <input> of a text type,
  <textarea>, contenteditable; Android is_editable elements; each with index, label, value and a signature of what may
  identify it across screens (Web: host, path, tag, type, id, name, placeholder, aria-label, label, autocomplete, role,
  form action; Android: package, resource name, class, hint, content description, tooltip). Live: Google Flights 4,
  Wolfram|Alpha 1, GitHub 1 (sign-up email), Apple 0 (search is a button), arXiv 0 at 1024 px (hidden), Booking 0
  under its pop-up; Android SMS search (resource id), other start screens none.
- run_search --from_run <run>: the run starts as another run's tree in its own directory (store.fork_run: jsonl logs
  and archive copied, frames/embeddings/states hard-linked; the source is never changed); config records from_run and
  from_expansions, and max_expansions, saturation and budgets count only the run's own expansions. An orchestrator
  interleaves expanders (random -> typing -> random) by forking each phase from the previous.
- --expander vlm_typing: the VLM executor with the env in action_mode "typing" (Web: Type, which presses Enter, and
  Scroll; Android: input_text, keyboard_enter, scroll, and hide_keyboard = Back while the keyboard is shown; anything
  else is invalid). It expands only nodes with a field whose identity (expanders.field_key: Web host, tag, type, name,
  placeholder, aria-label, label, form action, not the per-load id nor the page path; Android package, resource,
  class, hint, description) has been typed into fewer than --typing_field_cap (10) times in the scene; a typing action
  into a capped field is not performed; the run stops when no node has an open field. Web: a Type that leaves the URL
  unchanged makes no node (info no_node: a saved state would not hold the typed text).
- First test failures: the cap only gated selection and the count skipped the expansion in progress (Wolfram|Alpha's
  box typed 69 times at cap 2); scripts/slurm/search.sh only accepted --model with --expander vlm; the M3A executor
  rejected hide_keyboard while parsing. After the fixes (cap 2): Web each of 9 fields typed exactly twice, forks
  hard-linked, random forked again from the typing run. Android: forked nodes mostly failed to replay in the new job
  (7 of 10 replay_diverged), so typing there is limited to restorable nodes (reported as "no restorable node with a
  typable field left").
- Web env construction retries opening a scene in a new browser (Apple's start page timed out its first screenshot
  twice with 30 browsers on one node).
- Android emulator deaths, now explained by the per-start logs: the emulator segfaults (killed by signal 11, no
  marker of ours before it), 4 times in 14 h in one search; each was recovered. That search then ended when switching
  scenes after a restart: connecting failed (adb `shell whoami` exit 1, the emulator running but not yet ready), which
  was not taken as a death; a failed connection on resume is now handled as one (restart and rebuild).
- GameBoy scenes are now the 40 start states of the pokemon_red benchmark (49 tasks); the 2-scene GameBoy trees,
  practice and fine-tunes were moved to storage/archive/gameboy_2scenes_2026-10-30.
- Reruns, all scoped: random and VLM, 3 seeds each, for Web (10 scenes), Android (20), GameBoy (40); 20 steps per
  expansion for both expanders, saturation after 50 fruitless expansions, node threshold 0.5 (Web, Android) / 0.1
  (GameBoy), 69 h each (Web and GameBoy as 3 x 23 h chains; Android as one long-lg job per seed, one Android job per
  node, CPU-only random without --exclusive). GameBoy random saturated at 4600 / 6300 / 7198 nodes (7.5k-21.6k
  expansions, 1-4 h), against 250-355 nodes from the 2 old scenes.

## 2026-10-31: Web/Android reruns launched with --max_expansions 1000 (failure, fixed)

- The 2026-10-30 reruns were meant to stop only on saturation (patience 50) or the 69 h cap, but Web and Android were
  submitted with --max_expansions 1000 (GameBoy got 100000). All three Web random trees stopped at exactly 1000
  expansions, unsaturated (1895 / 1344 / 1824 nodes), after 17-24 h of their 69 h.
- Fix: max_expansions is an operational key, so the runs resume with a larger value. Web random s0-s2 resubmitted as
  chains with --max_expansions 100000 for the rest of their 69 h; the pending Web VLM and Android random s0 segments
  were replaced the same way. The running Android single jobs keep 1000, which they cannot reach (~8 expansions/h).
- A first resubmission failed in seconds: --extra was passed unquoted, so search.sh saw its flags as stray arguments.
  --extra must be one quoted string.
- Editing scripts/slurm/search.sh while jobs run it (the 2026-10-30 tidy) makes those jobs end with "syntax error:
  unexpected end of file" after run_search returns: bash reads the script as it goes. The search itself finished
  (summary.json written) but sacct shows FAILED. Don't edit a job script under running jobs.
- Android random s2 (289240) died after 20 h inside recovery: a snapshot load's UI tree never caught up, _recover
  restarted the emulator, and rebuilding the scene on the fresh emulator failed ("Could not get a11y tree": its
  accessibility service was not up yet). _recover now restarts up to RECOVER_ATTEMPTS = 3 times before giving up.
  Resumed as 289550 on lg01 for the remaining 49 h (the other running Android jobs still have the one-attempt code).

## 2026-11-01: Practice on the 40-scene GameBoy random trees: a seeded 750-task sample per tree

- The saturated trees give 14.6k / 20.1k / 22.9k practice tasks. Measured on the first ~70 per tree: ~160 Gemma-31B
  calls per task (1.9 tries x up to 50 steps x supervisor + executor, plus judge and critique), 149k prompt tokens,
  18 min median wall; one 2-A6000 vLLM server with 8 workers does ~15 tasks/h (prefill-bound, ~1.2k prompt tok/s,
  ~8 requests running, none waiting). A full pass would take 700-950 h per tree.
- --max_tasks takes the first tasks in node (discovery) order, biased toward shallow nodes, so it is not a sample.
  New: run_practice --sample_tasks N --sample_seed S attempts the first N of a seeded shuffle; samples are nested
  (raising N extends one), and a resume with a different seed (or sampling on/off) is refused. On s0, 750-1500 cover
  39-40 of the 40 scenes in proportion.
- Launched with --sample_tasks 750 --sample_seed 0 on sat_random_s0-s2 (~50 h each, 3 segments). The node-order
  attempts (~70 per tree) were deleted. My first estimate (1,500 in ~2 days) used an early progress-bar rate and was
  off by 2x.
- 2026-11-01: Android VLM s1 (289242) died the same way after 50.7 h (pre-fix code); resumed as 289699 on lg19 for
  the remaining 18.3 h with the retrying _recover.

## 2026-11-02: Cluster maintenance 2026-11-03 16:00 -> 11-04 04:00 UTC (all eclairlg nodes)

- Slurm starts no job whose time limit overlaps the window, so the queued 23 h segments would wait until Nov 4.
  Practice third segments shortened to fit (s0 8 h, s1 5.5 h, s2 5 h: each tree needs ~4-5 h more at ~15 tasks/h)
  with a 23 h backstop segment after the window (290134-290136); ann-web-random_s0 shortened to 12 h. Annotation,
  diversity and practice all resume where they stop, so a job killed at the window loses nothing.

## 2026-10-31: sync_data sets for trees, practice and fine-tunes; packed dirs; only finished runs pushed

- Sets: evals (eval/<env>/<run>[_s<i>]), trees (search/<env>/<run>, incl. annotations, task_coverage,
  task_diversity; viewer/ skipped unless --with_viewer), practice (practice/<env>/<model>/tree/<run>) and practice-ft
  (models/finetune/<env>/<run>; checkpoint/ skipped unless --with_checkpoints). No archive set. Push and pull are dry
  runs unless --no_dry_run.
- The Hub caps files per repo (~100k) and per folder (~10k): a 40-scene GameBoy tree is ~50k files and one practice
  dataset ~112k. Each set's many-file dirs go up as one uncompressed tar each (trees: the WRITE_ONCE_DIRS; practice:
  episodes, dataset; evals: episodes, trajectories and WebVoyager's task*/ dirs), rebuilt per push with fixed
  metadata and order so an unchanged dir uploads nothing new, and unpacked (tarfile filter="data") on pull. Round
  trip checked byte-identical on real trees, evals and practice runs.
- Only finished runs are pushed (skipped, or an error when named): trees need summary.json newer than
  expansions.jsonl and any started annotation / task_coverage / task_diversity finished; evals need every task of
  their config summarised, none failed to run, and every Android shard; practice needs dataset/, fine-tunes adapter/;
  and nothing may have been modified in the last 30 min. Unannotated trees may be pushed.
- Open: trees stopped by their Slurm time limit have no summary.json, so the gate refuses them; and pushing a run
  already on the Hub as loose files leaves those files behind.

## 2026-11-03: Concurrent annotation (--workers, default 8); cancelling practice to free GPUs backfired

- annotate_run made one VLM call at a time: its vLLM servers ran 1 request ~86% of the time (~200 prompt tok/s, vs
  ~1,200 for practice's 8 workers). Now each node's calls run on a worker thread and the main thread writes the
  updates one by one, merging task lists at write time (a node's own tasks keep the edge tasks its children added;
  an edge task joins its parent's current list). Tested on a de-annotated copy of android/sat_random_s2 (277 nodes,
  4 A5000s): 0 unannotated, 0 task-list inconsistencies, 980 vs 985 tasks from the serial run, ~650 nodes/h vs ~160.
- Failure: to focus compute on annotation I cancelled the three GameBoy practice jobs (629 / 569 / 538 of 750 kept),
  but the 12 A6000s freed (practice + the restarted serial annotations) went to another user's queued jobs within
  seconds (our fairshare is lower). Only lg10's A5000s were ours to use. Check the queue for other users' pending
  jobs before releasing GPUs.

## 2026-11-03: The VLM explorer typed; expanders renamed vlm_w_typing / vlm_no_typing / vlm_typing_only

- Failure: the VLM trees were described as "typing off", but `--expander vlm` ran the env with every action. Only the
  typing-only explorer changed the action set. Across the sat_vlm trees the VLM typed in 5.4-6.5% of steps on Web
  and Android, mostly its objective into search boxes. 17-30% of their nodes were reached by a typing step and 38-81%
  descend from one. Random never types (its samplers only click, scroll, go back or home); GameBoy has no typing. So
  random vs VLM on Web and Android compared no typing with some typing.
- Expanders renamed: vlm -> vlm_w_typing (Web, Android), vlm_typing -> vlm_typing_only (Web, Android), and a new
  vlm_no_typing (GameBoy: the same explorer, there being nothing to type; Web and Android: refused until a no-typing
  action mode exists). The GameBoy sat_vlm trees are vlm_no_typing.
- The 9 sat_vlm trees were migrated (config.json expander, each expansions row's expander and its steps' extra key,
  task_coverage / task_diversity run_dir; checked equal to a backup up to the renamed fields) and renamed
  sat_vlm_w_typing_s* (Web, Android) / sat_vlm_no_typing_s* (GameBoy), with their practice outputs. Practice was not
  cancelled: running jobs load the tree at start and then open files under its path, so symlinks under the old names
  keep them going; the queued segments (no GPUs held) were resubmitted under the new names. Remove the symlinks once
  the old-name jobs end.

## 2026-11-03: vlm_no_typing (Web, Android)

- Env action_mode "no_typing", with typing both refused and removed from the agent's prompt:
  - Web: Type is refused (MSG_NO_TYPING). WebVoyager's system prompt and the action menu drop the Type entry, its
    format line and the two typing guidelines (without_typing, anchored like without_google).
  - Android: input_text and keyboard_enter are refused. M3A's action prompt drops both actions and the guidance on
    entering or editing text (typing, deleting with backspace, pasting, auto-complete), and "input_text" from the
    index rule (m3a.without_typing, every cut checked at import).
- Android's on-screen keyboard keys are ordinary clickable elements (key_pos_*: letters, Space, Delete, Enter / Done
  / Search), so no_typing also refuses a click or long press on any element of the keyboard's package
  (com.google.android.inputmethod.latin). In the existing trees, 3.9-4.4% of random's Android clicks hit the
  keyboard (letters, Space, Delete: gibberish typing), and 3.5-4.4% of vlm_w_typing's (mostly Enter / Done / Search
  after input_text). So random on Android is not strictly typing-free either. Random on Web cannot type.
- Text-field contents are not a typing signal: random trees, which cannot type text, show fields "holding text"
  (Android placeholders reported as values; Web fields the site pre-fills, e.g. Google Flights' "Where from?").
- Live tests (Gemma 4 31B on a practice job's vLLM, via srun --overlap):
  - Web (Google Flights, Wolfram|Alpha; 2 expansions, 40 steps): no typing attempts, only clicks and scrolls.
    Format errors 5/40 vs 7.6% in the vlm_w_typing trees.
  - Android (Contacts, SMS; 4 expansions, 80 steps): no typing attempts and no keyboard taps.
  - A scripted check on the emulator (SMS search field open, keyboard shown): a key tap, a long press,
    input_text and keyboard_enter were all refused; navigate_back worked; the field kept its placeholder.
  - The first Web vlm_no_typing submission (291319-291327) left out `--gres=gpu:rtxa6000:2`, so vLLM got one A6000 and
    failed at startup; each afterany segment then failed in ~2 min. Resubmitted with the gres as 291829-291837.
    slurm/search.sh defaults to one GPU, so every VLM search must pass the gres.
  - Web vlm_no_typing s1 failed 3 times on lg16 (alongside web practice): apple.com's start page timed out on all 3
    browser attempts while building roots. curl from the node was fine. Resubmitted with --exclude=eclairlg16 (291844-291846).

## 2026-11-04: Fine-tunes on the saturation practice datasets (max_length 6144 on A6000)

- The 2026-10-28 settings (r16, alpha 32, lr 5e-5, 64 rows/step, 1 epoch) with --max_length 3072 dropped 44% of Web rows:
  Web rows run to a 2.8k-token median (99th percentile 6.5-7.4k), GameBoy's to ~0.8k. Now --max_length 6144
  --max_batch_tokens 6144 for every env (drops ~2% of Web rows, none on GameBoy). On 4x A6000: peak 39-40 GB of 48,
  ~300 s/step (0.2 rows/s, ~600 tokens/s), so a Web fine-tune (~110-120 steps) takes ~10 h.
- 3072 was an A5000 limit (8x 24 GB); a 4x A5000 node cannot hold Gemma-4-31B for training, and 2x A6000 was not tried.
- The old-name symlinks (storage/{search,practice}/<env>/.../sat_vlm_s<k>) must stay: practice datasets written before
  the rename hold absolute image paths under sat_vlm_s<k>. Removing them (2026-11-04 17:10) crashed the Web VLM s0/s1
  fine-tunes with FileNotFoundError; restored, and the two fine-tunes resubmitted (291960, 291961).

## 2026-11-04: Practice datasets store image paths relative to the dataset directory

- Rows held absolute image paths (<storage_dir>/practice/.../dataset/images/...), so a renamed run or a dataset pulled
  to another storage_dir could not be trained on (the Web VLM s0/s1 fine-tunes crashed when the old-name symlinks went).
- Now the dataset stage writes images/<name>.png, and cusi.finetune.data.read_rows resolves paths against the dataset
  directory: an absolute (older) path keeps its part after the last /dataset/. read_rows raises if any image is missing
  (one directory listing per images dir), so a broken dataset fails at startup.
- Migrated the 11 datasets (enumerated by hand: web random s0-2, web vlm_w_typing s0-2, android random s1-2, gameboy
  random s0-2; tmp_migrate_paths.py; originals in storage/tmp/dataset_paths_backup). Checks: a dry run asserted every
  row unchanged but for its image paths and every image present; before vs after on all rows, the new reader resolves
  to the same file (realpath of the directory + name) as the stored path; the processor's encodings (token ids, pixel
  values, masks) of 6 sampled rows per split are identical; absolute paths 682,043 -> 0; a rerun finds nothing to
  rewrite; android sat_random_s1 rebuilt from its episodes with the new code equals the migrated rows (images
  byte-identical). Running fine-tunes kept their in-memory rows and were not disturbed.
- The old-name symlinks (sat_vlm_s<k>) are removed (18, each checked as a symlink); the Web VLM s0/s1 datasets load
  without them. Android VLM s2's practice segment started before the change and will write absolute paths: read_rows
  handles them; rerun the migration on it once built.
- llm-utils submodule removed (no CUSI code, script or job used it; its env was never created).
- Correction: Android VLM s2's dataset (built 2026-11-05 01:26) already had relative paths (the dataset stage ran with
  the new code); a dry run of tmp_migrate_paths.py on it found nothing to rewrite and every image present.

## 2026-11-05: Search segments starved between 23 h segments; don't pin a chained search to one node

- The chained vlm_no_typing search segments (priority 87, 2x A6000) lost every freed A6000 to another user's 1-GPU
  arrays (fairshare priority ~264) and to our own practice / fine-tune jobs (92). Web s0, s1, s2 and Android s1 sat
  queued after their first segments ended (01:56-02:33); Web s0 resumed on lg06 at 02:41.
- Android s1's later segments were pinned to lg17 (--nodelist, to keep one emulator per node), and a 4-GPU Web fine-tune
  took lg17 at 02:42, which would have blocked the search for ~9 h. Changed both pending segments to
  ExcNodeList=lg02,lg05,lg06,lg10,lg18 (the nodes that run or will run an emulator) instead of a single node.

## 2026-11-05: Test-set evals of the saturation-practice adapters

- Submitted (tmp_eval_adapters.sh) evals of the 8 finished final_r16_lr5e-5 adapters: Web random s0-2 and
  vlm_w_typing s0-2 (eval runs lora_<source>), Android random s1-2. With them, greedy base evals
  base_gemma-4-31b-it on Web and Android: the existing full_gemma-4-31b-it_r0 runs predate greedy decoding and the
  current executor, so they are not a fair baseline (GameBoy already has base_gemma-4-31b-it_greedy).
- Web: 2x A6000, 6 browsers, Gemma-4-31B judge on the same server, lg16 excluded (apple.com timed out there for the
  Web s1 search). Android: exclusive 4x A6000 node, 3 emulators, as the earlier Android evals.
- Android vlm_w_typing s0-2 evals queued behind their fine-tunes (afterok; tmp_eval_android_vlm.sh).
- Deleted the old Web and Android base runs full_gemma-4-31b-it_r0 (Android with its shards _s0-_s2; ~1.6 GB) on
  request: superseded by base_gemma-4-31b-it (greedy, current agent). Their scores remain in
  plans/archive/results.md. Then also GameBoy's full_gemma-4-31b-it_r0 and Android's wait1_gemma-4-31b-it_r0 (with
  shards; ~0.6 GB).
- Android random s0's practice finished 2026-11-05 22:47 (4600 success train rows, 21 of 250 unrestorable): fine-tune
  293004 and its eval lora_sat_random_s0 (afterok) queued.

### 2026-11-06: Web test-set results of the saturation-practice adapters (final_r16_lr5e-5)
WebVoyager (429 tasks, greedy, Gemma-4-31B judge). Base 62.7%. Random s0/s1/s2: 64.1 / 62.5 / 61.5% (mean 62.7%).
vlm_w_typing s0/s1/s2: 62.5 / 62.5 / 65.7% (mean 63.6%). Fine-tuning on practice successes does not move Web success
beyond seed noise (spread across seeds ~3-4 points, i.e. ~15 tasks); VLM vs random differs by under 1 point on average.
Android base (AndroidWorld, 116 tasks in 3 emulator shards, greedy): 80.5 / 115 scored = 70.0% (1 task failed to run).
GameBoy vlm_no_typing s1 practice done (291308 + 291309): 750 tasks, 0 unrestorable, 113588 success train rows; paths
relative, all images present. Fine-tune 293008-293010 (3 segments).
Android adapters so far (of 115 scored): random s1 81.0 (70.4%), random s2 77.5 (67.4%), vlm_w_typing s2 74.0 (64.3%).

### 2026-11-06: Why the saturation-practice fine-tunes don't move test success (diagnosis, in progress)
- Eval noise: any two Web runs (base or adapter) disagree on 17-21% of the 429 tasks, adapter-vs-adapter as often as
  base-vs-adapter; 195 of 429 tasks are solved by some of the 7 runs and not others. Android: 9-13% of 116. The SE of a
  difference between two runs is ~2 points (Web) / ~3 (Android). Base repeats queued to measure the floor directly
  (tmp_eval_repeat.sh: web base_gemma-4-31b-it_rep1/rep2 293180/293181, android rep1 293182).
- The adapters do change the policy: on Android, adapter and base differ at the first call (the plan) in 89 of 116 tasks.
- Templates match: Android action / plan / summary prompts, train vs eval, no template line on one side only.
- Web, split by whether practice visited the site (practice scenes: Apple, ArXiv, Booking, ESPN, GitHub, Wolfram
  Alpha; next to nothing on Coursera, Google Flights, Google Map, Huggingface):
  practiced 261 tasks: base 62.1%, all six adapters >= base, mean 64.5% (+2.4; GitHub 30 -> 31-36 of 41);
  unpracticed 168 tasks: base 63.7%, all six <= base, mean 61.0% (-2.7; Google Map 40 -> 34-38 of 41).
  Gains where practice went, losses where it did not: the net is zero. Pending: whether the base repeats keep this split.
- Training is small: one epoch; Android 27-51 steps of 64 rows (84-~200 train tasks, 27 of random s1's 84 in Settings),
  15-36% of Android rows dropped at max_length 6144 (the longest, i.e. late steps of long tasks); Web 106-120 steps.
  The logs have no untrained validation loss: probes with --lr 0 --max_rows 64 (run probe_lr0, tmp_ft_probe.sh,
  293175-293179) give the base model's loss on the same 200 validation rows.

### 2026-11-06: GameBoyRL's world model, and a SigLIP world model on the random Red trees
- GameBoyRL's world-model benchmark arm (results/benchmark/<game>/subgoal_world_model_*.csv, Sep 13-20) solved 196 of
  498 tasks over 10 games vs 203 for the same agent without it. Its decisions (report.pkl.gz per episode) hold the
  predicted frame per action and the real next frame. Pokemon Red, 4939 steps: decoded prediction error 63.2 grey
  levels (0-255) where the screen changed vs 23.1 for copying the current frame (the copy wins 96%), 59.4 vs 0.0
  where it did not; the decoder's own round trip of the real frame 30.6; predictions for different actions differ by
  11.2. The predictions are a tiled texture (overworld) or a blurred battle screen, the same for every action: the
  VLM chose among near-identical noise.
- The Oct 21 SigLIP fine-tune (1540 frames of the old 2-scene trees, val pixel MAE 3.9; its world model cos 0.85 vs
  copy 0.55 on 140 held-out transitions) was deleted on 2026-10-26, so it is retrained.
- Changes (cusi/explore): one held-out split for everything trained on a replay (replay.is_validation_episode: hash
  of run dir, episode, seed; 10%), used by the SigLIP fine-tune and the world model (before, the fine-tune split
  frames at random and the world model drew its episodes separately, so the embedder could train on the world
  model's validation frames). The SigLIP fine-tune samples its frames across every training episode
  (sample_canvases; before, the first 5000 in file order, capped in run_explore) and validates on up to 512
  held-out frames every 250 steps (before, 16 frames). Re-embedding skips the search run's own embedder pass
  (iter_replay embed=False) and batches frames across episodes. wm_eval embeds only the held-out episodes (before,
  every frame, twice) and adds GameBoyRL's pixel measures (prediction / copy / decoder round trip, split on screen
  change; spread across actions).
- Runs (tmp_wm_siglip.sh, Quadro RTX 8000): smoke chain 293295-293297; fine-tune siglip_red_random (siglip2-base,
  50k frames of sat_random_s0-s2, 15000 steps / 100 min) 293298; residual world models red_random_s0 (150k
  transitions) 293299 and red_random_all (890k) 293301, 100 epochs; wm_eval 293300, 293302.
- Fine-tune diagnosis, probe 1 (293175): the untrained model's loss on Web random s0's 190 validation rows is 0.2790;
  the adapter's on the same rows 0.2969. One epoch of training made the held-out practice rows (validation tasks of
  the same practice run) less likely, not more.
- Probe 2 (293176): Web vlm_w_typing s0, untrained 0.2636 vs adapter 0.2805 on the same 199 validation rows: worse
  again, by the same ~6%.
- GameBoy base, current agent, greedy (293168): 20/49 (40.8%). The Oct 29 greedy base (base_gemma-4-31b-it_greedy)
  had 15/49; the two disagree on 9 of 49 tasks (18%), the same model and decoding: GameBoy's run-to-run noise floor
  is as large as Web's. Any GameBoy adapter difference under ~5 tasks is noise.
- SigLIP fine-tune siglip_red_random (293298): 15000 steps, 62 min, 50k frames; held-out decoder round trip 10.7 grey
  levels (train 0.053 / val 0.059 loss). World model red_random_s0 (293299, residual, 100 epochs, 150k transitions;
  eval 293300 on 14540 held-out transitions): embedding cos 0.879 vs copy 0.770 (screen changed: 0.850 vs 0.707).
  Decoded pixels, screen changed (1589 sampled): predicted 22.4, copy 33.6, decoder floor 10.2, prediction beats copy
  on 56%; unchanged (411): predicted 7.2 vs floor 5.6; predictions for different actions differ by 33.6 (GameBoyRL:
  63.2 / 23.1 / 30.6, beats copy 4%, spread 11.2). The grid shows map scrolls, battle text and the START menu
  predicted for the action taken, and different screens for other actions.
- Android vlm_w_typing s1 (292905): 76.0 / 114 scored = 66.7% (2 failed to run); base 70.0%.
- GameBoy random s1 adapter (293170): 23/49 (46.9%) vs base 20/49; 9 tasks differ (6 adapter-only, 3 base-only), the
  same count as base vs base repeat.
- Probe 3 (293177): Android random s1, untrained 0.1134 vs adapter 0.0886 on the same validation rows: here training
  did help the held-out practice tasks (-22% loss), yet the test eval is flat (70.4% vs base 70.0%). Android's
  adapters learned their practice distribution; it does not reach AndroidWorld's tasks.
- GameBoy random s0 adapter (293169): 19/49 (38.8%) vs base 20/49; with s1's 23/49 the mean is at the base.
- Probe 4 (293178): Android vlm_w_typing s0, untrained 0.1006 vs adapter 0.0804 (-20%), as Android random.
- Probe 5 (293179): GameBoy random s0, untrained 0.4690 vs adapter 0.3052 (-35%).
- Probes, summary: Web adapters are worse than the untrained model on held-out practice rows (+6%, both sources);
  Android (-20 to -22%) and GameBoy (-35%) adapters are better on them. Yet none moves its test eval. So on Android
  and GameBoy the adapters learn the practice distribution and it does not transfer to the test tasks; on Web the
  training does not generalise even within practice (Web has ~110 steps over 7-8k rows of a few sites).

- All-seeds SigLIP WM (293301 train, 293302 eval; red_random_all, 3 random seeds, held-out episodes): cos 0.901 vs copy 0.773 (s0: 0.879); changed-screen pixel error 19.2 vs copy 31.9, decoder floor 11.0, beats copy on 59% (s0: 22.4 / 33.6 / 10.2 / 56%); spread across actions 39.1 (s0 33.6). More data helps on every metric. Grid: menus open and close correctly per action; outdoor scenes blur; screen transitions (black bars, fades) are missed.
- Android vlm_no_typing s2 search (291330) ended at its time limit 23:39 UTC with a 197-node tree (final segment, as planned).
- Android vlm_no_typing s0 search (291328) ended at its time limit 23:45 UTC (final segment, as planned).
- Android base repeat (293182, base_gemma-4-31b-it_rep1): 73.5/114 scored = 64.5% vs the original base 80.5/115 = 70.0%; 11 of 116 tasks flip (9.5%). Same model, 5.5 pts apart: Android adapter differences of that size (e.g. vlm_w_typing s1 66.7%) are within run-to-run noise.

### 2026-11-07: world models on four more GameBoy games (Links Awakening, Deja Vu 1, Sword of Hope 1, Bomberman Quest)
- run_search --game: a GameBoyWorlds game other than pokemon_red, its benchmark start states as scenes (specs.gameboy_spec),
  with no on-screen text (GameBoyPlayEnv read_text=False; glyph tables exist but are only checked for Red).
- First searches (293311-293331, Red's settings, saturation_patience 50) stopped "saturated" within seconds: 316 / 50 /
  55 / 71 expansions for s0 of the four games vs Red s0's 7509. Without text the novelty score finds few new nodes, so
  50 empty expansions come at once. Their fine-tunes / world models were cancelled. Rerun with saturation off and a
  fixed 7500 expansions per seed (Red s0's size, ~150k transitions): searches 293336-293356, fine-tunes 293339 /
  293345 / 293351 / 293357, world models 293340 / 293346 / 293352 / 293358, evals 293341 / 293347 / 293353 / 293359
  (tmp_wm_games.sh).
- wm_eval adds PSNR, SSIM and LPIPS (alex; lpips 0.1.4 added to setup/) for the prediction, copy and decoder round
  trip; Red rerun 293335.
- 2026-11-07: four-game reruns cancelled at the user's request (focus on Red); no results.
- Red all-seeds WM with PSNR / SSIM / LPIPS (293335, same 2000 sampled transitions as 293302: MAE identical). Screen changed (1572), predicted / copy / decoder floor: PSNR 17.55 / 14.63 / 20.21 dB, SSIM 0.742 / 0.657 / 0.813, LPIPS 0.228 / 0.274 / 0.172 (gap to the floor closed: 52% / 55% / 45%; MAE 61%). Unchanged (428): PSNR 21.43 vs floor 21.87, SSIM 0.888 vs 0.900, LPIPS 0.136 vs 0.101 (copy exact).
- GameBoy random s2 adapter (293171): 22/49. Random adapters s0/s1/s2: 19/23/22 (mean 21.3) vs base 20/49 (base repeat 15/49): within the noise floor; s2 vs base 10 flips.
- Red WM on Pokemon Crystal (unseen game): random search sat_random_crystal_s0 (293385, --game pokemon_crystal, 1500 expansions) -> wm_eval red_random_all --run_names search:sat_random_crystal_s0 (293386). wm_eval on another replay now writes eval_<replay>.json / predictions_<replay>.png beside the model's own.
- Red WM on Crystal (293386, 30000 transitions of 1500 expansions, none seen in training): does not transfer. Embedding cos (changed) 0.546 vs copy 0.695. Pixels, changed (1506): MAE 68.8 vs copy 33.4, floor 44.8; PSNR 8.80 / 15.77 / 11.86; SSIM 0.237 / 0.648 / 0.318; LPIPS 0.660 / 0.327 / 0.597; beats copy 15%. Even the decoder's round trip is worse than copying: Red's fine-tuned SigLIP + decoder do not represent Crystal's tiles, and the predictions drift to Red screens (Red's battle sprite, Red's START menu).
- Android vlm_w_typing s0 (293310): 78.0/115 = 67.8% (s1 66.7%) vs base 70.0% / base repeat 64.5%: within noise. Zelda s0 search (293361) slowest of the four (5531/7500 at 40 min).
- Web base repeat 1 (293180): 251/429 = 58.5% vs original base 269/429 = 62.7%; 110 of 429 tasks flip (25.6%). Same model, 4.2 pts apart: Web's run-to-run noise is larger than the ~18% flips estimated from adapter pairs.
- Sword of Hope 1 WM (293378; 43340 held-out transitions, 32081 changed): changed-screen cos 0.986 vs copy 0.759; MAE 0.64 / copy 7.29 / floor 0.51; PSNR 31.6 / 23.2 / 32.4; SSIM 0.989 / 0.900 / 0.991; LPIPS 0.007 / 0.128 / 0.005 (91-98% of the gap closed); beats copy 66%. A menu game: changes are small (copy MAE 7.3 vs Red 31.9), cursor and dialogue updates, which the model predicts almost to the decoder floor.
- Web vlm_no_typing s0 search (291831) ended at its time limit 05:37 UTC, final segment as planned: 887 expansions, 1529-node tree.
- Deja Vu 1 WM (293372; 45720 held-out, 35791 changed): cos 0.987 vs copy 0.871; MAE 2.32 / copy 10.69 / floor 1.40; PSNR 27.28 / 23.05 / 27.94; SSIM 0.979 / 0.901 / 0.988; LPIPS 0.022 / 0.120 / 0.013 (86-91% of the gap); beats copy on only 27%: the mean is carried by large changes, while on small ones copying is closer.
- Bomberman Quest WM (293384; 43320 held-out, 32880 changed): cos 0.938 vs copy 0.609; MAE 5.17 / 22.23 / 1.55; PSNR 29.87 / 20.37 / 31.52; SSIM 0.945 / 0.782 / 0.987; LPIPS 0.070 / 0.213 / 0.032 (79-85%); beats copy 40%. Grid: START opens the menu, the map and menu are right, the player sprite blurs away.
- Android random s0 fine-tune failed again (293291): NCCL 'unhandled cuda error' in the step-50 checkpoint save, on eclairlg05, the same step and node as 293004. Resubmitted excluding lg05: fine-tune 293395 (from scratch, ~6.5 h; no checkpoint was ever written) -> eval 293396.
- Link's Awakening WM (293366; 45760 held-out, 42757 changed): cos 0.965 vs copy 0.580; MAE 4.64 / 25.63 / 2.77; PSNR 27.30 / 17.56 / 28.35; SSIM 0.937 / 0.738 / 0.963; LPIPS 0.071 / 0.230 / 0.050 (88-92% of the gap); beats copy 65%. Grid: inventory fades and title screen right; a screen-edge scroll to the next room is missed (predicts the old room), a fade-in comes out blurred.
- Web base repeat 2 (293181): 250/429 = 58.3%. The three base runs: 62.7 / 58.5 / 58.3%; pairwise flips 25.6 / 21.7 / 17.0% (mean ~21%, SE of a difference ~2.8 pts). The two repeats (run a day after the original) agree; the original sits ~4 pts above both: live sites drift, so a Web comparison should use runs made close together.
- Android random s0 fine-tune, third failure (293395, eclairlg06): the same NCCL "unhandled cuda error" in the dcp gather of the step-50 checkpoint save, so it is not node lg05. The code is unchanged since 292127, which saved at step 50 with a 43 GB peak, so memory headroom is not clearly the cause (and max_batch_tokens cannot go below max_length 6144). Resubmitted as 293462 (eval 293463) with save_every 5 (the save path is tested at step 5, and a later crash resumes from the last save) and NCCL_DEBUG=INFO.
- vlm_no_typing searches all done (final segments ended at their time limit 13:05 UTC, as planned): Web s0 / s1 / s2 887 / 886 / 872 expansions (1529 / 1680 / 1523 nodes); Android s0 / s1 / s2 264 / 257 / 288 (341 / 352 / 338 nodes). No practice or fine-tunes run on them yet.
- Android random s0 retry 293462: the step-5 checkpoint save succeeded, so dcp saving works on this run; the failure appears only near step 50.
- GameBoy vlm_no_typing s0 fine-tune done (292922, resumed run): 1652 steps, final validation loss 0.2695. Stale follow-on 292923 cancelled; eval 293172 started.
- wm_eval gains pixel_shifted: the changed transitions that are scrolls (a vertical or horizontal shift of the current frame, up to 1/8 of it, matches the next with under 1/4 of the unshifted error; checked on real Red frames: synthetic shifts caught, an unrelated frame not; 162 / 617 changed). Rerun on the five world models (293546-293550; other buckets reproduce). Shifted transitions: Red 393 / 1572 changed, Link's Awakening 13 / 1949, Deja Vu / Sword of Hope / Bomberman Quest 0 (no scrolling in these replays).
  - Red, shifted (393): MAE 28.6 / copy 44.5 / floor 18.2 (60% of the gap); PSNR 13.93 / 10.26 / 17.26 (52%); SSIM 0.584 / 0.426 / 0.689 (60%); LPIPS 0.340 / 0.259 / 0.286 (worse than copy, and the floor is too: LPIPS tolerates a shift but not the decoder's blur); beats copy (MAE) on 89%.
  - Link's Awakening, shifted (13, too few to lean on): MAE 18.7 / 55.6 / 6.1 (74%); PSNR 21.08 / 11.87 / 23.99; SSIM 0.787 / 0.257 / 0.938; LPIPS 0.091 / 0.224 / 0.032; beats copy 85%.
- Sequel transfer evals, as Red -> Crystal (screen-changed transitions): Link's Awakening -> Oracle of Seasons (search 293551 -> eval 293552), Deja Vu 1 -> 2 (293553 -> 293554), Sword of Hope 1 -> 2 (293555 -> 293556), Bomberman Quest -> Bomberman Max (293557 -> 293558); 1500-expansion random searches, tmp_wm_sequels.sh.
- Sequel transfer (screen-changed; pred / copy / floor; cos pred vs copy): none transfer, as Red -> Crystal. Each sequel's decoder round trip is itself worse than copying, so the trained game's SigLIP + decoder cannot represent the sequel's screens and the dynamics are not tested.
  - Link's Awakening -> Oracle of Seasons (1714 px): cos 0.333 vs 0.744; MAE 89.97 / 32.93 / 62.79; PSNR 7.48 / 15.76 / 15.84; SSIM 0.122 / 0.690 / 0.297; LPIPS 0.741 / 0.304 / 0.576; beats copy 11% (trained game: 0.965 cos, MAE 4.64, 65%).
  - Deja Vu 1 -> 2 (1684): cos 0.560 vs 0.871; MAE 53.63 / 11.03 / 32.64; PSNR 10.10 / 22.78 / 12.66; SSIM 0.467 / 0.898 / 0.593; LPIPS 0.452 / 0.120 / 0.267; beats 8%.
  - Sword of Hope 1 -> 2 (1419): cos 0.240 vs 0.802; MAE 95.05 / 9.48 / 98.67; PSNR 4.83 / 20.74 / 5.07; SSIM 0.207 / 0.893 / 0.179; LPIPS 0.580 / 0.174 / 0.559; beats 2%.
  - Bomberman Quest -> Max (1602): cos 0.123 vs 0.744; MAE 105.76 / 35.62 / 78.45; PSNR 6.24 / 15.36 / 11.00; SSIM 0.046 / 0.605 / 0.163; LPIPS 0.775 / 0.269 / 0.564; beats 2%.
- GameBoy vlm_no_typing s0 adapter (293172): 17/49 (34.7%) vs base 20/49 (repeat 15/49); 11 tasks differ (4 adapter-only, 7 base-only). Random adapters s0/s1/s2: 19/23/22. Within the noise floor; s1/s2 vlm evals pending.

## 2026-11-07: why practice adapters do not help (diagnosis)

No practice adapter beats its base beyond the noise (GameBoy 17-23/49 vs base 20; Android 64-70% vs 70.0 / 64.5; Web
61.5-65.7% vs 62.7 on the same day). The practice -> test chain, step by step: (1) proposal (practice tasks like the
test's?), (2) labels (judge successes real? new behaviour in them?), (3) learning (does the policy change on test
screens?), (4) measurement (can the eval see a change?).
- Plan: practice on the GameBoy test set itself (attempt_test: the 49 pokemon_red tasks from their benchmark start
  states, run as the eval runs them, 175 steps; judged with hint retries as practice; each try also records the
  benchmark's success), 6 repeats (293568-293573; a first submission 293560-5 died on a command-name typo). From it: judge precision / recall against the benchmark, per-task
  success rates, and three oracle adapters with the tree adapters' fine-tune: judge labels on all 49 tasks (proposal
  taken out), benchmark labels on all 49 (proposal and judge out), benchmark labels on half the tasks evaluated on the
  other half (in-distribution generalisation). Two more greedy base evals (293566-7) for the noise floor.
  New: cusi/practice/stages/attempt_test.py, run_practice.py attempt_test and dataset --labels gt, practice.sh
  --command / --extra.
- Early findings from the existing runs:
  - Measurement: of the 49 GameBoy test tasks, 19 were solved by none of 8 runs (base, base greedy rerun, 6
    adapters) and 12 by all 8: only 18 can move. Evals are greedy, and the base still swings 20 vs 15 between runs.
  - Proposal: practice tasks are micro-steps ("Move one tile down after finishing the conversation", "Finish reading
    the current dialogue"), test tasks are goals ("Defeat Gym Leader Brock", "Sell exactly 1 Psychic TM"). Practice
    succeeds on 62-75% of its tasks at the first or a hinted try; the base solves 41% of test tasks.
  - Practice episodes never end themselves: 182 of 183 judged-successful tries (random s0, first 300 tasks) ran to
    the 50-step budget; the dataset keeps the calls up to the judge's safe point + 2 (median 41% of the try).
  - Training rows (GameBoy random s0, 22617 originals, x4 with paraphrases): action decisions 35%, frame_diff 34%,
    done_check 15%, supervisor calls 16%.
  - Learning: the adapters do change test behaviour. Greedy from the same start, a base rerun shares a median 10-action
    prefix with the base, the adapters 3-4 (first action the same on 80-90% of tasks vs 98%); the action mix barely
    moves (TV 0.04-0.07, as the base rerun's 0.06).

- 2026-11-07 (diagnosis, cont.): interim test-set practice at 129/294 pkls: judge vs benchmark over all tries (judge,gt)
  TT 76, TF 19, FT 10, FF 119 -> judge precision 80%, recall 88% (worse than the 92/97% at 77 pkls; ~1 in 5 judged
  successes is false). First-try benchmark success 64/129 (biased up: quick tasks finish first). Added repeats r6-r10
  (293575-293578, 293581) and greedy base reps 3-4 (293579-293580) on free GPUs. Oracle pipeline (tmp_oracle.sh) fires
  when r0-r5 finish: datasets testset_judge / testset_gt / testset_gtA (half A = seeded 25 of 49 tasks,
  tree/testset_halves.txt) -> fine-tunes with the tree adapters' recipe -> 2 greedy evals each on all 49.
  GameBoy vlm_no_typing s2 fine-tune 293002 completed; stale follow-on 293003 cancelled, eval 293174 unblocked.

- 2026-11-08 (diagnosis): test-set practice r0-r5 done (294 task-repeats, 554 tries). Judge vs benchmark success:
  all tries precision 0.71 / recall 0.91 (TT 137, TF 57, FT 13, FF 347); first tries 0.74 / 0.95; hinted tries are
  worse: try 2 precision 0.62, try 3 0.57 / recall 0.57. So the judge's false successes concentrate on exactly the
  hinted tries that are practice's only new signal. Hints rescue 29 of 179 first-try benchmark failures (16%).
  Sampled first-try success 0.39 per task = greedy mean 0.39 (5 greedy base runs: 15, 19, 21, 19, 22 of 49 -> the
  noise band all adapters, 17-23, sit in). Per task: 15 never solved by greedy or 6x3 practice tries; 6 tasks
  (17, 19, 23, 29, 31, 36) are never solved greedily but practice solves them sometimes -> the only room for new signal.

- 2026-11-08 (diagnosis): GameBoy adapters vs 5 greedy base runs, per task. Adapter - base success rate +0.024,
  permutation p = 0.57. Success agreement adapter-base 0.845 = base-base 0.849; 5-action openings identical
  base-base 0.82 but adapter-adapter 0.46 / adapter-base 0.48: an adapter is statistically a re-roll of the base
  (trajectories change, outcomes do not). By group: base-always 12/12 kept, never-solved 15 stay 0, base-mixed 16:
  adapters 35 vs base 36 (per-run means), practice-new tasks (17,19,23,29,31,36; greedy 0/5 each, sampled first try
  0.23): adapters 7/30 = 0.23 = the base's sampled rate (task 19: greedy 0/5, sampled 9/11, adapters 4/5; gains are
  offset by losses, e.g. task 3 greedy 4/5 -> adapters 0/5, all five stuck in a RIGHT/LEFT oscillation). Not
  scene-habit replay: adapter first action matches the practice scene's modal first action 0.19 = base 0.19.
  No generic shift (output tokens/call, calls/decision, legs unchanged). Training does fit practice: val loss
  0.41-0.47 -> 0.27-0.31 on held-out practice tasks. Only ~35% of rows are executor action calls (rest
  frame_diff, done_check, supervisor). Next: per-token NLL of base vs tree adapters on test-task success vs failure
  rows (tmp_token_loss.py, job 293666).

- 2026-11-08 (diagnosis, token loss; tmp_token_loss.py, job 293666): base vs tree adapters (sat_vlm_no_typing_s2,
  sat_random_s1) on 400 rows of test-task trajectories (testset_gt: 120 action + 80 other rows from benchmark-success
  tries, same from failures). Mean per-row NLL base 0.415 -> adapter 0.281, but without each row's last token
  0.312 -> 0.288: ~82% of the loss the adapters remove is the end-of-turn token. Cause: prompts make the model end
  with "[STOP]", inference cuts at [STOP] and strips (lm_inference.py), the dataset stores the cut response, and
  RowEncoder appends <turn|>: the base puts ~9 nats on <turn|> right after "Action: X" (5.9 after done_check), the
  adapters ~0. Harmless at inference (the turn ends either way) but it is most of what fine-tuning learns and why
  val loss falls 0.43 -> 0.29. On content: action tokens given the reasoning are already ~0.01 nats under the base;
  reasoning tokens drop only 0.006-0.011 on success rows and 0.036-0.039 on failure rows (adapters fit failed
  test trajectories better than successful ones); every action row's taken action gets likelier, success and
  failure alike. Conclusion: self-generated practice data has almost nothing for SFT to learn beyond formatting,
  and what little it learns is not selective for success. Fix to make later: keep "[STOP]" (the text the model
  actually produced) in dataset responses.

- 2026-11-09 (oracle results): testset_gtA (fine-tuned on benchmark-success tries of half A of the test tasks, from
  their own start states, r0-r5; 11 h) scores 16 and 18 of 49: half A (trained, 25 tasks) 8 and 11 vs base 8-11;
  half B (unseen, 24) 8 and 7 vs base 7-12. Training on correct trajectories of the very test tasks does not raise
  their success: the learning leg, not task proposal or labels, is the binding failure. Gains (task 16 0.4 -> 1.0,
  19 0 -> 0.5) are offset by losses (task 10 0.8 -> 0, 1 1.0 -> 0.5): the re-roll pattern again.
  testset_judge / testset_gt fine-tunes hit the 23 h limit (judge at step 1110, resumed as 293874); testset_gt went
  NaN at step 650 (every later checkpoint NaN), retrained from scratch as final_r16_lr5e-5_rerun (293876, follow-on
  293877; cancel it if 293876 succeeds) after train.py now skips optimizer steps with a non-finite grad norm.
  GameBoy vlm s1 fine-tune finished; stale follow-on 293010 cancelled, eval 293173 released.

- 2026-11-09 (corrections to the Nov 8 token-loss entry, found while writing the week's report): "every action row's
  taken action gets likelier" was wrong: the action segment included the appended <turn|> token, which carries the
  drop. On the action tokens alone (after "Action:", before the end token) base is 0.000-0.011 nats and the adapters
  0.002-0.017, i.e. unchanged or marginally worse; they lower that loss on 54-60% of rows (chance). Also: the 400
  rows sampled action / frame_diff / done_check tags only, no supervisor rows. And the "re-roll" reading is an
  aggregate: adapter-adapter outcome agreement 0.882 vs 0.875 expected for independent base-like runs, but task 3 is
  failed by all 5 tree adapters (base 5/6) and task 5 solved by all 5 (base 3/6); the gtA oracle does not share
  these flips (2/2, 1/2), so the tree adapters carry a consistent data-specific shift with no bias toward success.
  Per-task numbers recomputed with all six greedy base runs (the Nov 8 analysis omitted base_gemma-4-31b-it, 20/49):
  task 19 greedy 1/6 not 0/5; task 3 5/6; conclusions unchanged.

- 2026-11-09 (world-model audit; tmp_leak.py, jobs 293882-3): the held-out split is held-out *expansions*, not
  held-out game states. Random search re-expands shared nodes on a deterministic emulator, so held-out transitions
  repeat training ones. Red s0 (seed 42): 44,364 unique frames among 142k training frames; of 14,540 held-out
  transitions, 75% have a next frame byte-identical to a training frame and 48% repeat a training (current frame,
  action) pair exactly. Sword of Hope (3 seeds): 5,326 unique frames among 427k; 99.5% / 98.3%. So every in-game
  "beats copy" number (Red 19.2 vs 31.9, 59%; SoH 0.64 vs 7.3) is measured largely on transitions the model trained
  on, and the near-floor results on the menu games are memorisation. Unaffected: the Crystal and sequel evals (never
  seen), and the direction of the in-game vs sequel contrast, which is now partly train/test contamination rather
  than only a visual shift. Code audit otherwise clean: embedder / world model / wm_eval share one split (seed 42,
  same replay_dirs strings); copy baseline frames aligned; GameBoy canvas native so no resize mismatch; predictions
  normalised before the cosine; no model selection on validation. Next: re-score each world model on the held-out
  transitions whose (frame, action) pair is absent from training (Red s0 ~7.6k, SoH ~700) and report that column
  as the honest in-game number; make the leak check part of wm_eval / the model's fitness gate.

## 2026-11-09: World Action Model (WAM) contract and the SigLIP lookahead executor

- Contract (cusi/world_action/base.py): state in, actions out. `begin(task, hint, obs, info)`, `plan(obs, info,
  error, call) -> Plan(actions | finish | invalid, record)`, `observe(action_text, obs_before, ..., valid)`,
  `history()/restore_history()` across legs. No latent / embedder / decoder type in the interface (user's design
  call: the executor must not know how the model is made). `meta` carries env, game, game_series (GAME_SERIES
  table), replay_dirs, fitness (eval.json headline, leak_check.json). `transfer_kind(game)` = same_game /
  same_series / other_series, recorded in run config and episode meta.
- Child (cusi/world_action/siglip_lookahead.py): GameBoyRL's planning arm with CUSI's fine-tuned SigLIP world
  model: one-step rollout for every button, decoded by the SigLIP run's decoder, VLM picks "Choice: k" among
  [current, 7 predictions]; single action per plan. Model + embedder + decoder cached per (dir, device) per process.
- Executor (cusi/agents/executors/world_action.py): `world_action_executor_class(base, model)` subclasses the env's
  own executor and replaces decide() only (history policy, done check, invalid handling unchanged, so base vs WAM
  runs differ only in who decides). Plan.record goes on CallRecord.extra["world_action"] (new field) and the model
  keeps filling it from observe(). `episode_summary(legs)` -> EvalRow.extra wa_* fields.
- Wiring: make_agent(..., world_action=); run_eval gameboy --world_action <dir or world-model name>
  --world_action_device (default cpu: vLLM holds the GPUs); eval.sh / EVAL_DEFAULTS --world_action,
  --world_action_device. train_world_model now records game / game_series / replay_dirs; the 7 existing models were
  backfilled (tmp_backfill_wm_meta.py; red_* -> pokemon_red/pokemon, zelda_la -> legend_of_zelda_links_awakening/
  zelda, deja_vu_1, sword_of_hope_1, bomberman_quest).
- Smoke (tmp_smoke_wa.py, CPU, real Red replay frames, canned choice): 8 images to the VLM, 0.14 s predict+decode,
  candidate spread 45 grey levels (predictions depend on the action); on a real UP transition the hindsight-best
  candidate was UP (cos 0.80 vs copy 0.62; A / B predictions = copy). Not yet run live.
- Diagnostics hooks added before the first live run (user's request): (1) the VLM's view of the real outcome:
  per-candidate pixel errors vs what happened, hindsight_best_pixel, chosen_beats_copy, chosen_predicted_similarity;
  (2) the base policy's own decision on every world-action screen (shadow call, tag shadow_<tag>, decision None,
  never executed; --world_action_shadow, default true, doubles VLM calls): base_action, base_reasoning,
  base_invalid, diverged (base != executed), base_action_predicted_similarity (the world model's own view of the
  base's action). Episode summary adds wa_diverged, wa_chosen_beats_copy, wa_base_invalid,
  wa_base_action_pred_sim, wa_chosen_pred_sim. The done check reads the executed decision's reasoning, not the
  shadow's. Smoke (stub base policy, real frames): executed DOWN vs base UP -> diverged True; the world model rated
  the base's UP 0.80 vs the executed DOWN 0.57 against the real outcome.
- Oracle lookahead (cusi/world_action/oracle_lookahead.py): candidate frames from the emulator itself (PyBoy
  save_state to memory, press each button as GameBoyWorlds' emulator does, read the screen, load_state), below the
  GameBoyWorlds step counters / trackers, so it costs no benchmark steps. The common lookahead loop moved to
  cusi/world_action/lookahead.py (LookaheadModel: prompt, choice parsing, pixel scores); siglip_lookahead adds the
  embedding scores. Contract gains attach_env(env), documented as diagnostic-only. Smoke on a live Red env: the
  chosen candidate reproduces the real next screen exactly (error 0.0 on two decisions), step counter unchanged by
  the lookahead (2 -> 2), real steps advance (-> 4).
- Runs (tmp_wa_runs.sh; Gemma-4-31B, greedy, 8 workers, shadow base on, model on CPU): wa_red_rep1 (293885 died:
  its workers imported the executor after the attach_env edit but the old model module; all 49 tasks errored in
  seconds; resubmitted with --overwrite as 293889), wa_crystal_rep1 293886 (Red world model on Crystal, same-series
  transfer), oracle_red_rep1 293887, oracle_crystal_rep1 293888. Baselines: six greedy Red base runs (15-22/49);
  crystal_base_gemma-4-31b-it_greedy.
- Don't edit modules a running eval imports lazily: workers import at first task, so an edit mid-startup splits
  them across versions.
- GLM-4.1V-9B-Thinking (zai-org) on all three test sets (tmp_glm.sh): the earlier 8-9B evidence is one Oct 29 sweep
  with sampled decoding (Qwen3-VL-8B: GameBoy 6%, Web 35%, Android 46%; Gemma-4-26B-A4B 25 / 38 / 48%), so no 8-9B
  model has been run with the current greedy agents. vLLM 0.30.0 here supports Glm4vForConditionalGeneration but
  has no reasoning parser, so lm_inference.get_output_final now cuts <think>...</think> blocks before [STOP]
  truncation (no-op for Gemma). Risk: Web / Android executors cap at 1000 new tokens, which a thinking model's
  think block may exhaust (then the reply fails to parse and counts invalid); GameBoy's cap is 8000.

- 2026-11-09 (diagnosis, Android / Web per task; tmp_aw.py, existing evals): Web (429 tasks; 3 base runs, 6 adapters):
  base 269 / 251 / 250, adapters 275 / 268 / 264 / 268 / 268 / 282; outcome agreement base-base 0.786,
  adapter-base 0.776, adapter-adapter 0.816; adapter - base +0.033 (permutation p = 0.06) but confounded by date: the
  two base repeats ran a day after the original and sit ~4 pts below it, and the adapters ran near the original's
  date; against the original alone the adapter mean is +1.8 tasks. 6 tasks flip in all 6 adapters (e.g.
  Huggingface--21, GitHub--22 0 -> 1). Android (112 tasks scored in all 8 runs; 2 base, 6 adapters): base 78.5 / 71.5,
  adapters 73-79; agreement base-base 0.884, adapter-base 0.893; +0.007 (p = 0.83); 68 always / 31 never / 13 mixed;
  4 consistent flips (2 gains, 2 losses). [STOP] does not appear in the CUSI Web / Android executor or supervisor
  files (prompts imported from the WebVoyager / M3A packages not checked), so the end-token finding may be
  GameBoy-specific.

## 2026-11-09: Literature baselines under the "self-judged only" constraint (code audits, storage/tmp/lit_code/)

User's constraint: drop every phase that uses a stronger model or human demonstrations; all signal from the agent
itself. Audited PAE, OpenWebVoyager, OS-Genesis, UI-Voyager, MobileRL (repos cloned to storage/tmp/lit_code/).
- No paper reports a run that meets the constraint. PAE: every run starts from SFT on Claude-3-Sonnet trajectories;
  closest is Qwen2VL-7B as proposer+judge (not self) 14.9 -> 21.7 WebVoyager. OpenWebVoyager: GPT-4o writes queries,
  judges, and its IL data is re-trained every round; 19.9 -> 22.6 -> 22.7 -> 25.8, single sampled run. OS-Genesis:
  GPT-4o does synthesis, rollouts and scoring in every condition; 9.82 is a GPT-4o baseline, zero-shot Qwen2-VL-7B 0.89.
  UI-Voyager: self rollouts after round 1 (round 1 likely uses larger Qwen3-VL), labels from AndroidWorld's rule-based
  checker, training tasks are perturbed instances of the 116 eval templates; ~37 -> 73 (RFT) -> 81 (GRSD), 64 seeds.
  MobileRL: human-demo SFT + reasoning SFT from another model, then RL with AndroidWorld's rule-based reward on the eval
  templates; RL step alone 66.2 -> 80.2 (GLM-4.1V-9B), 56.8 -> 72.0 (Qwen2.5-VL-7B).
- The large AndroidWorld gains share ground-truth rewards and training on the eval templates; the web gains share a
  stronger-model judge and stronger-model demonstrations. Released code: PAE and OpenWebVoyager have training loops;
  OS-Genesis, UI-Voyager and MobileRL release eval / data code only (UI-Voyager also GRSD's fork builder).
- Training intensity in the papers vs ours (LoRA r16, lr 5e-5, 1 epoch): PAE LoRA r256 lr 2e-5 constant, 4 epochs per
  iteration, buffer never cleared; OpenWebVoyager full FT lr 1e-5 cosine, ~2-3 epochs per phase; MobileRL full FT
  (inferred) RL lr 1e-6 ~100 steps after 2-epoch SFT; OS-Genesis and UI-Voyager not reported.
- Variance: only UI-Voyager (64 seeds) and MobileRL ablations (3 runs) average runs; none reports variance.
- 2026-11-09: user's request for tomorrow morning: (1) action reproduction check first (cusi/finetune/repro.py,
  run_diagnostics.py action_repro, slurm/action_repro.sh pair; jobs 293928-293933: web random_s0 / vlm_w_typing_s0,
  android random_s1 / vlm_w_typing_s0, gameboy random_s0 / vlm_no_typing_s0; base vs adapter, greedy, 300 training
  and 300 held-out practice action rows each); (2) then PAE-style aggressive fine-tunes. Of the three PAE changes,
  shortest-success filtering and the never-cleared buffer are no-ops on our data (practice stops at the first judged
  success, so a task has at most one success; one practice round). Only the recipe applies: run_finetune
  --lr_schedule constant and --train_tasks N (seeded task subset) added; recorded in config only when set, so older
  runs resume unchanged. At measured speeds 4 full epochs take ~12 h (Android), ~39 h (Web), ~110 h (GameBoy).

- 2026-11-09 (cheap check #1, action_repro, jobs 293928-33): does an adapter reproduce its own training actions?
  Greedy base vs adapter on 300 executor action rows of the adapter's practice dataset (train) and held-out practice
  tasks (validation), first "Action:" line compared. Train base_match / adapter_match: GameBoy random 0.84 / 0.87,
  vlm 0.87 / 0.88; Android random 0.94 / 0.92, vlm 0.96 / 0.93; Web random 0.73 / 0.71, vlm 0.75 / 0.74.
  Validation alike (adapter equal or lower). On every env the base already picks the practice action 73-96% of the
  time, and the adapters do not pick it more often even on the rows they trained on: the fine-tunes (r16, lr 5e-5
  cosine, 1 epoch) did not fit the content, consistent with the token-loss finding (the drop is the end token).

- 2026-11-09 (six-task focus experiment, user's request: make learning work on the six practice-new tasks 17, 19,
  23, 29, 31, 36 with aggressive fine-tuning plus mix-in). Judge on these six (all 11 test-set practice repeats, 133
  tries): precision 0.53 (21 real of 40 judged), recall 1.00, vs 0.72 / 0.90 on the other 43.
  New stage run_practice focus_dataset (cusi/practice/stages/focus.py): the focus tasks' success tries from an
  attempt_test run (every one in train, kept whole under gt labels) x focus_repeat, plus mix_ratio x as many success
  rows sampled from another practice dataset (images copied in), validation = that dataset's held-out success rows.
  Built (no paraphrases), source sat_vlm_no_typing_s0 for mix-in and validation:
    testset_six_gt_mix     21 benchmark-success tries -> 4,782 rows + 4,782 mix-in (t19 has 10 of the 21 tries;
                           t17, t23, t36 one each)
    testset_six_gt_nomix   the same 4,782 rows, no mix-in
    testset_six_judge_mix  40 judged successes (19 false; t23 11 judged / 1 real) -> 8,890 rows + 8,890 mix-in
  Recipe (focus_r64_lr1e-4_e3): LoRA r64 a128, lr 1e-4 constant after warmup, 3 epochs, 32 rows/step,
  max_length 6144. Each -> action_repro on its own rows -> 2 greedy evals on all 49 (capability on the other 43)
  + 4 greedy evals on the six only (6 reps on the six, as the base's 6 greedy runs: 1/36 there).
  Commands:
    python run_practice.py --env gameboy --model_name google/gemma-4-31b-it --source testset_six_gt_mix \
        focus_dataset --from_source testset --tasks 17,19,23,29,31,36 --labels gt \
        --mix_source sat_vlm_no_typing_s0 --mix_ratio 1        (nomix: --mix_ratio 0; judge: --labels judge)
    sbatch slurm/finetune.sh --env gameboy --model google/gemma-4-31b-it --source testset_six_gt_mix \
        --run focus_r64_lr1e-4_e3 --extra "--lora_r 64 --lora_alpha 128 --lr 1e-4 --lr_schedule constant \
        --epochs 3 --rows_per_step 32 --max_length 6144 --max_batch_tokens 6144 --val_rows 400"
    sbatch slurm/eval.sh --env gameboy --model google/gemma-4-31b-it --run <name> --workers 6 \
        --tasks pokemon_red_017,pokemon_red_019,pokemon_red_023,pokemon_red_029,pokemon_red_031,pokemon_red_036 \
        --lora <adapter>
  Jobs: gt_mix ft 293986 (repro 293987, evals 293988-93), gt_nomix ft 293994 (293995, 293996-294001),
  judge_mix ft 294002 (294003, 294004-09). At 0.87 rows/s (4 A6000): ~9 h, ~5 h, ~17 h.
  The PAE-recipe per-env fine-tunes (previous request) are not launched: the A6000s go to this first.
  Caveat: this trains on the test tasks' own trajectories from their own start states, so it measures whether SFT
  can fit the six at all, not generalisation.
- 2026-11-09 (sync): pushed to the Hub search/gameboy/sat_vlm_no_typing_s0 (time-limited VLM tree, no summary.json:
  sync_data push --time_limited <tree>), practice/gameboy/gemma-4-31b-it/tree/testset (attempt_test tries; now
  finished once results.csv is newer than every episode) and the three testset_six_* focus datasets (self-contained:
  mix-in images copied in). attempt_test --tasks (benchmark rows) and practice.sh --lora (practise with an adapter,
  every role) added. More base practice on the six (repeats 11-40): 294010 (2x A6000), 294011 (8x A5000).
- 2026-11-09 (failure): oracle_red_rep1 (293887, 6 h) and oracle_crystal_rep1 (293888) lost every task: the episodes
  ran, then episode_summary raised KeyError 'chosen_is_hindsight_best' (a SigLIP-only record field) before the
  trajectory pickle was written. Fixed (the field is skipped when absent; the eval now saves the trajectory before
  computing world-action summaries); 293888 cancelled, both resubmitted --overwrite as 294013 / 294014. The SigLIP
  runs (293886, 293889) carry the field and were unaffected. Lesson: smoke the summary path for every WAM child,
  not just the SigLIP one.
- 2026-11-09 (action_divergence 294012, base greedy on the 1,749 action rows of the 21 benchmark-success tries on the
  six): base takes the trajectory's action on 85.6% (t17 137/137, t19 675/740, t23 76/89, t29 189/229, t31 380/488,
  t36 41/66): 251 divergent decisions. t17's single success is matched on every action row, so whatever makes
  greedy base fail t17 is not in the action calls (done check / supervisor rows, or decoding noise across steps).
  New arm testset_six_gt_div10_mix: the gt focus rows (29 tries now, repeats 11+ included) with the 251 divergent rows
  x10 extra, equal mix-in, 18.8k rows/epoch, focus_r64_lr1e-4_e2 (2 epochs), ft 294015 -> repro 294016, evals
  294017-22. judge_mix 294002 deprioritised (nice 500) behind it.
- 2026-11-09 (oracle results, cont.): testset_judge (judge labels, all 49, r0-r5, gentle recipe) 23 and 19 of 49
  (mean 21; base greedy 15-22): within noise. Six-task hits: rep1 019 and 031, rep2 none.
- 2026-11-09 (GLM-4.1V-9B-Thinking, Android, 293892): 36/114 scored = 31.6% (2 failed to run) vs Gemma-4-31B greedy
  64.5-70%; 23% of steps invalid (432/1862, Gemma 0.4-9%), 37 tasks hit max_steps; 17.7k output tokens per task.
- 2026-11-09 (GLM-4.1V-9B-Thinking, Web, 293891): 1/429 (0.2%), 285 tasks hit max_iter. Not a capability number:
  1,684 of 5,228 replies carry no "Action:" (1,497 "Format ERROR" feedbacks): WebVoyager's call_model caps every
  reply at max_new_tokens=1000 (the original harness's setting, same for every model) and GLM's thinking runs past it.
  The replies that do act wrap the action as "Action: <|begin_of_box|>- Click [15]<|end_of_box|>", which
  extract_information still matches (re.search). WebVoyager calls the model itself, not through lm_inference.
  lm_inference.get_output_final now also strips GLM's <answer>/<|begin_of_box|> wrappers and the "- " before a boxed
  action (CUSI-parsed envs: GameBoy, Android); it does not change Web. A fair Web number for a thinking model needs a
  larger reply cap, which changes the harness for that model only; not rerun.
- 2026-11-09 (six-task focus, first result): testset_six_gt_nomix (r64 a128, lr 1e-4 constant, 3 epochs, 448 steps,
  490M trainable) barely fits its own data: train loss ~0.33 (epoch 1) -> ~0.29 -> ~0.30 (epoch 3); val loss on
  held-out tree rows 0.42 -> 0.35 (step 100) -> 0.40 (step 400). action_repro 293995 on its own training action rows:
  base picks the row's action 0.87, the adapter 0.81 (agree 0.82): after 3 epochs on these exact rows the served
  adapter reproduces them *less* often than the base (per task t29 36 -> 28 of 41, t31 72 -> 66 of 87, t19 114 -> 109
  of 125). Focus successes are mostly first tries (28 of 41 real so far; hints are not the cause), from *sampled*
  practice, so the response text carries sampling entropy. Open question: trainer fails to fit vs. served adapter
  != trained adapter. Check: HF teacher-forced NLL of base vs this adapter on 150+ of its own training rows
  (tmp_token_loss.py --dataset, job 294055).
- 2026-11-09 (WAM oracle, Red, 294013): emulator-oracle one-step lookahead (the true next screen for every button)
  19/49, inside the six greedy base runs (15, 19, 19, 20, 21, 22). Per episode: 116 world-action decisions, 41 with
  no parsable "Choice: k" (36%); executed action differs from the shadow base action on 20% of decisions; the chosen
  candidate always beats copy (the oracle is exact). So a perfect one-step world model, offered as 8 candidate next
  screens, does not raise Gemma's success on Red; the 36% unparsed choices are a confound to check before reading
  this as "lookahead cannot help". SigLIP world model on Red (293889) so far 10/42.
- 2026-11-09 (GLM-4.1V-9B-Thinking, GameBoy, 293890): 0/49, 6,962 of 8,312 decisions invalid (84%). Of 1,281 replies
  in 6 trajectories: 649 wrapped in <answer>/<|begin_of_box|> (unparsed before the lm_inference unwrap), 448 cut
  inside an unclosed <think> (the call's token budget), 158 plain "Action:". Not a capability number. Reruns with the
  unwrap (runs base_glm-4.1v-9b-thinking_unwrap): GameBoy 294091, Android 294092, nice 300 (after the six-task work);
  truncated thinking will still count as invalid (the harness's budget, same for every model).
- 2026-11-09 (six-task focus, nomix eval rep1, 293996): 21/49; focus tasks solved 019, 029, 031 (greedy base: 1 of
  36 task-runs on the six across 6 runs); other 43: 18 (base 15-22 overall, so no visible capability loss). One run;
  the gentle testset_judge adapter also hit 2 of the six once. Six-only reps 293998-294001 restored to nice 0.
- 2026-11-09 (six-task focus, nomix eval rep2, 293997): 24/49 (the highest single GameBoy greedy run so far; base
  15-22); focus solved 017, 019, 031; other 43: 21. Two runs: six 6/12 (base greedy 1/36; untrained tree adapters
  7/30 = 0.23), other 43: 18 and 21 (base runs 15-22 overall). Six-only reps 293998-294001 pending.
- 2026-11-09 (failure, WAM parser): the "36% invalid choices" were a parser bug, not the model. Every rejected
  world-action reply (oracle Red 2,031 of 2,031; SigLIP Red 2,526 of 2,527) answered "Choice: Image N" with a valid N;
  the regex took only "Choice: N". Each rejection spends a decision of the step budget (executor run loop: invalid
  -> n_steps + 1, no button), so the WAM runs played with ~2/3 of base's budget. The oracle Red 19/49 and all partial
  WAM numbers are invalid as estimates. Fix: Choice:\s*\**\s*(?:Image\s*)?(\d+), case-insensitive; checked on 761
  replies of 10 oracle trajectories (agrees with every old parse, recovers every rejection). Cancelled 293889,
  293886, 294014; rerun as oracle_red_fixparse 294105, oracle_crystal_fixparse 294106, wa_red_fixparse 294107,
  wa_crystal_fixparse 294108 (same world model red_random_all, shadow base on, 8 workers).
- 2026-11-09 (six-task focus, nomix six-only rep3, 293998): 019, 029, 031 solved (3/6). Running total over 3 runs:
  9/18 (t19 3/3, t31 3/3, t29 2/3, t17 1/3, t23 0/3, t36 0/3) vs greedy base 1/36 and untrained tree adapters 7/30
  (0.23); binomial P(>=9/18 | 0.23) = 0.011 (treats task-runs as independent, which they are not).
- 2026-11-09 (nomix six-only rep4, 293999): 019, 029 (2/6). Total over 4 runs 11/24 (t19 4/4, t29 3/4, t31 3/4,
  t17 1/4, t23 0/4, t36 0/4).
- 2026-11-09 (nomix six-only rep5, 294000): 019, 029 (2/6). Total over 5 runs 13/30 (t19 5/5, t29 4/5, t31 3/5,
  t17 1/5, t23 0/5, t36 0/5).
- 2026-11-09 (tokloss 294055: HF teacher-forced NLL, base vs testset_six_gt_nomix adapter, 250 of its own training
  rows, the trainer's RowEncoder): action tokens given the reasoning: base 0.0015, adapter 0.0019 nats (P(trained
  action) 0.996 both): the action is fixed by the reasoning. Reasoning tokens 0.347 -> 0.302, adapter lower on only
  69/150 rows. Whole row incl. the end token 0.469 -> 0.282 (137/150): the end token again. done_check without end
  0.213 -> 0.203. So after 3 epochs at lr 1e-4 the trainer has not fit the reasoning of its own data (no
  serving/training mismatch is needed to explain the vLLM repro: HF agrees). Yet the adapter's six-task rate is
  13/30 vs 0.23. Sanity check queued: overfit one try (task 17's single benchmark success, 359 rows,
  testset_t17_gt_nomix) 20 epochs, lr 2e-4, 16 rows/step: ft 294127 -> repro 294128 -> 6 greedy t17 evals
  294129-34 (tmp_t17_overfit.sh).
- 2026-11-09 (nomix six-only rep6, 294001; final for this arm): 019, 029, 031. Six runs: 16/36 (t19 6/6, t29 5/6,
  t31 4/6, t17 1/6, t23 0/6, t36 0/6). Per task, the six tree adapters (untrained on these tasks, one greedy run each):
  9/36 (t19 5/6, t29 3/6, t31 1/6, others 0); gentle oracle adapters (testset_judge, testset_gtA, 4 runs): 3/24;
  base greedy 1/36. Against the tree adapters, one-sided Fisher p = 0.068 (task-runs not independent): most of the
  gain over greedy base is what any perturbation of the base gets on t19 and t29; what the six-task training adds is
  t31 (4/6 vs 1/6), t29 (5/6 vs 3/6) and one t17. The earlier 0.23 baseline was the 5-adapter, 30-run figure;
  this per-task table (6 runs) supersedes it.
- 2026-11-09 (correction): the "vs 0.23" comparisons and binomial p-values (0.011, 0.0114, 0.0108, 0.0036) in the
  nomix six-task entries above are wrong. 0.23 was itself an estimate (5 tree adapters x 6 tasks, 7/30), not a known
  rate, so a one-sample binomial overstated the evidence; it also collides with base's sampled first-try rate (also
  ~0.23), a different quantity. Correct comparison, per task, all six tree adapters one greedy run each: 9/36 vs the
  nomix adapter's 16/36, one-sided Fisher p = 0.068, task-runs not independent; the only clear per-task difference is
  t31 (4/6 vs 1/6). Pooled six-task rates are dominated by task difficulty (t19 near-always solved by any perturbed
  base, t23/t36 never); report per task.
- 2026-11-09 (proposal, not adopted yet): train/val/test split of the three benchmarks, tmp_split.py ->
  storage/tmp/split_proposal.json. Train envs: Red + Link's Awakening (GameBoyWorlds marks all their tasks
  can_train_from_init_state=True and pairs them with shifted test games Crystal / Oracle of Seasons, both
  can_train=False), Web sites Apple, ArXiv, Booking, Coursera, ESPN, GitHub, Huggingface (test sites Google Map,
  Google Flights, Wolfram Alpha), Android apps calendar, settings, markor, broccoli, opentracks, tasks, sms, joplin,
  retro music, clock, audio recorder, files (test: osmand, chrome, pro expense, vlc, camera, contacts, draw, gallery,
  clipper, and every multi-app task touching one). Inside train envs: train 60 / val 15 / in-env test 25 by task,
  GameBoy grouped by start state, balanced on base solve rate and category by search; the six focus Red tasks forced
  into train.
- 2026-11-09 (six_gt_mix repro, 293987): on its own training action rows base 0.857 vs adapter 0.837 (six-task rows
  0.82 vs 0.81; mix-in rows 0.89 vs 0.86); held-out tree rows base 0.91 vs adapter 0.79. Same pattern as nomix: the
  adapter does not reproduce its training actions, and drifts from base on held-out practice rows.
- 2026-11-09 ~22:10 UTC (benchmark change): GameBoyWorlds pulled c3efc1f -> 37b80e3 (user's request; the data pull
  `python -m gameboy_worlds.setup_data pull --game pokemon_red` ran first and synced 230 files into
  ../GameBoyRL/GameBoyWorlds/storage/rom_data/pokemon/pokemon_red). New Red task 49 "Read the blackboard in the school
  nearby" (navigation, interaction; location_viridian_city_starting_charmander), appended, so 0-48 keep their ids;
  Red is 50 tasks now. One Oracle of Seasons task reworded. Also changes the Pokemon emulator (screen-fade waits, Gen 2
  movement, battle-HP text on shake, memory signal tracking). Every GameBoy result before this is on the old emulator;
  jobs started after it (from 293989 on) run the new one, and full Red evals now have 50 tasks. Jobs running across
  the pull loaded the old code at start (mixed-version risk for lazily imported modules).
- 2026-11-09 ~23:55 UTC (user's call, refocus): cancelled the six_gt_mix evals (293988-89, 293992-93), the WAM
  parser-fix reruns (294105-08) and the 49-task oracle fine-tune testset_gt (293876, its follow-on 293877, evals
  293878-79). Priorities now: (1) six-task fine-tune on judge labels (294002) and its evals (294003-09); (2) faithful
  reproduction of the baseline methods (plans/baselines_reproduction_spec.md).
- 2026-11-10 (baseline for the judge arm): 6 greedy base runs of the six focus tasks on the new emulator
  (base_gemma-4-31b-it_six_newemu_rep1-6), since every six-task eval from now on runs on GameBoyWorlds 37b80e3.
- 2026-11-10 (base, six tasks, new emulator 37b80e3, 6 greedy runs): t17 5/6, t19 4/6, t23 1/6, t29 0/6, t31 0/6, t36 0/6 (10/36). The emulator update alone makes
  t17 (Defeat Blue) and t19 (Get onto the water) mostly solvable greedily (old emulator 0/6 and 1/6). t29 and t31
  stay at 0, so they are the tasks where an adapter can show a gain; the mix-in adapter's t17 solves (new emulator)
  are explained by the emulator, not training.
- 2026-11-10 (six_gt_div10_mix repro, 294016): first adapter that moves toward its training actions. Six-task training
  action rows (sample dominated by the x10-repeated divergent rows, so base is low by construction): base 0.46 ->
  adapter 0.78; mix-in rows 0.81 -> 0.84; held-out tree rows 0.86 -> 0.75 (drift, as the other arms). So up-weighting
  the decisions where greedy base leaves the successful trajectory does make the adapter take them; the plain arms
  (no up-weighting) did not (0.87 -> 0.81, 0.82 -> 0.81).
- 2026-11-10 (dirty PAE overnight, user's call; throwaway files tmp_pae.py, tmp_pae_train.sh, tmp_pae_launch.sh):
  PAE on the 50 Red test tasks (new emulator), no proposal. roll0 = base, 2 repeats x 50 tasks, one sampled try, no
  hints (294141-44); then per condition (gt = benchmark checker, judge = Gemma judge) 4 iterations of: shortest-20%
  successes per task over the cumulative buffer -> dataset (executor rows, no paraphrases) -> LoRA r256 a256 lr 2e-5
  constant, 4 epochs, retrained from the base each time (PAE continues the adapter) -> rollout with the new adapter.
  gt: trains 294145/50/55/60; judge: 294165/70/75/80. Read progress with `python tmp_pae.py report`.
  Task-17 overfit evals 294129-34 cancelled to free GPUs.
- 2026-11-10 ~05:00 UTC (dirty PAE, change): first update had 61 filtered trajectories / 5,226 executor rows; at 4
  epochs that is ~6.7 h per update, so the chain was cancelled (294145-84) and relaunched at 1 epoch per update (PAE's
  code value; paper says 4), rebuilding the dataset every update: gt trains 294192/97, 294202/07; judge 294212/17,
  294222/27, each followed by 4 rollout jobs. roll0 (base): 217 attempts at the first count, benchmark success 0.525,
  judge precision 0.84, 31 tasks solved.
- 2026-11-10 (GLM-4.1V-9B-Thinking, Android, rerun with the answer-wrapper unwrap, 294092): 0.341 (113 scored) vs
  0.316 before; invalid steps 118/1755 (6.7%) vs 432/1862 (23%); tasks hitting max_steps 18 vs 37. The unwrap fixed
  most invalid actions; success barely moved. Gemma-4-31B: 0.645-0.70.
- 2026-11-10 (six_judge_mix repro, 294003): same as the plain oracle arms. Own six-task training rows base 0.83 ->
  adapter 0.81; mix-in rows 0.86 -> 0.82; held-out tree rows 0.84 -> 0.76. No uptake of its training actions.
- 2026-11-10 (GLM-4.1V, GameBoy, unwrap rerun 294091, new emulator, 50 Red tasks): 3/50 (was 0/49); invalid decisions 993/8038 (12%, was 84%). The unwrap fixed parsing; GLM still solves almost nothing. Gemma-4-31B base 15-22/49 (old emulator).
- 2026-11-10 (six_judge_mix, 4 six-only evals, new emulator): t17 3/4, t19 4/4, t31 1/4, t29 0/4, t23 0/4, t36 0/4. vs base new emulator (6 runs): t17 5/6, t19 4/6, t29 0/6, t31 0/6, t23 1/6. Self-judged labels: one t31 solve, otherwise base-level. Full-benchmark evals 294004-05 pending.
- 2026-11-10 (base, all 50 Red tasks, new emulator, greedy, 294237): 25/50 (old emulator 15-22/49). The emulator update
  alone adds ~5 tasks. Against it the divergence-weighted adapter's 18 and 18 of 50 are BELOW base. Second base run
  294236 pending.
- 2026-11-10 (base, all 50 Red, new emulator, second greedy run 294236): 23/50. New-emulator base: 25, 23 (mean 24).
  Divergence-weighted adapter 18, 18 is ~6 tasks below.
- 2026-11-10 (six_judge_mix full-benchmark evals 294004-05, new emulator): 21 and 20 of 50 vs base 25, 23. Six tasks: only t17 each run. Judge-labelled aggressive SFT: no gain on the six, ~3-4 tasks below base overall.
- 2026-11-11 (split adopted, configs/splits/benchmark_splits.json): {task id: {split, text}}, 745 tasks. Ids:
  gb.<game>.<row:03d>, web.<WebVoyager id>, aw.<AndroidWorld task class> (text: the seed-0 instance). Splits
  train / val / test / test_env. From the 2026-11-09 proposal plus: Red task 49 -> train (shares its start state with
  train tasks 0, 4, 10); train -> test to reach 15 test tasks in Red and Link's Awakening and 25 in-app Android test
  tasks, picked to keep test's base solve rate and category mix close to the env's: Red 21, 25; Link's Awakening 1,
  9, 11, 15 (category only: no base eval of it); Android MarkorChangeNoteContent, RecipeAddSingleRecipe,
  SimpleCalendarAddOneEventInTwoWeeks, SimpleCalendarAddRepeatingEvent. GameBoy tasks sharing a start state stay in
  one split. Counts: GameBoy 55/15/30 + 100 test_env (Crystal, Oracle of Seasons); Web 180/45/75 + 129 (Google Map,
  Google Flights, Wolfram Alpha; the 5 bot-blocked sites are out); Android 48/13/25 + 30. Base success train / val /
  test: Red 0.44/0.39/0.40, Android 0.70/0.69/0.69. No version or drift check by choice: the benchmarks are assumed
  fixed. Plan for using it: plans/user_sandbox_plan.md.
- 2026-11-11 (benchmark data): `python -m gameboy_worlds.setup_data pull` for legend_of_zelda_links_awakening (153
  files), pokemon_crystal (129), legend_of_zelda_the_oracle_of_seasons (115). Before it, only Red had been pulled after
  the 37b80e3 update, and the other three games failed at env start on missing capture files (e.g.
  sword_pickedupinit.npy, lost_battle.npy). After it, one task per game starts and steps.
- 2026-11-11 (User sandbox built, plans/user_sandbox_plan.md; review list plans/user_sandbox_review.md): env server
  per env (cusi/env_server, run_env_server.py) with dev (train + val, free play, state on request, save/load) and
  test (test + test_env, success only) policies, per-session tokens, client + envctl (user_template/), Apptainer
  sandbox (storage/containers/sandbox.sif, only the workspace mounted), launcher slurm/user_env.sh, harness
  run_user_eval.py. Recording off by default (--record_dir). E2E (294745/294746, Gemma-4-31B): the final agents
  solved 6 of 8 tasks through the servers (both misses GameBoy at a 60-step cap), no state in any response, sandbox
  isolation and the held-out blocks held.
- 2026-11-11 (User sandbox, user's calls): (1) dev sessions undo steps into held-out apps/sites instead of ending the
  session: envs' blocked_packages / blocked lists, undone before observation, invalid with a message that names
  nothing; live-checked (294747 Android, 294749 Web). (2) env server `state` = env.full_state(): GameBoy adds
  GameBoyWorlds' memory snapshot (all four games, Zelda included) and agent_state; true_state() unchanged (exploration
  records it every step). (3) prime-agent 0.10.2 in the sandbox (image docker://python:3.12; agent home
  storage/env_server/agent_home mounted as $HOME); login pending.
- 2026-11-11 (trees, seeds 3 and 4 of every combination; tmp_submit_trees.sh / tmp_submit_vlm.sh): GameBoy random +
  vlm_no_typing, Web and Android random + vlm_no_typing + vlm_w_typing, same settings as s0-s2 (configs checked equal
  but for the seed), 69 h caps as 3 chained 23 h segments (GameBoy random one segment: it saturates in hours); VLM on
  2x A6000 as before, random without GPUs, Android --exclusive. Failures on the way: (1) run_search's defaults had
  changed since s0-s2 (random 100 steps / threshold 0.2, VLM 20 / 0.1; the old trees used 20 steps and 0.1 GameBoy /
  0.5 Web, Android): the first submission was cancelled and the values are now passed explicitly
  (--<expander>_steps, --<expander>_node_threshold). (2) 10 vLLM servers starting together hit the HF Hub API rate
  limit (429) and every VLM segment failed in seconds (afterany ran the whole chains): resubmitted with
  HF_HUB_OFFLINE=1. (3) Web vlm_no_typing s4's first segment failed on a stale NFS file handle in the shared torch
  inductor cache (concurrent compiles); its chain got a 4th segment (294869) to keep 69 h. Scenes include the test
  start states and held-out sites/apps, as in s0-s2 (parity; they predate the split).
- 2026-11-11 (VLM testbed, storage/tmp/testbed/testbed.json; tmp_testbed.py, tmp_testbed_pick.py): 28 test-split tasks
  picked from the results on disk, per env 3 easy + 3 hard held-out tasks and 2 easy + 2 hard held-out-env tasks
  (GameBoy: Red test + Crystal; Zelda has no Gemma results yet): easy = Gemma-4-31B solved every run, hard = never,
  hard preferring tasks another model solved. Existing models on it (easy / hard): Gemma-4-31B 14/14, 0/14;
  Qwen3-VL-32B 10/14, 5/14; Qwen3-VL-8B 8/14, 6/14; Gemma-4-26B-A4B 10/13, 3/14; Muse-Glimmer-30B 8/14, 4/14;
  Qwen3-VL-4B 2/14, 3/14; Qwen3-VL-2B 1/14, 0/14; GLM-4.1V (no unwrap) 1/13, 1/13. New models run with
  slurm/testbed.sh (one vLLM per model on 4x A5000, the three envs in parallel, Web judged afterwards by
  Gemma-4-31B): job 294912 with Gemma-4-31B (anchor), Gemma-4-12B / E4B / E2B, Gemma-3-27B, GLM-4.1V (unwrap),
  Qwen3-VL-30B-A3B, MiniCPM-V-4.5, InternVL3.5-8B, UI-TARS-1.5-7B, Kimi-VL-A3B-Thinking, Mistral-Small-3.2-24B.
  Caveat: the older full_* GameBoy runs predate the 37b80e3 emulator update.
- 2026-11-11 (Link's Awakening trees, tmp_submit_zelda.sh): random + vlm_no_typing s0-s5 as sat_zelda_la_<exp>_s<k>
  under the Red protocol (20 steps, threshold 0.1, patience 50, 69 h); VLM on 2x A6000 except vlm_no_typing s5 on 4x
  A5000 (user's call: no A6000 pair free). Older sat_random_zelda_la_s0-s2 are 7500-expansion world-model data runs.
- 2026-11-11 (Red random trees s3-s4 smaller than s0-s2): same config.json but one seed; saturated at 2507 / 2871 nodes
  vs 4600-7198. Not saturation luck: nodes after 1000 / 2000 / 4500 expansions are 1002-1041 / 1471-1528 / 2259-2335
  (s3, s4) vs 1271-1411 / 2035-2360 / 3172-3652 (s0-s2). Between them GameBoyWorlds moved c3efc1f -> 37b80e3 (Nov 9),
  whose Pokemon emulator waits out map scrolls and fades before an action returns. Not the fades: blank / low-contrast
  node frames are 0.2% / 3-4% in every tree (tmp_fade_nodes.py). Suspected: frames caught mid-step and presses
  dropped mid-step in the old emulator. Check: s0 and s1 rerun on 37b80e3 as sat_random_s{0,1}_gbw37b (295014-5).
  Testbed: Gemma-3-27B skipped (snapshot lacked preprocessor_config.json; now downloaded).
  Confirmed: sat_random_s1 rerun on 37b80e3 saturated at 2270 nodes (4289 expansions) vs 6300 originally. Red
  sat_random_s0-s2 and sat_vlm_no_typing_s0-s2 (with their annotations, task_coverage, task_diversity) deleted and
  rerun on 37b80e3 with the s3-s4 settings (tmp_submit_red_s012.sh: random 295045/49/53, VLM chains 295046-8,
  295050-2, 295054-6). Their LoRA fine-tunes and evals (storage/models/finetune/gameboy/
  sat_*_s0-s2, storage/eval/gameboy/lora_sat_*_s0-s2) and practice runs
  (storage/practice/gameboy/gemma-4-31b-it/tree/sat_*_s0-s2) deleted too.
- 2026-11-11 (Qwen3-VL-8B trees, tmp_submit_qwen_trees.sh, jobs 295159-295173 in storage/tmp/qwen_trees_jobs.txt):
  vlm_no_typing trees for Qwen3-VL-8B on Red, Web and Android, seeds 0-4, as sat_vlm_no_typing_qwen3-vl-8b_s<k>
  (first queued for all five Qwen models, 75 trees, ~3 months; cut to 8B by the user). User's call: all on one node
  (eclairlg14, 4x Quadro RTX 8000, fp16: no bf16 before Ampere), low priority, packed on its GPUs (1 GPU each),
  Android one at a time (singleton cusi-qwen-android: one emulator per node). The Gemma trees' 69 h cap is one budget
  (--budget_metric wall_s --budget 248400: expansion time, kept across resumes) in a 3-day long-lg job instead of 3
  chained 23 h segments. scripts/slurm/search.sh gained --vllm_extra (for --dtype half). Caveat: fp16 on Quadro vs
  bf16 on A6000 for Gemma. Then cut to seeds 0-2 (s3-s4 cancelled) and widened to every Qwen model, with Link's
  Awakening added ("GameBoy trees" = Red + LA): tmp_submit_qwen_trees3.sh (jobs in storage/tmp/qwen_trees3_jobs.txt),
  60 trees on two nodes: lg14 (long-lg, one 3-day job per tree) 8B / 2B / 32B, lg15 (medium-lg: 4 chained 23 h
  segments, the wall_s budget stops the tree) 4B / 30B-A3B; 2 GPUs for 30B-A3B / 32B; Android one at a time per node
  (singletons cusi-qwen-android, cusi-qwen-android-b; 2B Android s1-s2 on lg15: 7 and 8 trees). ETA ~3.5 weeks
  ideal (lg14 ~2,500 GPU-h on 4 GPUs), mid-December with scheduling slack.
  Then cut to 8B and 32B (user's call): 2B / 4B / 30B-A3B cancelled; 8B (Red, LA, Web, Android s0-s2) on lg14, 32B
  moved to lg15 (tmp_submit_qwen32_trees.sh, storage/tmp/qwen32_trees_jobs.txt; 4 chained 23 h segments per tree).
  Test-set evals of Qwen3-VL-4B / 2B first (tmp_submit_qwen_test.sh: test + test_env, runs test_<model>_r1, Quadro
  fp16, lg14 / lg15 excluded, Web judged later); the five-model full evals (tmp_submit_qwen_full.sh) not submitted.
- 2026-11-11 (eval compute): run_eval (GameBoy, Web, Android) records each task's compute tally in EvalRow.compute
  (LLM calls, tokens, images, FLOPs; NVML energy / busy time, which with parallel workers overlaps other tasks);
  summary.json adds llm_flops_total / llm_flops_mean. llm_flops returns None (counted unknown) for a model config it
  cannot read instead of failing the call: InternVL's crashed every testbed episode that way.
- 2026-11-11 (wall clock visible to the agent): the harness passes ENVCTL_DEADLINE (unix time the task's wall-clock
  limit ends, set when the sandbox starts); envctl shows the seconds left on every step's summary line, writes
  seconds_left into each obs json and envctl info, and envclient.seconds_left() returns it. README and the live-agent
  prompt say so. Steps were already visible (steps / max_steps in every response). Checked live (job 295357: GameBoy
  test task, fixed_repo in the sandbox: 299 s then 293 s left after a 5 s sleep).
- 2026-11-11 (naming: states vs screens vs tasks): a state is the env's ground truth (URL, activity, map tile, flag),
  a screen is a frame the agent saw, a task is a text goal; each metric is named after what it counts.
  cusi/search/coverage.py -> cusi/search/state_coverage.py (state_coverage(), state_coverage.json,
  STATE_COVERAGE_FILE); Android's "screens" measure (activity + fragments) -> "activity_states".
  cusi.search.screen_coverage is reserved for distinct screens from frames. Older entries above that say "true
  coverage" mean state coverage.
- 2026-11-11 (screen coverage, cusi/search/screen_coverage.py + slurm/screen_coverage.sh, configs/screen_coverage.yaml):
  compares search runs of one env by the screens (frames) they saw. Every frame in env-step order (scene start
  frames, then every expansion step's frame; restores save none; curiosity sessions log states only, no frames) is
  embedded with random_patch (the search's own, fixed seed 42, image only; CPU) or SigLIP (--embedder siglip;
  --encoder_model defaults to google/siglip2-base-patch16-224: the user's call, the one model default allowed here).
  Same screen: 1 - cosine <= tau. No per-run distinct-screen counts (user's call): per run, the frames no frame of
  another run comes within tau of (per other run; for 3+ runs, by none of the others), by scene, at tau/2, tau, 2tau,
  optionally within the first N env steps; frame-weighted (a screen counts as often as its run saw it). Output:
  storage/search/<env>/_compare/screen_coverage/<hash>/ (meta, summary.json, summary.md, sheets of sample unmatched
  frames beside their nearest frame of the other run). Embeddings cache at <run>/screen_coverage/<key>/parts/
  (blocks of 200 expansions, resumable; a grown run is embedded from where it stopped). Web random s0 (77,710
  frames): 12 min random_patch (CPU), 14 min SigLIP (1 Quadro).
  tau calibration (calibrate --stage sample / fit): 60 same-scene pairs per env from the random and VLM s0 trees,
  spread over distance bands of both embedders, labelled by me (Claude) blind to the distances, rule: same screen =
  the same page / app screen / map room with the same overlay (dialog, menu, keyboard) open or closed; ignored:
  clock / status bar, animations and sprite or NPC positions, carousels and ads, blinking cursors, scroll under ~1/4
  viewport, typed text in a field, the player moving within the same room view, the same message part-typed; a
  different dialogue line or menu, a large scroll, or a mid-fade / half-drawn transition frame = different. Labels
  with notes: storage/tmp/screen_calibration/<env>/labels.jsonl (pairs.jsonl, sheet_*.png, fit.json beside them).
  tau (errors of 60): gameboy random_patch 0.0314 (10), SigLIP 0.0328 (3); web 0.0019 (8), 0.0401 (7); android
  0.0065 (8), 0.0275 (2). SigLIP separates better everywhere; random_patch is weakest on Android (white settings pages
  look alike pixel-wise) and Web (small dropdowns barely move either embedder).
  Test (random s0 vs VLM s0; % of each run's frames unmatched by the other at tau/2 / tau / 2tau):
  gameboy LA (random vs vlm_no_typing): SigLIP 16.3/6.1/1.6% vs 3.1/2.3/0.9%; random_patch 24.9/13.7/11.0% vs
  9.5/3.1/2.4%. android (random vs vlm_w_typing): SigLIP 72.1/63.5/50.7% vs 65.7/56.2/41.0%; random_patch
  49.6/32.2/17.2% vs 46.0/26.3/11.9%. web: SigLIP 87.9/85.0/72.4% vs 63.1/59.1/48.7%; random_patch 88.8/86.4/80.4%
  vs 64.9/63.0/59.1%. The embedders agree on direction everywhere and on size for web; android's SigLIP unmatched
  sheets show genuinely different screens (random_patch matches distinct white settings pages). Caveats seen in the
  sheets: GameBoy dialogue frames count once per typed-out step (frame-weighted); web random's localised pages
  (Booking, App Store in ~30 languages) count as different screens and inflate its unmatched share.
- 2026-11-11 (task dedup, cusi/eval/task_dedup.py): word overlap alone tested and rejected (lemma Jaccard > 0.7:
  ~74% precision, merges other-value tasks such as Enterprise Server 3.20 / 3.22, misses 35-61% of cosine >= 0.95
  pairs; overlap coefficient > 0.7 chains into one group holding 45% of web random's tasks). Adopted: after the
  exact-text merge, two tasks merge when embedding cosine >= 0.95 AND content-word Jaccard >= 0.7 (stopwords minus
  direction / negation words) AND their values agree (quoted strings, numbers, capitalised words); direct pairs only
  (a leader and the tasks that pass against it, most-sourced first), no chains. Runs between the embed stage and the
  judge on the cached vectors (CPU; no GPU, nothing shared with vLLM). task_coverage / task_diversity: --dedup
  semantic (default) | exact, --dedup_cos, --dedup_jaccard; one embeddings.npz per embedder, semantic results in
  <embedder>/dedup_cos<c>_jac<j>/ (with dedup.jsonl), exact in <embedder>/ as before. First run (embed stage):
  web random 25,382 -> 24,356, web VLM 7,200 -> 7,126, android random 1,244 -> 1,239, android VLM 1,000 -> 993;
  sampled merges are true rewordings. Jaccard blocks translations and synonyms (web random has 5,445 pairs at cosine
  >= 0.95; 1,026 merged). Android task_coverage embed now needs a GPU: AndroidWorld goals at seed 30 changed since
  the cached run (date-dependent), so the benchmark side should come from the split file's fixed texts.
- 2026-11-11 (screen coverage default): --embedder siglip (google/siglip2-base-patch16-224) is the default (user's
  call; it fits the labelled pairs better in every env); random_patch stays as an option.
- 2026-11-11 (task dedup revised): the word-overlap test is dropped (user's call): reading cosine >= 0.95 pairs it
  blocked (~85% were true rewordings / translations) while the values test caught most wrong merges. Rule now:
  cosine >= 0.95 AND values agree, direct pairs only; --dedup_jaccard removed; output dir dedup_cos<c>/. Rerun
  (embed stage): web random 25,382 -> 23,605, web VLM 7,200 -> 7,058, android random 1,244 -> 1,234, android VLM
  1,000 -> 990. Known misses: tasks adding or dropping a part ("monthly price" vs "... and the sign-up page") and
  unquoted named variants (Copilot vs Copilot CLI).
- 2026-11-11 (task judges v2, cusi/eval/judge_prompts.py): the six prompts of plans/task_judge_prompts.md (reviewed
  by the user): one shared scale on mechanics / capabilities (10-9 paraphrase, 8-7 interchangeable for execution,
  6-5 similar but different, 4-3 same area, 2-1 different), per-env coverage prompts with test-set examples and
  internal prompts with made-up-environment examples at 10 / 8 / 6 / 4 / 2; judges see only the two texts. Constants
  COVERED_AT = 7 (coverage@7 replaces coverage@8), SAME_FUNCTION_AT = 7 (functions@7 replaces functions@8), DEDUP_AT
  = 9 (not used yet). No prompt versioning (user's call): judge outputs stay in judge_<model>/.
  task_coverage compares GameBoy trees with their own game's benchmark rows (config "game"; Link's Awakening
  supported). Not done: the five-anchor judge sanity check (still paraphrase vs random), judge-based dedup at >= 9.
- 2026-11-11 (annotation: Android app names): the annotator named other apps' settings "the Settings app" and many
  apps "the app" (audit: 54 / 39 and 49 / 104 tasks, random / VLM android s0 trees), which the judges (shown
  only the two texts) read as a different app. Node and edge notes now get a sentence naming the app on the screen
  (cusi.search.annotate node_app: the foreground package on the step that reached the node, from the env's true
  state, mapped by cusi.agents.specs android_app_label; a root's scene app). An exception to "the annotation prompts
  see only screens", for the app name only.
- 2026-11-11 (annotation: one-action rule dropped): the edge prompt no longer says an outcome one action away gets
  NO TASK. Gemma ignored it anyway (1,209 of 1,346 one-action segments on web random had a task), and most such
  tasks are legitimate (open a page, talk to someone, read a sign). GameBoy screen-relative tasks ("Talk to the
  person to your left") merged across nodes: left as is (user's call).
- 2026-11-11 (annotation: context frames): the node-notes call now also sees the last 3 distinct screens before the
  node (the expansion's earlier steps, preceded by the screen it started from; none for a root; frames are stored
  once per distinct image, so repeats of the node's screen are dropped), and is told to describe only the last
  screen and to write NO TASK when it is blank, loading or mid-transition (parse_node drops NO TASK). Not recorded in
  annotations.json (user's call): the annotation check stays model and max_features only. Live check
  (job 295384, Qwen3-VL-8B fp16, 10 nodes): the 3 darkest Red s3 nodes (black screen, player sprite) were described
  as transitions with no tasks; ordinary battle and Web pages kept their tasks. Caveat: a genuinely dark place (a cave
  without Flash) could look the same.
- 2026-11-11 (GameBoy screen text on for every game): gameboy_spec no longer sets read_text False, so Link's
  Awakening, Crystal and Oracle of Seasons scenes read their screen text from their glyph tables, as Red always did
  (GameBoyPlayEnv's default; GameBoy evals already read text for every game). Link's Awakening's table (73 glyphs,
  labelled by eye earlier, never validated) was checked rather than rebuilt (its harvest work dir was gone):
  `harvest check` on 16 saved states x 80 steps (1,296 frames) read a dialogue box at 130 / 130 text moments; a probe
  of the benchmark start states (talking, START, SELECT; 1,452 frames) read 38 / 38 complete dialogue lines exactly
  (e.g. "Thanks for call- ing! ...Well...", "You've got a Piece of Heart", "YIP YIP!"), with 84 distinct strings in
  all, the rest prefixes captured while text types out. No glyph is missing: the 18 "unknown" cells are keys in the
  table ("0", "!", "1", "L-") rejected by the screen check mid-fade. Not read: sprite text ("PUSH SELECT" on the
  subscreen), single characters (dungeon numbers 1-8; min_chars 2); the save menu and shop dialogue were not
  reached. The HUD (item levels "L-1", rupees "000") reads on nearly every frame. Crystal: dialogue / menu text at
  470 / 527 text moments, 1,966 unknown cells over 19 glyphs. Oracle of Seasons: 1 / 9 text moments (8x16 font;
  "The" reads "lhe"). Effect on LA trees: obs text_boxes filled, the novelty score's text half (region_alpha 0.5)
  active, so new dialogue lines and changed HUD numbers make nodes; annotation prompts get the screen text. The LA
  trees are rerun with it (user's call).
- 2026-11-11 (Link's Awakening trees rerun with screen text): the 13 LA trees (Gemma random / vlm_no_typing s0-s5,
  a partial Qwen 32B s0) ran with read_text off and were deleted with their two screen comparisons (user's call:
  screen text must be in every tree). Resubmitted with text on (tmp_submit_la_text.sh, jobs in
  storage/tmp/la_text_jobs.txt): Gemma random + vlm_no_typing s0-s5 (VLM on 2x A6000, 3 x 23 h), Qwen3-VL-8B s0-s2
  on eclairlg14 and 32B s0-s2 on eclairlg15 as their other trees. Expect bigger trees that saturate later: new
  dialogue lines and HUD changes (rupees, item levels) now count toward the novelty score's text half.
- 2026-11-11 (text coverage; distinct screens): cusi/search/text_coverage.py (+ slurm/text_coverage.sh,
  scripts/slurm/text_coverage.sh, CPU only): every observation's screen text in env-step order (scene starts, then
  each step), items = web element labels ('"a", "b"' counts as two), android element text / content_description (not
  resource_id), gameboy text_boxes lines; NFKC, lowercase, whitespace collapsed, punctuation / symbols / icon-font and
  control characters trimmed; digits kept and no language reporting (user's calls); GameBoy drops items that are a
  strict prefix of a longer item in the same expansion (text typing out). Per run: distinct items, per scene,
  --at_steps; 2 runs: shared / only-A / only-B; 3+: unique to each; most frequent unique items as examples.
  screen_coverage now reports each run's distinct screens too (user reversed the earlier call): a greedy pass over
  its frames in the order seen, new when no kept screen is within tau (overall, and per scene on that scene's frames),
  at tau/2, tau, 2tau; in compare and in a new `distinct` command / --stage distinct.
  Tests (random s0 vs VLM-with-typing s0 for web and android; Red random s3 vs vlm_no_typing s3, the latter still
  running, 7,480 steps):
  text, distinct items (not seen by the other): web 72,146 (65,779) vs 17,726 (11,359), shared 6,367; android 4,436
  (2,498) vs 4,035 (2,097); Red 1,302 (737) vs 913 (348); Red at 7,000 steps each 446 (132) vs 884 (570).
  Numbers-only items (HUD, HP, money) are 19% / 30% of Red's items. Web random's unique items are led by its
  localised pages.
  screens (SigLIP), distinct at tau/2 / tau / 2tau: web 22,316 / 17,097 / 10,253 vs 6,110 / 5,321 / 3,834; android
  2,021 / 1,414 / 917 vs 1,459 / 1,126 / 725; Red 7,446 / 2,818 / 766 vs 1,311 / 748 / 294, and at 7,000 steps each
  1,266 / 643 / 221 vs 1,249 / 716 / 281.
- 2026-11-12 (grouped tree comparison, cusi/search/groups.py): text, screen and state coverage compare groups of runs
  (--groups "name=r1,r2;name=r3,r4"; --runs = each run its own group; a run in one group only). Each run is first
  reduced to its distinct items (text: its set of normalised items; states: its sets per set measure; screens: its
  greedy distinct screens at tau), a group's distinct items are the union over its runs (screens: a greedy pass at tau
  over the runs' pooled distinct screens), and groups are compared: 2 groups both / only-A / only-B (screens: a group's
  screen is matched when any frame of the other group's runs is within tau), 3+ groups unique to each and shared by
  all; each run's own count with the group's mean and sd; per scene; --at_steps per run. Screen --runs compares now go
  through this too (screen-based, no longer frame-weighted unmatched counts). state_coverage became a click group:
  counts (the old per-run command) and compare (sets recomputed from the run's steps each time, no cache); max / count
  measures (badges, party, dex, battles, js_functions) per run with group mean and sd only. New Slurm pair
  slurm/state_coverage.sh + scripts/slurm/state_coverage.sh; text / screen Slurm scripts take --groups.
  Test (jobs 295440-295452): Web random s0-s2 / VLM w-typing s0-s2 / VLM no-typing s0-s2, distinct text 148,045 /
  36,359 / 32,976, screens at tau 40,205 / 12,390 / 11,223; Android text 7,557 / 6,995 / 4,641, screens 3,271 / 2,597
  / 1,482, activity states 566 / 520 / 389; Red random s0-s4 vs VLM no-typing s0, s1, s3, s4 (s0, s1 still rerunning,
  s3, s4 still running) text 3,164 / 1,252, screens 5,568 / 1,333 (VLM screens matched by random 1,147), tiles 3,622 /
  638. Group distinct >= each of its runs; single-run counts match the earlier per-run ones.
- 2026-11-12 (Gemma Link's Awakening VLM trees on Quadros): the six Gemma-4-31B LA vlm_no_typing s0-s5 trees (text on)
  waited for A6000 pairs; moved to Quadro RTX 8000 pairs in fp16 (user's call; tmp_la_quadro.sh, jobs in
  storage/tmp/la_quadro_jobs.txt). Caveat: the other Gemma trees ran bf16 on A6000s, so these differ in precision.
  FAILED: every segment died at vLLM startup (cudagraph memory profiling) with Triton OutOfResources: shared memory,
  required 98304, hardware limit 65536 (Turing). The kernel itself is too big, so --enforce-eager would not help:
  Gemma-4-31B does not run on Quadro RTX 8000s with this vLLM (0.30.0). Resubmitted unchanged to A6000 pairs, bf16
  (tmp_la_a6000.sh, jobs in storage/tmp/la_a6000_jobs.txt); no tree data was lost (none had started).
- 2026-11-12 (Qwen trees unpinned; 4B added): user's call. 32B non-Android chains: any 2 GPUs except lg10 (2 x 24 GB
  A5000 can't hold 32B fp16), via scontrol update (ReqNodeList cleared, TresPerNode gres/gpu:2, ExcNodeList eclairlg10);
  the 7 unstarted ones took the 14 idle Quadros (lg09, lg11-13) at once. 8B: the 6 unstarted non-Android long-lg jobs
  could only ever run on lg14 (long-lg's only Quadro node), so they were cancelled and resubmitted as 4 x 23 h
  medium-lg chains on any Quadro (ports kept). 4B (new): Red, Link's Awakening, Web, Android x s0-s2, same settings
  (tmp_submit_qwen_quadro.sh, jobs in storage/tmp/qwen_quadro_jobs.txt). Android stays one per node (search trees'
  emulator is always port 5554): 8B on lg14, 32B on lg15, 4B on lg11, each a singleton. Budget per tree: 69 h
  exploration wall-clock (--budget_metric wall_s --budget 248400); ~3,100 GPU-hours left over the three models.
- 2026-11-12 (Qwen 32B set aside): user's call, so 8B and 4B get the Quadros. All 47 32B jobs cancelled
  (storage/tmp/qwen32_cancel_ids.txt), including the running Red s0 (segment 2) and Web s0 (~21 h) on lg15; their run
  dirs are kept and resume with --overwrite false. The 7 started minutes earlier have ~1 min of data.
- 2026-11-12 (4B Android unblocked): the 4B Android chain (96 GB) could not fit on lg11 next to three 64 GB trees (64 GB
  left), so 4B Web s0 (~2 min of data) was cancelled and resubmitted excluding lg11 (tmp_move_4b_web.sh, 295770-3, now
  on lg12); 4B Android s0 (295729) started on lg11. 96 GB kept: the 8B Android tree peaked at ~96 GB RSS.
- 2026-11-12 (oracle lookahead -> GameBoyWorlds peek): cusi/world_action/oracle_lookahead.py now takes each button's
  candidate frame from GameBoyWorlds' env.peek_high_level_action(LowLevelAction, low_level_action=...) instead of its
  own PyBoy press/release copy. Why: on the current emulator (37b80e3) Pokemon's step adds screen-fade waits, the
  options-menu nudge, naming-screen exit and dialogue skip, which the copy missed. Live check (tmp_oracle_check.py,
  60 random steps from Red task 0's state): the old copy's chosen candidate != real next screen on 14/60 steps, peek
  on 0/60; Link's Awakening: no differences; peek leaves step_count unchanged. The earlier oracle runs (*_rep1,
  *_fixparse) started before the 37b80e3 pull and show pred_pixel_error 0.00 on Red, so were not hit by this.
- 2026-11-13 (how much of vlm_w_typing's novelty is typing; tmp_typing_attr.py, job 295790, out
  storage/tmp/typing_attr_2026-11-13/{android,web}.md): leave-one-out over vlm_no_typing s0-s2, base_i = random
  s0-s2 + the other two no-typing trees (full trees); each w_typing tree's (s0-s2) exclusive items are attributed to
  the step that first reached them: typing step / later in an expansion that typed / no typing. Random never types;
  no_typing's 13 Web Type attempts were blocked. Typing steps are 6% of w_typing steps (after typing 24-26%).
  Web (one more no-typing tree vs a w_typing tree; typing / after / none): URLs 1,103 vs 1,655 (29/28/43%); routes
  324 vs 369 (17/32/52%); skeletons 249 vs 402 (20/30/51%); text 5,586 vs 7,432 (19/28/53%); screens 2,541 vs 3,199
  (13/26/61%). The no-typing part alone is below one more no-typing tree on every Web measure, so W's Web excess is
  typing-related. Android: activities 7 vs 13 (0/32/68%); activity states 38 vs 65 (1/24/75%); text 465 vs 1,199
  (34/30/36%); screens 249 vs 615 (22/36/42%). Android's extra activities / activity states are mostly NOT typing
  (no-typing part 9 vs 7, 49 vs 38); its extra text is mostly typing, and the top typing items are keyboard
  autocomplete fragments ('navigatio', 'setting so', 'barbara'), i.e. inflated.
- 2026-11-13 (oracle WM feasibility, Web / Android; tmp_web_typing_tasks.py): Web test + test_env (204 tasks), Gemma
  base 3 reps: 189 tasks typed in >=1 rep (182 in every rep); of 153 tasks solved at least once, 133 typed in every
  successful episode. 1,271 Type steps: new URL holds the typed text 48%, URL unchanged 29%, new URL without the text
  23%. Google Flights (42 tasks, 564 Type steps): URL unchanged 55% (form fields typed before search), rest the
  encoded tfs URL; Huggingface 42% unchanged (in-page filters); Booking 22%; ArXiv / GitHub / Coursera / ESPN /
  Google Map / Wolfram mostly GET URLs. Restore costs from sat_vlm_w_typing_s0: Web load_state (new browser, URL +
  cookies + storage + scroll) median 7.2 s; Android snapshot load ("cached") median 13.9 s, replay median 73 s.
  Android test + test_env (55): ~7 read-only (Q&A / open app); the rest write app data, settings, files, SMS,
  camera or timers, which Back does not undo.
- 2026-11-13 (peek: lookahead on every env without stepping it): TextActionEnv.peek(actions=[...]) in cusi/envs/base.py
  returns each candidate's step result with the env put back (episode bookkeeping in _peek_keep restored; no step
  counted); info peek_restored / peek_seconds, and on envs where a peek can be inexact peek_synced / peek_failed.
  Default _peek_impl: save once, step + load per candidate. Web (cusi/envs/webvoyager.py): load_state is lossy for
  in-page state (typed fields, pickers, filters), so each candidate runs in a second, sandbox Chrome synced by
  replaying the episode (from the last new-document snapshot, else from the start, else URL + cookies + storage),
  checked against the main browser (same page + identical element rows = peek_synced); the main browser is only
  read (checked: 355/355 states read-only commands). Element indices are remapped by (tag, text, position); a
  candidate whose element is missing in the sandbox returns peek_failed rather than clicking another element. Live
  test (tmp_peek_web.py, storage/tmp/peek_web/run5; 12 test tasks, 120 states, 533 candidates, eval waits): synced
  77%; peek vs the real step: same page 92%, same element text 80%, text + pixels 70%; two peeks of the same action
  agree on text 82%, so most misses are the live site differing between loads (Booking results / language, Google
  Maps ads and viewport, ESPN live scores, Wolfram banner, Apple carousel). Per site (synced / text match): Google
  Flights 1.00/1.00 (forms, date picker, pushState), GitHub 1.00/1.00, ArXiv .93/1.00, Apple .90/1.00, Wolfram
  .98/.80, Huggingface .72/.95, Coursera .90/.75, Google Map .83/.75, Booking .39/.50, ESPN .15/.35 (ESPN reloads
  itself). peek_failed 12/533. Cost: 90 s mean / 76 s median per candidate (max 337 s), growing with episode length
  (in-page-URL sites and GoBack replay the whole episode). Failed: on an overloaded node (24 Chromes / 16 CPUs) the
  sandbox's Wolfram open hit the 120 s WebDriver timeout (12-14 peek_failed per run).
- 2026-11-13 (Android peek, cusi/envs/android_world.py): snapshot-based _peek_impl (save once, step + load per
  candidate), _peek_keep = (_steps, _at_initial_state), interaction_cache put back, raises on an emulator restart, and
  a device fingerprint check (peek_verify; UI tree minus status bar, activity + fragments, global/secure/system
  settings, md5 of task + foreground apps' /data/data, shared storage, clock, is_successful). Known post-load system
  writes (Play services boot counter, launcher DBs, WebView metrics, Google telemetry stores) are PEEK_LOAD_NOISE.
  base.peek now restores _peek_keep in a finally. Live test (tmp_peek_android.py, results/peek_android/; 10
  state-changing test tasks, runs 3-6: 129 states, 499 candidates, scripted Save/Delete/toggle + 2 random + status
  complete): activity, settings, files and is_successful restored 100%; UI tree 91-100% (misses are elapsed-time text:
  stopwatch, Contacts time label); main state unchanged 100%; peeked vs real activity 97-100%, reward / valid 100%,
  elements 87-100%; status reward = is_successful 21/21. Cost: ~21-27 s per candidate (save 3.6-3.9 s once, step 7-8.5
  s, load 11-15 s median, check ~3 s), 3.6 GB per snapshot (one at a time). The load puts the device clock back (shift
  median 6-7 s over ~100 s peeks; eclairlg09 7-94 s ahead). Fixed in the shared load path (also trees / resets):
  _wait_until_restored ignores the keyboard package (every load after input_text timed out at 120 s) and re-sends the
  a11y kick every 10 s. Not fixed: (1) emulator segfault on snapshot load, ~2% of loads (host GL renderer errors);
  (2) after a load a 1 s long press registers as a tap (4-10 of 6-19 trials vs 0/9 without a load; 3 s or motionevent
  works) - opt-in long_press_ms (3000 makes MarkorDeleteNote's peek match), not on by default since it changes
  actuation vs stock AndroidWorld and also affects tree search loads; (3) after a load GPU-texture views go blank on
  redraw (home 10.6%, Markor editor 11%, Pro Expense form up to 91% of pixels) while the UI tree is right;
  swiftshader_indirect worse (99.5%), guest -> lavapipe same, SurfaceFlinger recompose no effect. Screen pixels
  restored 56-82%.
- 2026-11-13 (Android long press, user's call): peek holds long presses PEEK_LONG_PRESS_MS = 3000 (at least) during the
  peek only; real steps, tree search and eval keep AndroidWorld's 1 s (long_press_ms=None).
- 2026-11-13 (oracle WM executor v1, "oracle_preview", cusi/world_action/oracle_preview.py): why the GameBoy oracle
  lookahead did not beat base (oracle_red_fixparse 10/27): its picking prompt has no history (the base's visual
  history block, plan step context) and asks the VLM to compare 7 near-identical frames; it diverged from the shadow
  base on 28% of decisions. New design, same for every env: the base executor proposes with its own prompt and memory
  (policy hook; Executor.feedback: GameBoy's error block, WebVoyager's chat, M3A's history line), the env's
  exact peek (TextActionEnv.peek; GameBoyPlayEnv._peek_impl added via peek_high_level_action: 30/30 peeks == real
  next screen) previews it, proposals the env refuses or that change nothing are rejected outright, else a check call
  (screen now, screen after, the action and reasoning, Web URL / Android activity) says good / bad; bad -> feedback
  with the previewed outcome and reason, re-decide; at most 3 proposals, the last one executed. Finishes are never
  previewed (a peek would leak the reward). Web/Android evals take --world_action (Android emulators start with
  snapshots). Dev sets (storage/tmp/wam_dev/): web_dev2 40 test tasks (30 mixed across the 3 base reps, 5 never, 5
  always; base 30/14/16 per rep), android_dev2 26 (4 mixed, 16 never, 6 always; base 10/6). Round 1 jobs: GameBoy Red
  49 preview_red_v1 (295980), Web preview_web_dev2_v1 (295981) with a same-day base base_web_dev2_r1 (295982), Android
  preview_aw_dev2_v1 (295983).
- 2026-11-13 (oracle_preview round 1 failed, fixed): (1) bug: the default feedback hook called self.decide, which on
  the world action executor plans again, so every revision recursed (GameBoy task 004: 5 proposals and 5 identical
  checks for one decision); now Executor.feedback() is a context manager around base.decide. (2) GameBoy's
  executor has a `_policy` attribute, which my policy method shadowed (every task TypeError). (3) Android
  ClockTimerEntry: the digit pad fills from the right, so after 1, 6 the click on 3 shows "00h 01m 63s"; the check
  called that intermediate screen bad, the agent backspaced and looped, failed. The check prompt now judges an action
  as one step of the agent's stated plan (an odd-looking unfinished screen is good; bad = the action did something
  other than intended, an error, undoing progress). Rounds 1 runs cancelled; same-day bases: base_web_dev2_r1 27/40.
  Round 2 (v2): preview_red_v2 295987, preview_web_dev2_v2 295988, preview_aw_dev2_v2 295989, base_aw_dev2_r1 295990.
  Android preview is slow: snapshot loads retry ("device did not come back", "UI tree did not catch up") and the
  emulator crashed on one load (task error).
- 2026-11-13 (oracle_preview v3, GameBoy): v2 partial (21/49 Red tasks before I stopped it): 17/21 vs 13 and 15 for
  the two new-emulator base reps on the same tasks (8-13 for older reps). Its 4 failures were battle / long tasks
  where the preview kept rejecting A: during battle text and animations a press shows no change yet, so it was
  auto-rejected as no-change or judged bad (Defeat Blue: 143 rejected A presses), stalling the agent. v3: no-change
  auto-reject only outside GameBoy dialogue / menu / battle (GameBoyPlayEnv peek info["agent_state"]); in those
  states the check is told repeated A presses are normal; a proposal the executor makes again after its rejection is
  kept ("insisted"). Run preview_red_v3 (296021, 10 workers). Web v2 check verdicts look sensible (wrong element /
  date / page); Android v2 7 done: 3/7 = base 3/7 on the same tasks; BrowserDraw's 34 no-change rejections were real
  (canvas clicks do nothing; base fails it too).
- 2026-11-13 (GameBoy v3 placement): all A6000s taken (another user's higher-priority arrays took lg03's pair; my
  CPU-only jobs pinned to lg06 stayed PENDING (Priority) even at 3 h). So preview_red_v3 runs as a step inside the Web
  preview job 295988 on lg06, using its localhost vLLM (port 8488). Failed first: `srun --jobid --overlap` without
  -n1 launched 3 copies (the job has 3 tasks) that raced on one run dir (004 twice, frames.zip.tmp FileNotFoundError);
  killed, relaunched with -n1 --overwrite. The step dies when the Web job ends; resume elsewhere then.
- 2026-11-13 (Web v2, Wolfram blocked): 14 of web_dev2's 40 tasks are Wolfram Alpha (base 14/14 the same day). In
  preview_web_dev2_v2 they end with 0 iterations (opening wolframalpha.com times out). From lg06 curl cannot connect
  to wolframalpha.com (exit 7 after 21 s) while the login node gets 200 in 0.25 s: lg06 looks blocked / rate-limited,
  plausibly by the preview's extra traffic (each peek replays the episode in a sandbox browser). A live-site hazard of
  peeking. v2 is compared without Wolfram; its Wolfram tasks are rerun from another node.
- 2026-11-13 (v4 check rubric; GPU holding): Android v2 = base on all 9 shared tasks so far (3/9 each).
  ClockTimerEntry v2 still failed: the check called "00h 01m 63s" (correct right-filling digit pad) a mistake and
  later "1 went to seconds" bad, steering the agent to a wrong theory (digits in reverse order). v4 rubric: bad only
  if image 2 shows the wrong target hit, an error / refusal, an unrelated place, or progress destroyed; never for a
  guess about later app behaviour or a different plan. New Slurm pair eval_server (slurm/ + scripts/slurm/): holds a
  GPU pair as one vLLM server and runs eval commands dropped in $tmp_dir/eval_queue/<job>/ (eval.sh --vllm_port
  $VLLM_PORT --keep_vllm true; eval.sh gained --keep_vllm), because released A6000 pairs go to another user's
  higher-priority arrays. Two submitted (296027, 296028).
- 2026-11-13 (failure: edited a running bash script): adding --keep_vllm to scripts/slurm/eval.sh while 295988 was
  running broke it: bash reads a script as it runs, so after the agent runs it hit "syntax error near unexpected token
  done" (line 141) and never judged preview_web_dev2_v2 (agent runs complete, unscored; judge later with
  --judge_only). Never edit a bash script that running jobs execute; copy it under a new name instead. The GameBoy v3
  step inside that job died with it at 15/49 tasks (resume). Android base base_aw_dev2_r1 finished all 26 (8/26).
- 2026-11-13 (Android preview: long press after a peek): MarkorDeleteNote (base 1, preview v2 0): the peek's long
  press (held 3 s) previewed "note selected, delete toolbar", but the real step that followed the peek's snapshot load
  used AndroidWorld's 1 s press, which after a load registers as a tap: the note opened instead, the agent then hunted
  for delete inside the editor and failed. Fix: AndroidPlayEnv._after_peek (set when a peek ends, cleared by the next
  executed action): a real long press right after a peek is held PEEK_LONG_PRESS_MS too, so the step does what its
  preview showed. Steps not preceded by a peek (base runs, trees) keep 1 s. Android v2 so far: 4/12 vs base 5/12.
- 2026-11-13 (piggybacking on tree jobs' vLLM, user's suggestion): no free A6000s for ~3 h (eval_server jobs
  296027/8 cancelled while still pending). Each Gemma tree job serves on localhost:8100 + (job id mod 800); an
  `srun --jobid=<tree job> --overlap -n1 --cpus-per-task=8` step on its node reaches it (eval.sh --vllm_port <port>
  --keep_vllm true). Steps are capped by the tree job's own 8 CPUs / 64 GB (asking 12 CPUs left the step waiting:
  "step creation still disabled") and die when it ends. Placement: lg03 294855 (port 8555) GameBoy v3 resume then the
  Web v2 judge; lg07 294849 (8549) Web v4, 3 workers (lg16's and lg06's Web trees are blocked by wolframalpha.com;
  lg07 and lg03 reach it); lg18 294840 (8540) Android v4, 2 emulators (no Android job on lg18; lg03's port 8555
  would clash with the 2nd emulator's gRPC port).
- 2026-11-13 (GameBoy v5): v3 so far 13/15 scored vs 10-12 for base reps on the same tasks, but three long
  free-roam tasks (005 aide, 016 map, 012 potion) hit 176 steps with ~59% of decisions revised and 75-103 no-change
  rejections each: the "insist" rule let the agent re-propose a blocked UP / A and execute it anyway. v5: insisting
  overrides only the check's judgement; a repeated refused / no-change action stays rejected (not previewed again);
  on a GameBoy no-change rejection the feedback names the buttons that would change the screen (all 7 peeked, exact
  and instant). preview_red_v5 runs on lg18 inside tree job 295055 (port 8755).
- 2026-11-13 (Android v4 on a shared 8-CPU step): with 2 emulators inside tree job 294840's 8 CPUs, 3 of the first 4
  tasks errored (emulator hung and restarted x2, "Could not get a11y tree" x1). Restarted with 1 emulator (run dir
  preview_aw_dev2_v4, the 2-emulator shards' rows are in preview_aw_dev2_v4_s0/_s1).
- 2026-11-13 (FAILURE, my fault: killed a tree job): running `eval.sh --judge_only true` as a step inside tree job
  294855 (gameboy sat_vlm_no_typing_s4) stopped the tree's own vLLM server: eval.sh's judge branch always restarts the
  server when judge_only is set (`judge_model != model || judge_only`), ignoring --keep_vllm, and the restart then
  failed inside the step. The tree's search died on connection errors (FAILED at 23:03, 13.6 h in). Resubmitted the
  same command without its old dependency (job 296076, --overwrite false, resumes). Fix later (not while my steps run
  eval.sh): with --keep_vllm, never stop / restart the server for the judge when judge_model == model. Until then, run
  judges by calling run_eval.py web --judge_* directly, not eval.sh --judge_only, in piggyback steps. GameBoy v3 was
  stopped on purpose just before (v5 supersedes it; v3 ended at 13 successes over its scored tasks).
- 2026-11-13 (Web v2 scored; Gemma judge, judged by calling run_eval.py web --judge_* directly in a lg07 step):
  preview_web_dev2_v2 21/40 overall (its 14 Wolfram tasks failed to open: lg06 blocked), base_web_dev2_r1 27/40. On
  the 26 non-Wolfram tasks: preview 18 vs base 13 (+6 Apple--8, ArXiv--15, Booking--5, Booking--8, GitHub--9, Google
  Map--33; -1 Google Flights--21). Android v2 so far on scored tasks 8/19 vs base 6/19 (+3 RetroPlayingQueue,
  SportsTrackerTotalDistance..., VlcCreatePlaylist; -1 MarkorDeleteNote, the long-press bug fixed for v4; 3 emulator
  errors). GameBoy v5 12/14 so far.
- 2026-11-13 (Android v4 placement failed): inside tree job 294840 (lg18) Android v4 errored on all 3 tasks run with 1
  emulator ("emulator hung and was restarted"): the step only has the job's 8 CPUs, and the tree's two vLLM servers
  keep ~6 busy, starving the emulator (qemu 41%). Idea of a CPU-only Android job on an idle medium node (eclair12:
  kvm, 40 CPUs) talking to the tree's vLLM through a TCP forwarder (tmp_tcp_forward.py) was blocked by the permission
  classifier (it exposes an unauthenticated endpoint on a node port); not pursued. Android v4 queued as its own eval
  job 296078 (1 emulator, resumes preview_aw_dev2_v4). Android needs its own allocation (16 CPUs) to be stable.
- 2026-11-14 (tree resubmit failed, then fixed): 296076 (the resubmit of the killed gameboy sat_vlm_no_typing_s4 tree)
  got lg03's GPUs at 00:15 and failed in 6 s ("Unknown argument: 50"): I rebuilt the command from sacct's SubmitLine,
  which drops the quotes around --extra "...". Resubmitted with the args from the original log's "Active variables"
  (job 296093). Lesson: rebuild a resubmit from the job log's parsed args, not sacct's SubmitLine. Its GPU pair went to
  the Android v4 eval (296078, lg03) meanwhile.
- 2026-11-14 (user): no cross-node forwarding of a job's vLLM server (the CPU-node Android idea is dropped for good);
  tmp_tcp_forward.py is unused.
- 2026-11-14 (GameBoy v6, battle rollouts): v5 so far 17/23 (base new-emu reps 13 / 16 on the same tasks); failures
  are battles (Lass, Brock, Blue: preview revised 0-9% of decisions; one press only moves a cursor, so a one-step
  preview shows nothing) and long navigation (bike 014, secret switch 018). v6: GameBoyPlayEnv.peek_rollout(action,
  follow="B"): the press then B (advances battle text, never commits a menu choice) until the base battle menu or the
  battle ends, emulator restored as peek_high_level_action does (live: restored exactly; a real replay of A then Bs
  reproduced the rollout's 7 screens). oracle_preview: an A in a battle away from the base menu is previewed by the
  rollout (up to 5 screens incl. the last) and the check judges the move's effect. preview_red_v6 runs on lg18
  inside tree job 294840 (port 8540).
- 2026-11-14 (Android v2 final): preview_aw_dev2_v2 26/26 run, 23 scored (3 emulator hangs: OsmAndTrack,
  MarkorTranscribeVideo, RecipeAddMultipleRecipes): 9/23 vs base_aw_dev2_r1 7/23 on the same tasks (+3
  RetroPlayingQueue, SportsTrackerTotalDistanceForCategoryOverInterval, VlcCreatePlaylist; -1 MarkorDeleteNote,
  long-press-after-peek bug fixed in v4). Base overall 8/26.
- 2026-11-14 (Web v5, sandbox browser reuse): lg07 got blocked by wolframalpha.com too (curl exit 7; login node 200)
  after ~10 Wolfram tasks of preview_web_dev2_v4, whose later Wolfram tasks then ran 0 iterations. Each sync of the
  sandbox started a new Chrome with a fresh profile and reloaded the site: cold sessions in bulk. v5
  (WebVoyagerPlayEnv._fresh_browser): the sandbox's live browser is reset instead (cookies + the current origin's
  storage cleared, about:blank, HTTP cache kept); a new Chrome only if it is dead or a full replay contains a GoBack.
  preview_web_dev2_v5 runs on lg10 inside GameBoy tree job 295078 (port 8778; lg10 reaches Wolfram).
- 2026-11-14 (host moves): tree job 294840 (lg18) hit its 23 h limit at 01:49 and took GameBoy v6's step with it;
  295055 (GameBoy v5's host) was ~25 min from its limit. Both resumed (the run config has no server URL, so a resume on
  another server is valid; errored tasks rerun): v5 on lg10 inside 295081 (port 8781), v6 on lg16 inside my
  resubmitted tree 296093 (port 8193). Avoided lg17 (an Android tree: 8 shared CPUs would starve its emulator).
  Web v5 first previews: 25 s mean vs 99 s in v4 (browser reuse).
- 2026-11-14 (Web v4 scored): preview_web_dev2_v4 19/40 (8 Wolfram tasks never opened: lg07 blocked). Non-Wolfram 26:
  v4 13 = base 13 (v2 18). The two previews intervened at the same rate on those tasks (revised 37 vs 38 decisions;
  bad verdicts 48 vs 43; no-change rejections 30 vs 15), so v2 - v4 = 5 is mostly noise: the Web executor samples at
  temperature 1.0 on live sites (the three historical base reps on these 40 tasks: 30 / 14 / 16). A second same-day
  base on the 26 non-Wolfram tasks (base_web_dev2nw_r2, lg07 step) to size the noise. Wolfram tasks that ran in v4:
  6/6 = base.
- 2026-11-14 (Web noise check): second same-day base on the 26 non-Wolfram dev tasks (base_web_dev2nw_r2): 16/26.
  Base 13 / 16 vs preview v2 18 / v4 13 on the same 26: means 14.5 vs 15.5, within run-to-run noise (temperature-1.0
  executor on live sites). Web gain from oracle_preview so far is not established; v2's +5 was mostly noise.
- 2026-11-14 (Web select mode, v6-v9): new executor hooks Executor.alternatives(n) / adopt(response) (WebVoyager
  implements them) and an oracle_preview select mode: the greedy base decision plus up to 2 alternatives, each
  previewed in the sandbox, and one call (PREVIEW_SELECT) picks the candidate whose previewed screen brings the agent
  closest; the chosen reply is adopted into the chat. v6: no alternatives at all (eval decoding is greedy, so the
  executor's temperature is ignored); v7: AgentVLM.chat(sample=True) at temperature 1: the model rewords its thought
  but keeps the same action (3/3 identical actions per decision); also fixed: the fallback loop now names its own
  call as the step's owner (record["owner_call"]). v8: alternatives asked for explicitly (ALTERNATIVE_ASK: "a
  different action, other than X, that could also make progress", appended to a copy of the chat, never to the chat):
  3 distinct candidates per decision, the select call reasons about each preview; but the base was always Image 2 and
  always picked. v9: candidates shown in a shuffled order (sorted by the action's hash). preview_web_dev2nw_v9 on the
  26 non-Wolfram tasks (lg07 step); compare with base r1 13 / r2 16, preview v2 18 / v4 13.
- 2026-11-14 (GameBoy v7): v6 lost 013 (use a not very effective attack on Seaking; v5 solved it) and 012 (potion).
  In 013 the rollout check was right (16 bad of 83: "Mega Drain is super effective on Seaking"), but the agent
  re-proposed Mega Drain and the insist rule let it through (Seaking fainted: task impossible), and the check also
  rejected A presses that only advanced text, judging them by an outcome already decided. v7: rollouts only for an A
  that commits a battle choice (parser.is_in_fight_options_menu / is_in_fight_bag / is_in_pokemon_menu); a rejection
  backed by a rollout cannot be overridden by insisting. preview_red_v7 on lg06 inside tree job 294864 (port 8564).
- 2026-11-14 (GameBoy v5 FINAL, Red 50 tasks): preview_red_v5 28/50, vs base new-emulator reps 23 and 25, older
  greedy reps 15-22 (all on the same tasks): the best GameBoy executor so far. On the 27 tasks the old oracle lookahead
  (oracle_red_fixparse, VLM picks among 7 next frames, no history) ran: it 10, v5 19. Failures are tasks every base rep
  fails (Lass and Brock battles, secret switch, Seafoam, Pallet without Surf, sell TM, ghost, minigame) plus a few
  coin-flips (Defeat Blue, bike).
- 2026-11-14 (GameBoy replicate): second run of the current code (v7) on lg10 inside 295081 (port 8781):
  preview_red_v7_rep2, to size run-to-run noise (base reps range 15-25). (A run first started as "v5_rep2" was
  stopped within seconds: the model code is v7 now, so that name would have been wrong.)
- 2026-11-14 (GameBoy v8, walk previews): v7 lost 000 / 005 (v5 solved them) in no-change loops: the agent re-proposed
  a blocked UP three times despite "buttons that would change the screen: DOWN, LEFT, RIGHT", the third executed. v8:
  the no-change feedback adds how far each open direction walks in a straight line ("UP 9 tiles+; DOWN 5 tiles;
  RIGHT 2 tiles; blocked: ..."), from GameBoyPlayEnv.peek_rollout(until="blocked") counting the player's RAM
  map / x / y (counting changed frames overcounted: animated tiles change the screen while blocked). Live: env
  restored exactly. v6 stopped at 19/32 (superseded by v7); preview_red_v8 on lg16 inside 296093 (port 8193).
- 2026-11-14 (GameBoy preview runs stopped, user's call): v7 13/20, v7_rep2 16/19, v8 10/12 at stop (base new-emu
  reps on the same tasks 12/14, 12/14, 10/9); v5 complete 28/50 remains the GameBoy result. Tree jobs untouched.
  State the GameBoy oracle used: the emulator itself (PyBoy save/load in memory: next screen per action, exact; v6+
  rollouts of several presses), the controller's "action unavailable", screen-parsed signals (GameBoyRL parser:
  agent state FREE_ROAM / DIALOGUE / MENU / BATTLE, battle-menu predicates, MoveSteps movement judge), and from v8 the
  player's map / x / y from RAM (walk distances). Never: reward, the test tracker, success, or other RAM.
- 2026-11-14 (RAM position removed from the GameBoy oracle, user's call: keep it only if it proves its worth): on the
  13 tasks v8 and both v7 runs finished, v8 (walk distances from RAM map / x / y) 11 vs v7 10 and 11; the one
  disagreement (000 Pokemon Center) also split the two v7 runs; the feedback fired on 55 proposals. No measurable
  gain, so removed (peek_rollout's until="blocked" and the walk feedback). The GameBoy oracle now reads no RAM: next
  screens from emulator peeks / battle rollouts, the controller's unavailability, and screen-parsed state only.
- 2026-11-14 (Web interim, judged on copies of the finished tasks: storage/tmp/web_interim/, tmp_web_interim.py): on
  the 26 non-Wolfram dev tasks: base 13 / 16; check-and-reject previews v2 18, v4 13, v5 15 (mean 15.3 vs 14.5).
  Best-of-3 select (v9) on its first 11 tasks: 8 vs base 5 / 8, v2 9, v4 4. Web v5's 4 finished Wolfram tasks: 4/4 =
  base, none failed to open (sandbox browser reuse). No Web gain established yet; differences are within the
  live-site noise.
- 2026-11-14 (Web push, user: "focus all efforts on WebVoyager, reach as deep into state information as needed, max
  GPU"): deeper state: the Web peek returns the previewed page's whole text (WebVoyagerPlayEnv.page_text():
  document.body.innerText, below the fold too, 6000 chars), shown to the check (2500 chars) and, per candidate, to the
  select call (1500 chars). Spec oracle_preview_check = check mode only (no select). Larger dev set web_dev3: 60
  non-Wolfram test tasks (all 37 mixed across the 3 historical base reps, 15 never, 8 always; historical base 31 / 29
  / 26). Runs (tree-job steps, 3 workers): base_web_dev3_r1 (lg06 294864), base_web_dev3_r2 (lg16 294869),
  preview_web_dev3_sel10 (select + page text, lg10 295081), preview_web_dev3_chk10 (check + page text, lg16
  296093); queued own-GPU jobs base_web_dev3_r3 (296114) and preview_web_dev3_sel10_rep2 (296115). Android trees'
  nodes avoided (their emulators share the 8 CPUs).
- 2026-11-14 (Wolfram blocks lg10 too): Web v5 (sandbox browser reuse) finished its 26 non-Wolfram tasks and 4
  Wolfram tasks fine, then lg10 got blocked by wolframalpha.com as well (0-iteration tasks from 06:53). Reuse delays
  the block, does not prevent it: previewing on Wolfram is not viable from a single node. v5 stopped (its non-Wolfram
  results stand); its slot runs preview_web_dev3_chk10_rep2.
- 2026-11-14 (hosts expiring): tree jobs 294849 (lg07), 294864 (lg06), 294869 (lg16) hit their 23 h limits ~08:30:
  Web v9 died at 17/26 (dev2nw), base_web_dev3_r1's judge was cut (agent runs complete; re-judged by run_eval.py
  directly on lg18 inside 295056, port 8756), an interim-judge step failed with them. dev3 so far: base r2 26/60, r3
  27/60. Own-GPU jobs (sel10_rep2 lg05, sel10_rep3 lg04, chk10_rep3 lg06) and steps on lg10 (sel10, chk10_rep2; ~6.7
  h left) and lg16 296093 (chk10; ~14.5 h left) continue.
- 2026-11-14 (Web dev3 interim): base 28 / 26 / 27 of 60 (three same-day reps, steady). Finished preview tasks judged
  on copies: check + page text (v10 chk, 3 runs) 34 solved over 50 task-runs vs the base mean 27.7 on the same tasks
  (+23%; each run >= base mean); select + page text (v10 sel, 3 runs) 19 over 32 vs 17.3. Early (Apple / ArXiv /
  Booking first in the list); check mode looks better and is cheaper.
- 2026-11-14 (Web check-mode losses; v11 = check + page text + today's date): of the interim check-mode tasks, losses
  vs base (base >= 2/3 solved, preview failed): Coursera--25, GitHub--11, Booking--2 (x2), Booking--35, GitHub--9;
  gains (base 0/3, preview solved): Booking--8, Coursera--40, GitHub--19, Google Flights--10. GitHub--9 ("created in
  the last week"): the agent searched created:>2025-05-15 (today is 2026-11-14) and the check approved it, not knowing
  the date. v11: check / select prompts carry today's date (Web only; an Android device keeps its own clock).
  preview_web_dev3_chk11 runs as a second step inside own-GPU job 296117 (lg04, port 8217; that job's own run is
  sel10_rep3); a fourth chk10 run (chk10_rep4) on lg18 inside 295056.
- 2026-11-14 (Android v4 final): preview_aw_dev2_v4 26 run, 20 scored, 6 emulator errors (hangs on snapshot loads):
  9/20 vs base 8/20 on the same tasks (+1 TasksDueNextWeek, no losses; MarkorDeleteNote now solved, the long-press fix
  works). With v2 (9/23 vs 7/23): the Android preview gains 1-3 tasks per 20-23 and never lost a task that was not a
  bug; its cost is ~20-50 min per task and 13-26% of tasks lost to emulator hangs.
- 2026-11-14 (Web dev3, first complete preview run): preview_web_dev3_chk10_rep3 (check mode + page text) 36/60 vs
  base 28 / 26 / 27 (mean 27): +9 tasks (+33%), above the best base rep by 8.
- 2026-11-14 (Web dev3 pooled, v12 = unsynced previews not judged): finals chk10_rep3 36/60, chk11_rep2 34/60 (base
  mean 27). Pooled with interim tasks, vs the base mean on the same tasks: chk10 95/182 (52%) vs 45%, chk11 56/101
  (55%) vs 45%, sel10 56/107 (52%) vs 46%. Select costs ~3 previews per step and is no better: check mode stays. One
  rep (chk10_rep4, 32 tasks) is below base (-1.7); live-site noise per run is large. Largest check-mode losses:
  ESPN--20 2/6 vs 3/3 (live stats, wrong table picked), Booking--2 1/6 vs 2/3, Coursera--4 / GitHub--9 / GitHub--11
  4/6 vs 3/3. In two final runs (1545 previewed proposals), 18% of previews were unsynced (the sandbox did not reach
  the live page's elements; in-page state such as filled inputs or an open calendar lost) and 4% failed. The check
  rejected 43% of unsynced previews vs 22% of synced ones, mostly for "progress destroyed" that was the sandbox's loss
  (Done "cleared the dates", typing "cleared the departure date", a scroll "closed the menu"). v12: an unsynced
  preview is not judged (the action executes, kept = "unsynced preview"); select drops unsynced candidates. Runs:
  preview_web_dev3_chk12 + _rep2.._rep6, sel12 + _rep2 (own 2-GPU jobs 296151-296158).
- 2026-11-14 (wrap-up, user's call): v12 runs cancelled before results; v12 is not in the code. Final oracle_preview =
  check mode only (Web v11, GameBoy v7 without RAM, Android v4); select mode and its executor hooks
  (alternatives / adopt / sampled chat) removed, spec oracle_preview_check folded into oracle_preview. All versions and
  results: plans/oracle_preview_versions.md.
- 2026-11-14 (Web chk10 final): preview_web_dev3_chk10 27/60 (= base mean 27). Complete check runs: 36, 34, 27; pooled
  with interim tasks 158/295 (54%) vs base 46% on the same tasks. Losses include tasks base always solves (ESPN--20,
  Google Map--5); the "no losses on always-solved tasks" note from the first run no longer holds.
- 2026-11-14 (tree resumes): leftover preview jobs (296115/7/20, step 295056.2) and the zombie 295161 (Qwen 8B Android
  s0, search dead since 02:17, job still holding a Quadro) cancelled. 19 unfinished trees resubmitted with
  --overwrite false (296187-296205; 295161 had been submitted with --overwrite true). The six Web trees that died in
  every job within ~15 min (random s3/s4, Qwen 4B s0/s1, Qwen 8B s1/s2) were not a code bug: opening
  wolframalpha.com timed out on lg09 / lg12 / lg15. Wolfram check before placing: reachable from lg01, 02, 03, 08,
  12, 13, 19; blocked from lg07, lg10. Web trees pinned to reachable nodes; GameBoy / Android to lg07 / lg10.
- 2026-11-14 (tree resume failed, resubmitted): 296195 (Android Gemma sat_vlm_no_typing_s4, lg05) died 47 min in at
  emulator setup: the accessibility-forwarder APK download was cut off (http IncompleteRead), so a network glitch,
  not code. Resubmitted as 296219 on lg15 (4 Quadros, no Android job there) without --exclusive, since another
  user's CPU job holds lg15; --overwrite false, args from the job log.
- 2026-11-14 (FAILURE, my fault: tree resumes past the 69 h cap): the seed 3-4 Gemma / random trees cap at 69 h as
  3 chained 23 h segments (2026-11-11), so a third segment ending in TIMEOUT is the designed stop, not a crash. I
  resumed every chain whose last segment timed out. Legit resumes (a failed segment in the chain): Web random s3/s4
  (296199/200, at 44.5/45.1 h of expansion wall time vs s0-s2 64 h), Android vlm_w_typing s3 (296196, 45.7 h vs
  56-57.5 h), plus the original 4th segment 294867 (w_typing s4, 51.8 h). Cancelled ~20:15 on the user's OK as over or at cap
  (expansion wall time): Red vlm_no_typing s0/s1/s3 (71.4 h each), s4 (296093, 78.9 h: its 4th segment after a failed
  one ran ~10 h over); Web vlm_no_typing s3/s4 (70.2/70.4 h), vlm_w_typing s3/s4 (70.1 h; s0-s2 ended 67.6-67.8 h);
  Android vlm_no_typing s3/s4 (61.4/60.6 h vs s0-s2 60.4-61.8 h) and random s3/s4 (48.8/50.9 h vs 48.2-52.2 h).
  Analyses of the over-cap trees must cut them at 248400 s of cumulative expansion wall time (Qwen trees stop on
  that budget themselves). Red vlm_no_typing s2 (295056) is its chain's real third segment and keeps running.
- 2026-11-14 ~20:30 (job moves, user: "watch for free GPUs and move jobs accordingly"): the Qwen Android trees were
  serialised by a singleton name (one emulator per node), so 4B s1/s2 waited on s0 (~5 / ~8 days) and 8B s2 on s1.
  With lg01, lg08 and lg15 free, the pending chains were re-pointed in place (scontrol update, stored args kept):
  4B s1 (295745-48) -> lg01, 4B s2 (295761-64) -> lg08, each with a per-seed job name; 8B s2 (295167) -> lg15 as a
  23 h medium-lg leg plus 296222-24 (afterany); 4B s0's next leg 295732 renamed per seed. All stay on Quadro RTX 8000
  (fp16) like the other Qwen trees: the budget is wall time, so the GPU type must match across seeds. The five
  single-job Qwen resumes (Web 4B s0/s1 and 8B s1/s2, Android 8B s0) got 3 follow-on legs each on their own nodes
  (296225-39, --overwrite false); their 69 h wall_s budget stops them, and spare legs exit at once.
- 2026-11-15 00:00 (Android vlm_w_typing s4 short of parity): its 4th segment 294867 timed out at 53.9 h of expansion
  wall time (s0-s2 56.3-57.5 h; it lost time to a11y reconnects). Resumed as 296240 on lg02 (--exclusive, args from
  the job log, --overwrite false) with --time 03:20:00, so it stops itself near 57 h; resumed at 253 expansions /
  301 nodes. vlm_w_typing s3 (296196, 49.1 h at 00:00) reaches parity ~08:00 but its job runs to ~16:15. Result: 296240 stopped at 56.89 h (268 expansions), in the s0-s2 range.
- 2026-11-15 16:15 (tree chain ends): Web random s4 ended at 64.12 h of expansion wall time (s0-s2 63.65-64.19 h): done.
  Web random s3 ended at 62.76 h: topped up as 296297 (lg12, --time 01:15:00, --overwrite false). Android
  vlm_w_typing s3 (296196) ran its job out to 63.37 h (277 expansions) vs s0-s2 56.3-57.5 h: no stop was approved
  after it reached parity at 08:45, so it needs the cut (to ~57 h) before analysis. Qwen follow-on legs started on
  schedule (296225, 296231, 296234, 296237; Android 8B s0 resumed at 70 expansions / 89 nodes).
- 2026-11-15 16:20 (Wolfram blocks lg12 and lg13): the Web jobs that started there at 16:15 (296225, 296231, 296234,
  296297) failed opening wolframalpha.com (ReadTimeout 120 s), as on 2026-11-14. curl from each lg node: blocked from
  lg12, lg13; reachable (200) from lg01-04, 06, 08, 09, 11, 14, 15, 16, 18, 19 (lg09, lg15 recovered). Pending Web
  Qwen legs re-pointed in place: 4B s0 / s1 (296226-30) -> lg09, 8B s1 / s2 (296232-33, 296235-36) -> lg15, all
  Quadro. Web random s3's top-up resubmitted on lg01 (296313).
  Web random s3 top-up 296313 stopped at 63.83 h (3038 expansions), in the s0-s2 range (63.65-64.19 h).
- 2026-11-16 15:30 (extra legs): four Qwen chains could not reach the 248400 s budget with the legs left (Web 4B s0,
  8B s1, 8B s2 lost a leg each to the lg12/lg13 Wolfram block; Android 8B s1's 3-day job ends with ~26 h of
  expansion time still to go plus emulator overhead). One more 23 h leg each, afterany, same nodes: 296403 (Web 4B
  s0, lg09), 296404 (Web 8B s1, lg15), 296405 (Web 8B s2, lg15), 296406 (Android 8B s1, lg14). The budget stops each
  tree; a spare leg exits at once.
- 2026-11-16 15:35 (Wolfram blocks lg15): Web 8B s1 / s2 legs 296233 / 296236 failed opening wolframalpha.com on
  lg15; their extra legs 296404 / 296405 had already started there (doomed). curl check: lg15 blocked; lg01, 08, 09,
  11, 12 (recovered), 13, 14 reachable. Both trees continue on lg11 (free Quadros): 296415-16 (8B s1) and 296417-18
  (8B s2), afterany on the lg15 legs so only one process writes each tree. Web 4B s0 / s1 on lg09 resumed fine.
- 2026-11-16 (old annotations deleted, user's call): the 13 annotated trees (android / web sat_random_s0-s2 and
  sat_vlm_w_typing_s0-s2, android tt_annot) were annotated Nov 2-3, before the Nov 11 prompt changes (app names,
  one-action rule, context frames). Their annotation rows were stripped from nodes.jsonl (rows with annotate_compute,
  and the parent {"id", "tasks"} rows), and annotations.json, task_coverage and task_diversity deleted. The search's
  own rows (nodes, search-time updates, flags) are kept, so the trees are as the search left them.
- 2026-11-16 (re-annotation of every finished tree): 67 cusi-annotate jobs (296488-296554, list in
  storage/tmp/annotate_jobs_2026-11-16.txt), google/gemma-4-31b-it as in the Nov 2-3 annotations, defaults
  (max_features 8, workers 8): Android / Web sat_random, sat_vlm_no_typing, sat_vlm_w_typing s0-s4, Red sat_random and
  sat_vlm_no_typing s0-s4, Link's Awakening sat_zelda_la_random / vlm_no_typing s0-s5, and the finished Qwen trees
  (Red and LA 4B / 8B s0-s2, Web 4B s2 and 8B s0, Android 4B s0). 2x A6000 each; Web random s0 / s2 on lg10's 4x A5000.
  Not annotated: the 9 Qwen trees still running, the set-aside 32B trees, the older sat_random_zelda_la world-model
  runs, side-game and smoke trees. Over-cap trees (Red Gemma s0/s1/s3/s4, Web s3/s4, Android w_typing s3) are annotated
  whole; a later cut must also drop edge tasks that cut children added to their parents' task lists.
- 2026-11-16 (task coverage + diversity on every re-annotated tree): one task_coverage and one task_diversity job per
  annotation job, afterok on it (296557-296690; storage/tmp/task_eval_jobs_2026-11-16.txt: annotate, coverage,
  diversity ids), judge google/gemma-4-31b-it for all (user's call), embedder Qwen/Qwen3-Embedding-8B, k 5, semantic
  dedup (defaults), v2 judge prompts. Caveat (open since 2026-11-11): Android test goals are drawn from AndroidWorld
  at seed 30 and some depend on the date, and each tree snapshots its own test set at its embed stage, so Android trees
  embedded on different days can be scored against slightly different goals. The 16 Android trees here embed within a
  few hours of each other; the 6 Qwen Android trees still searching will embed later.
- 2026-11-16 (image-only oracle preview): --world_action oracle_preview_image (OraclePreviewImageModel, image_only):
  the check call is given the previewed outcome as images only: no page / app facts (Web URL and title before and
  after, Android activity before and after), no page text, no GameBoy dialogue / menu / battle note. oracle_preview is
  unchanged. Rendered for all three envs: 2 images each, none of those texts in the prompt. Then (user's call) the
  rest of the non-image channels off too: every previewed proposal goes to the check (no env-refusal or no-change
  rejection, so no env error text or buttons-that-change list in the executor's feedback; no RAM game-UI gate), and
  no battle rollout (a battle choice is previewed as its one next screen). Live check on a Pokemon Red emulator
  (tmp_preview_image_check.py, fixed-action policy, check call recorded): free-roam B (no change), an unparseable
  action and a battle A each went to one 2-image check with none of the texts, and the feedback carried only the
  check's own outcome / reason; oracle_preview still rejected the first two without a check, as before. The
  rollout path was not reached in this check (the fight menu was not open); it is gated off in code.
- 2026-11-16 (clicked element recorded): Web and Android steps now carry info["target"] for indexed actions (the
  element as on the screen acted from: Web tag / text / value / box in viewport CSS px; Android resource name / text /
  content description / class / package / box in screen px), kept in trees (tree.INFO_KEYS). Older trees lack it.
- 2026-11-17 (cut and sweep fill, GameBoy only; user's call): cusi/search/cut.py cuts a run at a compute budget into
  a new run (fork, then the store's commit truncation; the expansion that crosses the budget is kept, as run_search's
  budget stop keeps it; annotated parents lose edge tasks of cut children; the archive is the source's).
  run_search --from_run <cut> --sweep_expansions k (selection.SweepSelector): every source node expanded k times in id
  order, only each expansion's last step a node (closing_node, threshold 1e9), new nodes never expanded, resumable.
  Tested on gameboy/sat_random_zelda_la_s0 (not in the annotation set): a 1000 s cut kept 3604 / 7500 expansions (an
  exact prefix, nodes 0-245) with the source's whole-directory sha256 unchanged; a k=2 x 10-step fill passed the audit
  (tmp_fill_check.py: every source node exactly k expansions, one child each at the last step, parent = the expanded
  node, no new node expanded, source prefix intact, cut tree's sha256 unchanged; a relaunch adds nothing); a
  k=100 x 100-step fill killed at 6 min and relaunched resumed mid-node (node 1: 74 -> 100) in id order. Measured on
  Link's Awakening: 1.34 s per 100-step expansion, ~0.4 MB on disk per expansion (steps ~105 KB, frames ~100 KB, the
  new node's saved state ~190 KB). Red steps are ~9 KB each (screen text boxes), so Red at k=100 x 100 is ~100 MB of
  expansions.jsonl per source node: not feasible as is (see the reply of 2026-11-17).
- 2026-11-17 (target field live): the smoke trees show info["target"] on every indexed action (web 218 / 218,
  android 185 / 185; jobs 296691-2).
- 2026-11-16 FAILURE (HF rate limit): annotate / task_coverage / task_diversity jobs were submitted without
  HF_HUB_OFFLINE=1, so each vLLM start queries the Hub; 296581 (task_coverage web sat_vlm_w_typing_s3) died at vLLM
  start with 429 Too Many Requests. Resubmitted offline as 296693 (its embed stage was done).
- 2026-11-17: task_diversity 296614 (gameboy sat_vlm_no_typing_s1) failed on the same HF 429 at vLLM start; resubmitted as 296694 with HF_HUB_OFFLINE=1.
- 2026-11-17: task evals 296656/296657/296658 (LA vlm_no_typing s3 div, s0 cov+div) failed on HF 429 at vLLM start; resubmitted with HF_HUB_OFFLINE=1.
- 2026-11-17: 12 more task evals failed on HF 429 at vLLM start (296642, 296671-76, 296683-87); resubmitted as 296698-296709 with HF_HUB_OFFLINE=1. Offline resubmissions verified: vLLM loads weights from HF_HOME with no Hub call.
- 2026-11-17: task_diversity 296670 also failed on HF 429 (missed by the watcher between restarts); resubmitted as 296759 offline. All 8 still-pending evals (web sat_random s0/s2/s3/s4, waiting on annotations 296488-91) cancelled and resubmitted with HF_HUB_OFFLINE=1 and the same afterok dependencies (296751-296758).
- 2026-11-17 (Qwen3-VL-32B trees back on): user's call ("send out the Qwen3-VL-32B trees across all GPUs ... all trees
  on all 3 envs generated fully"). 12 trees: Red, Link's Awakening, Web, Android x s0-s2, vlm_no_typing, same settings
  and ports (8200-8211) as the 2026-11-12 32B jobs and the 8B / 4B trees (fp16, 20 steps, thresholds 0.1 GameBoy /
  0.5 Web and Android, saturation_patience 50, 69 h wall_s budget). tmp_submit_qwen32_full.sh, jobs in
  storage/tmp/qwen32_full_jobs.txt: 5 chained 23 h medium-lg legs each, 2 GPUs, HF_HUB_OFFLINE=1. Non-Android on any
  2 GPUs except eclairlg10 (A5000s too small); Android pinned one per node to eclairlg02 / 03 / 04 (A6000; emulator
  port 5554). Red s0 (24.1 h used, 214 expansions) and Web s0 (20.2 h, 88 expansions) resume their run dirs; the rest
  start fresh. Caveat: hardware is mixed (A6000 and Quadro pairs, both fp16), and Red s0 / Web s0 began on Quadros, so
  per-expansion speed differs between and within trees; the budget is wall-clock, so faster pairs give bigger trees.
  All 12 first legs started at once (296762 ... 296817).
- 2026-11-17 04:05 (Wolfram still blocks lg15): Web 32B s1 leg 1 (296792) landed on lg15 and timed out opening
  wolframalpha.com. curl from every lg node: only lg15 blocked (lg12, lg13 recovered). 296792 cancelled before any
  expansion; leg 2 (296793) took over on lg12. lg15 (and lg10) excluded from all pending Web 32B legs (s0 296773-76,
  s1 296793-96, s2 296813-16).
- 2026-11-17 (random fill: k=20 for Red and Link's Awakening): user's call (k=20, 100 steps per expansion, GameBoy only).
  The 34 GameBoy trees (Red and LA: random, Gemma vlm_no_typing, Qwen 4B / 8B) are cut at 69 h first
  (<tree>_cut69h; Red random s0 used 4,809 s, so its cut is the whole tree). Red test (296761, 20 min, GPU-free on
  extra-long, sat_random_s0_cut69h): 128 fill expansions in order, 100 steps and 1 child each, but 3.95 s per expansion
  (env 2.06, novelty score 1.36 against a 3k archive, embed + record 0.5) and 13 GB RSS. Scoring grows with the
  archive, and every fill node joins it: Red s0's archive would go 3k -> 61k, ~11 days for that tree. Fix: sweep runs
  skip step scoring (TreeSearch score_steps=False from run_search --sweep_expansions): fill steps get value 0 and
  nothing enters the archive; novelty plays no part in a fill (endpoints are nodes regardless). Default unchanged,
  no new config key (sweep_expansions already marks a fill), so every other run behaves as before. Retest 296842.
- 2026-11-17 04:21 (32B LA s1 vLLM hang): leg 1 (296782, lg19) hung at vLLM startup: one tensor-parallel worker died
  with OSError Stale file handle (Errno 116) reading the shared Triton cache on the NAS while Red s1 (same node)
  compiled the same kernels; the other worker waited forever (shm_broadcast every 60 s). No tree data; cancelled,
  leg 2 (296783) took over.
  Retest 296842 (no scoring): flat ~2.8-4.2 s per Red fill expansion (all emulator; varies with node load), RSS
  +1.63 MB per fill expansion (10.44 -> 10.55 GB over 69). All 34 cuts done (Red random and Qwen trees and every LA
  tree kept whole: random under the cap, Qwen exactly at it, LA saturated early). 34 fills submitted
  (tmp_submit_fill.sh, jobs in storage/tmp/fill_jobs.txt: 296845-296878), GPU-free on the rtx2080ti partitions,
  memory 1.3 x (source load + 1.63 MB x fills) + 2 GB (Red random 91-139 GB, other Red 49-77 GB, LA 16 GB), time
  1.3 x 4.5 s x fills (Red random 65-96 h on extra-long). Output <tree>_random_filled; the cut and source trees are
  untouched.
- 2026-11-17 (test trees deleted, user's call): tt_fill_cut1000s, tt_fill_cut1000s_filled_k2 / _k100, tt_fill_cut1h, tt_red_fill_k20, tt_red_fill_k20b (cut and fill tests; sources untouched, their write-once dirs were hard links).
- 2026-11-17 (random fill cancelled, user's call): the fill would dominate the data (~97% of a filled GameBoy tree's steps would be random fill). The 28 still-running fill jobs were cancelled; 6 LA fills had already completed. The <tree>_random_filled dirs (complete and partial) are kept; the <tree>_cut69h trees and the source trees are untouched.
  Then resumed on the side (user's call: CPU only, so keep them): the 28 cancelled fills resubmitted unchanged (k=20, same memory / time), resuming their partial dirs: 296881-296908 (storage/tmp/fill_resume_jobs.txt).
- 2026-11-17 (image-only oracle evals, user's call): oracle_preview_image with Gemma-4-31B (executor = check = judge),
  same settings as the final oracle_preview runs: GameBoy Red (preview_image_red_v1, 296909, lg07), Web dev3
  (preview_image_web_dev3_v1, 296910, lg06), Android dev2 (preview_image_aw_dev2_v1, 296911, lg05, no Android tree on
  that node). Link's Awakening added (user's call): preview_image_la_v1 on its 50 tasks, plus a Gemma base run
  base_la_gemma-4-31b-it (no Gemma LA eval existed to compare against; only Qwen 2B / 4B test runs). The oracle
  preview was built and tuned on Red; LA is its first run.
- 2026-11-17 (tree world models, cusi/world_model + run_world_model.py + slurm/world_model.sh): user's spec. Data from
  the original trees (not the fills): every committed expansion step -> (screen before with the one before it, action,
  screen after); split 85/15 by expansion (user's call: seeded hash of run + expansion); val dedup (user's call): a val
  transition is dropped when a train transition has pixel-identical before AND after screens (action not compared).
  Actions (user's call: type + target box): discrete type (GameBoy 7 buttons; Web click / type / scroll element or
  window / wait / goback / answer; Android click / long_press / input_text / scroll x4 / back / home / enter / wait /
  open_app / answer / status), typing one type whatever the text, plus the target element's box on the screen before
  ([x0, y0, x1, y1] / frame size, from that screen's text_boxes by ref). Steps with no type are left out: Android
  no-typing trees' refused typing (23% of android Gemma s0, no action performed), Web unparsed replies (Format ERROR,
  1241 of 17.7k in web Gemma s0), Web bot checks. Box found for 97% (Android) / 96% (Web) of indexed actions.
  SigLIP 2 base fine-tuned per model (reconstruction with the EmbeddingDecoder, as run_explore's train_siglip) on the
  train screens, val loss on held-out screens every 500 steps, early stop after 6 without a new best, best kept.
  World model: BoxActionWorldModel ([emb t-1, emb t] + type embedding + box MLP -> emb t+1, residual), MSE, val every
  epoch, early stop after 15 epochs, best kept; reports val cos vs the copy baseline (overall and changed only).
  Per policy (user's call): red_random / red_gemma / la_random / la_gemma / web_random / web_gemma / android_random /
  android_gemma (tmp_submit_wm.sh, jobs 296916-296923, storage/world_model/<env>/<name>); GameBoy trees at their 69 h
  cuts, Web / Android whole trees.
  Test (tt_la_random_s2, LA random s2 only, 600 SigLIP steps, 40 WM epochs): val pixel MAE 87 -> 18; WM val cos 0.921
  vs copy 0.857 (changed-only 0.920 vs 0.855), still rising. Residual start dips at lr 1e-3 (0.29 after epoch 1, then
  recovers past copy by ~epoch 12); early stopping patience covers it.
  Executor: wm_preview_image:<name> (cusi/world_action/wm_preview.py) = oracle_preview_image with the peek replaced by
  the WM's decoded prediction (resized to the screen); the check is told image 2 is a blurred prediction, not an exact
  preview (OraclePreviewModel.one_after). Reads only the executor's observation, the env's action parser and its
  description. Test eval tt_wm_preview_la (296915, 2 LA tasks).
  Executor test fix: GameBoy eval workers reload the model from wam.directory, which load_world_action_model treated
  as a siglip_lookahead dir (FileNotFoundError world_model_meta.json, 0 tasks run in 296915); a directory holding a
  tree WM (wm_meta.json, or wm/wm_meta.json) now loads as wm_preview_image. Retest 296924 (tt_wm_preview_la, the
  tiny LA random s2 test WM, 2 LA tasks): ran end to end, 1/2 (stuck-behind-bed solved in 158 steps; Tarin task
  truncated at 176). 330 decisions all wm_preview_image, 360 previews checked, 43 judged bad (e.g. LEFT into the bed:
  "the predicted image shows no movement"), 42 decisions revised, 13 insisted, 0 preview failures; ~4 s per preview
  (SigLIP + WM + decoder on CPU, vLLM holds the GPUs).
- 2026-11-17 08:10 (GPU fault on lg14): Web 32B s2 leg 1 (296812, lg14 Quadro pair) lost its vLLM after 4 h 14 min
  (OpenAI connection error), and every restart (legs 296813-16) died at vLLM startup with pynvml NVMLError_Unknown:
  a GPU fault on lg14, not code. The other lg14 jobs (Android 8B s1 tree 295164, red_gemma WM 296917) kept running.
  lg14 added to the exclusions of all 30 pending 32B legs; Web 32B s2 resubmitted as 4 x 23 h legs excluding lg10,
  lg14, lg15: 296925 296926 296927 296928. Its tree resumes (--overwrite false).
- 2026-11-17 (Link's Awakening base): base_la_gemma-4-31b-it (296913) 20/50. preview_image_la_v1 so far 11/22 vs base 6/22 on the same tasks (+5 tasks, none lost: 000, 002, 012, 015, 024); one base run, so base noise is unknown.
- 2026-11-17 (Web image-only oracle): preview_image_web_dev3_v1 (296910) 34/60 on web_dev3, vs base 28 / 26 / 27 and the full (text) oracle_preview's complete runs 27 / 36 / 34 / 32 (chk10, chk10_rep3, chk11_rep2, chk11_rep3). One run: inside the full preview's spread, ~7 above base mean.
- 2026-11-17 (image-only oracle, Red 50 tasks, 296909): preview_image_red_v1 21/50, vs the full text+image preview
  (preview_red_v5) 28/50 and base reps 23 / 25. On Red, dropping the text part of the preview costs about 7 tasks and
  falls slightly below base. LA image-only 23/44 so far (base 20/50); Android image-only 4/10 scored after 5 h (slow).
- 2026-11-17 (WM la_gemma, 296919, done in 4.5 h): SigLIP fine-tune ran the full 50k steps (stopped on max_steps, not
  patience: val pixel MAE still falling, 1.82 at the end). WM early-stopped at epoch 55 (best 39): val cos 0.935 vs
  copy-the-current-frame 0.757; on changed transitions 0.935 vs 0.752 (train 0.956). Usable as
  wm_preview_image:la_gemma.
- 2026-11-17 (image-only oracle, LA 50 tasks, 296912): preview_image_la_v1 23/50 vs base 20/50 (same tasks, same
  Gemma-4-31B). Image-only so far: Web +7 over base mean, LA +3, Red -2 to -4 (full text preview +3 to +5 on Red).
- 2026-11-17 (WM la_random, 296918): SigLIP early-stopped at 43.5k steps (val pixel MAE 1.98). WM best epoch 75 of
  91: val cos 0.932 vs copy 0.648; changed-only 0.931 vs 0.642 (train 0.968; small set, 17k train). Bigger gain over
  copy than la_gemma (random steps change the screen more per action).
- 2026-11-17 (WM web_gemma, 296921): SigLIP hit the 50k-step cap (val pixel MAE 6.92, still falling). WM best epoch
  63 of 79: val cos 0.625 vs copy 0.415; changed-only 0.600 vs 0.362 (train 0.711). Far below GameBoy (0.93): web
  next pages (new sites, search results) are much harder to predict from a 5-dim box + type.
- 2026-11-17 (WM red_gemma, 296917 on lg14): SigLIP ran the full 50k steps (val pixel MAE 2.93, best at the last
  step) and saved vision_model.pt / decoder.pt / embedder_meta.json, then CUBLAS_STATUS_EXECUTION_FAILED on lg14
  while drawing val_reconstructions.png (lg14 GPU fault again, cf. NVMLError_Unknown), so no train_summary.json.
  Weights are intact: resubmitted only the wm stage (296929, --exclude=eclairlg14) instead of `all`, which would
  redo the 6 h fine-tune. lg14 should not get new GPU jobs.
- 2026-11-17 (WM red_gemma, wm-only rerun 296929 on lg03): best epoch 77 of 93: val cos 0.826 vs copy 0.751;
  changed-only 0.816 vs 0.726 (train 0.905). Smaller gain over copy than LA (0.93); untested guess: Red Gemma steps are often
  dialogue / menu text advances whose next frame depends on hidden game state.
- 2026-11-17 (WM web_random, 296920): SigLIP hit the 50k-step cap (val pixel MAE 6.08). WM best epoch 80 of 96:
  val cos 0.730 vs copy 0.519; changed-only 0.693 vs 0.449 (best_train not recorded). Better than web_gemma (0.625)
  on 4x the training transitions; Web stays far below GameBoy.
- 2026-11-17 (WM red_random, 296916): SigLIP hit the 50k-step cap (val pixel MAE 2.20). WM best epoch 58 of 74:
  val cos 0.858 vs copy 0.776; changed-only 0.840 vs 0.747 (460k train transitions). Above red_gemma (0.826), below
  LA (0.93).
- 2026-11-17 (WM test on real Gemma task attempts, user's request before the WM oracle runs): new
  `run_world_model.py test` (cusi/world_model/evaluate.py) + Slurm pair world_model_test: every valid step of eval
  episodes (base Gemma-4-31B runs: LA base_la_gemma-4-31b-it 5,890 steps; Red newemu rep1+rep2 10,628; Web
  base_web_dev3_r1-3 1,985), one-step embedding cos vs copy, decoded pixel MAE vs copy and the decoder's floor, and
  open-loop rollouts k=1..5. Eval episodes do not store element boxes, so indexed Web/Android actions get NO_BOX
  (reported apart as box "missing"). Results (one-step cos vs copy; pixel MAE vs copy):
    la_gemma 0.848 vs 0.799; 14.1 vs 10.5 (floor 7.6). Rollout falls below copy from k=2.
    la_random 0.834 vs 0.814; 15.5 vs 10.5. Same pattern.
    red_gemma 0.826 vs 0.785; 19.7 vs 16.5. Above copy to k=3.
    red_random 0.846 vs 0.790; 14.6 vs 16.5 (only model that beats copy in pixels). Above copy to k=5.
    web_gemma 0.587 vs 0.643 (below copy); 29.8 vs 19.9.
    web_random 0.474 vs 0.686 (below copy; Type -0.01, scroll_down 0.01); 32.9 vs 19.9.
  So on real test steps the GameBoy WMs are a little better than "nothing changes" in embedding space (tree val
  overstated them: LA 0.93 there), worse than copying in decoded pixels except red_random; Web WMs are worse than
  copying even on unboxed actions (scroll window, goback, answer), not only because of the missing boxes.
- 2026-11-17 (WM oracle, Web dev3 60 tasks): wmprev_web_gemma_v1 (296974) 25/60, wmprev_web_random_v1 (296975)
  24/60, vs base 28 / 26 / 27 and the exact image-only preview 34/60. Previews ran on every check (WM_AFTER prompt in
  all 60 episodes, 0 failed); the check rejected about half as often (71 / 81 bad vs 140 with exact images). As the
  test predicted (both Web WMs below copy), a learned Web preview does not help: at or slightly below base.
- 2026-11-17 (WM seed study, user's question "is Gemma ever statistically better than random at generating trees
  for WM training, or is it noise?"): the pooled WMs mix all seeds (5 or 6 trees each), so there was no variance
  estimate. New: cusi/world_model/seeds.py + run_world_model.py seeds_data / seeds_wm / seeds_compare + Slurm pair
  world_model_seeds. One dataset per tree seed (and a copy with train cut to the smallest tree's size, whole
  expansions in a seeded order: build_dataset max_train), one WM per dataset, each tested on the same base Gemma
  eval steps; Welch t / Mann-Whitney U / Cohen's d, gemma minus random, per metric (one-step cos margin over copy,
  changed-only, pixel MAE and vs copy, rollout k=3/5 margins, tree val margin). Embedder held fixed within a
  comparison (pooled models' fine-tuned SigLIPs, so only the WM training transitions differ); run once with each
  policy's embedder. The WM stage seed (42) is the same for every model, so the spread is the trees'.
  LA: 6+6 trees, cap 2,100 train transitions (per-seed train: gemma 2.7k-8.4k, random 2.1k-4.1k). Jobs: LA wm
  296979/296980; Red data 296981 -> wm 296982/296983; Web data 296984 -> wm 296985/296986; Android data 296987 ->
  wm 296988/296989 (also after the pooled Android models 296922/296923 finish, for their embedders).
  Not covered: the oracle success rate per seed (8+ GPU-hours per eval run).
- 2026-11-17 (WM seed study, LA, 6 vs 6 seeds, both embedders): on the real Gemma test steps no significant
  gemma-vs-random difference. Capped (2,100 train each): cos margin over copy +0.0025 (la_gemma emb, p 0.58) /
  +0.0022 (la_random emb, p 0.70); pixel MAE and rollout k=3 within noise (p 0.39-0.94). Full trees: gemma ahead
  +0.014 / +0.016 (p 0.11 / 0.10 Welch, 0.13 / 0.09 MWU), pixel MAE -1.06 / -0.54 (p 0.15 / 0.19): the direction of
  gemma's larger trees (2.7k-8.4k vs 2.1k-4.1k), gone at equal size. Only significant metric: tree val margin,
  random higher (p 0.002 capped) -- each on its own val set, so it measures how predictable each policy's own
  transitions are vs copy, not which data trains a better model. Per-seed WMs are weak: margin 0.01-0.02 (pooled
  la_gemma 0.049), all below copy at k=3.
- 2026-11-17 (WM android_gemma, 296923): SigLIP stopped on the 600-min cap at 34k steps (val pixel MAE 1.97). WM
  best epoch 52 of 68: val cos 0.793 vs copy 0.567; changed-only 0.777 vs 0.528 (train 0.881). Test on base Android
  steps 296990, then the WM oracle wmprev_android_gemma_v1 296991 (afterok). Seed-study Android wm job 296988 (gemma
  embedder) started.
- 2026-11-17 (WM android_gemma test, 296990): on the 421 base Android steps cos 0.528 vs copy 0.551 (below copy),
  pixel MAE 35.6 vs 32.1, below copy at every rollout k (80% of steps indexed with no box: see the box caveat).
  WM oracle wmprev_android_gemma_v1 (296991) started on lg03 anyway (the user asked for all WMs in the oracle).
- 2026-11-17 (WM android_random, 296922): SigLIP stopped on the 600-min cap at 38.9k steps (val pixel MAE 1.95).
  WM best epoch 83 of 99: val cos 0.823 vs copy 0.697; changed-only 0.783 vs 0.623 (train 0.861). Test 296992,
  then WM oracle wmprev_android_random_v1 296993 (afterok). All 8 pooled WMs done.
- 2026-11-17 (WM android_random test, 296992): 421 base Android steps, cos 0.515 vs copy 0.539 (below copy; also
  on the 87 unboxed steps 0.424 vs 0.506), but pixel MAE 27.4 vs copy 32.1 (beats copy in pixels, android_gemma
  35.6 did not). Below copy at every rollout k. WM oracle 296993 running on lg05.
- 2026-11-17 (WM seed study, Red, 5 vs 5, both embedders): full trees, random significantly better on every test
  metric in both embedders (cos margin -0.024 / -0.021 gemma minus random, Welch p 0.0003 / 0.003, MWU 0.008 = the
  5 seeds fully separated; pixel MAE +2.3 / +1.9, p 0.0002 / 0.007; rollout k=3 p 0.0006 / 0.012) -- random trees
  are 2.4-6x larger (52k-129k train vs 21k-31k). Capped at 21,452 each: no difference (cos margin p 0.71 / 0.99,
  pixel MAE p 0.46 / 0.46, rollout p 0.76 / 0.37). Same conclusion as LA: tree size decides, the policy does not.
- 2026-11-17 (WM seed study, Web, 5 vs 5, both embedders): Gemma (vlm_no_typing) significantly better on the real
  Gemma test steps, also at equal size (capped 13,272): cos margin over copy +0.105 / +0.086 (web_gemma / web_random
  embedder; Welch p 0.0005 / 0.010, MWU 0.008 = fully separated), rollout k=3 +0.16 / +0.13 (MWU 0.008); pixel MAE
  not significant capped (p 0.14 / 0.49). All models below copy (margins -0.09 to -0.21). By type (capped, web_gemma
  emb): Type -0.19 vs -0.53 (random trees contain no Type step; gemma no-typing trees a few), scroll_window_down
  +0.11 vs +0.00, click -0.002 vs -0.012 (p 0.008 there, 0.056 with the random embedder). So unlike GameBoy, Web
  shows a policy effect beyond size, much of it from action types random never or rarely takes.
- 2026-11-17 (WM seed study, Android, 296988/296989 done): random trees give better WMs on the base Gemma test steps.
  Capped (equal train size), gemma embedder: cos margin -0.150 vs -0.070 (Welch 0.0007, MWU 0.008), pixel MAE 46.9 vs
  36.1 (MWU 0.008), rollout k=3 -0.21 vs -0.10 (MWU 0.008). Random embedder: same direction, cos margin MWU 0.032,
  pixel MAE n.s. Full trees: same direction, weaker (MWU 0.016-0.31). Gemma trees score higher on their own tree val
  (+0.10), so they are easier in-distribution but transfer worse. All models below copy on test.
- 2026-11-17 (WM oracle, partial, matched task ids): Red wmprev_red_gemma 19/30, wmprev_red_random 18/29 vs base reps
  15-17 / 15-18 and exact image-only 13 / 14 on the same tasks. LA wmprev_la_gemma 3/16, wmprev_la_random 8/21 vs base
  2/16, 6/21 and exact image-only 6/16, 9/21. Partial, still running; Android not yet judged.
- 2026-11-17 (split revised, user's call; supersedes the 2026-11-11 split): no val split, only train / test /
  test_env, and test is the larger side in every train env. configs/splits/benchmark_splits.json rewritten
  (tmp_split_revise.py -> storage/tmp/split_proposal_v2.json, copied in). Rules: old val counts as train; no old test
  task leaves test (moves only train -> test); test_env unchanged; GameBoy start states stay in one split; among seeded
  candidates, the one whose test and train base solve rate (base Gemma-4-31B, mean over base runs) and category / site
  / app mix are closest to the env's. Counts train / test / test_env: Red 10 / 40, Link's Awakening 10 / 40, Crystal
  and Oracle of Seasons 50 each test_env; Web 140 / 160 / 129 (7 sites, each 17-22 train / 21-24 test); Android
  40 / 46 / 30 (every same-env app on both sides). Base solve env / train / test: Red 0.48 / 0.50 / 0.47, LA
  0.40 / 0.40 / 0.40, Web 0.61 / 0.61 / 0.61, Android 0.69 / 0.69 / 0.70. Red train: 000, 004, 007, 008, 010, 026,
  032, 033, 047, 049 (the six old focus tasks 17/19/23/29/31/36 are now test). LA train: 004, 013, 014, 021, 022, 023,
  024, 027, 028, 045. cusi/eval/splits.py SPLITS and env_server/policy.py (dev = train only) drop "val". The Web /
  Android WAM dev sets (storage/tmp/wam_dev/) are all test / test_env tasks and should be redrawn from train.
  Exploration scenes unchanged (user: not a concern).
- 2026-11-17 (split, user's call): test_env merged into test; the split is only train / test (GameBoy 20 / 180, Web
  140 / 289, Android 40 / 76). splits.SPLITS = (train, test); env_server test policy reads test only. Held-out games,
  sites and apps are still the ones with no train task (policy.py's HELD_OUT_* lists unchanged).
- 2026-11-17 (oracle PAE, user's call: "the other side of PAE"): practice on the TEST split with google/gemini-3.5-flash-lite
  (OpenRouter) as both the agent and its own judge (practice judge -> critique hint -> retry, max 3 tries), LoRA-tune
  google/gemma-4-31b-it on the judged successes with the aggressive six-task recipe (r64 a128, lr 1e-4 constant, 3
  epochs), benchmark on the same test tasks. This is an ORACLE / upper bound (trains on the test tasks themselves), not a
  result; names carry "oracle_test". New: run_practice.py attempt_split (cusi/practice/stages/attempt_test.py
  attempt_split: any env, every task of a benchmark split, each try run as that env's eval runs it; GameBoy and Android
  record the benchmark's success, Web writes each try as an eval run for auto_eval), score_web_tries, judge_agreement;
  dataset --val_copy (validation rows copied from train when no task is held out); Slurm pair practice_split (API
  model, no GPU). OpenRouter model google/gemini-3.5-flash-lite exists ($0.30 / $2.50 per M tokens in / out).
  Spend cap (user): 200 USD on this effort; baseline key limit_remaining 8400.762 (usage 246.62) at 20:53; spend is
  tracked as the drop in limit_remaining (usage lags), log storage/tmp/pae_cost_log.txt.
  Smoke (2 tasks per env, max 2 tries): GameBoy Red 016 judge success = benchmark success on try 1 (25 steps); Web
  Apple--12 / --19 judged success on try 1 (auto_eval pending); dataset stage on the Web smoke: 22 success rows, rows
  carry Gemini's replies in the Gemma agent's format. Base Gemma test-split evals (2 reps per env, run names
  testsplit_base_gemma_r1/r2; task id files storage/tmp/oracle_pae/test_<env>.txt): 297013-297018.
- 2026-11-17 (API-model test-split evals, user's request): google/gemini-3.5-flash-lite and google/gemini-3.6-flash
  (OpenRouter) as the final agent on the test split (GameBoy 180 = Red 40, LA 40, Crystal 50, OoS 50; Web 289; Android
  76; task files storage/tmp/eval_api/test_<env>.txt, identical to the practice fork's), same agent and settings as the
  base Gemma test-split runs (testsplit_base_gemma_r1/r2: greedy, GameBoy 8 workers, Web 6 workers + Gemma-4-31B judge
  on vLLM, Android 3 emulators). New Slurm pair eval_api (scripts/slurm/eval_api.sh + slurm/eval_api.sh): the agent
  stage needs no GPU (medium partition; Android --exclusive on an idle medium node: eclair16 has /dev/kvm, 40 CPUs);
  Web is scored by a second GPU job (--stage judge) that serves the judge and runs the judge on the host with the same
  agent flags (config unchanged, as eval.sh's judge pass). EVAL_API_DEFAULTS in scripts/utils.sh;
  gemini-3.6-flash added to lm_inference's rate table (600 qpm, as flash-lite). Both ids answer a live call;
  3.6-flash spends hidden reasoning tokens (46 output tokens for "OK"). Cap for these evals: 150 USD, separate from
  the practice fork's 200; the key is shared, so cost is attributed from per-run token counts x price, and the key's
  limit_remaining (8400.340 at the start) is the overall check. Smoke jobs (2 tasks per env per model): 297021-297028.
- 2026-11-17 (oracle PAE, smoke done, full runs launched): smoke (2 test tasks per env, max 2 tries): GameBoy Red 016
  judge = benchmark success; Link's Awakening 002 judged success but benchmark failure (truncated at 174 steps: a false
  positive of the self-judge); Web Apple--12 / --19 judged success; Android SimpleSmsReplyMostRecent and
  SimpleCalendarAddOneEventTomorrow judge = benchmark success. Measured cost per try (compute snapshots, list prices,
  matching the key's own drop): GameBoy $0.23 (~240 calls, ~680k input tokens), Web $0.033 (15 calls), Android $0.079
  (30 calls); smoke total $0.83. Projection for the full test split (545 tasks, max 3 tries): ~$90-110 expected
  (GameBoy ~$60-85, Web ~$15, Android ~$9), worst case above the 200 USD cap only if GameBoy needs 3 tries nearly
  everywhere; a watchdog (tmp_pae_watch.sh) logs real spend every 30 min to storage/tmp/pae_cost_log.txt and cancels
  the practice jobs at $190. Full practice (source oracle_test): Web 297020 (6 browsers), GameBoy 297029 (12 workers,
  all 4 games' 180 test tasks), Android 297030 (4 emulators, exclusive). Chains (storage/tmp/oracle_pae/chain_jobs.txt):
  dataset (--labels judge --n_paraphrases 0 --val_frac 0 --val_copy 200) -> fine-tune
  oracle_test_gemini_focus_r64_lr1e-4_e3 (--practice_model google/gemini-3.5-flash-lite, r64 a128 lr 1e-4 constant,
  3 epochs, max_length 6144; GameBoy two 23 h segments) -> 2 adapter evals per env (runs
  testsplit_oracle_test_gemini_focus_r64_lr1e-4_e3_r1/r2): Web 297031 / 297032 / 297033-34, GameBoy 297035 / 297036-37
  / 297038-39, Android 297040 / 297041 / 297042-43. Web tries scored by auto_eval with Gemma-4-31B (the benchmark's
  judge) then judge_agreement for all envs: 297044 (after Web practice; rerun judge_agreement for GameBoy / Android if
  they finish later). Base Gemma test-split evals 297013-18.
- 2026-11-17 (oracle PAE spend cap, fix): a separate Gemini eval job now shares the OpenRouter key (own 150 USD cap), so
  the key-wide drop in limit_remaining no longer measures practice spend. The watchdog (tmp_pae_watch.sh) now caps
  practice only: attributed from the practice pickles' token counts x list price (tmp_pae_spend.py; oracle_test and
  oracle_test_smoke runs) + a 10 USD margin for tasks in flight, cancelling 297020 / 297029 / 297030 at 190; the
  key-wide figure is logged beside it as a sanity check. First check 21:10: practice $1.23, key-wide $3.11.
- 2026-11-17 (API-model evals: smokes, launch): smokes 297021-297028 all ran end to end. GameBoy (Red 016, OoS 000)
  flash-lite 2/2 (42 and 168 steps; 218k / 960k input tokens), 3.6-flash 2/2 (31 and 52 steps; 227k / 317k input but
  69k / 46k output: hidden reasoning ~2.2k output tokens per step). Web (Apple--12, Google Map--3) both models finish
  both tasks, 70-109k input tokens per task (Gemini tiles screenshots: ~2x Gemma's count). Android
  (AudioRecorderRecordAudioWithFileName, MarkorDeleteNewestNote) flash-lite 0/2, 3.6-flash 2/2. Cost per task from
  the smokes x the base runs' ~115 GameBoy steps per test task: flash-lite GameBoy ~$0.21, Web ~$0.03, Android
  ~$0.03 (all envs ~$48); 3.6-flash Web ~$0.08, Android ~$0.08, GameBoy ~$1.6 (all 180 GameBoy tasks ~$285).
  Under the first 150 USD cap only Red's 40 GameBoy tasks fit for 3.6-flash; the user then raised the cap to 300.
  Full runs (run testsplit_<model>_r1; Android shards _s0-_s2; 3.6-flash GameBoy split into _red_r1 and _gbrest_r1):
  flash-lite GameBoy 297045, Web 297046 (+ judge 297047), Android 297048; 3.6-flash GameBoy Red 297049, Web 297050
  (+ judge 297051), Android 297052. The other 140 3.6-flash GameBoy tasks (LA, Crystal, OoS) launch from
  tmp_evalapi_watch.sh once the Red run's measured cost per task keeps the projection <= 290 (auto-cancel of all API
  eval jobs at 290 attributed); spend log storage/tmp/eval_api/cost_log.txt, launches storage/tmp/eval_api/jobs.txt.
- 2026-11-17 (API evals, finding): gemini-3.5-flash-lite's Web smoke (Google Map--3) had 10 of 15 steps invalid: 9
  of 15 replies were reasoning only (OpenRouter returns message.reasoning, content None; lm_inference writes
  "Reasoning: ..." with no "Response:"), so the executor's "Format ERROR: Both 'Thought' and 'Action'" retry spends a
  step. Replaying the context (tmp_evalapi_probe.py, text only: the stored episode keeps no image data):
  finish_reason "stop", not "length", even with 34 reasoning tokens of a 1000-token budget, and reasoning effort
  "low" did not stop it; a larger max_tokens is no fix. It is the model failing the reply format under the same
  harness as every other model, so the harness is unchanged; invalid-step rates are reported with the results.
  3.6-flash: 1 invalid step per Web smoke task. The Web smoke judges ran on 4x A5000 (lg10; A6000s all taken):
  both models 2/2.
- 2026-11-17 (API evals, user's change): the Web judge for the Gemini test-split evals is
  google/gemini-3.5-flash-lite via OpenRouter, not Gemma-4-31B on vLLM: no GPU jobs for judging. The pending GPU
  judge jobs 297047 / 297051 were cancelled. New Slurm pair eval_api_judge (CPU, medium): run_eval.py web with the
  run's agent flags and an API judge; --from_run judges a copy, so the agent runs testsplit_<model>_r1 stay unjudged
  (a Gemma judge pass can still be added for a like-for-like comparison) and the judged runs are named
  testsplit_<model>_r1_judge-gemini-3.5-flash-lite. Base Gemma's Web runs were judged by Gemma-4-31B, so the Web
  comparison has different judges. Tested on copies of the two Web smokes (297054, 297055): 2/2 each, as Gemma's
  verdicts; 6-12k judge input tokens per task (~$0.003). Note: auto_eval reads 'NOT SUCCESS' / 'SUCCESS' anywhere in
  the reply, and lm_inference prefixes Gemini's reasoning ("Reasoning: ...") to it, so reasoning text can sway the
  parse; to be checked on the full judge logs. Full judges 297056 (flash-lite run), 297057 (3.6-flash run), afterany
  their agent jobs; judge tokens count under the 300 USD cap (tmp_evalapi_cost.py parses auto_eval.log).
- 2026-11-17 (annotation v2 built, user's request; plans/annotation_v2_plan.md): train-task-guided path tasks by
  forward extension. New: cusi/search/task_context.py (phase 0: flavours from the split's train tasks only ->
  configs/annotation/<env>_train_flavours.json; GameBoy by benchmark task_category per game, Web by site, Android by
  app; a seeded sample of 3 per flavour, fixed for every call and tree; split guard + split hash in meta),
  cusi/search/path_notes.py (BUILD: TOO_SHORT / TASK; EXTEND: CONTINUE / REVISE / NEW; parser),
  cusi/search/annotate_paths.py (every node starts a chain on every branch, copied at forks; stops at NEW, leaf,
  challenge page or --max_steps 30; REVISE keeps the earlier name as a nested task; cached verdicts make reruns
  resume), Slurm pair annotate_paths, task_coverage / task_diversity --tasks v1|v2 and task_coverage
  --test_split all|test (v2 sees train tasks, so it must be scored on the test split; v1 rescored there too).
  Deviations from the plan: phase 1 (separate edge tags) dropped, the calls read v1's transition reports
  (`description`, present on every edge of all 52 main trees); results go to <run>/annotations_v2/p<prompt
  version>/ (calls.jsonl, path_tasks.jsonl, meta.json, summary.json), not nodes.jsonl, so prompt versions never mix
  and v1 is untouched; flavours are site / app / category (no hand-reviewed "kind" split).
- 2026-11-17 (annotation v2 pilot, live Gemma-4-31B on eval_server 297007, lg07; Android sat_random_s0 unless noted):
  p1: Gemma copied the examples' layout ("Task: [Settings]\n- ...": most tasks were the label) and named page
  visits as tasks; REVISE piled unrelated wandering onto a task. p2: stricter BUILD (a change or an answer; reaching a
  page only if the examples do), REVISE only for the same goal, tolerant parser (0 mismatches re-parsing p1's 163
  replies): 53 -> 11 tasks on 30 starts, much closer to AndroidWorld. p3: no accidental / garbled typed text, no
  trailing navigation; also on Red and Web (25 starts each). Code fixes from reading p3: CONTINUE no longer moves a
  task's end (Red dialogue tasks stretched to 14 edges), a REVISE to the same name moves the end without a nested
  copy (Red 35 -> 26 tasks). p3 on the full trees: random s0 1,080 calls / 357 nodes (3.0 per node), 125 tasks (89
  distinct); VLM w_typing s0 979 / 280 (3.5 per node), 108 (72). ~34 min per tree, two at a time on one 2-GPU server
  without prefix caching.
- 2026-11-17 (v1 vs v2 p3, Android s0, test split = 76 tasks): judged coverage@7 v1 0.197 (random) / 0.145 (VLM)
  vs v2 0.013 / 0.000 — v1 has 1,177 / 969 distinct proposals, v2 77 / 58 (after dedup), so coverage, a max over
  proposals, falls with the count. Per proposal (max cosine to the test tasks, Qwen3-Embedding-8B): v1 node 0.543 /
  0.546, v1 edge 0.560 / 0.560, v2 0.530 / 0.559 — no per-task gain either. Reading samples: v1 edge tasks are
  mostly "open the X page"; v1 node tasks are benchmark-shaped but imagined (never done, invented values); v2 tasks
  are grounded and closest in form to the train tasks ("Set the recurrence of the 'Renew Driver's License' task to
  'Every year' in Tasks app"), with some vague ("Toggle a switch in the Public API settings"), typed-fragment
  ("Search for events containing 'be'") and page-view ones left. p4: reaching a place counts only on GameBoy, tasks
  must name their target, no fragment typing.
- 2026-11-17 (annotation v2 p4, final prompt for the full run): Android full trees: random s0 1,068 calls / 357
  nodes (3.0 per node), 128 tasks (81 distinct); VLM w_typing s0 1,003 / 280 (3.6 per node), 70 (41 distinct);
  ~44 min each, four runs sharing one server without prefix caching (the Slurm jobs have it: 1.4 calls/s alone).
  Red and Web random s0, 25 starts each: 176 / 163 calls, 20 / 21 distinct tasks. Reading: Android VLM tasks are
  AndroidWorld-like and grounded ("Turn on Airplane mode in Settings", "Export events to an .ics file in Simple
  Calendar Pro", "Set the App start folder to 'AppData (private)' in Markor"); Web WebVoyager-like ("Find the birth
  date, death date, and place of death for Harriet Tubman on WolframAlpha", "Find the price of the AirPods 5 on the
  Apple website"); Red in the benchmark's short form ("Speak to Professor Oak", "Enter the Gym"). Left: the VLM
  explorer's typed intent strings become searches ("Search for 'Playlist management' in the VLC app"), a few vague
  ("Change the currency in the Pro Expense app") or hallucinated places ("Poke Mart in Pallet Town"), Red dialogue
  tasks spanning 8-12 edges. Android test split (76 tasks): coverage@7 v2 0.000 / 0.000 vs v1 0.197 / 0.145,
  coverage@5 0.105 / 0.105 vs 0.789 / 0.684; per proposal, max cosine to the test tasks v2 0.537 / 0.569 vs v1 node
  0.543 / 0.546, v1 edge 0.560 / 0.560 (share >= 0.6: v2 0.17 / 0.32, v1 node 0.19 / 0.17). So v2 tasks are as
  test-like per task or more (VLM), grounded by construction, but 10-20x fewer: most chains never complete a task
  (Android random: 431 of ~600 dropped while still TOO_SHORT), and coverage, a max over proposals, falls with the
  count. v1 Red s0 on the test split (40 tasks): coverage@7 0.85.
- 2026-11-17 (annotation v2 full run submitted, tmp_submit_v2.sh, ids in storage/tmp/annot_v2_jobs.txt): p4 on the
  52 main trees (Red, Web, Android 5 seeds per group, Link's Awakening 6; Android s0 random / vlm_w_typing already
  done in the pilot): 50 annotate_paths jobs (297070-297272), each followed (afterok) by v2 task coverage on the test
  split and v2 task diversity; plus v1 task coverage on the test split for all 52 (independent). All --nice=100 so
  tree continuations keep priority; A6000 x2 each. Summarise with tmp_v2_compare.py (per env / group means of v1 vs
  v2 coverage, v2 tasks, functions@7, Vendi). eval_server 297007 stopped.
- 2026-11-17 (annotation v2 p4, final prompt for the full run): Android full trees: random s0 1,068 calls / 357
  nodes (3.0 per node), 128 tasks (81 distinct); VLM w_typing s0 1,003 / 280 (3.6 per node), 70 (41 distinct);
  ~44 min each, four runs sharing one server without prefix caching (the Slurm jobs have it: 1.4 calls/s alone).
  Red and Web random s0, 25 starts each: 176 / 163 calls, 20 / 21 distinct tasks. Reading: Android VLM tasks are
  AndroidWorld-like and grounded ("Turn on Airplane mode in Settings", "Export events to an .ics file in Simple
  Calendar Pro", "Set the App start folder to 'AppData (private)' in Markor"); Web WebVoyager-like ("Find the birth
  date, death date, and place of death for Harriet Tubman on WolframAlpha", "Find the price of the AirPods 5 on the
  Apple website"); Red in the benchmark's short form ("Speak to Professor Oak", "Enter the Gym"). Left: the VLM
  explorer's typed intent strings become searches ("Search for 'Playlist management' in the VLC app"), a few vague
  ("Change the currency in the Pro Expense app") or hallucinated places ("Poke Mart in Pallet Town"), Red dialogue
  tasks spanning 8-12 edges. Android test split (76 tasks): coverage@7 v2 0.000 / 0.000 vs v1 0.197 / 0.145,
  coverage@5 0.105 / 0.105 vs 0.789 / 0.684; per proposal, max cosine to the test tasks v2 0.537 / 0.569 vs v1 node
  0.543 / 0.546, v1 edge 0.560 / 0.560 (share >= 0.6: v2 0.17 / 0.32, v1 node 0.19 / 0.17). So v2 tasks are as
  test-like per task or more (VLM), grounded by construction, but 10-20x fewer: most chains never complete a task
  (Android random: 431 of ~600 dropped while still TOO_SHORT), and coverage, a max over proposals, falls with the
  count. v1 Red s0 on the test split (40 tasks): coverage@7 0.85.
- 2026-11-17 (annotation v2 full run submitted, tmp_submit_v2.sh, ids in storage/tmp/annot_v2_jobs.txt): p4 on the
  52 main trees (Red, Web, Android 5 seeds per group, Link's Awakening 6; Android s0 random / vlm_w_typing already
  done in the pilot): 50 annotate_paths jobs (297070-297272), each followed (afterok) by v2 task coverage on the test
  split and v2 task diversity; plus v1 task coverage on the test split for all 52 (independent). All --nice=100 so
  tree continuations keep priority; A6000 x2 each. Summarise with tmp_v2_compare.py (per env / group means of v1 vs
  v2 coverage, v2 tasks, functions@7, Vendi). eval_server 297007 stopped.
- 2026-11-17 ~23:44 (API-model evals stopped, user's call): the user stopped the Gemini evals; 297060 (3.6-flash
  GameBoy rest) and 297052 (3.6-flash Android) were cancelled mid-run, the spend watcher stopped, nothing new
  launched. Final, solved / scored on the test split (Web judged by gemini-3.5-flash-lite; base Gemma's Web runs by
  Gemma-4-31B, so Web differs in judge as well as agent):
    gemini-3.5-flash-lite (complete): Web 194/289; Android 47/75 (1 not scored); Red 19/40; LA 25/40; Crystal 24/50;
      Oracle of Seasons 20/49 (1 failed to run).
    gemini-3.6-flash (partial where cut): Web 224/289; Android 54/72 (cut, 4 of 76 not done); Red 31/40; LA 26/40;
      Crystal 22/25 (cut at 25 of 50); Oracle of Seasons not run.
    Base Gemma-4-31B on the same ids: Web 193 / 158 / 162 of 289 (three runs); Android 52/75 and 46/74 (two runs);
      Red 18 / 20 / 16 / 20 of 40 (four runs); LA 16/40 (one run); Crystal and OoS: no base run on the new emulator.
  Invalid steps: flash-lite Web 298 / 3099 (reasoning-only replies), 3.6-flash Web 310 / 3007, base Gemma Web ~95 /
  2600. Spend, attributed from finished tasks' tokens x price: $147.34 (flash-lite $52.9 incl. smokes; 3.6-flash
  $92.6; Web judge $1.9); tasks in flight at the cancel are not counted (a few dollars). Key-wide drop in
  limit_remaining since this work started: $259.01, which includes the practice fork's spend on the same key.
- 2026-11-17 ~23:47 (oracle PAE, user's decision): the Gemini 3.5 Flash-Lite test-split practice was stopped by the
  user (297020 / 297029 / 297030 and their whole chain 297031-297044 cancelled, watchdog killed) and is to be deleted
  completely: storage/practice/{gameboy,web,android}/gemini-3.5-flash-lite (the user runs the deletion; nothing is
  read from or built on those dirs). At the stop: practice-attributed $97.5 (GameBoy 145 / 180 tasks, Web 183 / 289,
  Android 73 / 76 done). Kept: the base Gemma test-split evals 297013-297018 (testsplit_base_gemma_r1/r2), the
  baseline for the redo.
- 2026-11-17 (oracle PAE redo, user's call): the same pipeline with google/gemini-3.6-flash as practice agent and self-
  judge (model dir gemini-3.6-flash, source oracle_test; fine-tune oracle_test_gemini36_focus_r64_lr1e-4_e3; 2 adapter
  eval repeats per env). Cap 500 USD, practice-attributed (pickled token counts x $0.75 / $3.75 per M; output tokens
  are OpenRouter's completion_tokens, which include the billed reasoning tokens), key-wide only as a sanity check;
  watchdog every 30 min, cancel at ~$480. Fix found before the redo: OpenRouter returns 3.6-flash's reasoning summary,
  which the API client writes as "Reasoning: ...\nResponse: <answer>" (seen in the eval job's 3.6-flash trajectories);
  the dataset stage now strips it (dataset.strip_api_reasoning: from each row's response and from earlier assistant
  turns of chat inputs; a reasoning-only reply gives no row), so Gemma learns Gemini's answers, not its summaries.
- 2026-11-17 23:58 (oracle PAE redo, 3.6-flash smoke, 1 test task per env, max 2 tries): GameBoy Red 016 try 1 judge
  FAILURE vs benchmark success, try 2 judge success: the judge's reply was cut before its "Success:" line because
  3.6-flash's reasoning used up the 2000-token judge budget (a false negative from the budget, not the model). Fixes:
  judge / critique replies are parsed with the reasoning summary stripped (cusi/utils/api_reasoning.py, used by
  cusi/practice/judging.py and the dataset stage), and practice_split.sh takes --max_new_tokens (judge / critique
  budget; 8000 planned for 3.6-flash). Agent calls: GameBoy frame_diff 7 / 21 and action 1 / 22 replies were reasoning
  only (the agent's own budgets, as in the 3.6-flash eval job; unchanged, such replies give no training row); Web and
  Android none. Web Apple--12 and Android SimpleSmsReplyMostRecent judged success on try 1 (Android = benchmark
  success). Smoke cost: $0.51 by token count (GameBoy $0.34 for 2 tries, Web $0.09, Android $0.08), key-wide $0.63.
  Projection for the full test split: ~$200 expected, ~$375 worst (all 3 tries), under the 500 USD cap. The full
  practice + chain submission (tmp_pae.sh) was denied by the permission system; nothing of the full redo is submitted.
- 2026-11-18 (oracle test-split practice, gemini-3.6-flash, submitted): tmp_pae.sh run after the blocked attempt.
  Practice web 297287, gameboy 297288, android 297289 (agent and self-judge google/gemini-3.6-flash, max 3 tries,
  max_new_tokens 8000); chains: web ds 297290 ft 297291 evals 297292/297293; gameboy ds 297294 ft 297295+297296 evals
  297297/297298; android ds 297299 ft 297300 evals 297301/297302 (fine-tune oracle_test_gemini36_focus_r64_lr1e-4_e3;
  ids also in storage/tmp/oracle_pae/chain_jobs_36.txt). Baseline: base Gemma test-split evals 297013-297018.
  Watchdog tmp_pae_watch.sh: 3.6-flash practice spend by tokens + $25 in-flight margin, every 30 min, cancels practice
  at $480 (user cap $500); log storage/tmp/pae_cost_log.txt.
- 2026-11-17 (user's call: full run cancelled; path tasks become the only annotation; p5 = looser prompt): p5
  lowers the TOO_SHORT bar, accepts reaching a specific named page / screen / setting in every env, and keeps each
  chain's best partial task (TOO_SHORT replies name one on a Partial line; a chain that never names a full task keeps
  its latest). Pilot (eval_server 297277; Android s0 full trees, Red / Web random s0 25 starts):
  Android random s0 807 calls / 357 nodes (2.3 per node), 411 tasks, 308 distinct (p4 81; v1 870 node + 323 edge),
  78 partial; VLM w_typing s0 592 / 280 (2.1 per node), 281 tasks, 217 distinct (p4 41; v1 724 + 255), 54 partial.
  Web 25 starts: 48 distinct (p4 21); Red 25 starts: 38 distinct (p4 20), 29 of them partial.
  Test split (76 tasks): coverage@7 random 0.039 / VLM 0.000 (p4 0.000 / 0.000; v1 0.197 / 0.145), coverage@5 0.566
  / 0.303 (p4 0.105 / 0.105; v1 0.789 / 0.684), judge max@5 4.58 / 4.07 (v1 5.47 / 5.12). Vendi 18.3 / 15.9 on 289 /
  206 deduped tasks (v1 33.9 / 30.5 on 1,177 / 969).
  Honest read: the yield went up 4-5x on Android, but realism slid back toward v1's edge tasks: 83-86% of p5's
  Android and Web tasks are navigation ("Open the Bluetooth Sensors settings in OpenTracks", "Go to the About page on
  Wolfram|Alpha"; p4 1-12%, v1 edge 53-77% by the same regex). A chain now names a navigation task at its first edge
  and usually ends at NEW on the next, so the change that followed is cut off or vaguer (p4 "Find the birth date,
  death date, and place of death for Harriet Tubman on WolframAlpha" -> p5 "Go to the About page on Wolfram|Alpha",
  "Search for a specific notable person on WolframAlpha and view their birth and death details"). REVISE turning a
  navigation into the change done there fires rarely (39 / 13 times) but gives the best tasks ("Open the Expense
  Detail view for 'Rent Payment'" -> "Delete the 'Rent Payment' expense in Pro Expense"). Partials bypass the
  fragment rule ("Search for 'ol' in Settings"). Red p5 is mostly partials ("Move toward the Pokemon Center").
  A fix to try next (not run): in EXTEND, a navigation task followed by an action on the reached screen is REVISE,
  not NEW.
- 2026-11-17 (consolidation, user's call): the path annotation is the annotation, as if v1 never existed.
  cusi/search/annotate.py now runs stage 1 (TransitionDescriber reports for edges without `description`, as v1 did,
  update rows "description" / "window_descriptions") and stage 2 (path tasks) in one command; results in
  <run>/annotation/p<prompt version>/ (calls.jsonl, tasks.jsonl, meta.json, summary.json last). Removed: v1's
  node / edge task proposals (cusi/search/notes.py deleted; app_note moved to path_notes.py), annotations.json,
  `unannotated`, `context_frames`, the features pass, the annotate_paths module and Slurm pair (moved to
  storage/tmp/superseded_code/: rm is not permitted here), `--tasks v1|v2`. Kept: describe.py, features.py (the
  VLM-explorer search uses them live), viewer.py untouched (lazily imported by running searches; it shows v1 fields
  when present). Readers: task_coverage / task_diversity proposed_tasks read tasks.jsonl (task_coverage keeps
  --test_split all|test -> task_coverage_testsplit/), practice attempt.tree_tasks reads tasks.jsonl by start node,
  sync_data's finished check uses annotate.unfinished. Slurm pair annotate = the new command (defaults in utils.sh).
  plans/annotation_v2_plan.md -> plans/annotation_plan.md (as built). The p5 pilot outputs were moved from
  annotations_v2/p5/path_tasks.jsonl to annotation/p5/tasks.jsonl on the 4 pilot trees; the Android s0 trees'
  task_coverage_testsplit/ and task_diversity/ now hold p5 results (v1's numbers are in the entries above).
  End-to-end check from scratch (tmp_e2e_annotate.py on storage/tmp/annot_e2e/android_sat_random_s0, a copy of
  android sat_random_s0 with every v1 update field and every report stripped): stage 1 wrote 337/337 reports, a
  rerun 0; stage 2 on 30 starts 121 calls, 66 tasks, a rerun all cached; proposed_tasks and practice tree_tasks read
  them.
- 2026-11-18 (annotation p6 pilot, user's call "run the fix"; eval_server 297318, lg07; `python -m
  cusi.search.annotate`, output annotation/p6/): p6 = p5 + in EXTEND an action on (or right after reaching) a screen
  the task only reached is REVISE (the navigation becomes the task done there), NEW only when the edge leaves for
  something unrelated; partials must pass the same filters as tasks (exact target, no fragment typing). Also: a
  --max_starts pilot's stage 1 now describes only the edges its chains can reach within --max_steps (Red 209 of
  2,870 edges, Web 182 of 5,844); a full run describes all.
  Android random s0 (stage 1 from scratch: 337 reports, the tree had been stripped): 965 calls / 357 nodes (2.7 per
  node), 687 tasks, 476 distinct (p5 308, p4 81, old 870 node + 323 edge), 71 partial, 209 nested, 325 REVISE
  verdicts. VLM w_typing s0 (0 reports needed: the search wrote them): 797 / 280 (2.85 per node), 519 tasks, 338
  distinct (p5 217, p4 41, old 724 + 255), 44 partial, 198 nested, 258 REVISE. Red random s0, 25 starts: 196 calls,
  30 distinct (p5 38, p4 20), 13 partial, 7 REVISE.
  Test split (76): coverage@7 random 0.039 / VLM 0.000 (p5 0.039 / 0.000; p4 0 / 0; old 0.197 / 0.145), coverage@5
  0.539 / 0.276 (p5 0.566 / 0.303; p4 0.105 / 0.105; old 0.789 / 0.684); Vendi 19.9 / 17.6 on 411 / 287 deduped
  tasks (p5 18.3 / 15.9; old 33.9 / 30.5). Navigation share of distinct tasks 0.77 / 0.79 (p5 0.86 / 0.83; p4 0.01 /
  0.12; old edge tasks 0.73 / 0.77).
  Read: the fix did not work. REVISE fires far more (325 / 258 vs 39 / 13), but mostly to turn a navigation into a
  deeper navigation ("Open the App notifications page in Settings" -> "Open the Special app access page in
  Settings"; "Open the Custom recurrence page ..." -> "Open the date picker for the end date ..."): the prompt's own
  example "opens or edits an item listed there" let opening a deeper screen count as doing something there. Real
  conversions exist but are a minority ("Open the sort options menu in Tasks app" -> "Sort tasks by priority in
  Tasks app"; "Open the Security update settings" -> "Set a pattern for the screen lock in Settings"; "Turn off the
  'Daily device usage' toggle in Digital Wellbeing"). Coverage and Vendi stay at p5's level. Red: mostly partials
  and dialogue / sign reading in the benchmark's form ("Read the full Pokédex entry for Golbat", "Read the sign in
  front of the Poké Mart"), a few low-value ones ("Send out Char", "Move toward the Pokemon Center").
  Next fix to try (not run): REVISE only for a change made or an answer read on the reached screen; opening another
  screen, menu or dialog from it is CONTINUE for the same task (its end does not move) or NEW.
  Web part not run: web/sat_random_s0 not yet stripped by the deletion (tmp_strip_annotations.py, still in gameboy
  after 3.5 h); server released rather than held idle. Queue script for it: tmp_p6_web_ready.sh.
  Failure, mine: while the deletion ran I re-ran its tmp.sh once by accident (a parallel `bash tmp.sh` after my
  Write to tmp.sh was rejected), a second strip racing the first on gameboy/sat_random_s0 (it crashed in rmtree);
  the coordinator verified and restored the trees. Rule kept since: own script names, run only after a confirmed
  write. Also: the first p6 attempt (02:50) died when the deletion removed annotation/ under it and stripped its new
  reports. Note: stage 1 writes description / window_descriptions update rows into nodes.jsonl, the keys the strip
  removes, so a later strip must not run over trees with new annotation output (or stage 1 could write its reports
  under annotation/ instead).
- 2026-11-18 (annotation p6, Web spot check after the strip; eval_server 298204, lg03): web/sat_random_s0, 25
  starts: stage 1 wrote the 182 reachable reports; stage 2 126 calls (5.0 per start), 64 tasks, 64 distinct (p5 48,
  p4 21), 6 partial, 22 nested, 28 REVISE verdicts; navigation share 0.80 (0.76 without nested; p5 0.85, p4 0.10).
  Same pattern as Android: REVISE mostly deepens navigation ("Go to the 'Search on GitHub' documentation page" ->
  "Navigate to the 'Searching users' page ..."; "Go to the 'Tutti i punti di noleggio auto' page" -> "Go to the car
  rental locations page for Costa Rica on Booking.com"); some real info-finding tasks come out of it ("Go to the
  bundled subscriptions help article on ESPN and find the link to view bundle plans and prices", "Go to the Arcade
  section on the App Store for Vision and view the ratings and reviews for 'Alto's Odyssey — Remastered'"), and one
  unachieved / vague one ("Fill out the complaint form on the Booking.com Dispute Resolution Center page"). p4's
  WebVoyager-style tasks on the same starts ("Find the birth date, death date, and place of death for Harriet Tubman
  on WolframAlpha", "Find the price of the AirPods 5 on the Apple website") are not recovered. Server released.
- 2026-11-18 (annotation p7, user: "maximally adjusted to avoid tasks that don't fit"; eval_server 298205, lg03):
  (1) Stage-1 reports now live in <run>/annotation/transitions.jsonl (shared by prompt versions; the VLM trees'
  search-written base-row descriptions are still read from nodes.jsonl); nodes.jsonl is never written. The pilot
  trees' appended rows (android sat_random_s0 337, gameboy sat_random_s0 273, web sat_random_s0 182) were moved there
  (tmp_p7_migrate.py) and each nodes.jsonl reset to exactly strip_rows(storage/tmp/annotation_strip_backup/...)
  (checked equal after the write). (2) p7 prompt: REVISE only for a change or a specific answer read out on the
  reached screens; opening another screen / menu / dialog / tab / article is CONTINUE (end does not move) or NEW.
  Navigation-only tasks only where the train flavours have them: GameBoy yes (Red "navigation" "Enter the Pokemon
  Center", "menu" "Open the Map"; Link's Awakening navigation), Web no (its 3 open/visit train tasks all ask for an
  answer there), Android no (none). (3) Fit gate after stage 2: one call per candidate (start and end screens, the
  path's reports, the same train sample as stage 2 rather than only its flavour's: Android / Web candidates have no
  reliable flavour label), ACCEPT only if (a) the path completes it, (b) the end screen shows it done or the answer,
  (c) it reads like a train task, (d) not a one-click wrapper; candidates.jsonl / tasks.jsonl (accepted) /
  rejected.jsonl (failed criteria + reason) / gate_calls.jsonl. Embedding signal (max cosine to train tasks,
  Qwen3-Embedding-8B, tmp_p7_embed.py): AUC accepted vs rejected 0.68 (Android random), 0.64 (VLM), 0.41 (Red), 0.56
  (Web) -> no clear separation, logged only, not a filter.
  Results (stage 2 calls/node; candidates -> accepted distinct; reject rate; accepted navigation share):
  Android random s0 1,156 calls (3.2/node), REVISE 0, TASK 18, 211 candidates (104 distinct, 199 of them partials
  kept) -> 16 accepted, 13 distinct (p6 476, p5 308, p4 81, old 870 + 323); 92% rejected (failed a 100, b 188, c 3,
  d 9); nav 0.08. VLM w_typing s0 1,026 (3.7/node), REVISE 5, 90 candidates (49) -> 26 accepted, 19 distinct (p6
  338, p5 217, p4 41, old 724 + 255); 71% rejected (a 43, b 58, c 2, d 5); nav 0.21. Red 25 starts: 26 -> 20
  accepted, 15 distinct (p6 30); 23% rejected (a 4, b 5, d 3); nav 0.27. Web 25 starts: 26 -> 16 accepted, 16
  distinct (p6 64); 38% rejected (a 4, b 4, c 6, d 2); nav 0.12 (p6 0.80).
  Android test split (76): coverage@7 0.000 / 0.000, coverage@5 0.053 / 0.013 (p6 0.539 / 0.276; p4 0.105 / 0.105;
  old 0.789 / 0.684); Vendi 7.0 / 7.1 on 12 / 16 deduped tasks (p6 19.9 / 17.6; old 33.9 / 30.5).
  Read: realism is the best of any version (Web: "Use WolframAlpha Pro Premium to find the step-by-step solution
  for the partial fractions of 1/(x^3+2x^2-x-2)", "Find out what happens after making a booking for an airport
  transfer on Booking.com"; Android: "Set the 'Renew Driver's License' task to repeat every year in Tasks app",
  "Turn on Show passwords in Settings"; Red: "Speak to Professor Oak", "Read the full Pokédex entry for Golbat"),
  but on Android the gate removes nearly everything. Most Android rejections are a granularity problem, not bad
  tasks: a task is done in the middle of an edge (random edges are 5-11 steps) and the edge goes on to another
  screen, so the end node's screen does not show it (b) and the recording "continues past" it (a): "Turn off the
  'Show notification' switch in Nearby Share settings" (ends on Default Print Service), "Enable 'Show hidden items'
  in Simple Gallery Pro settings" (done in transition 4, recording continues), "Set the snooze time to 2 hours in
  Simple Calendar Pro" (ends on the Default event type menu). Stage 2 also names few full Android tasks (BUILD said
  TOO_SHORT 1,129 of 1,156 times on random): most candidates were partials. Gate misses: accepted "Send out Char"
  (Red), "Search for 'Map exploration' in the OsmAnd app" and vague "Find information about the US Constitution on
  WolframAlpha". Not loosened. Fix to try next (not run): end a task at the step that completes it, not the edge's
  end node (cut the path inside the last edge, show that step's frame to the gate), and let BUILD look at the
  step frames inside a long edge.

- 2026-11-18 PAE (Gemini 3.6 Flash practice -> Gemma LoRA): practice cost $347.56 (cap $500). Judged success: Web 244/289, Android 65/73, GameBoy 139/179 (benchmark 130). Android LoRA eval r1: 49/74 vs base Gemma 47/74 (r1) and 53/74 (r2); on 70 matched tasks 45 vs 46/49 base, 52 Gemini 3.6: no gain even though it was trained on the test tasks. Android LoRA r2 (297302) failed at emulator startup on lg06 (infra); resubmitted as 298213 excluding lg06.
- 2026-11-18 (annotation p7, Link's Awakening pilot, unchanged p7 code; eval_server 298212):
  gameboy/sat_zelda_la_random_s0 (nodes.jsonl checked clean first: 1 update row, no annotation key), 25 starts:
  stage 1 83 reports; stage 2 201 calls (TOO_SHORT 164, TASK 11, CONTINUE 13, REVISE 13); 50 candidates (22
  distinct) -> 28 accepted, 17 distinct, 10 partial, 5 nested; 44% rejected (a 10, b 6, c 2, d 13); navigation
  share 0.00 (no navigation task at all, though 6 of LA's 10 train tasks are navigation).
  Accepted include "take the Piece of Heart", "check the chest", "talk to Bucket Mouse", "talk to rabbit", "talk to
  the telephone in the bedroom". Flags: unnamed target accepted: "talk to the NPC in the room" (the gate rejected
  the bare "talk to NPC" 7 times, under c or d); vague "check the chest" (which chest); forced / no-choice: the
  scripted phone call ("answer the phone", "read the phone ringing text", "read the phone ringing and answering text
  and the click sound" / "... and the subsequent conversation": advancing a forced dialogue, 5 accepted variants on
  one start); near-duplicate names of one task ("take the Piece of Heart" / "take the heart piece" / "take the piece
  of heart"; "talk to rabbit" / "talk to rabbit NPC" / "talk to rabbit kid"). Rejections are mostly (d) "only
  advances already-open dialogue" and (a) "conversation not finished".

- 2026-11-18 (diagnosis: Android oracle_test_gemini36 LoRA = no gain on its own training tasks). Checked, in order:
  (1) LoRA applied: yes. vLLM 297301 log "Loaded new LoRA adapter" (max-lora-rank 64 = r64), agent_model = the LoRA
  name; plans are closer to Gemini's training plans than base's (difflib 0.744 vs 0.554 / 0.554; 6 vs 1 exact of 64)
  and the LoRA replays Gemini's action sequence further (mean matched prefix 0.47 vs base 0.26 / 0.30; full replay 21
  vs 8 / 11 of 65 tasks). (2) Format: train action prompts and the eval agent's prompts for the same task and step
  are the same template (diff = status-bar a11y text, notes, subgoal wording; same Gemma chat template with the empty
  thought channel; labels start at the response). One mismatch: 584 / 1522 success rows carry Gemini's raw
  "Reasoning: **...**\nResponse: ..." step summaries inside the history text of the single-prompt executor calls
  (dataset.call_messages strips only assistant turns, cusi/practice/stages/dataset.py:55-70); Gemma's eval histories
  never have it (0 / 967 action prompts). (3) Labels: 12 of the 65 judged-success tasks are benchmark failures on the
  accepted try (BrowserMultiply, CameraTakePhoto, ExpenseAddMultipleFromMarkor, ExpenseDeleteDuplicates2,
  MarkorAddNoteHeader, MarkorTranscribeVideo, RecipeAddMultipleRecipesFromImage, RecipeAddMultipleRecipesFromMarkor2,
  RetroPlayingQueue, SaveCopyOfReceiptTaskEval, SportsTrackerTotalDurationForCategoryThisWeek, TasksDueNextWeek): 386
  rows, 214 of the 889 trained. Targets are aligned (no off-by-one: base takes the target action on 85% of rows).
  (4) Training: only success rows are used (failure rows unused); --max_length 6144 dropped 633 / 1522 rows (42%;
  314 / 758 action rows), the longest = late steps (kept/dropped by fifth of episode 143/36, 88/66, 88/58, 77/77,
  48/77): the final action row is dropped in 36 / 65 episodes, 20 / 38 status rows dropped. Rows are long from the
  a11y list + 2 images + history, not the leak (removing it would rescue ~33 rows). Val (copies of train) loss
  0.876 -> 0.198. (5) Memorisation, real decodes (job 298214, tmp_dbg_repro.py / tmp_dbg_repro.sbatch, output
  storage/tmp/dbg_lora/repro_oracle_test.json; 60 kept + 60 dropped action rows, greedy, base and LoRA on one vLLM):
  action match (normalised: clear_text false, scroll index, app-name case) kept base 51 / LoRA 57 of 60, dropped base
  50 / LoRA 57 (7 fixed, 0-1 broken). The adapter does learn Gemini's decisions, but base already takes them 83-85% of
  the time on Gemini's own states. (6) Per task (one greedy run each): where the LoRA fully replays Gemini's path it
  wins 21 / 21 (base avg 18.5); on the 32 benchmark-true tasks where it leaves the path it wins 20 vs base avg 24.2
  (left at a dropped row 8 vs 10 of 14; at a kept row 12 vs 14.2 of 18; kept-row divergences are mostly small
  observation differences, e.g. one extra a11y element shifting indexes, or equivalent actions). Off the path the
  LoRA loops more (7 loop episodes vs 3 / 2; 3 with the same full response 3x in a row vs 0 / 0, e.g.
  RetroCreatePlaylist "click 271" 17x ignoring "index out of range"). Headroom is small: base already solves 41-44 of
  the 53 really-solved tasks (best case ~+10), and base r1 vs r2 differ by 6 tasks.
  Read: not a serving or format bug. (a) the length filter removes the late / terminal rows of long tasks, so the
  adapter is trained on the part base already gets right and untrained where the hard tasks are decided; (b) 18% of
  "success" tasks are false; (c) off the path the adapter is a bit worse than base (repetition), cancelling the +2.5
  on replayed tasks. Fix to try (not launched): keep every row (max_length >= 14.6k, or trim a11y list / history so
  rows fit), strip API reasoning from history text inside prompts, benchmark labels for the oracle run; verify with
  tmp_dbg_repro.py on terminal rows, the replay analysis, and >= 3 eval repeats.
- 2026-11-18 (annotation p8, Web and Android; GameBoy keeps p7; eval_server 298216, lg06): path_notes.PROMPT_VERSIONS
  = {gameboy 7, web 8, android 8}; output annotation/p8/. Changes: (1) step-level ends: stage 2 sees the latest
  edge's stage-1 window reports (frame k+1 = the screen after action k) and names the completing action ("Step:");
  a task ends at (edge end node, end_step), records "mid_edge", "replay" {from_node = the last edge's start, actions =
  its first end_step actions} and "restorable" (from_node not flagged unrestorable / replay_diverged); the gate is
  shown the last edge's screens up to that action (at most 11) and judges (b) on the last one. (2) (c) also needs a
  checkable outcome (no "find information about X"), no navigation tacked onto a done task. (3) no search whose query
  only labels the app or an activity. (4) accepted paraphrases of one start merged (content-word Jaccard >= 0.8,
  alt_ends / paraphrases). GameBoy check: Red and Link's Awakening rebuilt from their p7 caches with the p8 code
  (0 model calls): identical accepted and rejected sets.
  Results (stage-2 calls/node; candidates -> accepted distinct; reject rate (failed a/b/c/d); navigation share;
  accepted ending mid-edge):
  Android random s0 1,156 (3.2/node); 224 (113) -> 78 rows, 51 distinct (p7 13, p6 476, old 870 + 323); 65% (72 /
  130 / 1 / 11); nav 0.00; mid-edge 70 of 78; 63 of 78 accepted are partials; restorable end 26 of 78 (the rest end
  in edges from replay_diverged nodes). VLM w_typing s0 1,018 (3.6/node); 93 (53) -> 40 rows, 24 distinct (p7 19,
  p6 338, old 724 + 255); 57% (33 / 41 / 5 / 6); nav 0.08; mid-edge 25 of 40; restorable 24 of 40. Web 25 starts:
  182 calls; 18 -> 10 distinct (p7 16, p6 64); 44% (2 / 3 / 5 / 0); nav 0.00; mid-edge 3 of 10.
  Test split (76): coverage@7 0.000 / 0.000 (p7 0 / 0; old 0.197 / 0.145), coverage@5 0.079 / 0.013 (p7 0.053 /
  0.013; p6 0.539 / 0.276; old 0.789 / 0.684); Vendi 12.1 / 8.0 on 45 / 21 deduped (p7 7.0 / 7.1; old 33.9 / 30.5).
  Paraphrase merge within a start merged 0 (paraphrases sit on different starts); tree-wide the same rule would
  take Android random 51 -> 43, VLM 24 -> 21 texts, and misses some ("Modify" vs "Change the mobile data usage cycle
  in Settings"). Leakage (tmp_p8_leak.py, Qwen3-Embedding-8B vs all train + test texts): 0 accepted tasks at cos >=
  0.85; max 0.839 (Web, a train task: "Compare the prices of the iPhone Duo and iPhone Air" vs "Compare the prices
  and chips for the iPhone 14 Pro and iPhone 15 Pro"), max 0.731 against a test task.
  Read: step-level ends work (mid-edge tasks are most of the Android gains: "Disable background data for Android
  System Intelligence", "Turn off Microphone access in Privacy settings", "Mark 'Learn to Play Guitar' as completed
  in Tasks app", "Set the preferred network type to 3G for the T-Mobile SIM in Settings"); Web keeps only
  checkable tasks ("Find the average word length of the US Constitution on WolframAlpha", "Search WolframAlpha for
  ... Harriet Tubman and find her birth date, death date, and place of death"). Still unrealistic or vague among
  accepted: typed fragments ("Search for 'aex' in Simple Calendar Pro", "Add the number 7 to the blocked list",
  "Search for cities starting with 'y'"), no target value ("Modify / Change the mobile data usage cycle", "Change
  the date and time formats in Joplin", "Set a specific end date for custom recurrence", "Toggle the 'Hide status
  bar icons' setting"), navigation that slipped ("Open the To-Do list manager in Markor", "View the properties of the
  image ...", "... and view the metadata of the image"), and an inconsistent gate ("Turn off Fade when not in use"
  rejected as a one-click wrapper on one start, accepted on another). Coverage stays near zero because few test-like
  goals are ever completed on these trees.
- 2026-11-18 (reporting change, user's call): annotation coverage is reported as (1) judge_max@k, each test task's
  best LLM-judge score over its k = 5 nearest annotation tasks, averaged over the test split, and (2) top1_cos, its
  best cosine, averaged the same way (task_coverage summary.json "overall"); coverage@7 / @5 are no longer headlined.
  Test split, current versions (p8 Web / Android, p7 GameBoy): Red 5.08 / 0.606, Link's Awakening 4.72 / 0.542, Web
  2.22 / 0.363, Android random 2.99 / 0.516, Android VLM w_typing 2.68 / 0.481. The Android s0 pilots by version, in
  these terms (random / VLM, judge_max@k, top1_cos): old v1 5.47 / 0.660 and 5.12 / 0.641; p4 3.09 / 0.521 and
  3.33 / 0.528; p5 4.58 / 0.582 and 4.07 / 0.546; p6 4.46 / 0.580 and 4.05 / 0.552; p7 2.64 / 0.485 and 2.87 / 0.485;
  p8 2.99 / 0.516 and 2.68 / 0.481.
- 2026-11-18 (reporting rule, user's call): always report three numbers: coverage@8 (share of test tasks whose best
  judge score is >= 8, from per_task.jsonl; the code's COVERED_AT is 7), judge_max@k and top1_cos. Current test
  split: Red 0.200, Link's Awakening 0.125, Web 0.000, Android random 0.000, Android VLM 0.000. Android s0 pilots,
  (coverage@8, judge_max@k, top1_cos), random / VLM w_typing, from the per_task files that still exist (no new runs):
  p4 (0.000, 3.09, 0.521) / (0.000, 3.33, 0.528) [task_coverage_v2_testsplit, its numbers match the logged p4 run];
  p8 (0.000, 2.99, 0.516) / (0.000, 2.68, 0.481) [task_coverage_testsplit]. Old v1, p5, p6, p7: coverage@8 not
  recoverable (their per_task files were overwritten by later runs or deleted by the annotation strip); their
  judge_max@k / top1_cos are in the entry above.
- 2026-11-18 (failure): Web PAE LoRA evals 297292/297293 finished all 289 agent runs but skipped the judge stage and
  timing: scripts/slurm/eval.sh was edited (compact_elements flag) at 18:01 while they ran, and bash reads scripts
  incrementally. Episodes are intact; judge-only reruns 298233/298234 (judge google/gemma-4-31b-it, as base). Rule
  restated to all agents: never edit a .sh that a running or pending job executes.
- 2026-11-18 (Android oracle LoRA fix, user-approved; two arms + 3 evals each, all with compact element lists).
  Code, opt-in, default off: m3a.compact_elements (drops False / None / "" / [] fields of "UI element k: {...}"
  lines; _parse_elements' text regex also accepts a line ending at its text), M3AExecutor(compact_elements=...)
  applies it to the action prompt's and both summary-prompt lists (env element text unchanged), reached through
  Agent / make_agent(executor_options=...) and run_eval.py android --compact_elements (eval.sh --compact_elements
  true); api_reasoning.strip_embedded_reasoning + dataset --strip_prompt_reasoning (user text of every row; regex
  bounded to one history step) and dataset --compact_elements; cusi/finetune/disagree.py + run_diagnostics.py
  base_disagree / disagree_dataset (Slurm pair slurm/base_disagree.sh). Failure of mine: editing
  scripts/slurm/eval.sh (even by atomic mv) while Web evals 297292/297293 ran it made both skip their judge stage;
  the coordinator resubmitted judge-only 298233/298234. Rule: never edit a .sh a running / pending job executes.
  Dataset A (source oracle_test_cgt, episodes linked to oracle_test's, labels gt, compact, reasoning stripped,
  val_copy 200): 59 gt-success tries (50 tasks x1, 3 x2, 1 x3), 1296 rows (648 action, 589 summary, 59 plan). Checks:
  all 584 leaked rows clean (2965 removed spans, every one right after "Step k- Action selected: {..}. "); every row
  = compact(strip(old row)) for the 1226 rows present in the old dataset; no element line keeps a False field.
  Memory smokes (r64, 4x A6000, 64-128 rows): 8192 (298226), 7680 (298230), 7168 (298231) OOM in backward (~5 GB
  alloc with ~42.6 GB used); 6656 (298232) ran step 1 at 41.0 GB then OOM at step 2. So 6144 (old 44.2 GB peak).
  Rows over max_length on dataset A (dropped / action / final = status rows of 59): 6144 235 / 112 / 15; 6656
  165 / 58 / 7; 7168 120 / 35 / 3; 7680 81 / 17 / 2; 8192 22 / 5 / 0. At 6144 the final row is still dropped for 15
  episodes (14 tasks): ExpenseDeleteMultiple2, MarkorChangeNoteContent, MarkorCreateNoteFromClipboard,
  MarkorEditNote, MarkorTranscribeReceipt, RecipeDeleteMultipleRecipesWithNoise, SimpleCalendarAddOneEvent /
  InTwoWeeks / Tomorrow / AddRepeatingEvent / NextMeetingWithPerson (2 tries), SimpleSmsReplyMostRecent,
  SimpleSmsResend, VlcCreateTwoPlaylists (old dataset: 36 / 65).
  Arm B (source oracle_test_cgt_dis): base Gemma, greedy, 1000 tokens, JPEG, on all 648 action rows of A (298225):
  agrees on 544 (84%), disagrees on 104 (42 tasks; by fifth 32/36/11/15/10; 3 of 59 final rows; base never
  unparsed). My call: plan and summary rows dropped in B (the arm is the corrections only). B has 104 rows, 23 over
  6144, so 81 trained x 3 epochs = 7 optimizer steps (A: 99) - a much weaker update; noted, not compensated.
  Jobs: compact base evals 298227-298229 (testsplit_base_gemma_compact_r1..3); A ft 298237
  (oracle_test_cgt_gemini36_r64_lr1e-4_e3) -> evals 298238-298240; B ft 298243
  (oracle_test_cgt_dis_gemini36_r64_lr1e-4_e3) -> evals 298244-298246 (run testsplit_<ft run>_r1..3).
- 2026-11-18 (failure + resubmit): android sat_vlm_no_typing_qwen3-vl-32b_s1 stopped at 95 nodes, with no summary:
  its five chained jobs 296797-296801, pinned to eclairlg03, all failed on resume. The snapshot was missing, then
  the a11y forwarder APK install broke ("Failure calling service package: Broken pipe"). Resubmitted on eclairlg06
  (298251, continuations 298252/298253). gameboy sat_zelda_la_vlm_no_typing_qwen3-vl-32b_s1 is really done: it
  stopped "saturated" at 129 nodes, and its continuations exited at once.
- 2026-11-18 (user's call): filled trees (*_random_filled) are never annotated.
- 2026-11-18 (annotation: final procedure, user's call "make it the proper final version"): cusi/search/annotate.py
  is the only tree annotation; no versions. Behaviour per env: GameBoy tasks end at the edge's end node (its edges are
  2-3 steps; navigation / menu tasks allowed, as in its train flavours); Web and Android (path_notes.STEP_ENVS) end at
  the completing action inside the last edge, with replay and restorable recorded, stricter fit (checkable outcome,
  no navigation tacked on, no exploration-label searches) and paraphrase merging. Stage 1 reuses the VLM explorer's
  base-row descriptions (user: "use the edge description from the explorer"); every other edge is described by the
  annotation model; meta.json counts both. Output per annotation model: <run>/annotation/<model short name>/
  (transitions, calls, gate_calls, candidates, tasks, rejected, meta, summary). task_coverage / task_diversity take a
  required --annotation_model (outputs under task_coverage[_testsplit]/<model>/ and task_diversity/<model>/); practice
  tree_tasks reads the practising model's own annotation; sync_data checks every model's annotation. Never annotated
  (enforced in annotate.py: annotated_run): *_random_filled, *_cut69h, smoke_*, tt_* (user's calls). Slurm pair
  annotate: --run takes a comma-separated list annotated --parallel (3) at a time on one vLLM server; --vllm_extra.
  plans/annotation_plan.md rewritten as the final procedure.
- 2026-11-18 (annotation models): google/gemma-4-31b-it, and "qwen3-8b" read as Qwen/Qwen3-VL-8B-Instruct (the
  annotation needs images; the trees were explored with it; in HF_HOME). Qwen3-VL-8B on a Quadro RTX 8000 in fp16
  generated ~0.5 tokens/s (stalled; Turing attention path), and on one A5000 vLLM had 1.56 GiB of KV cache left
  (engine failed to start): Qwen runs on 2 Ampere GPUs, bf16. Smoke, android sat_vlm_no_typing_s0 (216 nodes, all
  196 reports from the explorer), full tree: Gemma (4x A5000) stage 2 726 calls, 0 unparsed (TOO_SHORT 687, TASK 29,
  NEW 10), gate 99 calls -> 38 accepted (15 distinct; 38%), 24 mid-edge, 23 min ("Enable the 'Disallow screenshots'
  option in Markor", "Enable the 'Power button ends call' toggle in Settings"); Qwen (2x A5000) stage 2 480 calls, 0
  unparsed (TASK 174, REVISE 75, NEW 127, TOO_SHORT 73), gate 254 calls -> 237 accepted (93%), 232 rows, 212 distinct,
  38 min, mostly navigation, which the prompt forbids ("Navigate to the Settings page in the Markor app", "Navigate
  to the Connected devices menu in the Settings app"). Kept as is (user: how lax Qwen is compared with Gemma is
  itself the finding; no prompt or gate change for Qwen).
- 2026-11-18 (annotation launch): scope = every finished plain search tree (no running / pending job writes to it,
  search log unchanged for 6 h; a missing summary.json is a run that ended at its time limit) whose game / env has
  train tasks: 20 Web, 37 GameBoy (Red, Link's Awakening), 20 Android in 8 batches per model (Web 4, GameBoy 3,
  Android 1; balanced by expected calls), plus android sat_vlm_no_typing_s0 (the smokes) and gameboy
  sat_zelda_la_random_s0, web sat_random_s0 (submitted once the strip has passed them, tmp_final_later.sh). Gemma
  batches 298260-298267 (A6000 x2 each), Qwen batches 298279-298286 (2 GPUs; pending ones held to the A5000 node lg10
  at nice 100 so Gemma keeps the A6000s). In-progress Qwen3-VL-32B search trees (11: android s0-s2, gameboy s0-s2,
  LA s0 / s2, web s0-s2) are annotated after their last search job ends (afterany): Gemma 298268-298278, Qwen
  298287-298297. Skipped: games without train tasks (Bomberman, Deja Vu, Sword of Hope, Crystal, Oracle of Seasons
  trees), the fills / cuts / smoke / test runs. Estimate per model (stage 2 + gate ~3.9 calls per node, stage 1 one
  call per edge without an explorer report, ~40k): Gemma ~460k calls, ~1 call/s per server -> ~130 server-hours
  (~260 A6000 GPU-hours), ~16-20 h wall if 8 A6000 pairs free (A6000s are contended: likely 1-2 days); Qwen ~330k
  calls at ~0.5 call/s -> ~180 server-hours (~370 A5000 GPU-hours), ~2 days on lg10 alone. Batches resume when
  resubmitted. Old annotations deleted: tmp_strip_annotations.py --apply on every tree not written to by a job
  (annotation/, task_coverage, task_coverage_testsplit, task_diversity dirs of the pilot trees; the 5 idle
  *_random_filled trees' old rows, annotations.json and viewers); the 5 Red *_random_filled trees under running fills
  keep their old rows (never annotated now). rm is not permitted in this session: task_coverage_v2_testsplit on
  android sat_random_s0 / sat_vlm_w_typing_s0, storage/tmp/annotation_strip_backup/, storage/tmp/superseded_code/
  and the root tmp_* annotation scripts are left for deletion by the coordinator.
- 2026-11-18 (user question: how often does base Gemma already agree with Gemini 3.6 Flash on the Web / GameBoy
  practice rows?). Base google/gemma-4-31b-it, greedy, eval budgets (Web 1000; GameBoy executor 8000, done_check
  1500, supervisor 5000), the exact training messages and images (Web chats with PNG in place, GameBoy single
  prompts as JPEG), stop [STOP]; cusi/finetune/disagree.py decode_all / agreement_report, run_diagnostics.py
  base_decode_all, debug job tmp_dbg_decode_all.sbatch (298247 Web all 1945 rows, 298248 GameBoy: all rows ~3.5
  GPU-h, so a seeded task-stratified sample of 2999, then cancelled). 0 request errors.
  Equivalence: GameBoy action = one button (7 values; order / multiplicity cannot differ), exact = normalised;
  verdict rows by their Complete / Flawed / Undone line; free text by difflib ratio. Web: action line lower-cased,
  brackets / spaces around ; ignored, trailing period dropped; typed / answer text must match (case-insensitive).
  GameBoy (actions n 1102): 0.662; fifths 0.585 / 0.691 / 0.688 / 0.701 / 0.651; final 0.800 (n 30); Red 0.741,
  Oracle of Seasons 0.707, Link's Awakening 0.652, Crystal 0.598. Verdicts: done_check 0.957, regression_check
  0.878, judge 0.846, plan_flaw 0.784. Text sim: judge_slice 0.42, plan 0.32, frame_diff 0.31, hint 0.16.
  Web (actions n 1945): exact 0.500, normalised 0.541; fifths 0.744 / 0.511 / 0.540 / 0.511 / 0.264; final 0.221
  (mostly ANSWER rows: same verb 0.874 but same text 0.157, text sim 0.586 when both answer); by Gemini verb: Click
  0.663 (verb 0.789), Type 0.549 (0.827), Scroll 0.471 (0.552), GoBack 0.31; base unparsed 44. Per site: Booking
  0.683, Google Flights 0.654, Apple / Coursera / ArXiv ~0.595, GitHub 0.549, ESPN 0.495, Google Map 0.470,
  Huggingface 0.429, Wolfram Alpha 0.393. (Android on the compact gt rows: 544 / 648 = 0.84.)
- 2026-11-18 (second annotation model -> Qwen/Qwen3-VL-32B-Instruct, user's call "instead run the 32B qwen model"):
  all Qwen3-VL-8B annotation jobs cancelled (298256, 298279-298297); their partial outputs annotation/
  Qwen3-VL-8B-Instruct/ remain in 10 trees (android sat_vlm_no_typing_s0; web sat_random_s2/s3/s4,
  sat_vlm_no_typing_s4, sat_vlm_no_typing_qwen3-vl-8b_s0, sat_vlm_w_typing_s0/s1/s2/s4) because rm is not permitted
  in this session (left for the coordinator). Qwen3-VL-32B, same prompts and gate, served with --dtype half. On 4x
  A5000 at the default memory use it had 1.2 GiB of KV cache (2.0 GiB needed for one 32k request: engine failed);
  with --gpu-memory-utilization 0.95 it serves. Smoke, android sat_vlm_no_typing_s0 (same tree as Gemma's): stage 2
  683 calls, 0 unparsed (TOO_SHORT 589, TASK 53, REVISE 10, NEW 23, CONTINUE 8); gate 166 calls, 0 unparsed -> 145
  accepted (87%), 92 distinct; 1 h 46 min on 4x A5000 (~0.17 calls/s). Gemma on the same tree: TASK 29, gate 38%, 15
  distinct. 32B accepts some navigation ("Navigate to the Legal information page in the Settings app") and
  descriptions of the recording ("... the user unchecked and then re-checked the 'Add completion date' option").
  Launch (tmp_final_launch.py): batches 298318-298325 (Web 4 on A6000 pairs at nice 100; GameBoy 3 and Android 1 on
  the A5000 node, 4 GPUs, --gpu-memory-utilization 0.95), in-progress Qwen3-VL-32B search trees afterany their last
  search job: 298326-298336. Gemma keeps the A6000s first.
- 2026-11-18 (idealized PAE, user's top priority): can practice SFT shift Gemma at all under ideal conditions? True
  test tasks, true rewards, and only the tasks Gemini 3.6 Flash solves (its test-split eval) but base Gemma failed in
  at least one of its two test-split runs. Web: 66 tasks (vs flash-lite-judged eval copies); Web has no programmatic
  reward, and by the user's call the rows keep Flash's own practice-judge labels: 58 tasks with a judged success,
  552 executor action rows (492 within max_length 6144). Red: 16 tasks (pokemon_red eval runs; 11 failed by Gemma in
  both runs, 5 in one), labelled by gt (benchmark success; a successful try kept whole): 15 tasks, 17 tries, 1617
  rows across all supervisor and executor roles (024: no gt success in 3 tries; 035: 3 gt successes, all judged
  failures, so it was missing from the judge-labelled dataset).
  New stage run_practice subset_dataset (cusi/practice/stages/subset.py): a practice run's success tries of listed
  split ids, rows by try_rows as the dataset stage makes them, val_copy train rows copied as validation. Datasets:
  practice/<env>/gemini-3.6-flash/tree/oracle_test_flashwins (task lists storage/tmp/oracle_pae/flashwins_*.txt).
  Aggressive LoRA (user's request, "make sure the behaviour shifts"): r128 a256, lr 2e-4 constant, 5 epochs,
  32 rows/step, val 50 copied rows, eval/save every 25 (run oracle_test_flashwins_r128_lr2e-4_e5; ~75 steps Web,
  ~250 Red). Evals: Web all 289 test tasks x2 (Gemma judge, then flash-lite copies *_judge-gemini-3.5-flash-lite,
  the scale of the Flash / base comparison); Red the 40 pokemon_red test tasks x2 (gt). Read: win tasks vs the rest.
  GPUs (user: clear them if needed): all pending annotation jobs held; Gemma annotation 298261-298264 requeued held
  (they resume from calls.jsonl); the first two only freed 2+2 GPUs on different nodes, so lg03 was cleared too.
  297296 (the GameBoy PAE ft continuation, likely unneeded) set to nice 100 so Red's ft takes lg07 first.
  Jobs: Web ft 298340 -> evals 298341/298343 -> flash-lite judges 298342/298344; Red ft 298345 -> evals 298346/298347.
- Failure (idealized PAE): the Web r128 ft 298340 ran out of GPU memory in the first backward (979M trainable; the r64 Web ft
  had already peaked at 43.8 of 47.4 GB on Web's long rows). Web is now r64 a128, lr 2e-4 constant, 8 epochs
  (oracle_test_flashwins_r64_lr2e-4_e8, ~120 steps; every row kept): ft 298349 -> evals 298352/298350 -> judges
  298351/298353; the old evals/judges 298341-298344 were cancelled. Red stays r128 (GameBoy rows peaked 29.6 GB at
  r64): 298345 took lg03.
- 2026-11-19 (idealized PAE, Red result + diagnosis): the aggressive Red LoRA (r128 a256, lr 2e-4, 5 ep, 1617 rows)
  broke the model: 2/40 and 3/40 (win tasks 1/16 each; rest 1-2/24) vs base 16 / 20 and Flash 31; 36-37 of 40
  episodes truncated, invalid actions 15-23 per run (base 1-5). Cause: the frame_diff rows. HistoryTracker calls
  frame_diff with max_new_tokens=60 (GameBoyRL/execution/executors/policies/history.py); Gemini 3.6 Flash spends
  that on reasoning, so its visible diffs are cut off (training targets mean 10 chars, "The dialogue"). The LoRA
  learned it: its frame diffs are "The player" (5115 times in run 1); base Gemma's are full ("nothing changed.",
  "The player character moved ..."). The executor's action history is then blind, and it repeats A next to the
  NPC until the step limit (027). Supervisor rows were also copied verbatim (identical judge_slice summaries across
  legs, self-contradictory). Every Flash GameBoy practice dataset has these rows; the full-data GameBoy PAE LoRA
  (14.8k rows, frame_diff diluted) is not broken: evals 297297/8 at 45/103 and 38/97 so far (base 39-44%).
  Reruns (gentle recipe r64 a128, lr 1e-4 constant, 3 ep, as the six-task focus that helped): subset_dataset
  --drop_tags (new): oracle_test_flashwins_act (executor action rows only, 529) ft 298410 -> evals 298411/298412;
  oracle_test_flashwins_nofd (frame_diff dropped, 1233) ft 298413 -> evals 298414/298415. Web LoRA (r64 lr 2e-4
  e8, final validation loss 0.122) evals 298350/298352 running; Web rows are action rows only (no frame_diff).
- 2026-11-19 (idealized PAE, Web result): the Web LoRA (r64 a128, lr 2e-4, 8 ep; final validation loss 0.122) broke
  the model: 25 / 21 of 289 (flash-lite judge; Gemma judge 19 / 20) vs base 187 / 198, Flash 224; win tasks 3/66;
  ~192 of 289 episodes hit max_iter 15. Not the frame_diff bug (Web rows are action rows only; 45 of 58 episodes
  end in ANSWER): the LoRA copies Flash's dithering ("Wait, let's look closely ...", 136 of 552 rows say "Wait"),
  re-clicks the same element and never answers. Over-fitting from the recipe. Gentle rerun (r64 a128, lr 1e-4, 3 ep,
  as the Red reruns): ft 298752 -> evals/judges 298753(judge 298754) 298755(judge 298756).
  Old full-data PAE LoRAs (r64 lr 1e-4 e3, all practice rows): GameBoy 73 / 68 of 179 vs base 70 / 78 (Red 12 / 12
  of 40 vs base 16 / 20); Web 180 / 179 (Gemma judge) vs base 177 / 186: no gain, no collapse.
- 2026-11-19 (GameBoyRL pull + frame diff 1000): pulled GameBoyRL to 0ce68ea (env server, execution/game_api.py, comments
  stripped) and set the frame diff budget 60 -> 1000 (history.py; pushed 50a814a; root bump b35b83f). GameBoy searches
  296789/296804/296809 and fills 296881/296885 requeue-held for it. Smoke tests: Flash 2-task Red eval failed on
  OpenRouter 402 (out of credits; flash-lite re-judges 298754/298756 will hit it too). Gemma on Quadro RTX 8000
  (298997) cannot start vLLM: Gemma-4 kernels need 96 KB shared memory, Turing has 64 KB. Gemma evals stay on A6000s.
- 2026-11-20 (Flash wins + mix, action only): Qwen3-VL-8B smoke (299312, Quadro) on the pulled GameBoyRL ran supervisor
  legs and saved an episode; holds on GameBoy searches/fills released. Overlap smoke inside annotation 298260 starved
  (shared vLLM ~25 tok/s across 16 requests): never smoke on a busy shared server. Android A/B (76 tasks, 3 runs):
  base compact 53/51/50, arm A (all gt-success rows, 1296) 56/54/52, arm B (104 disagreement rows) 47/45/46.
  New training sets: Flash-win tasks + seeded 25% of the other Flash-success tasks (Gemma succeeds in both runs),
  executor action rows only. Red 16 + 4 (005 009 039 042), 587 rows, gt labels; Web 66 + 40, 792 rows, Flash judge.
  Source oracle_test_flashwins_mix25_act. Gentle r64 lr 1e-4 e3 / aggressive lr 2e-4 e5 (Red r128, Web r64 since r128
  OOMs). Red gentle ft 299321 (3 A6000s: no node had 4 free; rows_per_step is global, so same recipe) -> 299322/3;
  Red aggressive 299324 -> 299325/6; Web gentle 299327 -> 299328/30 (judges 299329/31); Web aggressive 299332 ->
  299333/5 (judges 299334/6). All other pending jobs at nice 100.
- 2026-11-20 (Red 5x): mix aggressive Red 12 / 12 of 40 (worse than base); Web aggressive mix chain (ft 299332 and its
  evals/judges) cancelled by request. Red r1/r2 differ only by vLLM batching nondeterminism (greedy, seed is a label).
  Base Red r1/r2 ran before the GameBoyRL pull + frame diff 1000, the mix evals after, so base is rerun on current
  code: 5x base (testsplit_base_gemma_fd1000_red_r1-5, 307582-6) and gentle mix r3-r5 (307587-9) on test_red.txt.
- 2026-11-20 (Red 5x result, current code): base Gemma 18/15/20/16/17 of 40 (mean 17.2; Flash-win 16: 3.8, range 2-6;
  4 added: 2.6; rest 20: 10.8, 10-11). Gentle mix (oracle_test_flashwins_mix25_act r64 lr1e-4 e3) 19/20/18/19/17 (mean
  18.6; wins 6.4, 6-7; added 3.0; rest 9.2, 9-10). Wins up +2.6 with no overlap below base's best; rest down -1.6
  (mix never above base's worst); total +1.4 within noise. Flash's action rows transfer to the tasks trained on, at a
  cost elsewhere; mixing in 4 Flash-success tasks did not protect the rest.
- 2026-11-21 (Web mix gentle + all-Flash Red): Web gentle mix (299327) 103 / 94 of 289 on the Gemma judge vs base 177 /
  186: hurts. Its flash-lite re-judge left 116 / 289 episodes 'unjudged' with no error (8 / 3 successes): not usable,
  cause not found yet. Flash never loses a Red task to Gemma (0 tasks where Gemma succeeds in any of 5 runs and Flash
  fails; 9 fail for both); on Web 11 tasks Gemma solves twice and Flash fails. New: Red gentle LoRA on all Flash
  success tries, action rows only (oracle_test_flashall_act: 31 of 40 tasks, 1129 rows; r64 lr1e-4 e3) ft 311166 ->
  5 evals 311167-311171 (testsplit_oracle_test_flashall_act_r64_lr1e-4_e3_r1-5).
- 2026-11-21 (why the All-Flash Web LoRA loses tasks base solves): not Flash's "Wait," style (it is more common in
  failed episodes for every model, Flash and base included; the LoRA's successes have Flash's rate) and not replay
  of Flash's trajectories (prefix overlap the same in successes and failures). Cause: training rows teach exploring
  but often not ending. The success cut (judge safe_success_point + safety_margin 2) removed the ANSWER step from 49
  of 244 successful Flash tries, and the 6144-token cut drops late rows of long tries, so only 146 / 244 tries keep
  their ANSWER row (84% of 1-5-step tries, 39-53% of longer ones). On the 158 tasks Flash and base both solve, the
  LoRA fails 31% of runs on the 52 tasks with no ANSWER row in training vs 8% on the 105 with one; 12 of the 17 tasks
  lost in both runs had none. The lost episodes scroll/click to the 15-step limit and answer from memory. Fix: keep
  the final ANSWER row of a success try (the cut should not apply past the end, or keep it as the last row), and
  lift --max_length so long tries' late steps survive.
- 2026-11-21 (sampled test evals): eval.sh / run_eval.py --temperature T (gameboy, web, android): every agent call
  (supervisor, executor, frame diff) at T via AgentVLM(temperature=T), recorded as decoding "t<T>" in the run config;
  default unchanged (greedy, decoding "greedy", so existing runs resume). The Web judge (auto_eval.py) stays at 0.
  Edited while all-Flash Red evals 311167-71 were queued/running (user's call; default path unchanged; the .sh
  files replaced atomically so running bash keeps the old copy). Live check against 311167's server (srun --overlap,
  127.0.0.1): greedy gave 1 distinct output pattern in 4 (one word differed: batching nondeterminism), t=1 gave 3.

## 2026-11-21 All-Flash action-only gentle LoRA on Red: no gain (5x)
- testsplit_oracle_test_flashall_act_r64_lr1e-4_e3_r1-5 (1,129 rows, all 31 Flash-solved tasks, gt labels):
  totals 17/15/18/16/14, mean 16.0 vs base 17.2 and mix gentle 18.6. Per group (mean of 5): Flash wins (16) 4.4
  (base 3.8, mix 6.4), extras (4) 2.6 (2.6, 3.0), other Flash-solved (11) 8.8 (10.8, 9.2), Flash fails (9) 0.2.
  Training on every Flash success dilutes the win-task signal and still costs tasks base solves; mix gentle stays best.

## 2026-11-21 Base Gemma 20x at t=1 on test (Red, Link's Awakening, Web, Android)
- Runs testsplit_base_gemma_t1_{red,la,web,android}_r1-20, --temperature 1 --seed r, base google/gemma-4-31b-it.
  Red test_red.txt (40) workers 8; LA test_la.txt (40, the links_awakening lines of test_gameboy.txt; its other
  50 "legend" lines are Oracle of Seasons) workers 8; Web test_web.txt (289) workers 3, Gemma judge, 5 dependency
  lanes of 4 so at most 5 hit the live sites at once; Android test_android.txt (76) --exclusive, 3 emulators,
  compact elements, lg06 excluded as before. Red r13-20 pinned to lg10 (gres rtxa5000:4, two lanes of 4).
  Job ids: scratchpad t1_jobs.txt (first ids 320383-...).

## 2026-11-22 Temperature sweep: Gemma holds up to T=1.8; T=2 starts breaking on Web
- The t=1 batch was cancelled at the user's request with Red 20/20, LA 20/20, Android 17/20, Web 5/20 done
  (means: Red 19.1/40, LA 19.1/40, Web 189.2/289, Android 52.8/~74.5; Red pass@20 30/40, LA 29/40).
- cusi/eval/temperature_sweep.py (run_diagnostics.py temperature_sweep; slurm/temperature_sweep.sh, job 332139):
  recorded test-set calls from the t=1 runs replayed through AgentVLM at T in 0,0.5,1,1.2,1.4,1.6,1.8,2 (vLLM
  rejects T>2 with a 400), 40 calls per tag x 2 samples, each reply through the env's own parser. vLLM applies
  Gemma's generation config (top_k 64, top_p 0.95) since the client sends only temperature.
- GameBoy (action, done_check, frame_diff, plan) 100% valid at every T; Android 99-100%; Web 95-99% with the same
  2-3 hard prompts failing format at every T including 0 (ANSWER without "Action:" after a CAPTCHA timeout).
- No degeneration at T<=1.8 anywhere. At T=2 on Web: 2/80 replies run to the 1000-token cap, 2/80 leak Dutch /
  Portuguese words, 4% glued words ("specificObservation"); Android 1/80 garbled. Agreement with greedy at T=1/1.8/2:
  GameBoy action 0.88/0.85/0.86, Web 0.64/0.56/0.51, Android 0.85/0.88/0.83. Highest safe T = 1.8 (format-level;
  task success at high T not measured). Results: storage/tmp/temperature_sweep/v1/.
- The sweep's arg tables are in scripts/slurm/temperature_sweep.sh, not scripts/utils.sh (pending jobs source it).

## 2026-11-23 Why the Web / Android tree world models fail: the action is an index (diagnosis, no code change)
- Measured on the committed steps of the Gemma trees (sat_vlm_no_typing_s0-4 + sat_vlm_w_typing_s0-4; JSON only:
  text_boxes, parsed_action). Web 162,083 steps, 66% indexed; Android 47,234 steps, 61% indexed.
- An index is an address in one observation's element list, not an element. The most common element behind an index
  covers 14% of its uses (Web) / 15% (Android), 29% / 48% within one site / app. Of elements seen on >=5 screens, 76% /
  60% carry more than one index (p90 8 / 7). One click on the same url / activity renumbers some surviving element
  in 29% of steps on both; a window scroll in 41% (Web) / 37% (Android).
- The box (what BoxActionWorldModel gets in place of the index) says where, not what. Position -> element within a
  site / app (64 px cells): 49% Web, 83% Android. On Android 49% of acted-on boxes lie inside another indexed box and
  11.5% have no text; 7.4% of indexed actions name an element text_boxes does not hold (merged container / label,
  text-less container) and the env executes 98% of them, so the WM sees NO_BOX for a real click.
- Lookup test, memory = 8 trees, test = the 2 s0 trees (same sites / apps, other seeds), valid indexed steps with a
  box; prediction = the next screen of a memory step with the same key taken on a DIFFERENT screen; score = Jaccard
  overlap of the screens' text sets (copy = the screen before):
  Web (20,231 steps): index 0.11, position 0.21, element (role, text) 0.63, copy 0.40; same screen + element 0.89
  (covers 46%). Android (4,197): index 0.18, position 0.38, element 0.53, copy 0.30-0.35; same screen + element 0.83
  (covers 62%). Index and position transfer worse than "nothing changes"; the element's identity transfers.
- So the box WM is a per-screen lookup: in-tree val (seen screens, held-out expansions) beats copy (Web 0.62 vs 0.42,
  Android 0.79 vs 0.57), real task steps do not. The WM test also cannot judge indexed steps: eval episodes store only
  parsed_action (info["target"] is not carried into the episode step's extra; 0 of 2,015 Web / 658 Android indexed
  steps of testsplit_base_gemma_t1_*_r1 have it), and on those the WM returns the copy (click cos 0.6645 vs copy
  0.6645), probably because in training a click without a box was usually a no-op (not checked).
- Other limits, not index-related: typed text is dropped although different texts in the same field give different
  screens in 87% (Web) / 98% (Android) of fields; the same screen content + same element gives a different next
  screen in 14% (Web) / 25% (Android) of repeated cases (text overlap < 0.8); 42% (Web) / 52% (Android) of clicks lead
  to a screen where over half the text is new; 81% / 68% of (screen, type, element) triples were taken once.
- Proposed, not built: (1) carry info["target"] into eval episode steps; (2) action = type + element descriptor (role,
  text, box) + typed text, the index resolved inside plan() / observe(); (3) WM keyed on the element (text / crop
  embedding in place of or beside the box), retrieval over tree transitions first, learned model as fallback.
- Scripts (scratchpad, not in the repo): index_diag.py, key_diag.py.

## 2026-11-23 Random-prefix practice on the test split (oracle), Red + Web, Gemini 3.6 Flash
- User's design: start at a test task's benchmark start state with the task known, take 10 random actions, then PAE
  practice (judge -> critique hint -> retry, max 3 tries) recording the judge and the benchmark's success; gt-labelled
  successes -> gentle LoRA -> eval. Within a repeat every try replays the same prefix (user's call); labels gt first;
  practising model Gemini 3.6 Flash (OpenRouter, BYOK Google key). An oracle (test tasks), as oracle_test before.
- New: run_practice.py attempt_split --random_prefix N (cusi/practice/stages/attempt_test.py random_prefix): N
  sample_action steps after reset, seeded by sha1 of the task id (which carries the repeat); a prefix that ends the
  episode is redrawn (next seed, up to 5); the env's step budget is raised by N; each try records entry["prefix"]; not
  in the agent report, so no rows. GameBoy and Web only (Android refuses). Web test mode is unscoped, so a random click
  may leave the site.
- Smoke (oracle_test_rp10_smoke, 336821/336822, max 2 tries): Red 017 judge + gt success on try 1 after the prefix
  (gb.pokemon_red.000 is not a test id, so only one Red task ran); Web Apple--12 / --19 judged successes, 10/10 valid
  prefix actions, Apple--12's two tries replayed the identical prefix.
- Full runs, source oracle_test_rp10 (repeat 0, max 3 tries, 8 workers): Red 40 test tasks 337129, Web 289 test
  tasks 337130. Next: score_web_tries (benchmark judge) for Web gt, subset_dataset --labels gt, gentle LoRA (r64 a128,
  lr 1e-4 constant, 3 ep), eval.
- 2026-11-23 (random-prefix chain): Web gt labels come from score_web_tries (auto_eval per try, web_scores.json);
  subset_dataset now reads them (attempt_test.load_web_scores / try_gt, shared with judge_agreement) and refuses Web
  gt without them. Judge for the Web tries: google/gemini-3.5-flash-lite (OpenRouter, no GPU); on the smoke it agreed
  with Flash's own verdict on 3/3 tries. Smoke datasets oracle_test_rp10_smoke_gt_act: Web 27 rows, Red 124 rows
  (executor action rows only, the 8 non-action tags dropped as in the Red action-only sets; no prefix rows).
  Chain (Slurm afterok): Red 337129 -> dataset 337616 -> gentle LoRA 337617 (r64 a128 lr1e-4 constant e3, source
  oracle_test_rp10_gt_act, run r64_lr1e-4_e3) -> evals 337618-22 (5x, greedy, test_red.txt); Web 337130 -> judge
  337623 -> dataset 337624 -> LoRA 337625 -> evals 337626/7 (2x, Gemma judge as the base runs). Eval run names
  testsplit_oracle_test_rp10_gt_act_r64_lr1e-4_e3_r<k>. attempt_test.py was edited while 337129/337130 ran: their
  workers are threads of the already-loaded module, so they keep the old code; the edit only adds functions.

2026-11-23 rp10 Red chain: practice 337129 done (3h36m). Gt dataset 337616: 69 tries, 33 benchmark successes, 29/40 tasks with rows, 1137 train + 50 val executor rows. Fine-tune 337617 got a 4x A6000 node at once, so the T=1.8 jobs were not held. 106 optimizer steps.
2026-11-23 rp10 Web chain: practice 337130 done (4h08m, 518 tries). Flash-Lite benchmark judge (337623): try1 198/289, try2 52/127, try3 34/102. Gt dataset 337624: 284 success tries, 230/289 tasks with rows, 2622 train + 50 val executor rows. Fine-tune 337625 pending on Priority; every T=1.8 job was already running (none left to hold), so Red eval r5 (337622) was held instead so the fine-tune takes the next 4x A6000 node; release it once 337625 runs.
2026-11-23 rp10: Web fine-tune 337625 started after about 1h pending; Red eval r5 337622 released.
2026-11-23 rp10 Web fine-tune 337625 failed (eclairlg17, 1h34m): NCCL "unhandled cuda error" in the dcp gather of the first checkpoint save (step 25/217), the same failure as the Android s0 fine-tunes (293004/293291/293395), again with long rows (mean 3590 tokens, 299 over 6144 dropped, 43 GB peak) where Red (mean 639, 29 GB) saved fine. No checkpoint written. Resubmitted as 341846 with save_every 5 (the fix that let 293462 complete) and NCCL_DEBUG=INFO; evals 337626/337627 repointed to afterok:341846. Web steps take ~210 s (vs Red 22 s), so 217 steps is ~13 h.
2026-11-23 rp10 Red result (5 greedy evals on test_red.txt, 40 tasks, seeds 1-5): LoRA on random-prefix Flash practice successes (gt labels, 1137 rows) 19/20/17/19/19 = mean 18.8 (sd 1.1) vs base Gemma fd1000 18/15/20/16/17 = 17.2 (sd 1.9). Diff +1.6 +- 1.0 SE, about 1.6 SE, not significant. Per task: LoRA better on 9, worse on 8, tied on 23, so the gain is not concentrated on a set of tasks; the main visible effect is lower run-to-run spread. Note the practice was on the same test tasks (oracle setting), so this is an upper bound on what the practice transfers.
2026-11-23 rp10x10 Red (user: 10 random prefixes per task, gentle LoRA, eval, max priority): repeats 1-9 added to practice/gameboy/gemini-3.6-flash/tree/oracle_test_rp10 (r0 reused; prefix seed is per task_id incl. repeat, so 10 distinct prefixes per task, each shared by its <=3 tries). 360 task-repeats, est ~$580 Flash. One job, 72 workers (337129 used 2.7 GB and 17 CPU-min over 3.6 h at 8 workers: API-bound). Chain: practice 345734 -> gt act dataset oracle_test_rp10x10_gt_act 345735 -> gentle LoRA r64 lr1e-4 e3 (same recipe as rp10; ~10x rows so ~10x steps) 345736 -> 5 greedy evals 345737-345741 (testsplit_oracle_test_rp10x10_gt_act_r64_lr1e-4_e3_r1-5).
2026-11-23 Resubmitted the two unfinished Android Qwen trees (user): sat_vlm_no_typing_qwen3-vl-32b_s1 (63 exp, its 298251-3 chain held after a launch failure pinned to lg06, left held) as 347157 -> 347158 -> 347159; sat_vlm_no_typing_qwen3-vl-8b_s1 (169 exp; 295164 hit the 3-day job limit before the 69 h wall_s budget, resume 296406 failed vLLM init on lg14) as 347160 -> 347161. Same args, --overwrite false (resume), --exclusive, lg06/14/15 excluded.
2026-11-23 Annotation restarted (user: annotate all trees, below the Red rp10x10 practice -> fine-tune). Reused the held 2026-11-18 jobs (they resume from calls.jsonl; Gemma batches 298261-298267 + post-search 298268-298278, Qwen3-VL-32B batches 298318-298321/298324/298325 + post-search 298326-298336; 298260/298322/298323 already completed; the cancelled Qwen3-VL-8B jobs stay cancelled). Every A6000 job now depends after:345736 (starts once the Red rp10x10 fine-tune has started) at nice 100. The Android batches (298267 Gemma, 298325 Qwen on A5000) include sat_vlm_no_typing_qwen3-vl-8b_s1, so they also wait afterany:347161 (its resumed search); android sat_vlm_no_typing_qwen3-vl-32b_s1 (298269, 298327) waits afterany:347159. The A5000 GameBoy Qwen batch 298324 does not compete for A6000s: released now (running on lg10). Old held 298251-298253 (the stuck 32B-s1 search) left held.
2026-11-23 rp10x10 Red: practice 345734 done (4h00, 360 new task-repeats, 3 transient API errors retried OK). 400 task-repeats: 295 benchmark-solved (273 on try 1); 35/40 tasks solved under >=1 prefix (vs 29 with one prefix); 22 tasks solved under all 10 prefixes, so easy tasks dominate the data. Dataset 345735: 341 success tries, 10702 train + 50 val action rows. Fine-tune 345736 started at once on eclairlg03: 1003 steps (~6.5 h at ~22 s/step). Annotation A6000 jobs became eligible when it started (after:345736), as asked; if the evals queue behind them, hold pending annotation until the evals start.
2026-11-24 rp10x10 Red fine-tune 345736 done (6h55, validation loss 0.3653). Its evals had no A6000s (annotation held 14 pairs; Slurm start estimates 14:25-21:20 UTC). User approved requeuing the 5 newest annotation jobs (298263, 298264, 298334, 298335, 298336; ~1.2 h in, resume from calls.jsonl); they are held, with pending 298261/298262, until the evals run. Evals 345737-345739 started at once (lg05, lg16, lg17); the GPUs freed on lg02 and lg07 went to other users' jobs, so 345740/345741 still wait. Web evals 337626/337627 running since 01:37 / 01:58.
2026-11-24 rp10x10 Red evals: r1 28, r2 29, r3 27 of 40 (mean 28.0 vs base 17.2, rp10 18.8, no-prefix 16.0). User then asked to stop all running practice jobs and evals: cancelled Red evals r4 345740 / r5 345741 (partial) and Web rp10 evals 337626 / 337627 (~5 h in, partial). Annotation and searches left running.
2026-11-24 rp10x10 Red, judge labels (user: redo the fine-tune on judged success instead of benchmark gt, eval 5x, max priority). Same practice run (oracle_test_rp10, 400 task-repeats), subset_dataset --labels judge (tries the practice judge called success, cut at its safe success point) -> oracle_test_rp10x10_judge_act 352827 -> same gentle LoRA recipe 352828 -> 5 greedy evals 352829-352832, 352834 (testsplit_oracle_test_rp10x10_judge_act_r64_lr1e-4_e3_r1-5). Compare with the gt-label run (28.0 over 3 runs).
2026-11-24 Judge-label dataset 352827: 292 judged-success tries (vs 341 gt), 9075 train + 50 val rows, 37 tasks with rows (vs 35 gt). Fine-tune 352828 needs 4 A6000s on one node: requeued+held annotation 298321/298334 (both on lg16, user OK'd requeuing for these jobs); it started on lg16.
2026-11-24 Gemma T=1.2 practice (user: same as the Flash rp10x10 practice, Gemma at T=1.2, more retries; 10 tries; judge = Gemma itself but greedy). New attempt_split --temperature: every agent call at that temperature, judge and critique on a separate greedy AgentVLM (config.json records temperature, judge_greedy). Same 40 Red test tasks, repeats 0-9, --random_prefix 10 (same prefixes as Flash: the seed is per task_id incl. repeat), --max_tries 10, source oracle_test_rp10_t1.2_tries10 under gemma-4-31b-it. 6 shards (slurm/practice.sh, vLLM on 2 A6000s each, 8 workers): 352950-352955. Requeued+held annotation 298336, 298262, 298263 for GPUs; shard 5 lowered to 8 CPUs (lg07 had 2 GPUs but only 8 CPUs free). First try ran OK (judge failure / benchmark success on 045_r0, 10 steps).
2026-11-24 FAILURE: Gemma T=1.2 practice shards 352950-352955 judged 0 of 202 tries successful (128 were benchmark successes). Cause: strip_api_reasoning (2026-11-18, for API thinking models' "Reasoning: <summary>\nResponse: <answer>") turned any reply starting "Reasoning: " without the Response separator into "" (meant for API reasoning-only replies). Gemma's judge format starts with "Reasoning:", so every verdict parsed as failure; Gemma's agent replies start with it too, so a Gemma dataset would have dropped them. Every vLLM-model practice judged since 2026-11-18 is affected. Fix: strip_api_reasoning(api=False) keeps such replies; judging passes api from the vlm (not vLLMModel), dataset/subset_dataset from --model_backend. Re-parsing the 440 saved tries: judge vs benchmark precision 0.91, recall 0.82. User chose to keep the 44 finished episodes (rejudge offline later) and continue: shards resubmitted 357419-357424 (8 CPUs each).
2026-11-24 rp10x10 Red, judge labels, evals: r1 22, r2 20, r4 21, r5 22 of 40 (mean 21.25 over 4; r3 352831 still running, 18 of 34 so far) vs gt labels 28.0 (3 runs) and base 17.2. Judge labels keep most of the tasks (37 with rows vs 35) but lose about 7 points: 292 judged-success tries vs 341 gt, each cut at the judge's safe success point instead of kept whole.
2026-11-24 Cleanup before commit (behaviour-preserving, checked offline: old vs new subset_dataset byte-identical on 2 real Red tasks for both label kinds; AgentVLM.api equals the old isinstance test on vllm and openrouter): AgentVLM stores model_backend and exposes .api, replacing judging's peek at vlm._model and run_practice's ctx "model_backend" key. No .sh that a running or pending job executes was touched.
