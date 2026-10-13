# Decisions taken while implementing plans/plan.md (autonomous run, started 2026-10-04)

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
