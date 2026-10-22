# End report: plans/archive/plan.md implementation (autonomous run, 2026-10-04)

Everything in plans/archive/plan.md's "Order of work" was built and run at small scale on eclair. Design
decisions are in `plans/decisions.md` (D0.x, D1.x, D2.x); this report covers what exists, what ran,
the results, and what needs your attention. **Nothing was committed** (your settings deny
`git add/commit`); see "Uncommitted changes" at the end.

## Needs your attention (most important first)

1. **PPO with the plan's sequence-level ratio barely trains.** The ratio is taken over the whole
   generated action (~150 tokens), so one optimizer step moves it a lot. At LR 1e-5 (first
   GameBoy update, and the Android/Web dev runs): clipfrac ~0.8, approx KL 0.2–0.5 per update.
   At LR 2e-6 with a 0.1 target-KL stop (the new default), every GameBoy update stops after
   about one minibatch, and over 12 updates the policy stayed at the reference (KL ~0). An opt-in `--ratio_mode token_mean` (mean per-token log ratio) updates
   stably at LR 1e-5 (approx KL ≈ 0, KL to the reference model slowly rising). **Decide which
   ratio to use** before long runs (D2.6, D2.7, D2.15). Comparison runs:
   `storage/explore/gameboy/dev_viridian` (sequence, LR 2e-6) vs `dev_pewter_tokenmean`.
2. **Task quality from the judge is lenient.** Some proposed tasks are trivially satisfiable and
   the judge accepts them (Android "Create a new contact by clicking the plus icon" succeeds
   after one click; GameBoy Viridian attempts "succeed" at the 50-step budget). Data volume is
   fine; whether these tasks are worth training on is your call.
3. **No validation rows in the Part 1 datasets.** With 3 practice episodes per task and
   `val_frac` 0.2, floor(0.6) = 0 episodes go to validation (GameBoyRL's rule). Use
   `run_practice.py ... dataset --val_frac 0.34` (or more practice attempts) if you want a split.
4. **Short dev runs only.** Exploration runs are 320 (Android/Web) and 1536 (GameBoy) steps;
   the world models are trained on 16–50 episodes. The world models are about level with the
   "nothing changes" baseline on held-out episodes (Android below it: cosine 0.883 vs 0.904) and the
   decoders give blurry layouts, no legible text (`storage/explore/*/world_model/*/predictions.png`).
   That is a data-size result, not a verdict on the architecture.
5. **Bug found and fixed mid-run (D2.16):** Web text novelty treated WebVoyager's tab-separated
   element list as one line in the Web dev run (`explore/web/dev_arxiv`). Fixed; that run's
   rewards can be recomputed with `run_explore.py tasks --rescore_text_alpha 0.5` (done for its
   curiosity tasks).
6. **Two environment changes to existing CUSI code** you may want to review:
   `AndroidPlayEnv(start_app=...)` (scenes open their app, D1.16) and recovery from a failed
   snapshot restore (D1.18); `WebVoyagerPlayEnv` serialises Chrome start-up (D1.15).
   `scripts/container.sh` now binds `$HF_HOME` and takes
   `CUSI_CONTAINER_NV=1` for GPUs.
7. **Not built:** a world-model executor (the plan lists the world model and the decoder, not
   the executor, D2.14). GameBoy OCR is still a TODO ([GB-OCR] slots are in place: GameBoy runs
   use frame novelty only and no text block).

## Part 0: shared vLLM — done

- `/nas/eclairnas01/users/ashokd/vllm/`: `.venv` (vLLM 0.30.0, torch 2.13.0+cu130,
  transformers 5.18.0, Python 3.12), weights in `$HF_HOME` (Gemma 4 26B-A4B-it, SigLIP 2 base,
  Qwen3-VL-2B-Instruct), `setup.sh`, `serve_vllm.sh` (subshell venv, `setsid nohup`, `-tp` =
  visible GPUs, waits for /health, returns), `stop_vllm.sh` (kills the process group), `env.sh`,
  `state/` (logs, PIDs, versions.txt).
- CUSI: `scripts/serve_vllm.sh` / `scripts/stop_vllm.sh`, port and server limits in `configs/project_vars.yaml`, install path (`shared_vllm_dir`) in `configs/private_vars.yaml`.
- Verified (job 282699, eclairlg03, 2x A6000): healthy in ~2–3 min; text and image requests
  through `cusi.utils.build_model(model_backend="vllm")` correct; ~725 output tok/s at batch 32.
- Two fixes needed: bind to 127.0.0.1 (D0.2) and `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer's
  JIT sampler cannot build without a matching CUDA toolkit, D0.5).

## Part 1: practice pipeline — done, small-scale test run on all three environments

Code: `cusi/practice/` + `run_practice.py`. Native executors behind one skeleton:
GameBoyRL's `single_actions` (prompts byte-identical to GameBoyRL's, checked in
`tests/practice_gameboy_prompt_test.py`), M3A (action + summary calls), WebVoyager (one chat,
3 screenshots). Six shared stages ported from GameBoyRL. Datasets are JSONL chats with image
paths; hint/guidance blocks strip to exactly the native prompt (`tests/practice_strip_test.py`).
All six stages also run with a scripted model (`tests/practice_pipeline_mock_test.py`, all envs).

Small-scale test (job 282702, one A6000 node, vLLM + 3 envs in parallel, 2 scenes x 3 tasks,
~75 min). Report: `results/practice_small_282702.md`.

| env | attempts succeeded | practice runs succeeded | calls accepted by clean | dataset rows (train) |
|---|---|---|---|---|
| GameBoy (Viridian, Pewter) | 6/6 | 8/18 | 176/191 | 528 |
| Android (Contacts, Settings) | 5/6 | 12/15 | 101/112 | 303 |
| Web (arXiv, GitHub) | 6/6 | 12/18 | 43/49 | 129 |

(Train rows include the paraphrase augmentation, 3x.) Outputs:
`storage/practice/<env>/gemma-4-26b-a4b-it/`. A first attempt (job 282701) was stopped after
5 minutes to fix the proposals (D1.16, D1.17); its Web/Android outputs are kept as
`*_superseded_run282701/`.

## Part 2: curiosity exploration + world model — done, dev runs on all three environments

Code: `cusi/explore/` + `run_explore.py`; Slurm: `slurm/explore.sh`,
`slurm/curiosity_practice.sh` (logic in `scripts/slurm/`).

Runs (outputs in `storage/explore/<env>/<run>/`; per-update metrics in `metrics.jsonl`):

- **GameBoy Viridian, sequence ratio, LR 2e-6** (`dev_viridian`, job 282710, 1536 steps, 51 episodes,
  ~4 h on one A6000, ~10 s/step): 95% valid actions; all 12 updates hit the KL stop; KL to the
  reference model stayed ~0, so the policy effectively did not change; mean reward flat (~0.012).
- **GameBoy Pewter, token-mean ratio, LR 1e-5** (`dev_pewter_tokenmean`, job 282711, 1536 steps):
  95% valid; approx KL ~0, no clipping; KL to the reference rose steadily 0.0008 -> 0.0076, so
  the policy is moving, slowly; mean reward flat (~0.014). Too short to show a novelty gain.
- **Android Contacts** (`dev_contacts`, job 282709, 320 steps, 16 episodes, LR 1e-5 sequence ratio)
  and **Web arXiv** (`dev_arxiv`, job 282708, 320 steps, 21 episodes): the Android/Web PPO paths,
  the container GPU path, text novelty (~0.3 per step) and resets all work; 77% / 58% valid
  actions; clipfrac ~0.8 every update (LR too high for the sequence ratio, see item 1).

World model (cosine of predicted vs real next embedding, against the "nothing changes" copy
baseline) and decoder:

| run | steps | valid actions | WM held-out episodes | held-out, frame changed | WM in-sample (all) | decoder val pixel MAE (0-255) |
|---|---|---|---|---|---|---|
| gameboy/dev_viridian | 1536 | 0.95 | 0.978 vs 0.978 | 0.977 vs 0.976 | 0.981 vs 0.981 | 20.4 |
| gameboy/dev_pewter_tokenmean | 1536 | 0.95 | 0.982 vs 0.979 | 0.982 vs 0.976 | 0.980 vs 0.977 | 14.4 |
| android/dev_contacts | 320 | 0.77 | 0.883 vs 0.904 | 0.886 vs 0.899 | 0.943 vs 0.888 | 11.7 |
| web/dev_arxiv | 320 | 0.58 | 0.900 vs 0.928 | 0.895 vs 0.805 | 0.912 vs 0.914 | 13.6 |

So far the world model is about level with the copy baseline on held-out data (a little better
on GameBoy Pewter and on Web frames that changed, worse on Android): far too little data
(16–51 episodes) to judge the architecture. Prediction grids: `world_model/<run>/predictions.png`.

Element-index cap K (decision 18): Android screens have 26 elements on average (p95 60, max 83;
3.3% of screens exceed K = 64) and Web 25 (p95 92, max 92; none exceed K = 128). No chosen
action used an index >= K in either run, so K = 64 / 128 hold for now; revisit with longer runs.

Curiosity task source -> Part 1 (`run_explore.py tasks` then `run_practice.py --source curiosity`,
outputs in `storage/practice/<env>/gemma-4-26b-a4b-it/curiosity/`):

- **GameBoy Viridian** (job 282714): 13 outlier trajectories -> 6 groups -> 5 tasks ("enter the
  poke mart", "open the in-game menu", "navigate to the save option in the game menu", "enter
  the building", "walk to the poke mart entrance"); 15 practice episodes, 6 succeeded; 97 calls
  accepted by clean (15 rejected); 291 train rows. Report: `results/curiosity_practice_gameboy_dev_viridian.md`.
- **Web arXiv** (job 282713, after rescoring and with z_min 2.5, D2.17): 3 outliers -> 3 groups ->
  1 valid task ("navigate to the general relativity and quantum cosmology page on arxiv"); 3/3
  practice episodes succeeded; 33 train rows.
- Android: not run (the 320-step dev run is short; same command works:
  `sbatch ... slurm/curiosity_practice.sh --env android --run dev_contacts --model <id> --z_min 2.5`).

## How to run

```bash
M=google/gemma-4-26b-a4b-it
bash scripts/serve_vllm.sh --model $M                       # on a GPU node
python run_practice.py --env gameboy --model_name $M all    # Part 1 (android/web: via scripts/container.sh)
sbatch ... slurm/practice_small.sh --model $M               # Part 1 small-scale test, all envs
sbatch ... slurm/explore.sh --env gameboy --scene viridian --run x --wm true \
    --policy_model Qwen/Qwen3-VL-2B-Instruct --encoder_model google/siglip2-base-patch16-224
sbatch ... slurm/curiosity_practice.sh --env gameboy --run x --model $M
```

## Uncommitted changes

New: `cusi/practice/`, `cusi/explore/`, `run_practice.py`, `run_explore.py`,
`scripts/{serve_vllm,stop_vllm}.sh`, `scripts/practice_report.py`,
`slurm/*.sh`, `scripts/slurm/*.sh`, `tests/{vllm_smoke,practice_gameboy_prompt_test,practice_strip_test,practice_pipeline_mock_test}.py`,
`plans/decisions.md`, `plans/archive/end_report.md`. Modified: `cusi/envs/{base,gameboy,android_world,webvoyager}.py`,
`scripts/utils.sh` (`parse_args` helper), `scripts/container.sh`, `setup/pyproject.toml` +
`uv.lock` (added `tabulate`), `README.md`, `configs/config.env`. Outside CUSI:
`/nas/eclairnas01/users/ashokd/vllm/`. `configs/private_vars.yaml` was not touched by me.
