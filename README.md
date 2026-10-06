# Project Starter

A template for Python research projects. Provides standardised environment setup, config-driven parameter management, structured logging, large file sync via HuggingFace, and artifact tracking.

---

## Checklist

1. Do [setup](setup/README.md)
2. Checkout llm-utils to the latest commit
3. Create symlink with llm-utils/setup/.venv and wherever your actual env is
4. Add all and commit



## Reproduction

This project uses Python with [uv](https://docs.astral.sh/uv/) for dependency management. See [setup/README.md](setup/README.md) for full instructions.

Quick start:
```bash
# 1. Clone with every submodule, recursively
git clone --recurse-submodules <url>
cd <project_name>

# 2. Project venv. Optional first: put the venv under your storage dir and symlink it,
#    e.g. ln -s <storage_dir>/venvs/cusi setup/.venv
cd setup && uv sync && cd ..
cd llm-utils/setup && uv sync && cd ../..
source setup/.venv/bin/activate

# 3. Fill in every PLACEHOLDER in configs/private_vars.yaml (storage_dir, results_dir,
#    shared_vllm_dir, ...), then generate the shell config
python configs/create_env_file.py

# 4. Install the vLLM helper scripts into shared_vllm_dir and check its venv
bash setup/move_vllm_scripts.sh
```

`HF_HOME` must already be set in your shell (model weights live there).

**vLLM.** vLLM runs from its own venv at `<shared_vllm_dir>/.venv`, separate from the project
venv (its torch version differs), and that directory can be shared by several projects. Step 4
copies `setup/vllm_scripts/` (`serve_vllm.sh`, `stop_vllm.sh`, `env.sh`) there and checks for a
`.venv` with vLLM. If there is none, it warns: create a venv there and install vLLM into it,
symlink `.venv` to a venv that already has vLLM, or install vLLM into the venv that is there.
`setup/vllm_scripts/` is the source of truth: after changing those scripts, rerun step 4
(`scripts/serve_vllm.sh` warns when the installed copies are out of date).

- Maybe a line on pulling data

---

## Config Files

All `.yaml` files in [`configs/`](configs/) are automatically merged into the `parameters` dict used throughout the codebase. See [`configs/README.md`](configs/README.md) for what each variable does and how to add new ones.

- `private_vars.yaml` — machine-specific paths and credentials. Never shared as-is.
- `project_vars.yaml` — project-level settings (seeds, result paths, etc.)

---

## Core Utilities

| Function / Class | Description | Example |
| - | - | - |
| `load_parameters` | Loads and merges all YAML configs into a single dict. Safe to call with an existing dict — returns it immediately if already loaded. | `parameters = load_parameters(parameters)` |
| `log_info`, `log_warn`, `log_error`, `log_dict` | Structured logging. Always pass `parameters` to write to the log file in addition to console. `log_error` terminates execution. | `log_info("Done", parameters=parameters)` |
| `write_meta` | Saves a hyperparameter dict to a YAML file, named by a content hash. Use this whenever you produce an artifact that has configuration you want to track. | `meta_hash = write_meta("results/run/", args, parameters)` |
| `add_meta_details` | Returns a copy of a meta dict with additional fields merged in. | `extended = add_meta_details(args, {"epoch": 5})` |
| `sync_data.py` | Syncs named sets of `storage_dir` artifacts with a HuggingFace Dataset repo, which mirrors the local layout. `pull` dry-runs unless `--no_dry_run`. | `python sync_data.py push --set evals --run_name <run>` / `pull` / `init` |
| `cusi/utils/lm_inference.py` | LM and VLM inference via OpenAI-compatible APIs (`OpenAIModel`, `vLLMModel`, `OpenRouterModel`), Anthropic (`AnthropicModel`), and local HuggingFace models (`HuggingFaceModel`). Supports text and image inputs. Returns `{"output": ..., "meta": ...}`, where `meta` holds per-record `input_tokens`/`output_tokens`. | `model = OpenAIModel(model="gpt-4o"); result = model.infer(texts, max_new_tokens=256); result["output"], result["meta"]["input_tokens"]` |
| `paired_bootstrap` | Statistical significance test comparing two systems via paired bootstrap resampling. Returns the achieved p-value. Optionally logs win ratios and 95% confidence intervals. | `p = paired_bootstrap(sys1_scores, sys2_scores, verbose=True, parameters=parameters)` |
| `llm-utils` (submodule) | Efficient, scalable, offline, batched LLM/VLM inference via HuggingFace Transformers and vLLM. Also supports pretraining, SFT, DPO, and unlearning. Entry point: [`llm-utils/infer.py`](llm-utils/infer.py). Called via [`scripts/llm-utils.sh`](scripts/llm-utils.sh). | `bash scripts/llm-utils.sh --input data.csv --model_name <name> hf --batch_size 8` |

---

## Running Code

Python entry points follow the click pattern (see `run_eval.py` for a small example):
```bash
python run_eval.py [--global_option value] subcommand [--subcommand_option value]
```

For bash scripts, see [BASH_TEMPLATE.md](BASH_TEMPLATE.md).

---

## CUSI pipelines (plans/plan.md)

| Part | Code | Entry point | Slurm |
|---|---|---|---|
| 0. Shared vLLM | `setup/vllm_scripts/` (installed into `shared_vllm_dir` by `setup/move_vllm_scripts.sh`) | `bash scripts/serve_vllm.sh` / `scripts/stop_vllm.sh` | `slurm/vllm_smoke.sh` |
| 1. Practice pipeline | `cusi/practice/` (native executors: GameBoyRL, M3A, WebVoyager) | `python run_practice.py --env {gameboy,android,web} [--source curiosity] {propose,attempt,guidance,practice,clean,dataset,all}` | `slurm/practice_small.sh`, `slurm/curiosity_practice.sh` |
| 2. Curiosity + world model | `cusi/explore/` (VLM LoRA PPO policy, e.g. Qwen3.5-0.8B; curiosity modules on `cusi.state`; replay, world model, decoder, embedder training) | `python run_explore.py --env X {ppo,tasks,world_model,decoder,wm_eval,elements,train_embedder}` | `slurm/explore.sh` |
| Shared state module | `cusi/state/` (image / text embedders, novelty archive with cells, novelty scorers; inference only) | used by `run_explore.py` and `cusi.search` | — |
| Pre-exploration search | `cusi/search/` (tree, energy-sampling selection, random / VLM expanders, VLM prior; library only — the toy entry point was removed) | — | — |
| Evaluation | `cusi/eval/` (test sets through `cusi.envs` in test mode, shown equivalent to the native harnesses: `plans/eval_plan.md`; episode artifacts + step videos per task: `cusi/eval/episode.py`) | `python run_eval.py {gameboy,android,web} --model_name ... --run_name ... [--supervisor baseline\|revision\|subgoal\|info_subgoal_*] [--workers N]` | `slurm/eval.sh`, `slurm/eval_parity_{gameboy,android,web}.sh` |
| Supervisors | `cusi/agents/supervisors/` (GameBoyRL's supervisor arms over any executor and env: `plans/agents.md`) | through `run_eval.py --supervisor` | `slurm/eval.sh` |

Outputs go to `storage_dir/practice/<env>/<model>/`, `storage_dir/explore/<env>/`,
`storage_dir/eval/<env>/<run>/`.

Plans and results:
- `plans/curiosity_plan.md`, `plans/skill_discovery.md`: curiosity PPO and pre-exploration search.
- `plans/eval_plan.md`: evaluation through our envs vs the native harnesses (done; parity results inside).
- `plans/agents.md`: GameBoyRL's supervisors on all three envs and episode artifacts for debug panels
  (built; status and test results in its section 8), and the supervisor ablation plan.
- `plans/results.md`: results of the supervisor ablation (done 2026-10-06; all envs × gemma-4 26B / 31B ×
  baseline / revision / subgoal): success per config, the best supervisor (subgoal on GameBoy 26B;
  baseline on Android and Web), why, costs, and how long each run took.

Design
decisions are in `plans/decisions.md`. Tests: `tests/practice_gameboy_prompt_test.py` (GameBoy prompts
identical to GameBoyRL's), `tests/practice_strip_test.py`, `tests/practice_pipeline_mock_test.py`
(all six stages with a scripted model), `tests/vllm_smoke.py`.
