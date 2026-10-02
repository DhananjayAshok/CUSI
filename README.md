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

Quick start (from project root):
```bash

git pull <url> --recursive
cd <project_name>
cd setup && uv sync
cd ..
cd llm-utils/setup && uv sync
cd ../../
source setup/.venv/bin/activate
```

Then fill in your local values in `configs/private_vars.yaml` (replacing any `PLACEHOLDER` entries) and generate the shell config:
```bash
python configs/create_env_file.py
```

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
| `sync_data.py` | Syncs the local `sync_dir` with a HuggingFace Dataset repo. Use for sharing large files (models, datasets) across machines. | `python sync_data.py pull` / `push` / `init` |
| `utils/lm_inference.py` | LM and VLM inference via OpenAI-compatible APIs (`OpenAIModel`, `vLLMModel`, `OpenRouterModel`), Anthropic (`AnthropicModel`), and local HuggingFace models (`HuggingFaceModel`). Supports text and image inputs. Returns `{"output": ..., "meta": ...}`, where `meta` holds per-record `input_tokens`/`output_tokens`. | `model = OpenAIModel(model="gpt-4o"); result = model.infer(texts, max_new_tokens=256); result["output"], result["meta"]["input_tokens"]` |
| `paired_bootstrap` | Statistical significance test comparing two systems via paired bootstrap resampling. Returns the achieved p-value. Optionally logs win ratios and 95% confidence intervals. | `p = paired_bootstrap(sys1_scores, sys2_scores, verbose=True, parameters=parameters)` |
| `llm-utils` (submodule) | Efficient, scalable, offline, batched LLM/VLM inference via HuggingFace Transformers and vLLM. Also supports pretraining, SFT, DPO, and unlearning. Entry point: [`llm-utils/infer.py`](llm-utils/infer.py). Called via [`scripts/llm-utils.sh`](scripts/llm-utils.sh). | `bash scripts/llm-utils.sh --input data.csv --model_name <name> hf --batch_size 8` |

---

## Running Code

Python entry points follow the click pattern. See [`main.py`](main.py) for the template:
```bash
python main.py [--global_option value] subcommand [--subcommand_option value]
```

For bash scripts, see [BASH_TEMPLATE.md](BASH_TEMPLATE.md).
