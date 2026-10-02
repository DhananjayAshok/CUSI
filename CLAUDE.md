# CLAUDE.md

This file provides guidance for AI assistants working on projects derived from the ProjectStarter template.

---

## ⚠️ Project Scope — Read First

**You are working on this project only.** This codebase shares file names and structure with other projects derived from the same template (`utils/parameter_handling.py`, `configs/private_vars.yaml`, etc. exist in multiple sibling projects). **This does not mean they are connected in any way.** Never read, reference, or modify files belonging to any other project. Even if you are aware of sibling projects, their parameters, configs, and logic are entirely irrelevant and must not influence anything you do here. You will never be asked to make edits to more than one project at a time.

---

## Environment Setup

This project uses `uv` for dependency management. The uv project lives in `setup/` (not the project root).

Whenever you start work, you must activate (always from project root):
```bash
source setup/.venv/bin/activate
```
Especially if you are going to consider uv pip installing anything, you must only do it in this projects setup env. 

If you want to make a package dependency official, add it with (from root):
```
cd setup
uv add <name>
cd ..
```
---

## Configuration System

All `.yaml` files in `configs/` are automatically loaded and merged into a single `parameters` dict at runtime. 

### Files
- `configs/private_vars.yaml` — sensitive or machine-specific values (storage paths, API keys, HuggingFace credentials). Committed with `PLACEHOLDER` values. You are NEVER to fill these values in yourself. Ask for them to be filled for you, even if it causes an error that completely halts your progress. 
- `configs/project_vars.yaml` — project-level settings (`random_seed`, `results_dir`, etc `WANDB_PROJECT`, etc.)
- Additional domain-specific YAML files may be added freely.

**All paths must be absolute.**

### Auto-Derived Parameters (do not set in YAML)
`compute_secondary_parameters()` derives and creates these directories automatically:
- From `storage_dir`: `data_dir`, `model_dir`, `tmp_dir`, `sync_dir`
- From `results_dir`: `log_dir`, `figure_dir`

You can use these in your code.

In general, you have access to and do not need to ask permission to edit anything in these directories, even if they are located outside of your current root. 
An important convention to follow is: if at any point of time, you must download a sizeable file etc, download to somewhere under storage_dir instead of ~, or root or tmp/ etc. If you are writing a file that has code to be run, always write it somewhere inside root as opposed to storage_dir 

### Generating `configs/config.env`
Run from project root whenever YAML configs change:
```bash
python configs/create_env_file.py
```
This exports all YAML parameters as shell environment variables for use in bash scripts.

---

## Bash Scripts

See [BASH_TEMPLATE.md](BASH_TEMPLATE.md) for the standard argument parsing template, `scripts/utils.sh` shared infrastructure pattern, `args_to_flags` usage, and `scripts/get_strings.py` for Python-based string generation (experiment names, etc.) from bash.

**Every bash script must source both of the following** (either directly or via `scripts/utils.sh`):
```bash
source configs/config.env || { echo "configs/config.env not found"; exit 1; }
source setup/.venv/bin/activate || { echo "Virtual environment not found."; exit 1; }
```

This ensures config variables (`storage_dir`, `WANDB_PROJECT`, etc.) are available as shell environment variables and the correct Python is active. Never run project bash scripts without this sourcing in place.

---

## Python Code Structure

### Philosophy
Code should be **modular** — split logic into files and subdirectories by concern. Do not create pip-installable packages (no `src/` layout, no entry points in `pyproject.toml`) unless explicitly requested.

### Entry Point Philosohpy (`main.py` structure)
The primary script click group for a given set of functionalities. The `main.py` file is just an example placeholder and you should create separate files of this nature for every independant script-like functionality you want in python. Commands are implemented in separate files and registered here:
```python
main.add_command(my_command, name="my_command")
```
Run as:
```bash
python main.py [--global_opt value] subcommand [--subcommand_opt value]
```
Global options (e.g. `--random_seed`, `--log_file`) override YAML values for that run.

### Adding a New Command
In a dedicated file:
```python
@click.command()
@click.option("--my_arg", default="val")
@click.pass_obj
def my_command(parameters, my_arg):
    # parameters is the fully loaded dict passed via ctx.obj
    pass
```
Register in `main.py`:
```python
from my_module import my_command
main.add_command(my_command, name="my_command")
```

This pattern allows you to have a single script that handles similar, but different functionalities (e.g. evaluation script that has different commands for different settings).

---

## Method Philosophy

Always force every argument to every method to be a named parameter with def func(*, arg1, arg, op_arg1=something, ....). 


## Parameter Handling

`load_parameters(parameters)` is always safe to call — if passed an already-loaded dict (one containing a `logger` key), it returns it immediately. This means functions can always accept `parameters=None` as a default and call `load_parameters(parameters)` safely without double-loading.

### Gold Standard: Class-Based Code
Load **once** at `__init__`, store as `self._parameters`, and thread it explicitly to all methods and child objects. Never re-call `load_parameters()` inside methods.

```python
from utils import load_parameters, log_info, log_error

class MyProcessor:
    def __init__(self, *, some_arg, parameters=None):
        self._parameters = load_parameters(parameters)

    def do_thing(self):
        log_info("Doing thing", self._parameters)

    def create_child(self, child_arg):
        return ChildClass(child_arg=child_arg, parameters=self._parameters)
```

### Script / Click Command Code
Load once at module level, or receive via `ctx.obj`:
```python
loaded_parameters = load_parameters()

@click.command()
@click.pass_obj
def my_command(parameters, arg):
    log_info("Running", parameters)
```

---

## Logging

```python
from utils import log_error, log_warn, log_info, log_dict
```

**Always pass `parameters` as a named argument** to ensure output goes to the configured log file:
```python
log_info("Processing complete", parameters=self._parameters)
log_warn("Unexpected value", parameters=parameters)
log_error("Unrecoverable failure", parameters=self._parameters)  # raises and terminates
```

Calling without `parameters` is safe but logs to console only — fine for quick scripts, not for production runs.

**`log_error` terminates execution.** Only use it for errors that are so bad, it is safer to discontinue execution. You do not need to manually handle termination once you call this. 

**Action required on new project setup**: Rename the logger from `"PROJECT_NAME"` to the actual project name in `utils/fundamental.py`.

---

## Experiment Naming

Experiment names must be **reproducible and interpretable**. Two approaches:

### Option 1: Hyperparameters directly in the name
When the hyperparameter set is small, encode them directly in the name and use it to construct save paths:
```python
exp_name = f"{algorithm}-{lr}-{batch_size}-{dataset}"
save_path = os.path.join(parameters["model_dir"], exp_name)
```
In bash, use `args_to_flags` + `scripts/get_strings.py` to build the name from ARGS and use `$storage_dir` from the sourced config:
```bash
arg_string=$(args_to_flags ARGS)
exp_name=$(python scripts/get_strings.py my_exp_name $arg_string)
model_save_path="$storage_dir/models/$exp_name/"
```

### Option 2: Hash-based naming (large hyperparameter sets)
When there are too many hyperparameters to encode in a name, use `hash_meta_dict` for the path and `write_meta` to maintain a human-readable record:

```python
from utils.hash_handling import hash_meta_dict, write_meta

args = {"model": "gpt-4", "temperature": 0.7, "max_tokens": 512, ...}
exp_hash = hash_meta_dict(args)
save_path = os.path.join(parameters["model_dir"], exp_hash)

# Always write the meta file so the hash can be interpreted later
write_meta(save_path, args, parameters)
```

**Never use a hash as an experiment identifier without a corresponding `write_meta` call.** The meta file is the record linking the hash to its configuration.

---

## Artifact Tracking

```python
from utils import write_meta, add_meta_details

args = {"lr": 1e-4, "batch_size": 32, "epochs": 10}
meta_hash = write_meta("results/model_outputs/", args, parameters)
# Saves: results/model_outputs/meta_{hash}.yaml
```

Use `add_meta_details` to extend a config dict non-destructively before passing to `write_meta`:
```python
extended_args = add_meta_details(base_args, {"val_loss": 0.42, "epoch": 5})
```

---

## Permissions
Read .claude/settings.json for the list of permitted commands. You can compose any of these freely while you work. Do not ask for permission if it is specified in this list. 
Never use compound commands (e.g. cat x && echo y etc.). Instead, write them to tmp.sh and do bash tmp.sh


So if you want to run some python code after sourcing the environment, don't try the compound command: "source setup/.venv/bin/activate && python -c ...", but rather write this string to a tmp.sh and run bash tmp.sh

When doing this, always use bash tmp.sh, and not the absolute path to tmp.sh
