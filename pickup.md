# Pickup notes (written 2026-10-03)

Where CUSI's setup work stopped, and what to do first when you come back.

## Goal

One command that sets up everything CUSI needs, on eclair and on other machines.
It may only write to `/tmp`, the CUSI folder and `storage_dir`.

Scope:
- **In:** GameBoyRL, AndroidWorld, WebVoyager.
- **Out:** OSWorld (dropped), and vLLM, which stays an external server you point at.

## What you should see

### `git status` in CUSI: lots of uncommitted changes, all expected

Nothing has been committed since `3505fdb` (WANDB_PROJECT → CU), and nothing has been pushed.

| item | state |
|---|---|
| `.gitmodules`, `GameBoyRL/`, `android_world/`, `WebVoyager/` | staged: 3 new submodules |
| `setup/pyproject.toml`, `setup/uv.lock`, `setup/.python-version` | new: the single venv definition |
| `scripts/commit_all.sh` | new: commits and pushes every repo |
| `pickup.md` | new: this file |
| `configs/private_vars.yaml` | modified: `storage_dir` set. Intentionally never committed. |

### Submodules

| path | remote | branch | local changes |
|---|---|---|---|
| `GameBoyRL` (+ GameBoyWorlds, cleanrl, llm-utils) | your repos | detached at recorded commits | none |
| `android_world` | fork `DhananjayAshok/android_world` (`upstream` = google-research) | `cusi` | SetupAttempt patches + Gemini lazy-import + `_pb2` gitignore |
| `WebVoyager` | fork `DhananjayAshok/WebVoyager` (`upstream` = MinorJerry) | `cusi` | SetupAttempt patches |

- Both forks were created on GitHub. Their `main` is an untouched mirror of upstream, and the patches will live on `cusi`.
- The `cusi` branches have **not been pushed yet**, so they don't exist on GitHub yet.

### The venv: `setup/.venv`, 5.4 GB, installed and tested

- One uv project, `SelfCU`, on Python ==3.12.6, with 210 packages.
- `[tool.uv] override-dependencies` lifts AndroidWorld's old pins:
  - numpy 2.5, pandas 3.0, protobuf 7.36, android-env 1.3.0;
  - `google-generativeai` and `opencv-python` are excluded (only `opencv-python-headless` is installed).
- On eclair the venv sits directly in `setup/.venv`, with no symlink. This is eclair only, because all work there is on the NAS.

Test results, all on the new venv:

| what | result |
|---|---|
| AndroidWorld unit tests | 507 pass / 1 fail. The old 3.11 venv gives the identical result, and the failure is a test-order caching quirk. |
| AndroidWorld end to end (Slurm 282669, emulator + M3A + mock model, 2 tasks) | ran cleanly, exit 0 |
| WebVoyager end to end (headless Chrome + `auto_eval.py`, mock model) | ran cleanly |
| GameBoyWorlds env demo (Pokémon Red, random actions) | 98 steps at ~450 steps/s |
| GameBoyRL entry points, cleanrl stack | all load |
| torch 2.11+cu130 on a GPU (Quadro RTX 8000) | works |

Test job outputs are in `storage/tmp/test_jobs/`. The job script is not in the repo; it's a copy of SetupAttempt's `slurm/test_mock_e2e`, AndroidWorld part.

## First thing to do

The commit script was written but **not run**: Claude's auto mode blocked it. Run it yourself:

```bash
cd /nas/eclairnas01/users/ashokd/projects/CUSI
bash scripts/commit_all.sh               # optional: --message "..."  --push false
```

Expected result:
- `android_world` and `WebVoyager` each get one commit, pushed to `cusi` on their forks.
- GameBoyRL is skipped, because it's clean and detached.
- CUSI gets one commit, pushed to `main`.
- `private_vars.yaml` is never staged.

## Decisions waiting on you

1. **GameBoyRL working copy.** `CUSI/GameBoyRL` is now a second checkout of the same repo as `../GameBoyRL`, which has uncommitted work: `runs/*_eclair.sh`, and GameBoyWorlds one commit ahead at `5f26dc6`.
   - Recommended: make the CUSI copy the only one you edit. Check out real branches in it and its submodules, set `branch = main` in `.gitmodules`, move the leftover work over, and retire `../GameBoyRL`.
   - Alternative: keep editing `../GameBoyRL`, and bump CUSI with `git submodule update --remote GameBoyRL`.
2. **`GameBoyRL/configs/private_vars.yaml` still says `storage_dir: PLACEHOLDER`.** Real GameBoyRL runs need it (`--help` and the demos work without it). Plan: have the setup script generate it from CUSI's config.
3. **`results_dir: "results"`** in `configs/project_vars.yaml` is still a relative path.

## Things the portable setup script must handle

Learned the hard way:

- **SSL certificates.** Set `export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")`. uv's standalone Python looks for certificates at `/etc/ssl/cert.pem`, which AlmaLinux doesn't have, and android-env's APK download fails without this (job 282668).
- **The Gemini import.** AndroidWorld's `agents/infer.py` imports `google.generativeai` at the top of the file. It's patched to make that optional, and the patch lives in the fork's `cusi` branch.
- **Generated protobuf files.** AndroidWorld's `_pb2` files are generated at install time for protobuf 5.29. They load fine under protobuf 7, so they don't need regenerating.
- **Eclair-only shortcuts used so far.** The portable script must provide these itself:
  - the Android SDK, the emulator device image, Chrome and chromedriver, the system-library sysroot, and the `tools/bin` wrappers, all borrowed from SetupAttempt;
  - `GameBoyRL/GameBoyWorlds/storage/rom_data`, a symlink to your existing ROMs (gitignored);
  - Python 3.12.6 from `~/.local/share/uv/python`;
  - the uv cache at `/nas/eclairnas01/users/ashokd/uv_cache`.

## After that

- Turn all of the above into the actual one-command setup (bootstrap order discussed: read `storage_dir` → standalone uv → venv → `create_env_file.py` → tools → benchmarks).
- Run the benchmarks with a real model behind them (vLLM endpoint), not just the mock.
