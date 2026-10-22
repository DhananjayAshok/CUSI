# Sync plan: artifact roots and `sync_data.py`

Written 2026-10-06. Not started.

## 1. Do all artifacts live under common roots? Mostly, not fully

**Under the two configured roots:**
- `storage_dir`:
  - `eval/<env>/<run>/`, `practice/<env>/<model>/[curiosity/]`, `explore/<env>/<run>/`;
  - `explore/<env>/{world_model,decoder}/<name>/`;
  - the derived `data/`, `models/`, `tmp/`, `sync/`.
- `results_dir`: `logs/` (Python and Slurm logs), `figures/`, report markdown.

**Outside them:**

| what | where | why it matters |
|---|---|---|
| GameBoyWorlds session directories: emulator videos (`--save_video`) and GameBoyWorlds' own run files | `GameBoyRL/GameBoyWorlds/storage/sessions/` | run output inside the submodule tree, configured by GameBoyWorlds' own `private_vars.yaml` |
| GameBoy custom saved states (`GameBoyPlayEnv.save_state` → `custom_<id>.state`) | `GameBoyRL/GameBoyWorlds/storage/rom_data/<game>/states/` | the saved scenes that practice / explore / search start from; lost if the submodule is recloned |
| vLLM server logs | `shared_vllm_dir/state/` | infrastructure; fine outside |
| emulator staging and pid files, WebVoyager browser profiles and downloads, container XDG dirs | `/tmp/<user>-cusi-*` | ephemeral by design. WebVoyager's downloaded PDFs are lost, but the answers derived from them are recorded. |
| model weights | `$HF_HOME` | by policy (CLAUDE.md) |

**Within `storage_dir`, infrastructure is mixed with artifacts.** These sit alongside the run outputs:
- `containers/` (the `.sif`), `apptainer/`, `uv/`, `container_home/`;
- `android/` (AVD state) and `rom_data/` (ROMs, which are copyrighted);
- `setup_run*.log`.

Any sync has to step around them. Worse, the Hub's `upload_large_folder` walks the whole folder it is given, so a push rooted at `storage_dir` would scan the uv and apptainer caches.

**Path conventions are uneven.**
- Only practice has a paths object (`PracticePaths`). Eval, explore, info docs and the filtered Web task file each build their paths inline, in about 10 places.
- Some files sit beside runs instead of in them:
  - `eval/web/WebVoyager_data.cusi.jsonl`, `eval/mem10_subsets/`, `eval/info_docs/`;
  - `eval/android_parity/`, `eval/web_native/`;
  - `explore/<env>/elements_*.json`.

## 2. How to manage it

1. **One paths module, `cusi/utils/paths.py`**, following GameBoyRL's `python_scripts/paths.py`. Every artifact path becomes a function of a small identity, for example:
   - `eval_run(env, run)`, `practice_root(env, model, source)`;
   - `explore_run(env, run)`, `world_model(env, name)`, `decoder(env, name)`;
   - `info_docs(game)`, `web_task_file()`, `ablation_subsets(name)`.

   Producers and consumers both call it; nothing joins `storage_dir` by hand. When a read finds a path missing, the error names the script that produces it, as GameBoyRL's `require()` does.
2. **Split infrastructure from artifacts.** A new config key `infra_dir` (default `<storage_dir>/../storage_infra`) holds `containers/`, `apptainer/`, `uv/`, `container_home/`, `android/`, `rom_data/` and the setup logs. `storage_dir` then holds only artifacts, so a sync can be rooted at it cheaply and safely.
   - Touches `scripts/container.sh`, `setup/` (the uv and apptainer cache paths) and `scripts/android_emulator.sh`.
   - Existing directories are moved once; `infra_dir` is a private var, which you fill in.
3. **GameBoyWorlds storage into our tree.** Set GameBoyWorlds' own `storage_dir` (its `private_vars.yaml`, a value you set) to `<infra_dir>/gameboy_worlds`.
   - Saved states that a run depends on are copied into the run's own artifact directory (`explore/<env>/<run>/states/`) when written, so a run is self-contained and syncable.
   - Session videos stay infrastructure; our own `episode.mp4` is the artifact.
4. **No pickles in synced artifacts.**
   - Today's refactor made all 597 pickles unloadable, which is why they were deleted.
   - Records written to disk become JSON + PNG / NPZ:
     - replay chunks: NPZ for frames and embeddings, JSONL for the rest;
     - practice legs and eval trajectories: the `cusi/eval/episode.py` format, which already exists.
   - Anything still pickled stays local-only and is never synced.

## 3. `sync_data.py`

**Keep GameBoyRL's design:**
- one Hugging Face dataset repo (`huggingface_repo_namespace` / `huggingface_repo_name`, already private vars);
- named file sets with allow / ignore patterns;
- `init` / `pull` / `push`;
- `pull` dry-runs by default and prints the exact command to run for real;
- `push` refuses `--set all`;
- global ignores for ROMs and caches.

**Change:**
- **The repo mirrors the local layout.** Every set is a subtree of `storage_dir` or `results_dir`, and its repo path keeps that prefix: `storage/eval/...`, `storage/practice/...`, `results/...`. Sets can't collide in the repo, which they could in GameBoyRL, where each set uploaded relative to its own root. Pull writes each file back where it came from.
  - **To check:** `upload_large_folder` has no `path_in_repo`. Two ways to get the prefix:
    - root the push at the parent of `storage_dir`. This is cheap once infrastructure has moved out, which is step 2.
    - Or upload from a staging directory of symlinks. Check whether the Hub client follows directory symlinks before relying on this.
- **`push --dry_run`** reports the files that would be uploaded and their total size. GameBoyRL has no push dry run.
- **`list`** shows, per set, the local files that match and their size.
- **Drop the unused `sync_dir`**, or make it the staging directory if the symlink route is taken.

**Proposed sets** (all under `storage_dir` unless noted):

| set | contents | notes |
|---|---|---|
| `eval_summaries` | `eval/*/*/{config.json,results.jsonl,summary.json,timing.jsonl,scores.json,process_episodes.md}` plus each episode's `meta.json` and `events.jsonl` | small; the numbers behind `plans/archive/results.md` |
| `eval_episodes` | `eval/*/*/episodes/*/{legs/*,frames/*,episode.mp4}` plus Web's `task*/` (auto_eval inputs) | large; pull per run with a run-name pattern |
| `practice_datasets` | `practice/*/*/dataset/**` (and `curiosity/dataset/**`) | the training data: the main product |
| `practice_runs` | the JSON / JSONL stage outputs of practice (proposals, attempts, guidance, cleaning) | audit trail behind the datasets |
| `explore_models` | `explore/*/{world_model,decoder}/*/**`, policy adapters under `explore/*/*/policy/` | trained models and their metadata |
| `explore_replays` | `explore/*/*/replay/**` | large; only after replays move off pickles (section 2.4) |
| `models` | `models/**` | anything saved to `model_dir` |
| `results` (root `results_dir`) | `*.md`, `figures/**` | not `logs/` |

**Never synced:** `tmp/`, anything under `infra_dir`, `*.gb` / `*.gbc` / `*.sav` / `*.state` (ROMs and emulator states, unless a set names states explicitly), logs, `*.pkl*`.

## 4. Order of work

1. `cusi/utils/paths.py`, with every artifact path routed through it. Mechanical; verified with the existing tests and the GameBoy mock eval.
2. Replay and practice-leg records off pickles (NPZ / JSON).
3. The `infra_dir` split, which needs one value from you (where infrastructure should live), plus a one-time move of the existing infrastructure directories.
4. GameBoyWorlds' storage set to `infra_dir`, and saved states copied into run directories.
5. `sync_data.py` as above, tested against a scratch private dataset repo:
   - push one small set, pull it into an empty `storage_dir`, and compare checksums;
   - then dry runs on the large sets.

**Needs from you:**
- the Hub repo (namespace and name; private?);
- the `infra_dir` location;
- GameBoyWorlds' storage value;
- whether `eval_episodes` and `explore_replays` are worth syncing at all, given their size.
