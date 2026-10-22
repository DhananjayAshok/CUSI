# Tree search: stats, persistence, env fixes

Turns the `cusi/search/` core (select → restore → expand → score) into something that can run on
real envs, survive a crash, and be resumed or extended. Background and the method survey:
`plans/skill_discovery.md` §1.4. This plan covers that section's open items 2 (stopping), 4 (stats),
7 (on-disk format), 9 and 10 (Android, Web) and 11 (durable states). Still out of scope: the viewer,
threshold calibration, parallel expansion (runs stay one step at a time).

---

## 1. Stopping rule

`--max_expansions N`. The search stops when the run has N expansions in total, counting the ones
from earlier invocations. It's the only stopping rule. `run(stop=...)` keeps its callable; the
entry point passes `lambda s: "max_expansions" if s.iteration >= N else None`.

## 2. Stats

These are pure functions of the tree (and its expansions), so nothing beyond the tree is persisted
for them. They live in `cusi/search/stats.py` as `TreeStats` (the interface `TreeSearch` and
`EnergySelector` already call: `q`, `n`, `n_total`, `expansion_yield`, `record_yield`).

- **No cells.** Nothing in the search reads archive cells, so the search no longer calls
  `cell_of`.
  - `Node.cell`, `Step.cell` and `Step.new_cell` are removed, along with the `n_cells` /
    `new_cells` entries in the records and logs.
  - The archive is built without `cell_threshold`.
  - `node_threshold` (novelty against every stored state) is the only threshold to calibrate.
  - The archive keeps its cell code for other users; it just isn't used here.
- **Yield of an expansion:** the mean novelty value of its steps, in [0, 1]. It's on the same scale
  as the novelty that creates a node.
- **`N(n)`:** the number of expansions started from n (`n_expanded`). **`N_total`:** total
  expansions in the run.
- **`Q(n)`, expanded:** an exponential moving average of the yields of expansions from n or any
  node in its subtree, in the order they happened: `Q ← (1 − α)·Q + α·y` for n and every ancestor
  of the expanded node. `α` = `--q_ema` (default 0.3). Yields shrink as the archive fills, and a
  plain mean would overrate nodes that were productive early.
- **`Q(n)`, unexpanded:** the novelty value of the step that created n. Roots have no creating
  step, so they start at 1.0, which makes sure every scene gets expanded early.

On load, `Q` and `N` are rebuilt by replaying `expansions.jsonl` in order through `record_yield`.

## 3. Persistence

### Run directory

`storage/search/<env>/<run_name>/`, from `cusi/utils/paths.py` (`search_root`, `search_run`).
`--run_name` is required, as for evals.

### Config guard

The guard is the same as `EvalRun`'s (`cusi/eval/records.py`): `config.json` is written on the
first run.
- `--overwrite` clears the directory.
- A different config raises, unless `--ignore_config_violation` is passed, which warns and keeps
  the stored values.
- `max_expansions` is operational (not a violation), alongside the vLLM URLs. This makes
  **resume and extend the same operation**: re-run the same command, with a larger
  `--max_expansions` to extend. The guard should be factored out of `EvalRun` and shared, not
  copied.

### Files

All tree files are append-only. One iteration writes, in this order:
1. the new nodes' embeddings;
2. the new nodes' rows;
3. the frames zip;
4. the expansion row;
5. the archive.

The archive save is the commit point: it records `n_expansions`, the number of expansions it
includes. On load, any expansion row with `id >= n_expansions`, and anything referencing such an
expansion or one with no row (nodes, embeddings, a zip), is discarded. A crash therefore loses at
most the iteration in progress, and the archive and tree always agree.

```
config.json
nodes.jsonl              one row per node
expansions.jsonl         one row per expansion (the commit point)
embeddings/<exp>.pt      the embeddings of the nodes created by expansion <exp> (roots: roots.pt)
frames/<exp>.zip         that expansion's step frames (FrameStore: lossless WebP, dedup; roots: roots.zip)
archive/                 NoveltyArchive.save(): archive.pt + region stores
states/<state_id>.*      durable env states (GameBoy, Web; none on Android, see §4)
```

- **`nodes.jsonl` row:**
  - `id`, `parent`, `depth`, `env_key` (the scene);
  - `state`, the file under `states/`, or null;
  - `embedding`, a ref `"embeddings/<exp>.pt#<row>"`;
  - `frame`, a ref `"frames/<exp>.zip/<k>.webp"` to the creating step's frame (`roots.zip` for
    roots);
  - `texts`, `segment`;
  - `prior`, `prior_reason`, `prior_raw`, `flags`, `created_at`.

  `n_expanded` and `yields` are not stored; they're derived from `expansions.jsonl`. Flags that
  change after creation (e.g. `unrestorable` after a failed restore, `replay_diverged`) are
  appended as `{"id": n, "flags": {...}}` update rows; the latest row wins.
- **`expansions.jsonl` row:**
  - `id`, `node_id`, `iteration`, `children`, `yield`, `stats` (seconds, load seconds, valid rate);
  - the selection record (candidates, energies, probs, `q`, `n`, `p`, `n_total`), which replaces
    the in-memory `search.records`;
  - `steps`: action, texts, compact info, value, components, `node_id`, extra, and the
    frame ref.
- **Embeddings** are kept as tensors, not JSON. A per-expansion file keeps them append-only.
  Only node embeddings are needed (each is the `prev` of its next expansion); step embeddings
  live in the archive.
- **Archive:** `archive.save()` after every expansion (step 5 above), into `archive.new/` with a
  `commit.json` (`n_expansions`, `n_nodes`, `iteration`, `total_steps`). It then replaces
  `archive/` through two renames. On load, the most advanced complete one of
  `archive/`, `archive.new/` and `archive.old/` wins, so a crash at any point leaves a usable
  commit.
- **Seeds:** the selector (`EnergySelector.reseed`) and the random expander
  (`RandomExpander.start`, which passes its generator to `env.sample_action(rng=...)`) are reseeded
  from `(seed, iteration)` on every iteration. A resumed random-expander run therefore replays
  exactly what an uninterrupted run does, on deterministic envs, without storing RNG state. VLM
  explorer runs replay only as far as vLLM sampling is deterministic.

### Env API (`cusi/envs/base.py`)

- `export_state(*, state_id, directory) -> file name` copies a saved state into the run dir.
  `import_state(*, path) -> state_id` registers such a file as a loadable state on this env.
  Because an exported state is a copy, `close()` deleting the env's own states doesn't matter, so
  there's no `keep_states` flag.
- Per env:
  - **GameBoy:** export copies `custom_<id>.state` out of GameBoyWorlds' `rom_data/<game>/states/`
    into `states/<id>.state`. Import copies it back under a fresh alphanumeric id. No submodule
    edit is needed. Files are a few hundred KB each.
  - **Web:** export writes the state dict (url, cookies, storage, scroll) to `states/<id>.json`;
    import loads it into `_states`. Restoring stays best effort on live sites.
  - **Android:** no exported states (§4).

### Resume

1. Run the config guard.
2. Build one env per scene.
3. Load the archive.
4. Read `nodes.jsonl` and `expansions.jsonl`, dropping uncommitted rows, and rebuild the `Tree`.
   Embeddings and frames load from their refs.
5. Rebuild the stats from the expansions.
6. Import each node's state. A missing file means the node is marked `unrestorable`, not dropped.
7. Continue from `iteration = len(expansions)` until `max_expansions`.

Roots aren't re-added: `add_roots()` runs only on an empty tree, and on resume each scene's root
`state_id` is re-imported (GameBoy, Web) or re-made by `reset()` (Android).

## 4. Env fixes

### All envs: skip the restore when already there

`TreeSearch` tracks, per env, the node whose state the env is currently in: the last child
created by the last expansion on that env, provided the episode didn't end. If the selected node
is that node, `load_state` (or replay) is skipped. This is the Web fix (about 17 s per restore
saved when it hits), and it helps Android's replays most.

### Web: challenge-page detector

In `cusi/envs/webvoyager.py`, `info["challenge"] = True` (with `info["challenge_reason"]`) when any
of these holds:
- the title or visible text matches a pattern list: "Just a moment", "Verify you are human",
  "captcha", "Access denied", "unusual traffic", "Attention Required", "are you a robot";
- the DOM has Cloudflare / hCaptcha / reCAPTCHA markers (`challenge-platform`, `cf-chl`,
  `cf-turnstile`, `h-captcha`, `g-recaptcha` iframes);
- optionally, the main document's HTTP status from CDP `Network.responseReceived` is 403, 429
  or 503.

Add `challenge` to `INFO_KEYS` in `cusi/search/tree.py`.

In the search, a challenge step:
- is scored with `add=False`, and its value is set to 0, so it never enters the archive;
- if it still becomes a node (the end of an expansion), the node gets `flags["challenge_page"]`
  and isn't selectable (`restorable` is false).

Counts go into the expansion stats. The detector is tested on saved HTML of a few known challenge
pages and on normal pages from the task sites (false positives matter more than misses).

### Android: root snapshots only, replay for everything else

No per-node snapshots, in the run or on disk. Each scene keeps only the root snapshot that scene
setup already makes (`cusi_<scene_id>`).
- **`save_state`** isn't called on Android. Nodes have `state = null` but are restorable by
  replay. `Node.restorable` becomes "has a state, or its env restores by replay", and isn't
  flagged `unrestorable`.
- **Restore** a node: `reset()` (load the root snapshot), then step through `tree.path_steps(n)`'s
  actions.
- **Check** afterwards: the similarity between the replayed state's embedding and the node's
  stored embedding must be at least `--replay_similarity` (default 0.95, to be calibrated). If
  it's lower, the node is flagged `replay_diverged`, isn't selectable, and the expansion doesn't
  happen (logged like a failed `load_state`).
- **Cost** grows with depth: an Android step takes seconds, so a depth-10 node costs about 10
  steps to restore. The skip-when-already-there rule above saves the common case. If deep restores
  dominate, the fallback is to snapshot every d levels, but that's not part of this plan.
- **Resume** needs nothing extra: scene setup re-makes the root snapshot, and paths come from
  `expansions.jsonl`. Whether a rebuilt root matches the original (device clock, seeded app data)
  is checked by the same similarity test on the root.

The replay restore is written generically (`restore="replay"` per env), so GameBoy (exact) and Web
could fall back on it when a state file is missing. Only Android uses it by default.

## 5. Entry point

`run_search.py` plus a `slurm/search.sh` / `scripts/slurm/search.sh` pair. It takes:
- `--env`, `--run_name`, `--max_expansions`, `--k_steps`, `--node_threshold`;
- the selector's `c`, `τ`, `roots_only`, and `--q_ema`;
- the `cusi.state` flags;
- `--expander {random,vlm}` and `--prior/--no_prior`;
- `--overwrite`, `--ignore_config_violation`.

`--model_name` is required when the VLM explorer or prior is used, and the same value goes to
vLLM and the client. Later: a `search` set in `sync_data.py` (`search/<env>/<run>/*`).

## 6. Order of work

1. Drop cells from the search (`search.py`, `tree.py`, the test) and add `TreeStats`, plus
   updating `tests/search_core_test.py` to use it. The test's cell-count checks become checks on
   archive size and node count.
2. Run dir, config guard (factored out of `EvalRun`), append-only writer and loader, and archive
   save. Test: kill a fake-env run mid-way, resume, extend; check the rebuilt tree equals one
   built without the interruption.
3. `export_state` / `import_state` for GameBoy and Web (the base interface is done). Skip-when-already-there.
4. Replay restore plus the similarity check (test on the fake env and GameBoy, then Android).
5. Web challenge detector.
6. Entry point and Slurm pair. First real run: GameBoy, random expander, `random_patch`.

**Status (2026-10-07): steps 1–6 are done.**
- **Tests:**
  - `tests/search_core_test.py`, `tests/search_store_test.py` and `tests/web_challenge_test.py`.
  - Per env: `slurm/search_env_test.sh --env <env>`, which runs a search, then resumes it in a new process with
    `--verify_restores` and extends it (Web also runs the live detector test).
- **Results:**

  | Env | Result |
  |---|---|
  | GameBoy | 46/47 nodes pixel-exact after resume, similarity ≥ 0.9994. 40 expansions in about 6 s. |
  | Web | All 10 nodes restore; 4/10 pixel-exact; similarity mean 0.94, min 0.75 (live pages change). About 9.6 s per restore. Detector: no false positives on the 10 included task sites; none of the excluded sites served a bot check that day. |
  | Android | All 7 nodes replay to similarity ≥ 0.99998 after the scenes are rebuilt; none diverged. 44–77 s per replay restore, about 40 s of it the root-snapshot reset. |
- **Fixed along the way:** `random_patch` embedded an all-black frame as the zero vector, similar to
  nothing (itself included). Every black transition screen therefore scored novelty 1 and became a
  node. All-zero vectors now map to one fixed unit vector; every other embedding is unchanged. This
  also affects `cusi.explore`'s curiosity reward.
- **Env changes:**
  - `sample_action(rng=...)`.
  - A `raw_frame` property: the unlabelled screen after a reset, step or load, used when a node is restored.
  - Exclusive `SceneEnvs` for Android: one live scene; switching scenes rebuilds it, about 50 s.
- **Calibration still to do** (the defaults are placeholders):
  - `node_threshold` 0.1: on GameBoy with `random_patch`, step-novelty quantiles were 50% ≈ 0.002,
    75% ≈ 0.03, 90% ≈ 0.13, so 0.1 makes about 15% of steps nodes.
  - `tau` 0.1: with Q values around 0–0.25, Android selection was nearly greedy (p = 0.998 on one
    step).

## 7. TODO: revisit the prior

The prior is kept as it is for now, with a TODO in `cusi/search/prior.py`. It sees only the node's
own frame and texts, so it scores in absolute terms, not relative to what's already been explored.
That is left to `Q` and `N`. Options when it's revisited:
- include the frames of the k nearest existing nodes ("is anything new reachable from here that
  isn't reachable from these?");
- include a short text list of the screens already explored.

Either way, a prior computed once goes stale, so it might be recomputed when the node is selected.

## 8. Open question

**Node granularity.** Only steps above `node_threshold` (and expansion ends) become nodes.
`node_threshold = 0` makes every step a node, at the cost of a VLM prior call and a `save_state`
per step, and a candidate set k times larger. With Android on replay and GameBoy states cheap, this
is mostly a question of prior cost and selection quality.

## 9. VLM exploration

There are two separate questions: what the VLM does once a node is chosen (the expander), and how
a node is chosen (selection: energy, prior). This section covers the first; the second is still
open.

**Done (2026-10-08): tree context in the action prompt.** `TreeSearch` passes the tree, the node
and the expansion id to `expander.start()`. `VLMExplorer` then adds two prompt sections, built by
`cusi/search/context.py`:
- **How the node was reached:** its root→node actions, oldest first. Only the last
  `max_path_actions` (20) are kept, with "(N earlier actions)" above them.
- **Earlier explorations from the same node:** the most recent `max_tries` (5). Each gives its
  first `actions_per_try` (5) actions and, when it was described, "what happened: <description>".
  The prompt says to do something different.

The random expander ignores the context. Tested on the fake env with a recording VLM
(`tests/search_core_test.py`).

**Done (2026-10-08): transition descriptions** (`--describe`, `cusi/search/describe.py`). After
every expansion, one VLM call gets:
- the whole sub-trajectory: the start node's frame, then every step's (unlabelled) frame, so
  `k_steps + 1` images;
- the numbered actions, with failed ones marked.

It describes in two or three sentences where the sub-trajectory starts, what the actions did and
where it ends. The description is stored as `Expansion.description` in `expansions.jsonl`, and
the explorer's "earlier explorations" section uses it. This replaces the earlier screen summary
(URL plus first element lines), which was dropped: on Web the first elements are the site header,
on Android the status bar, and on GameBoy there is no text. A failed call (after the client's 3
retries) crashes the run, which resumes from its last commit.

Calls that will fail every time, not handled yet:
- `k_steps + 1` is more than `vllm_max_images` (16).
- Full-resolution Android frames: 11 of them are about 36k tokens, over `vllm_max_model_len`
  (32,768).

Possible fixes: a startup check, and downscaling the frames to a pixel budget.

**Cost:** one call per expansion with `k_steps + 1` images (11 at the default). Web screenshots
are 1024×768, so a describe call is long; if context or latency becomes a problem, downscale the
frames or subsample them. Not yet run against a real VLM.
