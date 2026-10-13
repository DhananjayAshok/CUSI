# Progress notes (plans/next_plan.md implementation)

Nothing is committed or pushed. The full status and per-variant results are in `plans/next_plan.md`, under
"Implementation status (2026-10-11)". This file lists the decisions you should know about, newest last.

## Decisions I made

1. **`obs["texts"]` is unchanged.** It still holds the labelled lists the eval and practice executors use, so their
   prompts are identical to before. The new text is a separate `obs["text_boxes"]`, which the search, novelty and VLM
   prompts read. The plan had rendered `texts` from the boxes instead.
2. **`TextBox` gained `ref`** (the env's element index for actions) **and `scrollable`**. The curiosity candidates
   need both.
3. **GameBoy glyph key = the tile's ink mask, not its raw 16 bytes.** The mask is the pixels that differ from the
   tile's majority colour index, so the key ignores palette and colour index.
   - Every match is checked against the screen pixels.
   - Roles come from the GameBoyWorlds OCR kinds active at that step.
4. **Pokémon tables are seeded from the gen1/gen2 charmaps**, not built as pure id tables.
   - Seeds come from tile ids on screen at text moments, plus the VRAM font when an anchor check passes.
   - The gen2 contraction ids are left out: Fool's Gold reuses them for box borders.
5. **The remaining GameBoy games were labelled by sub-agents** viewing glyph sheets, with label proposals from the
   other tables. I spot-checked the results through the `check` strings.
   - Two fallbacks were added: pixel segmentation for variable-width fonts (Harry Potter CoS, partial) and 8×16 pairs
     (Oracle of Seasons).
6. **Not portable**, besides the 4 missing ROMs:
   - `survival_kids_2`: its ROM is missing.
   - `harry_potter_philosophers_stone`: its ROM fails PyBoy's checksum.
   - `bomberman_pocket`: its only text is outlined title cards, so it is recorded as unreadable.
   - Brown, Starbeasts and CoS only load in GameBoyWorlds' `debug_mode`. The harvest tool enables that for them;
     normal envs don't.
7. **Each screen keeps at most its 64 largest text boxes** for novelty. This bounds the archive gather on dense web
   pages.
8. **Streaming dialogue rule.** A box that extends the previous screen's box gets credit only for its new
   characters. A line typed over several frames therefore scores about 1.4×, not exactly once.
9. **Selection was recalibrated** after the first Gemma run: it only ever chained freshly created nodes.
   - P is now normalised over the candidates (a PUCT prior, `--normalise_p`).
   - New defaults: `c = 2`, `tau = 0.05`. `node_threshold` stays 0.1.
   - Tool: `python -m cusi.search.calibrate --env <env> --run <run>`.
10. **Gemma explores GameBoy poorly.** It walks into walls and its yields are near 0. That is the model, not the
    search.
11. **Curiosity's best episode is materialised by replay on every env**, Web included. GameBoy is checked exactly;
    Web and Android by embedding similarity, cut at the first divergence and flagged with `materialise_diverged`.
    The plan had saved a state at every Web step instead.
12. **The curiosity encoders are required flags**, per the no-default-model rule: `--policy_vision_encoder
    mobilenet_v3_small` and `--policy_text_encoder sentence-transformers/all-MiniLM-L6-v2`.
    - The MobileNet weights were fetched with curl into `$HF_HOME/torch_hub`, because Python's SSL failed on the node.
13. **Search runs can be read as replay** by `run_explore.py world_model / train_embedder / decoder / tasks`: pass
    `search:<run>`. `tasks` also needs `--scene`. Search runs now write `action_space.json`.
14. **Deleted:** the old data (`storage/explore/*/*`, `storage/search/*/search_test_*`), `cusi/search/prior.py` and
    `tests/web_state_check.py`.
15. **New Slurm pairs:** `env_run`, `gameboy_text` and `curiosity_parity`.
16. **Web roles:** submit-type inputs and selects are now buttons, so they get clicked, not typed into.

## Running / pending (2026-10-11 ~02:40)

- `283830`: GameBoy parity check, our expander vs GameBoyRL `ppo_curiosity` (local buffer, 100k steps).
- `283831`: `curiosity_global_v1`, the GameBoy Go-Explore configuration (global buffer, 12 expansions, ≤1 h per
  session).
- `283835`: `curiosity_web_v1`, Web curiosity (1,500 steps per session).
- `283829`: `gemma_vlm_v2`, the Gemma rerun with the calibrated selection.
- Android, run serially to avoid emulator port clashes:
  - `283839`: search env test;
  - `283840`: live candidate check;
  - `283841`: `curiosity_android_v1` (300 steps per session).
- The earlier Android attempts failed only because two jobs shared an emulator port on one node.

## Results log
- **GameBoy labelling caveats (from the labelling agents):**
  - **Oracle of Seasons:**
    - Before I added 8×16 pairs, `T`/`l` and `f`/`t` shared their lower tile.
    - Pairs now fix the pairs they saw. Rare capitals are still missing.
    - HUD digits 2–7 and 9 were copied from Link's Awakening's table (same HUD font).
  - **Bomberman Max:** its menus use a variable-width font, which is not read. Its timer digits 6, 8 and 9 are
    medium-confidence labels.
  - **Bomberman Quest:** `C` is low confidence. Many capitals have never been seen.
  - **Harvest Moon 1:** the date bar is drawn half a tile off the grid, so it reads only partly.
  - **Harvest Moon 1–3:** boxes never get the dialogue/menu role (all come out as `text`). This is a role/region
    issue in the reader, not a labelling problem; I'm checking it.
  - **Déjà Vu:** the two games share one font, and the glyphs were merged between their tables.
  - Scratch scripts `tmp_label_*.sh` / `.py` from the agents are left in the repo root (no `rm` allowed).
- **Harvest Moon roles fixed.** Table `kinds` entries can now give an explicit pixel `box` instead of a GameBoyWorlds
  region name. Harvest Moon 1–3 use dialogue = bottom band and menu = top-right area. Dialogue/menu roles now appear
  (HM2: 344 of 376 text moments).
  - Caveat: in Harvest Moon 3, GameBoyWorlds often reports the dialogue kind while only the bottom HUD is on screen,
    so the HUD gets the dialogue role.

## Results log
- **GameBoy parity** (job 283830; Pokémon Red at Viridian; local buffer; image-only novelty; 100k steps; 30-step
  episodes; same reward definition):

  | | Episodes | Return, first 100 | Return, last 100 | Max |
  |---|---|---|---|---|
  | GameBoyRL `ppo_curiosity` (1 env) | 3,447 | 1.55 | 3.08 | 5.84 |
  | CUSI curiosity expander (8 envs) | 3,328 | 1.35 | 3.31 | 4.29 |

  - Both learn to a similar level; ours reaches a slightly higher final mean and a lower peak.
  - Ours took 48 min on one A6000 node.
  - I fixed the parity script's parsing of GameBoyRL returns (`tensor([x])`).
- **Gemma rerun with the calibrated selection** (`gemma_vlm_v2`, 60 expansions), against `gemma_vlm_v1`
  (60 expansions, old raw-P selection):

  | | Mean yield | Distinct discovered features | Nodes | Max depth |
  |---|---|---|---|---|
  | `gemma_vlm_v1` | 0.0015 | 24 | 63 | |
  | `gemma_vlm_v2` | 0.020 | 55 | 85 | 18 |

  It reached signs, the Pokémon Center healing dialogue and ledges. The calibrated defaults stay.
- **Search env tests, rerun with trigram + combination:**
  - GameBoy: all restores verified (11/12 pixel-exact; the other at similarity 0.9994).
  - Web: no restore failures; similarity mean 0.94, min 0.53 (live pages change).
  - **Android: 9 of 16 replay restores diverged**, the root included, at total similarity ~0.5. The text channel is
    the suspect. A debug job (`tests/android_replay_debug.py`, job 283843) runs before the Android curiosity run.
- **GameBoy Go-Explore curiosity** (`curiosity_global_v1`): sessions take ~30 min per 100k steps with 8 copies.
  Expansion yields so far are 0.38 and 0.26, against ~0.02 for the Gemma explorer and ~0.03 for the random
  expander.
- **Web curiosity** (`curiosity_web_v1`): an hour gets ~850 steps (85 episodes), not the planned 1,500; it runs
  ~4.2 s per step including restores. The best episode replayed and committed with yield 0.59.
- **Android stale a11y tree.** After a snapshot load the screen is right (image similarity 0.995), but the a11y tree
  sometimes still describes the pre-reset screen (Settings' scrolled list).
  - This breaks replay checks and also the index-based actions.
  - Sleeping and harmless key events don't refresh it. I'm testing draining the forwarder, `refresh_env` and a
    no-op swipe, compared against a uiautomator dump (job 283846).
  - I cancelled the Android curiosity run (283841) until this is fixed.
- **Android fix:** `_load_snapshot` now sends the `CLOSE_SYSTEM_DIALOGS` broadcast after every load (then waits 1 s).
  - In the debug run (`tests/android_replay_debug.py`) it was the only trigger that refreshed a stale tree: text
    similarity 0.48 → 1.00, matching a uiautomator dump.
  - Drain, sleep and taps did not help. `refresh_env` and the no-op swipe were only tried on trees that were already
    fresh, so they are untested.
  - This also affects evaluation restores (Android `reset_mode=snapshot`) and makes them more correct; eval prompts
    are otherwise unchanged.
  - Rerunning: env test 283849, then Android curiosity 283850.
- **Android env test after the fix** (job 283849):
  - Replay divergence fell from 9 of 16 nodes to 4 of 23. The root and most nodes now replay.
  - Verify on resume: 17 nodes checked, 3 failed; the restored ones have similarity min 0.957, mean 0.995.
  - The remaining divergences (similarity 0.5–0.8) look like genuine replay nondeterminism of random actions on
    Android. They are flagged `replay_diverged` as designed.
- **Web curiosity, expansion 2:** the replay diverged at step 5 (typing a search on live arXiv). It was committed up
  to there and flagged `materialise_diverged`, as designed.
- **Comparison tool:** `python -m cusi.search.compare --env <env> --runs a,b,...` reports archive states, distinct
  text keys, nodes, depth, env steps (committed + session) and hours.
- **Random baseline:** `random_k30_v1` (job 283851): random expander, 30-step expansions, 4,000 expansions (~120k env
  steps), the same trigram + combination novelty. It is the comparison for the Go-Explore curiosity run.
- **Snapshot at 04:30** (3 curiosity expansions so far):

  | Run | Committed steps | Env steps | Archive states | Distinct texts | Max depth | Mean yield |
  |---|---|---|---|---|---|---|
  | `curiosity_global_v1` | 90 | 300k | 78 | 84 | 23 | 0.26 |
  | `gemma_vlm_v2` | 600 | 600 | 160 | 49 | | 0.020 |
- **Android `open_app` list fixed.** "Calendar" and "Expense" are not app names AndroidWorld can launch. The list now
  uses AndroidWorld's `_PATTERN_TO_ACTIVITY` names for the AVD's apps (Simple Calendar Pro, Pro Expense, ...).
  - The running Android curiosity job still uses the old list: those candidates fail and get the invalid-action
    penalty. The adb tracebacks in its log are non-fatal retries.
- **Go-Explore curiosity vs random on GameBoy (05:20). This is a finding you should look at:**

  | Run | Expansions | Env steps | Committed steps | Archive states | Distinct texts | Max depth | Hours |
  |---|---|---|---|---|---|---|---|
  | `curiosity_global_v1` | 5 | 500k | 150 | 119 | 130 | 23 | 2.9 |
  | `random_k30_v1` (still running) | 2,581 | 77k | 77k | 6,843 | 830 | 66 | 0.74 |

  - Each committed curiosity step is far more novel (mean yield 0.22 against 0.003).
  - But the run as a whole covers much less: only the best 30-step episode of a 100k-step session enters the tree
    and archive, so 99.97% of what the session explored is discarded.
  - Random expansion covers far more per env step and per hour.
  - Options, if you want curiosity to compete on coverage (not done; a design change for you to decide):
    - commit the top-k episodes;
    - add every session state above `node_threshold` to the archive or tree;
    - run much shorter sessions.

## Android fix (2026-10-11, morning)
**What went wrong in `curiosity_android_v1`:**
- The first session's best episode diverged at replay step 1.
- In the second session the emulator stalled and was restarted. After the restart the a11y tree came back empty,
  so the root's replay looked different (similarity 0.5).
- The session's restore then raised an error, and that crashed the whole search.

**Fixes:**
1. **`AndroidPlayEnv._refresh_a11y`**: send the `CLOSE_SYSTEM_DIALOGS` broadcast, then wait (up to 30 s, repeating
   the broadcast) until the a11y tree has app elements again.
   - It runs after every snapshot load and at the end of `_go_to_start`, which covers scene builds, emulator
     restarts and the `reinit` reset.
   - If the tree never refills it logs a warning instead of failing.
2. **Replay check looks twice**: `TreeSearch._replay` re-observes once after 2 s before declaring a replay diverged
   (envs with `observe()`, i.e. Android), because the tree can lag the screen.
3. **Curiosity sessions no longer crash the run** when their node can't be restored (`RestoreFailed`):
   - the session stops early and keeps its best episode so far (`expander_info["stopped_early"]`);
   - if no episode finished, or the node can't be restored to materialise the best episode, the iteration is skipped
     (no expansion) and the node is flagged `replay_diverged`. The search continues.
4. **New fake-env test** of both paths (`tests/curiosity_expander_test.py`). The curiosity, core, VLM and store test
   suites all pass.

**Live runs:** Android env test (job 283854), then `curiosity_android_v2` (job 283855; 4 expansions, ~300 steps or
1 h per session).
- **Android env test after the second fix** (job 283854): 10 expansions, no crash, 0 unrestorable nodes.
  - The same 4 nodes diverged on replay as in the previous run, with identical similarities (0.45–0.81). That is
    deterministic mismatch between a live trajectory and reset+replay of its actions, not a11y lag.
  - They are flagged `replay_diverged`. The restored nodes verify at similarity min 0.957, mean 0.995.
- **`curiosity_android_v2`** (job 283855, still running; 23 h limit):
  - Session 1 (node 1) completed with no crash: 300 steps, 30 episodes, 3 PPO updates, 3265 s (inside the 1 h cap).
    Best return 6.75 (10 actions).
  - Materialising the best episode diverged at step 3 (similarity 0.936), so the expansion keeps 3 steps:
    3 children, yield 0.611.
  - The log has tracebacks, but they are android_env's own ADB retries. They are logged as "try 1 of 2", for example
    `am start -W` on OsmAnd/Chrome timing out after 5 s even though the app launched. They were retried and were not
    fatal.
  - The remaining 3 sessions should take about 1 h each.

## GameBoy curiosity_global_v1 finished (job 283831)
- 12 expansions, 137 nodes, max depth 29, archive 290 states, 238 distinct texts.
- About 1.2M env steps over 7.2 h. Mean yield 0.152. 0 unrestorable, 0 diverged.
- vs `random_k30_v1`: 4000 expansions, 5715 archive states, 947 texts, 120k env steps in 1.2 h.
- Random still covers far more per env step and per hour. The curiosity coverage design question stands.
  - Session 2 (node 0, restored by a 221 s replay): no crash. 300 steps, 30 episodes, 3095 s. Best return 5.46.
    All 10 steps materialised cleanly: 8 children, yield 0.547. The tree now has 13 nodes and the archive 15 states.
  - Iteration 3 picked node 4. Its restore replay reached similarity 0.927, so it was flagged `replay_diverged` and
    skipped without a session, which is the intended handling.
  - Session 3 (node 3, restored by a 30 s replay): no crash. 290 steps, 29 episodes, 3617 s; it stopped on the 1 h
    cap. Best return 5.42. All 10 steps materialised: 9 children, yield 0.542. The tree has 22 nodes and the
    archive 25 states.
  - Session 4 (node 18, restored by a 95 s replay): 180 steps, 18 episodes, 3710 s; stopped on the 1 h cap. Best return
    3.62. All 10 steps materialised: 6 children, yield 0.362.
  - **Final: job COMPLETED (exit 0), no crash.** 4 expansions, 28 nodes (27 selectable), max depth 14, archive 33
    states, 0 unrestorable, 1 replay_diverged (node 4), 4.1 h total.
  - The only errors in the log are android_env's internal ADB retries (`am start -W` timeouts). Every one recovered.
  - **Verdict:** Android curiosity now runs end to end. Remaining cost: env steps on Android are slow (~10 s each), so
    a 1 h session is only ~200–300 steps.

## Remaining plan items (2026-10-11, afternoon): real-VLM Web/Android runs, curiosity coverage setup, §D viewer
### 1. Real-VLM runs on Web and Android (Gemma 4 31B, same settings as GameBoy `gemma_vlm_v2`)
- `web/gemma_vlm_v1` (job 283856) and `android/gemma_vlm_v1` (job 283857).
- Settings: VLM explorer, features, describe and decision scores; 40 expansions, `k_steps` 10, calibrated selection
  (normalised P, c=2, tau=0.05); 3 GPUs (search + vLLM on 2).
- Both started cleanly: Web runs about 1 min per expansion, Android about 2 min. Results below when finished.

### 2. Curiosity coverage: commit several episodes per session (`--rl_commit_episodes`)
- **Problem:** only the best episode of each session entered the tree and archive.
- **What changed:** the session now keeps its top-k distinct episodes (by return, distinct action sequences,
  `PPOConfig.keep_episodes`). The search materialises each one as its own expansion of the node, best first: restore
  the node, replay the episode and score it against the global archive, then commit.
  - One record per expansion.
  - `max_expansions` caps how many expansions one iteration may add.
  - `expander_info` gains `plan`, `plans`, `episode` and `episode_return`.
  - `--rl_commit_episodes 1` (the default) behaves exactly as before.
- **Code:**
  - `session.py`: `keep_top`, `SessionResult.top`.
  - `expander.py`: `n_plans`, `begin_plan`.
  - `search.py`: `iterate` loops over the plans; the step loop moved into `_materialise`.
  - `run_search.py`: `--rl_commit_episodes`, `search.max_expansions`.
- **Config note:** `rl_commit_episodes` is now part of the run config, so resuming a curiosity run made before this
  needs `--ignore_config_violation`.
- **Tests:** keep_top, three expansions from one session, archive adds and records, and the `max_expansions` cap
  were added to `tests/curiosity_expander_test.py`. All curiosity, core, VLM and store tests pass.
  - The store test must run on local disk: on the NAS the `--overwrite` cleanup at its end trips over `.nfs`
    files (the run's frame zips are still open).
- **Coverage setup run:** `gameboy/curiosity_topk_v1` (job 283858) uses shorter sessions and commits more of each:
  10k steps per session, 8 copies, top 10 episodes committed, k=30, 300 expansions (30 sessions).
  - After 2 sessions it had archive 420, against `curiosity_global_v1`'s 290 after all 12 sessions.
  - Sessions take about 3 min, against about 36 min.
- **Not done:** I didn't add the other option (session states above `node_threshold` going straight into the
  archive without a tree node). It would put unrestorable states in the archive. Committing the top episodes gets
  most of that coverage while keeping every archived state in the tree.

### 3. §D viewer
- `python -m cusi.search.viewer --env <env> --run <run>` writes `<run dir>/viewer/index.html` with `data.js` and
  `frames/`. The frames are copied out of the run's zips as they are (WebP). The page has no network or server
  dependencies, so it opens straight from the NAS.
  - `run_search.py` now rebuilds it at the end of every run; a failure there only warns.
- **Tree tab:** the tree with node thumbnails (dots above 800 nodes), edges coloured by node value, flagged nodes
  dashed red, pan/zoom.
  - Clicking a node shows its frame, depth, value, Q / N / P, latest selection probability, latest decision score
    with per-feature verdicts and reasons, flags, the expansion and steps that created it, its parent, its
    children, and its expansions.
- **Expansion player:** the step strip (dashed frames are steps that became nodes), Prev / Next and the arrow keys,
  and per step the action, value with its components, created node, validity and on-screen text.
  - Also the description, discovered features, window descriptions and expander info (e.g. the curiosity session
    stats).
- **Timeline tab:** one chart per measure, against expansion: yield, the chosen node's selection probability,
  archive size, candidates per decision, the chosen node's decision score, discovered features, and seconds per
  expansion. Each has hover tooltips; clicking a point opens that expansion.
- **Expansions tab:** a table of all expansions.
- Light and dark themes, and URL hashes (`#n=12`, `#e=5&s=3`) for linking to a node or a step.
- **Checked** in headless Chrome (`tests/viewer_screenshot.py`, run in the container) on `gameboy/gemma_vlm_v2` and
  `web/curiosity_web_v1`: all views render, no JS errors. Viewers have also been built for `curiosity_global_v1`.

### Real-VLM runs: two failures fixed (17:50–18:10)
- **Android `gemma_vlm_v1` crashed** (job 283857, after 3 expansions).
  - `TreeSearch.rescore` read `self.envs[key].env_description` for every top-M candidate. On Android
    (`SceneEnvs(exclusive=True)`) that *rebuilds the other scene's env*, switching scenes on every decision, just to
    read a class attribute.
  - One of those rebuilds failed to launch Contacts (`am start` exit 1) and the exception killed the run.
  - **Fix:**
    - `SceneEnvs.description(key)` caches each scene's description when its env is built.
    - `TreeSearch.env_description` uses it, so scoring never touches the envs.
    - `AndroidPlayEnv` now retries a failed scene build: 2 retries, the last after restarting the emulator
      (`_build_scene_with_retries`).
  - Resumed as job 283859 from the committed expansions.
- **Web `gemma_vlm_v1` hung** (job 283856, after 21 expansions).
  - py-spy showed the search blocked in a Selenium `execute_script` (the challenge-page check in `_observe`) on a page
    whose renderer froze; vLLM was idle. Selenium has no client timeout by default, so the call never returned
    (one earlier freeze cost 20 min).
  - **Fix:** the web env's driver connection now has a 120 s client timeout (`WEBDRIVER_TIMEOUT`, set only while the
    driver is created, so eval drivers made elsewhere are unchanged).
    - A timed-out call (urllib3 `ReadTimeoutError`) is handled like a dead browser: restart on the current URL in
      free play; save/load state failures mark the node unrestorable, as before.
  - Checked with `tests/web_hang_check.py`: a `while(true){}` page raises after the timeout.
  - The hung job was cancelled and resumed as job 283860.
- The core and VLM search tests pass after these changes.

### Web `gemma_vlm_v1` finished (resumed as job 283860)
- 40 expansions, 150 nodes, max depth 47, archive 144 states, 1,055 distinct texts, 188 discovered features.
  - Mean yield 0.130, against 0.020 for Gemma on GameBoy.
  - 9 challenge (bot-check) pages; 0 unrestorable; 0 diverged.
  - ~1 h for 400 env steps.
- The resumed half had no browser hangs or restarts.
- The viewer was built automatically at the end of the run (`storage/search/web/gemma_vlm_v1/viewer/index.html`).
- **Against `curiosity_web_v1`** (4 one-hour sessions, best episode each): Gemma reached 144 states and 1,055 texts in
  ~1 h. Curiosity reached 32 states and 296 texts in 4 h.

### GameBoy `curiosity_topk_v1` finished (job 283858): the coverage setup
- 30 sessions of 10k steps each, top 10 episodes committed per session: 300 expansions in 2.0 h. No crash,
  0 unrestorable, 0 diverged.

  | Run | Env steps | Hours | Archive states | Distinct texts | Nodes | Max depth |
  |---|---|---|---|---|---|---|
  | `curiosity_topk_v1` (new) | 309k | 2.0 | 3,498 | 427 | 441 | 24 |
  | `curiosity_global_v1` (best episode only, 100k-step sessions) | 1.2M | 7.2 | 290 | 238 | 137 | 29 |
  | `random_k30_v1` | 120k | 1.2 | 5,715 | 947 | 4,434 | 68 |
  | `gemma_vlm_v2` | 600 | 0.2 | 160 | 49 | 85 | 18 |

- **Effect of the setup:** 12× the archive states and 1.8× the texts of the old curiosity setup, in under a third of
  the time and a quarter of the env steps.
- **Random is still ahead** on coverage per hour and per env step: 1.6× the states and 2.2× the texts. Curiosity's
  per-expansion novelty dropped (mean yield 0.010) because the lower-ranked episodes of a session overlap the best.
  Whether to push further is a design call for you; the next things to try would be fewer committed episodes with
  more sessions, or k=10.
- **`compare.py` fix:** a session's steps are now counted once, not once per committed episode (it reported 3.0M env
  steps for this run before the fix).
- Viewers built for `curiosity_topk_v1` and `random_k30_v1`. The job started before the automatic viewer hook
  existed, so they were built by hand.

### Android `gemma_vlm_v1` finished (resumed as job 283859; COMPLETED, exit 0)
- 40 expansions, 105 nodes, max depth 16, archive 204 states, 286 distinct texts, 178 discovered features.
  - Mean yield 0.083. 0 unrestorable; 7 replay-diverged nodes, flagged and skipped.
  - 2.8 h for the resumed part: a deeper node's restore replays its whole path, up to ~6 min.
- The old crash did not return. One emulator stall (a snapshot load failed) was recovered automatically by an
  emulator restart.
- The viewer was built automatically (`storage/search/android/gemma_vlm_v1/viewer/index.html`).
- **Against `curiosity_android_v2`** (4 one-hour sessions): 204 states and 286 texts in ~2.2 h of expansion time,
  against 33 and 218 in 4 h.
- **All plan items requested are now done:** real-VLM runs on GameBoy, Web and Android; the curiosity coverage
  setup; the §D viewer. Nothing is committed.

### Final check (2026-10-12)
- No jobs running. Every failed or cancelled job (283850, 283856, 283857) was a crash or hang that was fixed and then resumed.
- Built the missing viewers for `android/curiosity_android_v2` and `gameboy/gemma_vlm_v1`. Smoke and test runs have no viewer.
- `android/curiosity_android_v1` is a stale crashed run (1 expansion, no summary), superseded by v2. It was not deleted because `rm` is denied.
- Curiosity expander, search core and state unit tests: ALL OK.

### Pre-commit cleanup (2026-10-13)
- **Keyword-only arguments:** the public helpers in the new modules now take keyword-only arguments (`*,`). All call sites were rewritten by an AST script. The Archive-style `add/novelty/max_similarity(e)` methods are unchanged, to match the existing Archive interface.
- **Lint:** pyflakes is clean on all changed files.
- **Tests:** core, state, curiosity, VLM and store tests all pass. The GameBoy text reader check on pokemon_red read 38 of 40 text screens with 0 unknown glyphs.
- **Repo hygiene:** `.gitignore` now ignores `/tmp*.sh`, `/tmp*.py` and `screenlog.*`. `tests/android_replay_debug.py` was moved out of the repo.
- **Not committed:** `configs/private_vars.yaml` (real values) and the `GameBoyRL` pointer, which is unchanged.
