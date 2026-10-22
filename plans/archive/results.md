# Supervisor ablation: results

Run 2026-10-06, plan in `plans/archive/agents.md` section 7, build and tests in `plans/archive/agents.md` section 8. Full tables
(including which tasks each arm won or lost against baseline): `storage_dir/eval/ablation_tables.md`,
written by `report_ablation.py` (removed 2026-10-06; the tables remain). Per-task episode artifacts (events, every
prompt and frame, step video): `storage_dir/eval/<env>/abl_<model>_<supervisor>[_s<k>]/episodes/`.

**Code state.** The ablation ran with GameBoyRL's leg design: 5-step legs, and executor memory reset at every leg. Both defaults changed afterwards (2026-10-06): legs are 10 steps, and every leg now inherits the previous leg's executor memory (plans/archive/agents.md section 9). The code that produced these numbers was never committed on its own. `--max_leg_steps 5` restores the leg length, but the per-leg memory reset has no switch in the committed code.

**Question:** for each config (env × model), which GameBoyRL supervisor arm gives the best test-set
success, and what does each cost?

**Setup.**
- 3 envs × 2 models (`google/gemma-4-26b-a4b-it`, `google/gemma-4-31b-it`) × 3 arms (`baseline`, `revision`,
  `subgoal`) = 18 runs. The same model is executor and supervisor.
- Fixed executors: GameBoy `single_visual`, Android M3A, Web WebVoyager.
- Native budgets: GameBoy 175 steps; Android `int(10 × complexity)` (10–30 steps); Web 15 steps.
- Supervised legs are 5 steps.
- **Test sets:**

  | env | test set | tasks |
  |---|---|---|
  | GameBoy | `pokemon_red`, all benchmark tasks | 49 |
  | Android | android_world family, 1 instance each, seed 30 | 116 |
  | Web | `WebVoyager_data.cusi.jsonl`: WebVoyager without Amazon, Google Search, Cambridge Dictionary, Allrecipes and BBC News | 429 |

  The runs used 471 Web tasks. BBC News was removed afterwards (2026-10-06), because every arm and both models scored 0/42 on it: WebVoyager cannot click its cookie modal. It is now in `configs/project_vars.yaml`, and `report_ablation.py` drops excluded sites, so every Web number below is over 429 tasks. Wall-clock times still include the BBC tasks, which were run.

- **Web judge:** fixed to `gemma-4-31b-it` through the unchanged `auto_eval.py` (15 screenshots).
- **Success** is always the env's verdict:
  - GameBoy: the task tracker terminated.
  - Android: `task.is_successful` after the agent's `status`.
  - Web: the judge's verdict.
- **Android excluded task:** `OsmAndMarker_0` fails at task setup in every run (`no such table: map_markers`), so it is excluded, as AndroidWorld's harness excludes failed-to-run tasks. That leaves 115 Android tasks per run.
- **Paired p-values:** `cusi.utils.paired_bootstrap` (10,000 resamples, full-size); CIs are a percentile bootstrap.

---

## 1. Success per config, and the best arm

| env | model | baseline | revision | subgoal | best arm |
|---|---|---|---|---|---|
| GameBoy | 26B | 22.4% [12.2, 34.7] | 26.5% [14.3, 38.8] (+4.1, p=0.27) | **34.7%** [22.4, 49.0] (**+12.2, p=0.03**) | **subgoal** |
| GameBoy | 31B | 36.7% [22.4, 51.0] | **40.8%** [26.5, 55.1] (+4.1, p=0.32) | **40.8%** [26.5, 55.1] (+4.1, p=0.29) | revision ≈ subgoal (neither significant; revision is cheaper) |
| Android | 26B | **39.6%** [31.3, 48.7] | 30.4% [21.7, 39.1] (−9.1, p=0.02) | 9.6% [4.3, 14.8] (−30.0, p<0.001) | **baseline** |
| Android | 31B | **61.3%** [52.2, 69.6] | 53.0% [43.5, 62.6] (−8.3, p=0.006) | 36.5% [27.8, 45.2] (−24.8, p<0.001) | **baseline** |
| Web | 26B | **28.7%** [24.5, 33.1] | 23.5% [19.6, 27.5] (−5.1, p=0.006) | 18.9% [15.4, 22.6] (−9.8, p<0.001) | **baseline** |
| Web | 31B | **53.4%** [48.7, 58.0] | 50.1% [45.5, 54.8] (−3.3, p=0.07) | 41.3% [36.6, 45.9] (−12.1, p<0.001) | **baseline** |

Differences are in points against baseline on the same tasks.

**Which method gives the best task performance:**
- **Baseline** (no supervisor) wins 4 of 6 configs: Android and Web, both models.
- **On GameBoy the supervisors help.** `subgoal` is the best arm for the 26B model (+12.2 points, the only significant gain in the grid). For 31B, revision and subgoal tie at +4.1 points, which is within noise.
- **Elsewhere, more supervision does worse:** `subgoal` is the worst arm in all four Android/Web configs, and `revision` sits between it and baseline.
- **The pattern follows the step budget.** Supervision pays off only where the budget is long relative to a 5-step leg (GameBoy, 175 steps = 35 legs). It costs on Android (10–30 steps, 2–6 legs) and Web (15 steps, 3 legs). The cause is in section 2.
- **The 31B model beats the 26B model** in every env and arm. Its baseline is ahead by 14–25 points, more than any supervisor adds.

## 2. Where the supervisors help or hurt (from the episode artifacts)

| env, model | subgoal: plan steps | reached the final target | success when reached / not reached | judge "complete" | regression "undone" | plan "flawed" |
|---|---|---|---|---|---|---|
| GameBoy 26B | 2.9 | 96% | 34% / (2 eps) | 22% of 397 | 52% of 754 | 22% of 371 |
| GameBoy 31B | 2.6 | 98% | 42% / 0% | 25% of 255 | 61% of 658 | 8% of 593 |
| Android 26B | 3.5 | 46% | 21% / **0%** | 43% of 400 | 42% of 238 | 14% of 358 |
| Android 31B | 3.1 | 74% | 49% / **0%** | 58% of 358 | 48% of 267 | 10% of 291 |
| Web 26B | 3.0 | 40% | 39% / 6% | 40% of 1188 | 37% of 505 | 18% of 899 |
| Web 31B | 3.2 | 59% | 64% / 9% | 54% of 1169 | 39% of 466 | 18% of 679 |

### Short budgets: the plan uses up the steps before the agent can finish
- Android and Web score the agent's own finishing action: M3A `status`, WebVoyager `ANSWER`.
  - Under `subgoal`, that action only goes through the env on the final target. On earlier targets it just ends the leg (plans/archive/agents.md decision).
  - With 3–3.5 plan steps, 5-step legs and retries, a 10–15-step budget often runs out on an intermediate target.
- **Android:** the final target was reached in only 46% (26B) and 74% (31B) of episodes. Success when it was not reached is exactly 0.
- **Web:** the final target was reached in only 40% / 59% of episodes. Success when it was not reached is 6–9%, from episodes whose last leg was a non-final leg that happened to answer; the judge reads that leg's chat.
- **How often the finishing action reached the env:**

  | env, model | baseline | subgoal |
  |---|---|---|
  | Android 26B | 70% | 16% |
  | Android 31B | 83% | 44% |
  | Web 26B | 33% | 17% |
  | Web 31B | 59% | 42% |

- **Typical Android failure:** a 4-step plan ("open the app drawer", "tap Clock", "tap Stopwatch", "tap Start") for a 10-step task. The first step fumbles twice and the budget is gone. Under `subgoal`, Android 26B solved 1 task baseline did not and lost 36 that baseline solved, mostly short one-app tasks (CameraTakePhoto, ClockStopWatchRunning, MarkorCreateNote, ...).
- **Revision has no plan but still loses on Android and Web.** Its retries come with hints, but every 5-step leg is a fresh executor. M3A's step history and WebVoyager's chat restart each leg, so the agent loses its own context every 5 steps.
  - On Web, 15 steps become three short chats.
  - Revision's losses concentrate on sites that need long, careful navigation: GitHub 46 → 27% (26B), Huggingface 47 → 33%, Apple 33 → 23%.
  - It gains where a fresh look helps: Coursera 12 → 24%, ESPN 16 → 23% (26B).

### GameBoy: a long budget and long tasks
- 175 steps leave room for plans. `subgoal` reaches its final target in 96–98% of episodes and still spends about 70% of the budget there.
- For 26B, `subgoal` solved 8 tasks baseline did not and lost 2. Its wins are multi-stage tasks:
  - Catch Pidgey;
  - Buy a Potion;
  - Use a not very effective attack on Seaking;
  - Open the map on the wall;
  - Leave Mewtwo in daycare;
  - Switch to your weakest Pokemon;
  - Play the PokeFlute;
  - Look into the binoculars.
- The 31B baseline already solves several of those tasks, which leaves less room: +4.1 points for either arm, with wins and losses mixed (revision +6/−4, subgoal +5/−3).

### Supervisor components (worth fixing before reusing these arms)
- **GameBoy regression check is noisy.** It answers "undone" in 52–61% of checks: the screen changes whenever the player walks, so "the state that made the earlier step complete" looks lost even when it isn't. Each "undone" then pushes a corrective hint.
- **GameBoy judge is strict** (22–25% "complete"), while plan-flaw replans are frequent (1.6 replans per episode for 26B).
- **Android/Web judge is laxer** (37–58% "complete"). Here regressions are rarer and plans are rarely judged flawed (10–16%).

## 2a. Would a bigger step budget fix `subgoal` on Android and Web? Mostly a step problem, but the 5-step leg cap matters more than the total

Split each `subgoal` run by whether the episode reached its final target, and compare with baseline on exactly the same tasks:

| env, model | tasks where subgoal reached the final target | subgoal / baseline on those tasks | subgoal / baseline on the rest | share of the gap from not reaching the final target |
|---|---|---|---|---|
| Android 26B | 53 / 115 | 20.8% / 40.6% | 0.0% / 38.7% | 70% |
| Android 31B | 85 / 115 | 49.4% / 68.8% | 0.0% / 40.0% | 42% |
| Web 26B | 171 / 429 | 38.6% / 50.3% | 5.8% / 14.3% | 52% |
| Web 31B | 254 / 428 | 63.8% / 72.0% | 8.6% / 26.4% | 60% |

**Reading it:**
- **Never reaching the final target explains 42–70% of the gap.**
- **The rest is also mostly steps running out, not wrong answers.** On tasks where `subgoal` reached its final target, it scores 8–20 points below baseline. Splitting those episodes by how they ended:

  | env, model | reached final leg | succeeded | answered wrong | ran out of steps in the final leg |
  |---|---|---|---|---|
  | Android 26B | 53 | 11 | 7 | 35 |
  | Android 31B | 85 | 42 | 9 | 32 |
  | Web 26B | 171 | 66 | 5 | 99 |
  | Web 31B | 254 | 161 | 18 | 68 |

  (A few remaining final legs ended otherwise, e.g. truncated.)
- **When `subgoal` does submit, its answers are as good or better.** Its precision matches or beats baseline's on every config:

  | config | baseline correct | subgoal correct |
  |---|---|---|
  | Android 26B | 57% | 61% |
  | Android 31B | 74% | 82% |
  | Web 26B | 87% | 93% |
  | Web 31B | 90% | 90% |

- **Final legs do not restart from scratch.** Only 0–11% open with home, back, open_app or Google.
- **Where the steps go: the 5-step leg cap.**
  - A non-final leg the agent ended itself ("this step is done") was confirmed by the judge 77–86% of the time.
  - A leg cut off at 5 steps cleared its step only 13–19% of the time, and these are about half of all non-final legs: Android 211 of 475 (26B), Web 811 of 1219 (26B).
  - On Android and Web one plan step often needs more than 5 actions. Examples: "enter Prague as origin, Tokyo as destination and March 20 as the date"; "scroll the whole expense list to find the duplicates".
  - The leg is cut off, the judge says no, and the retry starts a fresh executor. That executor has M3A's history or WebVoyager's chat reset and does not know which fields were already filled.
  - **Evidence that the reset makes retries redo work** (scripts in the session scratchpad: `repeats.py`, `retype.py`). Each later unit is compared with the unit just before it. The control is baseline's own consecutive 5-step windows, where the agent keeps its memory.
    - *Re-typing a string the previous unit already typed* (units where both typed something):

      | | baseline | revision retry | subgoal retry |
      |---|---|---|---|
      | Android 26B | 36% | 43% | 77% |
      | Android 31B | 28% | 45% | 43% |
      | Web 26B | 47% | 71% | 84% |
      | Web 31B | 28% | 39% | 71% |

      A subgoal leg on the *next* step re-types at about baseline's rate (32–34% on Web).
    - *Exact repeats of a move* (the same parsed action on a pixel-identical screen; Android status bar cropped):

      | | baseline | revision retry | subgoal retry |
      |---|---|---|---|
      | Android 31B | 5.3% | 9.3% | 12.6% |
      | Web 31B | 2.9% | 5.3% | 10.2% |

      For 26B there is no clear increase, because 26B's baseline already loops with its memory intact (11–15% of moves repeat). On Web 26B, revision retries even repeat less than baseline (9.5% against 15.1%).
    - *Caveats:*
      - Retries happen on steps that already failed, which tend to be harder, so part of the increase is selection.
      - Re-typing can be legitimate when the page cleared the field.
      - The Android typing samples are small (37–73 units).
      - So the reset effect is clear on re-typed input and on the 31B model, and not shown for 26B's exact-screen repeats.
  - GameBoyRL's 5-step cap suits GameBoy, where a step is one button press; it is too short here.
- **Other plan problems seen in the stuck steps:**
  - Compound steps (several fields plus a search in one step).
  - Conditions the judge cannot check from one screen ("until no more duplicates can be found").
  - Steps the agent cannot do at all (a GitHub sign-up CAPTCHA slider).
  - Gesture instructions M3A cannot carry out ("swipe down from the top of the screen to open quick settings"; M3A has no swipe-from-edge action). The planner is told not to name actions, but it names gestures.
- **Revised conclusion:**
  - `subgoal`'s deficit on Android and Web is almost entirely steps running out: mid-plan from capped legs, and again in the final leg.
  - A bigger total budget would recover more than half the gap, but it would be spent inefficiently while legs stay capped at 5.
  - Longer or uncapped legs, plus executor memory kept across retries of the same step, attack the cause directly.
- **Baseline would gain too.**
  - Web baseline ran out of steps in most episodes: 287/429 for 26B, and 159 truncated plus 14 at max steps for 31B.
  - Android baseline was truncated in 35/115 (26B) and 19/115 (31B).
  - A fair test raises the budget for both arms.
- **How much budget subgoal used:** before reaching the final target, it spent 31–41% of the budget (Android 7.5–12 steps, Web 4.6–5.8). That is about 60% of the steps baseline used in total on the same tasks (Android 12.5–20, Web 8–11).
  - So roughly 1.5–2× the budget would let most final legs start with what baseline had.

**Experiments that would settle it** (not run):
1. **Native budget, longer legs.** `subgoal` with `--max_leg_steps` 10, and with legs capped only by the budget (the leg ends when the agent says the step is done). This tests the cap alone, and stays comparable to the benchmark.
2. **Memory across retries.** `subgoal` keeping the executor's memory across retries of the same target (M3A history, WebVoyager chat), with the hint added.
3. **2× budget.** Baseline and `subgoal` at 2× the native budget (Android `int(20 × complexity)`, Web 30). The env step limit must be raised with it. This answers the design question, not the leaderboard one.
4. **Planner changes.** Tell the planner the step budget and the action set, and ask for steps of at most about 5 actions with single-screen termination conditions.

## 3. Cost

| env, model | tokens/task (baseline → revision → subgoal), supervisor share | seconds/task mean (baseline → revision → subgoal) | successes per agent GPU-hour (baseline / revision / subgoal) |
|---|---|---|---|
| GameBoy 26B | 150k → 221k → 293k (supervisor 30% / 35%) | 506 → 574 → 775 | 5.9 / 6.3 / 6.0 |
| GameBoy 31B | 134k → 184k → 256k (30% / 38%) | 1059 → 1108 → 1564 | 4.7 / 4.8 / 3.5 |
| Android 26B | 158k → 192k → 211k (5% / 9%) | 291 → 335 → 348 | 3.4 / 1.8 / 0.6 |
| Android 31B | 131k → 172k → 193k (4% / 9%) | 337 → 456 → 500 | 4.1 / 2.8 / 1.8 |
| Web 26B | 60k → 50k → 58k (13% / 23%) | 154 → 160 → 178 | 12.4 / 9.9 / 7.2 |
| Web 31B | 34k → 34k → 46k (12% / 25%) | 172 → 182 → 250 | 20.3 / 18.0 / 11.0 |

**Notes on the cost figures:**
- **GameBoy executor token budget:** 8000 under revision/subgoal against 2000 for baseline. These are GameBoyRL's own per-arm settings (`--executor_max_new_tokens`), kept for parity; they confound the GameBoy comparison slightly.
- **Web tokens:** a fresh chat per leg keeps fewer screenshots in context, so executor tokens can fall under a supervisor.
- **GPU-hours** count agent wall-clock × GPUs only; vLLM start-up and judging are excluded.

## 4. How long each run took (for future reference)

**Hardware:** NVIDIA RTX A6000 (48 GB). Jobs ran through `slurm/eval.sh` with a fresh vLLM 0.30 server per job, its own port, and prefix caching off.

**Per-env configuration:**

| env | GPUs | vLLM | parallelism | partition |
|---|---|---|---|---|
| GameBoy | 2 | tp=2 | 8 workers | `medium-lg` |
| Android | 4 (exclusive node) | tp=4 | 3 emulators, one adb server | `medium-lg` |
| Web | 2 | tp=2 | 4 browsers | `scavenge-lg` (6 h limit) |

- **Android:** AndroidWorld always uses adb's default port, so Android jobs cannot share a node.
- **Web:** runs over 6 h were finished by a continuation job.

| env | model | baseline | revision | subgoal | vLLM start-up |
|---|---|---|---|---|---|
| GameBoy (49 tasks) | 26B | 0.93 h (53 tasks/h) | 1.04 h (47/h) | 1.41 h (35/h) | 3–3.5 min |
| GameBoy | 31B | 1.91 h (26/h) | 2.08 h (24/h) | 2.89 h (17/h) | 6.6 min |
| Android (116 tasks) | 26B | 3.4 h (35/h) | 4.8 h (24/h) | 4.7 h (25/h) | 2.5–5 min |
| Android | 31B | 4.3 h (27/h) | 5.5 h (21/h) | 5.8 h (20/h) | 2.5–3.5 min |
| Web (471 tasks) | 26B | 5.0 h + 0.5 h judge (95/h) | 5.1 h + 0.4 h (93/h) | 5.6 h + 0.6 h (84/h) | 2.5–5 min |
| Web | 31B | 5.7 h + 1.2 h judge (83/h) | 6.0 h + 0.9 h (79/h) | 8.1 h + 0.7 h (58/h) | 2.5–5 min |

**Rules of thumb:**
- Supervisors cost 1.0–1.5× baseline wall-clock (subgoal up to 1.5×).
- 31B takes about 2× the 26B time on GameBoy, but only about 1.1–1.3× on Android and Web, where env time dominates. In the Android smoke episodes, a task took about 25 s of model time against 80–150 s of env time.
- The whole grid took about 74 h of agent wall-clock and about 205 GPU-hours (about 215 with Web judging and reruns), done in 10 h of calendar time (00:20–10:30 UTC) with up to 14 jobs in parallel.

**Timing records:**
- Per run: `timing.jsonl` in each run dir.
- Per task: `seconds` in `results.jsonl`.
- Per episode: model vs env seconds in each episode's `meta.json`.
- The four Web jobs cut off by the 6 h limit had their timing reconstructed from their logs (flagged `reconstructed`).

**Infrastructure incidents** (did not change any score, but did add time):
- In three Android runs (26B revision, 26B subgoal, 31B baseline), shard 2's emulator died on an SMS task and its restart failed. The 10–13 tasks behind it on that shard failed to run. They, and 1–2 isolated adb failures in other runs, were rerun in a second job per run (0.5–0.7 h).
- Six duplicate Web continuation jobs were queued by mistake. They were chained after their originals, and the two not needed were cancelled before writing anything.

## 5. Caveats

- **GameBoy is small (49 tasks).** Only 26B `subgoal` vs baseline is significant; the 31B differences are 2 tasks out of 49.
- **Web on live sites is noisy run to run.** In the parity check the native harness agreed with itself on 20 of 24 tasks. The paired differences on 429 tasks are still significant for every 26B comparison and for 31B `subgoal`; 31B `revision` (−3.3, p=0.07) is not.
- **The Web judge is the 31B model**, so the 31B Web runs are judged by their own model. Comparisons within a model are unaffected, but the 26B vs 31B gap may carry some self-preference.
- **BBC News** scored 0% for every arm, from the known cookie-modal bug in WebVoyager's element mapping (`plans/archive/web_voyager_plan.md`). It is excluded from the Web numbers (see Setup) until that bug is fixed.
- **Fixed settings.** The supervisors ran with GameBoyRL's settings throughout (5-step legs, 3 attempts per target, final-attempt multiplier 3, 2 replans, 8 frames per judge slice). Prompt wording for Android and Web is our own (plans/archive/agents.md decision 1).
- **Possible next steps:** longer legs, keeping executor memory across legs, or sending the agent's finish through the env when the budget is nearly spent. Each could change the Android and Web picture; none was tested here.
