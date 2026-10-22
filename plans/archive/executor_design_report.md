# Executor design: results (plans/archive/design_plan.md)

The best executor for a fixed model (`google/gemma-4-31b-it`, also the Web judge), supervisor off, on Web
(WebVoyager), Android (AndroidWorld, M3A) and GameBoy (GameBoyRL, Pokémon Red and Crystal). No fine-tuning; GameBoy
reads nothing from RAM/ROM except the allowed VRAM text reader. Every decision and its rule, written before the
stage ran, is in `plans/decisions.md` §E; the feature catalogue and sources are in `plans/archive/executor_lit.md`; per-arm
tables are in `storage/eval/design/tables_*.md` (from `tmp_design_tables.py`).

## 1. Final designs and the full benchmark

| env | final design | dev: base → design | full benchmark: base → design | held-out (full − dev) |
|---|---|---|---|---|
| Web | **F10 + F14=google + F08=5** | 52.8% → 65.7% (+12.9 [+9.2, +16.7], 82/19, p < 0.001; 250 tasks, 3 repeats) | 56.4% → 66.4% (+10.0 [+5.8, +14.2], 66/23, p < 0.001; 429 tasks) | 57.0% → 66.5% (+9.5 [+2.8, +16.2], 28/11, p = 0.009; 179 tasks) |
| Android (stage 6, E6.9) | **Minitap bundle F27 + F55–F61** (prototype) | 60.5% → 67.8% (+7.3 [−0.6, +15.2], 11/4, p = 0.029; 57 tasks, 3 repeats) | not run | 60.3% → 68.4% (**+8.0 [+0.9, +15.8]**, 11/3, p = 0.006; 58 tasks, **3 repeats each**) |
| Android (stage 3, superseded) | F27 | 60.5% → 65.5% (+5.0 [+1.2, +9.6], 8/2, p = 0.039; 57 tasks, 3 repeats) | 62.2% → 63.0% (+0.9 [−5.2, +7.0], 8/7, p = 1.0; 115 tasks) | 62.9% → 62.9% (+0.0 [−8.6, +8.6], 3/3; 58 tasks) |
| GameBoy | **baseline executor** (nothing promoted) | 34.8% (5 repeats) | 31.3% [22.2, 40.4] (baseline only; 99 tasks) | 24.5% [12.2, 36.7] (49 tasks) |

Full benchmark: 1 repeat of base and design per env (user decision E4.1), clean implementation (§4), same tasks,
paired. Diffs are points; CI = bootstrap over tasks; W/L = tasks won/lost; p = exact McNemar.

**Against the earlier eval (`plans/archive/results.md`, 31B, same budgets):** Web baseline then 53.4% (best arm: baseline;
the supervisors cost 3–12 points); the Web design (66.4%) is 13 points above every earlier arm. Android baseline then
61.3% (supervisors −8 to −25). GameBoy baseline then 36.7% (49 tasks, not the same task set as the 99 here); there the supervisors were the only thing that helped
(revision/subgoal 40.8%, n.s.), and no executor feature did.

### The components

| ID | what it does | source | dev evidence |
|---|---|---|---|
| F10 | each Web observation ends with "This is iteration k of 15; n more after this one", plus a warning in the last 3 | catalogue F10 (evidence weak); motivated by the truncation share of baseline endings (lit review §5) | alone +7.6 (1 rep), +7.2 (2 reps); removing it from D −7.9 |
| F14=google | the `Google` action is removed from WebVoyager's system prompt and rejected if used | catalogue F14 action pruning (AgentOccam); the target is `ours`: the Web failure taxonomy found the Google action lands on a CAPTCHA in ~29% of episodes (E2.4) | alone +5.6; on top of F10 +3.6; removing it from D −7.6 |
| F08=5 | 5 past screenshots kept in the chat (WebVoyager keeps 3) | catalogue F08 (design choices in GUI-Owl-1.5, MobileRL, WebVoyager; no ablation) | alone +5.2 (F08=1 −7.2, F08=8 +4.4); on top +3.2; removing it −2.7 |
| F27 | Android: a `Notes:` field the model writes in each action reply, shown back in the next action prompt | catalogue F27 (Mobile-Agent-v2 memory unit, Mobile-Agent-E notetaker) | alone +7.0 (2 reps), +5.0 (3 reps); F19 and F21 add nothing on top |

Code: switches on `WebVoyagerExecutor(max_attached_imgs, show_budget, allow_google)` and `M3AExecutor(notes)`,
off by default; `cusi/eval/features.py` maps `--features` IDs to them (`CLEAN`); `tests/executor_design_test.py`.

## 2. What was tried

Screening: each feature alone vs base, paired, on dev; threshold = noise floor / √repeats (Web 4.8 pt at 1 repeat,
Android 6.2 pt, GameBoy 10.5 pt). Then greedy combination in rank order (keep if > t/2), same-target swaps, and
leave-one-out with 3 repeats.

**Web** (250 dev tasks). Promoted: F10 +7.6, F11 forced answer +6.4, F14=google +5.6, F08=5 +5.2. Dropped: F08=1
−7.2, F08=8 +4.4, F04 page text, F09 change report, F21 plan, F19 structured output, F49 task rewrite +0.0, F29
self-written tips +2.4, F37 vote +0.8 (3× calls), F14=wait +1.2, F32 loop warning +1.6, F27 notes +1.2, F41 answer
self-check −2.8. F11 lost its swap with F10 (−2.8). F12 (a `GoTo url` action): +2.7 on 100 tasks, unfinished
(E2.37; same target as F14=google).

**Android** (57 dev tasks). Promoted: F27 notes +7.0, F19 structured output +5.3, F21 plan +4.4. Dropped: F31
no-op note +0.0, F41 finish check −3.5, F08 previous screen +3.6, F29 tips +2.6, F38 critic +3.5 (1.55× calls), F32
+3.2, F34 reflector +1.8, F11 +2.6, F10 +2.0. In combination F19 (−4.4) and F21 (+1.8) added nothing on top of
F27. F25 (self-summary inside the action call, half the calls) −1.8 alone; F27+F25 −6.4 vs F27 (3 reps), so not a
valid cost variant of D, though it is only −1.5 vs base at half the calls and tokens (E3.13; untested on held-out).

**GameBoy** (50 dev tasks, 3–4 repeats). Nothing promoted. F07=3 upscale +1.2 (4 reps), F32 −0.1, F31 −0.8, F38
critic −1.5 (1.73× calls), F28 objectives log −2.1, F10 −2.8, F16 multi-action −2.8, F29 −2.8, F04 VRAM text
−3.8 (4 reps), F54 mode detection −3.8 (4 reps), F37 vote −4.1 (2.16× calls), F19 −4.1, F17 macros −5.3 (4 reps),
F26=15 longer history −5.5, F25 −5.5, F21 −6.8, F30 game guide from priors −12.8 (dropped early).

## 3. Findings

- **Web: a real, confirmed gain; Android: none confirmed.** The Web design keeps about +10 points on the full
  benchmark and on held-out tasks. Android's +5.0 on dev shrank to +0.9 on the full set and +0.0 held-out (and
  +1.8 on the dev tasks within the same full run): the dev estimate was mostly selection plus noise. F27 is
  cost-neutral and harmless, so it is still the reported design, but there is no evidence it helps.

- **Web gains come from fixing the harness, not from more reasoning.** The three kept features are a budget
  counter, removing an action that traps the agent, and more visual history. Every reasoning add-on (plan, notes,
  structured output, self-check, vote, rewrite) stayed under the threshold (−2.8 to +1.2).
- **Gains overlap.** On Android, F19 and F21 each helped alone (+5.3, +4.4) but added nothing on top of the notes;
  on Web, F11 and F10 target the same failure and only one survives.
- **GameBoy resists executor changes.** 17 features, none above the noise; the failure mass is loops and screen
  misreads, and neither loop handling (F32 warning, F31 no-op ban, F38 critic) nor text or mode inputs (F04, F54, F07) moved
  them. An
  offline check (§7, E5.2) found loop signals, even the repo's embedding archive, flag revisits on most steps and
  do not separate loop failures from others.
- **Literature priors transferred poorly.** The top-ranked GameBoy feature (F04) and Android's second (F34) did not
  help this model in this harness; Web's best features were rated weak evidence.

## 4. §7 interaction with the state-space features

- **Recovery (E5.2):** an in-episode novelty archive does not make a better loop detector than the frame hash
  (offline, 150 GameBoy baseline episodes; `storage/eval/design/s7_loop_signal.md`).
- **Observation:** `text_boxes` as executor input (F04) did not help on GameBoy (−3.8).
- **Not run:** exploration-derived knowledge as executor context (self-written tips, F29, were borderline or
  negative on all three envs: Web +2.4, Android +2.6, GameBoy −2.8; so this was not pursued); novelty as a supervisor trigger (supervisor off throughout).

## 5. Caveats

- One full-benchmark repeat per arm (E4.1); a ~5 pt effect on Android or ~10 pt on GameBoy would not be resolved.
- The Web judge is the same Gemma model; base and design are judged alike.
- Selection on dev inflates dev gains; the held-out rows are the unbiased estimate.
- Early-drop (E2.19) and the GameBoy 4th-repeat retests were added or applied mid-stage; each is flagged in
  decisions.md.

## 6. State of the code, and what is left

- The scratch prototypes (`tmp_exec_features.py`, `tmp_submit_design.sh`, `tmp_design_tables.py`, `tmp_s7_*.py`,
  `tmp_check_clean.py`, watchers) are still in the repo root: deleting files is not permitted in this session.
  `cusi/eval/features.py` still falls back to `tmp_exec_features.py` for any arm with a non-clean feature; once the
  prototypes are deleted, that fallback (and `PROTOTYPE_MODULE`) should go, leaving only the `CLEAN` switches.
- Nothing is committed.
- Open questions worth a run each: (a) Android baseline + F25 on held-out, since it looked close to baseline at half
  the cost (E3.13); (b) more held-out repeats if a ~2 pt Android effect matters; (c) the text half of the novelty
  archive as a GameBoy loop signal, which needs trajectories that store `text_boxes` (E5.2).

## 7. Stage 6: AndroidWorld with Minitap and ARTEMIS ideas (E6.0–E6.9)

- **Sources:** Minitap mobile-use (116/116 reported) and Google ARTEMIS (>99% reported, built on mobile-use). Both
  self-reported, frontier models, several calls per step; ARTEMIS has no step limit and uses ADB-shell/content-URI
  probes that read the evaluator's state (excluded). Mapping and exclusions: decisions.md E6.0, E6.3.
- **Screening (57 dev tasks, threshold 4.4 / 3.6):** F61 device context promoted (+5.6 at 3 repeats). Dropped: F55
  clear_text −0.9, F56 read-back +1.2, F57 no-Enter +2.6, F58 rules +0.9, F59 subgoals +3.2, F60 resource ids +2.3,
  F62 expectation +3.5, F63 list scan +3.5, F64 controlled swipe +0.9.
- **Candidate:** greedy design = F61 (66.1% on dev); bundle F27 + F55–F61 67.8% → bundle (E6.1 rule).
- **Held-out (58 tasks, 3 repeats each): +8.0 [+0.9, +15.8], p = 0.006 → confirmed.** Calls ×1.07, tokens ×1.13.
  Robust to the one unscored (emulator-error) episode: +7.5 [+0.3, +15.2] if counted as a failure. Which components
  carry it is not established.
- **GameBoy transfer:** only F62 was new there; 4 repeats, +1.2 [−4.8, +6.9] → dropped. GameBoy stays baseline
  (18 features, none promoted). A reviewer-facing write-up of the GameBoy search is in
  `appendix_notes/gameboy_executor_search.tex`.
- **Left to do:** clean implementation of F55–F61 as off-by-default `M3AExecutor`/env switches (as E5.1), with a
  prompt-identity check and a live smoke; the confirmation above used the prototypes in `tmp_exec_features.py`.
- **Infrastructure fix:** the vLLM port is now 8100 + (job id mod 400), below the emulator gRPC ports.
