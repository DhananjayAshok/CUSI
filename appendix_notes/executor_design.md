# Appendix: executor design search for a fixed model

This appendix documents the search for the best *executor* (the per-step acting agent) on three benchmarks, with
the model held fixed. It lists every option tried, how each fared, and what survived. All decision rules were
written down before the corresponding stage was run; deviations are flagged where they occurred.

## A.1 Setup

**Model.** A single open-weight vision-language model, Gemma-4-31B-it, is used for every role: the acting agent,
the failure labeller, and the Web success judge. No fine-tuning. Sampling is unseeded; repeats are independent runs.

**Paradigm.** The agent framework has a supervisor/executor split. The supervisor is switched off in every arm, so
all differences come from the executor alone.

**Benchmarks and native harnesses.**

| Benchmark | Native agent (baseline executor) | Step budget | Success | Tasks (full) |
|---|---|---|---|---|
| Web (WebVoyager, live sites) | WebVoyager's own agent: set-of-mark screenshot + element text, chat history with the last 3 screenshots | 15 iterations | VLM judge (same model) on the final answer and screenshots | 429 |
| Android (AndroidWorld) | M3A: set-of-mark screenshot + UI-element list, one-line summaries of past steps, a separate summary call per step | AndroidWorld's `10 × complexity` | Programmatic check of device state | 116 (115 scorable) |
| GameBoy (Pokémon Red and Crystal) | GameBoyRL's executor: screenshot + action history, one button press per step | 175 steps | Programmatic task check | 99 |

On GameBoy, the agent may not read game memory or ROM. The one exception is an on-screen text reader that decodes
the tile layer the console is displaying (the same text a player sees). There is no save-state lookahead.

**Dev and held-out splits.** Each benchmark's task list was split once, before any feature work:

| Benchmark | Dev | Held-out (full − dev) |
|---|---|---|
| Web | 250 tasks (150, then grown to 250 for power; stratified by site) | 179 |
| Android | 57 scorable (58 listed) | 58 |
| GameBoy | 50 (25 Red + 25 Crystal) | 49 |

All feature selection used dev only. Held-out tasks were touched only in the final confirmation.

**Noise floors.** The baseline was run 3 times on dev (5 times on GameBoy). The floor is the largest pairwise gap
between baseline repeats, or 1.96 × SD(paired diff)/√n if larger:

| Benchmark | Baseline on dev | Floor (1 repeat) | Screening threshold |
|---|---|---|---|
| Web | 52.8% (52.8 / 50.4 / 55.2) | 4.8 pt | ±4.8 at 1 repeat, ±3.4 at 2 |
| Android | 60.5% (60.5 / 61.4 / 58.9) | 6.2 pt | ±4.4 at 2 repeats, ±3.6 at 3 |
| GameBoy | 34.8% (34 / 32 / 34 / 38 / 36) | 10.5 pt | ±6.0 at 3 repeats, ±5.25 at 4 |

**Statistics.** Arms are compared task by task on the same tasks. Success is the mean over tasks of the per-task
mean over repeats. 95% CIs bootstrap over tasks (5,000 resamples). p is an exact McNemar test on seed-matched
discordant pairs. W/L counts the tasks an arm wins or loses against its comparison.

## A.2 Procedure

1. **Literature review and catalogue.** We surveyed agent papers and open-source harnesses for executor-side ideas.
   54 features were catalogued with their evidence and an implementation note per benchmark, then ranked per
   benchmark by evidence ÷ cost into a shortlist.
2. **Failure taxonomy.** About 50 failed baseline dev episodes per benchmark were labelled with one main failure
   bucket by the same model, and about 20% of labels were spot-checked against the trajectories (all spot-checked
   labels were consistent).
3. **Smoke test.** Every new prototype first ran live on 4 tasks, and the episode records were read to confirm it
   did what it claimed. This checks function, not effect.
4. **Screening.** Each feature alone vs the baseline, on dev, at the repeat count the floor requires.
   - **Promote** if diff ≥ threshold and calls per task < 2× baseline; or if neutral and it clearly fixes its
     target failure bucket.
   - **Drop** if diff ≤ −threshold, or if calls ≥ 2× with no gain.
   - **Retest** once with one more repeat if borderline and in the top half of the prior ranking; otherwise drop.
5. **Combination.** Greedy, in order of screening gain. A feature is kept if design + feature beats the design by
   more than half the threshold, at < 2× calls (2 repeats; 3 on GameBoy). Features aimed at the same failure were
   then tried as swaps for the kept one.
6. **Leave-one-out.** Remove each component in turn (3 repeats). A component is dropped if removing it costs
   less than half the threshold; ties go to the simpler design.
7. **Clean implementation and confirmation.** The surviving features were re-implemented as off-by-default
   switches on the executors, checked to build exactly the same prompts as the prototypes, and smoke-tested live.
   The final design and the baseline then ran on the full benchmark, and results are reported on the full set and
   on held-out tasks.

**Failure taxonomies** (labelled baseline failures on dev):

| Benchmark | Main buckets |
|---|---|
| Web (50) | environment error 25 (22 of them the agent's own `Google` action landing on a CAPTCHA page), loop/no-op 14, out of steps 6, other 5 |
| Android (49) | wrong plan 11, early finish 10, out of steps 10, loop/no-op 7, screen misread 5, none clear 6 |
| GameBoy (50) | loop/no-op 27, screen misread 15, wrong plan 4, out of steps 4 |

The Web CAPTCHA trap was the single largest failure cause found. The agent's `Google` action often lands on a
CAPTCHA, `GoBack` does not escape it, and the agent then spends its remaining iterations there and answers from
memory.

## A.3 Feature catalogue (screened subset)

IDs follow the catalogue. W/A/G marks where each was screened.

| ID | Feature | Prior evidence (catalogue) | Screened on |
|---|---|---|---|
| F04 | On-screen text as a text block (Web: page text; GameBoy: decoded on-screen text) | lmgame-Bench perception gains; BALROG: vision often hurts | W, G |
| F07 | Screenshot upscaling (GameBoy 3×) | weak (RegionFocus, coordinate grounders) | G |
| F08 | Number of past screenshots kept (Web k = 1, 5, 8; Android: previous screen as an extra image) | weak; design choices in GUI-Owl-1.5, MobileRL, WebVoyager | W, A |
| F09 | Programmatic change report (what changed after the last action) | weak (Agent-E, browser-use) | W |
| F10 | Step budget shown each step | weak | W, A, G |
| F11 | Forced or nudged finish near the budget end | weak; shipped in browser-use and GUI-Owl-web | W, A |
| F12 | Extra action: `GoTo url` | medium in general (AgentOccam action alignment) | W |
| F14 | Action-space pruning (Web: drop `Wait`; or drop `Google`) | medium (AgentOccam action reduction) | W |
| F16 | Multi-action decisions (several presses per call) | weak | G |
| F17 | Macro actions (walk n tiles, advance dialogue) | weak | G |
| F19 | Structured reasoning template | medium (AndroidLab, in XML mode) | W, A, G |
| F21 | Persistent plan field written by the model and shown back | medium (Mobile-Agent-v2, Agent S2, AgentOccam; all multi-call) | W, A, G |
| F25 | History form: self-written summary inside the action call instead of M3A's separate summary call (Android); summarised history (GameBoy) | medium (M3A ablation) | A, G |
| F26 | Longer action history (GameBoy 15 steps) | weak | G |
| F27 | Notes buffer written by the model and shown back | weak to medium (Mobile-Agent-v2 memory unit, Mobile-Agent-E notetaker) | W, A |
| F28 | GameBoy objectives-and-places log | weak | G |
| F29 | Self-written tips doc from the model's prior knowledge, written once per task | medium for exploration docs (AppAgent); parametric docs unproven | W, A, G |
| F30 | Game guide written from priors (GameBoy) | medium for M3A guidelines + reflection | G |
| F31 | No-op detection (screen unchanged → note, or ban on repeating the press) | weak to medium (MobileUse) | A, G |
| F32 | Loop warning (same screen ≥ 3 times in the last 20 steps) | weak to medium (MobileUse; WebVoyager failure analysis) | W, A, G |
| F34 | Outcome-classifying action reflector | medium to strong (MobileUse +5.7) | A |
| F37 | Self-consistency vote over k = 3 samples | medium | W, G |
| F38 | Pre-execution critic | weak to medium | A, G |
| F41 | Completion self-check before finishing | medium to strong in direction (TTI, MobileUse, Voyager) | W, A |
| F49 | Task rewrite before acting (Web) | none found | W |
| F54 | Game-mode detection (overworld / dialogue / menu / battle) from the screen | weak | G |

## A.4 Results: Web

**Screening** (250 dev tasks; threshold 4.8 at 1 repeat, 3.4 at 2):

| Arm | Repeats | Success | Diff vs base [95% CI] | W/L | p | Calls/task | Decision |
|---|---|---|---|---|---|---|---|
| baseline | 3 | 52.8% | — | — | — | 10.0 | — |
| F10 budget shown | 2 | 60.0% | +7.2 [+3.3, +11.1] | 64/28 | <0.001 | 9.8 | **promote** |
| F11 forced answer | 1 | 59.2% | +6.4 [+2.0, +10.9] | 48/26 | 0.017 | 9.9 | **promote** |
| F14=google (no `Google`) | 1 | 58.4% | +5.6 [+1.1, +10.3] | 49/27 | 0.049 | 9.7 | **promote** |
| F08=5 screenshots | 1 | 58.0% | +5.2 [+0.8, +9.7] | 46/28 | 0.060 | 9.9 | **promote** |
| F08=8 screenshots | 1 | 57.2% | +4.4 [+0.1, +8.7] | 43/29 | 0.099 | 9.9 | drop (below F08=5) |
| F04 page text | 1 | 55.6% | +2.8 [−1.5, +7.1] | 43/29 | 0.36 | 9.8 | drop |
| F29 tips doc | 1 | 55.2% | +2.4 [−2.0, +6.7] | 42/32 | 0.45 | 11.1 | drop |
| F12 `GoTo url` | 1 | 57.0% (100 tasks) | +2.7 [−4.3, +10.0] | 17/15 | 0.54 | 9.8 | unfinished (same target as F14=google) |
| F32 loop warning | 2 | 54.4% | +1.6 [−1.6, +4.9] | 41/38 | 0.14 | 9.8 | drop after retest |
| F14=wait (no `Wait`) | 2 | 54.0% | +1.2 [−2.1, +4.5] | 46/36 | 0.24 | 10.0 | drop after retest |
| F27 notes | 2 | 54.0% | +1.2 [−2.8, +5.1] | 47/43 | 0.25 | 10.3 | drop after retest |
| F21 plan | 1 | 53.6% | +0.8 [−3.3, +4.9] | 36/34 | 0.88 | 9.8 | drop |
| F37 vote (k = 3) | 1 | 53.6% | +0.8 [−3.3, +4.9] | 36/34 | 0.87 | 30.1 | drop (3× calls) |
| F49 task rewrite | 1 | 52.8% | +0.0 [−4.4, +4.3] | 38/36 | 1.0 | 10.6 | drop |
| F09 change report | 1 | 51.6% | −1.2 [−5.6, +3.2] | 34/40 | 0.78 | 9.8 | drop |
| F19 structured output | 1 | 51.6% | −1.2 [−5.5, +2.9] | 35/36 | 0.75 | 10.3 | drop |
| F41 answer self-check | 2 | 50.0% | −2.8 [−6.7, +1.1] | 42/48 | 0.47 | 11.1 | drop after retest |
| F08=1 screenshot | 1 | 45.6% | −7.2 [−12.3, −2.3] | 31/54 | 0.018 | 10.3 | drop |

The screenshot count matters in both directions: 1 screenshot hurts (−7.2) and 5 helps (+5.2), with no further gain
at 8.

**Combination** (2 repeats; keep if > +1.7 over the current design):

| Step | Design | Success | Gain over previous design | Decision |
|---|---|---|---|---|
| 0 | F10 | 60.0% | +7.2 over base | start |
| 1 | F10 + F14=google | 63.6% | +3.6 (47/32, p = 0.10) | keep |
| 2 | F10 + F14=google + F08=5 | 66.8% | +3.2 (48/34, p = 0.15) | keep |
| swap | F11 in place of F10 | 64.0% | −2.8 vs step 2 (31/43) | F10 stays |

**Leave-one-out** (3 repeats; drop if removal costs < 1.4):

| Removed | Success | Change |
|---|---|---|
| none (design) | 65.7% | — |
| F10 | 57.9% | −7.9 (24/62) |
| F14=google | 58.1% | −7.6 (24/63) |
| F08=5 | 63.1% | −2.7 (39/53) |

All three components survive. **Final Web design: F10 + F14=google + F08=5.**

**Full benchmark** (1 repeat each, clean implementation):

| Tasks | Baseline | Design | Diff [95% CI] | W/L | p |
|---|---|---|---|---|---|
| Full, 429 | 56.4% | 66.4% | +10.0 [+5.8, +14.2] | 66/23 | <0.001 |
| Held-out, 179 | 57.0% | 66.5% | +9.5 [+2.8, +16.2] | 28/11 | 0.009 |

Calls per task are unchanged (9.8 vs 9.9) and seconds per task similar (297 vs 290). Input tokens rise 31% (44.1k
vs 33.6k), from the two extra screenshots. The held-out gain is close to the dev gain (+12.9), slightly smaller as
expected after selection on dev.

## A.5 Results: Android

**Screening** (57 dev tasks; threshold 4.4 at 2 repeats, 3.6 at 3):

| Arm | Repeats | Success | Diff vs base [95% CI] | W/L | p | Calls/task | Decision |
|---|---|---|---|---|---|---|---|
| baseline | 3 | 60.5% | — | — | — | 23.3 | — |
| F27 notes | 3 | 65.5% | +5.0 [+1.2, +9.6] | 8/2 | 0.039 | 22.3 | **promote** |
| F19 structured output | 2 | 65.8% | +5.3 [−0.6, +11.4] | 8/4 | 0.27 | 24.9 | **promote** |
| F21 plan | 3 | 64.9% | +4.4 [−0.9, +10.5] | 7/4 | 0.17 | 25.6 | **promote** (after retest) |
| F08 previous screen | 2 | 63.4% | +3.6 [−3.0, +10.4] | 5/4 | 0.55 | 23.3 | drop |
| F38 critic | 2 | 64.0% | +3.5 [−1.5, +9.1] | 7/4 | 0.58 | 36.0 | drop (not shortlisted, 1.55× calls) |
| F32 loop warning | 3 | 63.7% | +3.2 [+0.3, +6.4] | 5/1 | 0.18 | 22.8 | drop after retest |
| F29 tips doc | 2 | 63.2% | +2.6 [−6.7, +11.7] | 7/5 | 0.63 | 25.0 | drop |
| F11 forced finish | 3 | 63.2% | +2.6 [−1.5, +7.3] | 4/3 | 0.39 | 23.0 | drop after retest |
| F10 budget shown | 3 | 62.6% | +2.0 [−2.0, +6.4] | 4/2 | 0.55 | 23.8 | drop after retest |
| F34 reflector | 3 | 62.3% | +1.8 [−3.2, +7.6] | 5/5 | 0.58 | 24.9 | drop after retest |
| F31 no-op note | 2 | 60.5% | +0.0 [−3.8, +3.8] | 4/5 | 1.0 | 23.8 | drop |
| F25 self-summary (half the calls) | 2 | 58.8% | −1.8 [−7.9, +3.8] | 3/4 | 0.51 | 11.4 | drop; cost variant tested later |
| F41 finish self-check | 2 | 57.0% | −3.5 [−10.2, +2.0] | 4/5 | 0.23 | 27.1 | drop |

The loop warning (F32) fixed only 1 of the 4 loop/no-op failures it touched, so it did not meet the bucket clause.

**Combination** (2 repeats; keep if > +3.1 over the design):

| Step | Design | Success | Gain over F27 | Decision |
|---|---|---|---|---|
| 0 | F27 | 67.5% (2 repeats) | — | start |
| 1 | F27 + F19 | 63.2% | −4.4 (3/6) | drop F19 |
| 2 | F27 + F21 | 69.3% | +1.8 (6/4) | drop F21 |
| cost | F27 + F25 | 59.1% (3 repeats) | −6.4 (3/10, p = 0.027) | not a valid cheaper variant |

Each of F19 and F21 helped alone but added nothing on top of the notes field. **Leave-one-out:** removing F27 (=
baseline) costs 5.0 > 1.8, so **final Android design: F27**.

**Full benchmark** (1 repeat each, clean implementation):

| Tasks | Baseline | F27 | Diff [95% CI] | W/L | p |
|---|---|---|---|---|---|
| Full, 115 | 62.2% | 63.0% | +0.9 [−5.2, +7.0] | 8/7 | 1.0 |
| Held-out, 58 | 62.9% | 62.9% | +0.0 [−8.6, +8.6] | 3/3 | 1.0 |
| Dev tasks within this run, 57 | 61.4% | 63.2% | +1.8 [−8.8, +12.3] | 5/4 | 1.0 |

**The Android dev gain did not replicate.** Even the dev tasks, re-run once more, gave only +1.8 against +5.0 over
the 3 dev repeats. The dev estimate was mostly selection plus run-to-run noise: two baseline runs disagree on 5–7% of
tasks (3–4 tasks), so a 3-task swing (about 5 points) is within the noise of one run. F27 is cost-neutral (22.4 vs 23.6 calls, −4% tokens) and harmless. It remains the
design the pre-registered rules chose, but there is no evidence that it helps. A side observation: F27 + F25 is
only −1.5 against the baseline at half the calls and tokens (11.8 vs 23.3 calls). This was not a pre-registered
test.

A follow-up stage on Android, using ideas from two scaffolds that report solving the benchmark, found a design that
does replicate on held-out tasks (+8.0 [+0.9, +15.8], 3 repeats); see A.9.

## A.6 Results: GameBoy

**Screening** (50 dev tasks; threshold 6.0 at 3 repeats, 5.25 at 4):

| Arm | Repeats | Success | Diff vs base [95% CI] | W/L | Steps/task | Calls/task | Decision |
|---|---|---|---|---|---|---|---|
| baseline | 5 | 34.8% | — | — | 115.9 | 227.9 | — |
| F07=3 upscale | 4 | 36.0% | +1.2 [−5.3, +8.0] | 8/7 | 117.5 | 231.0 | drop after retest |
| F32 loop warning | 3 | 34.7% | −0.1 [−5.6, +6.0] | 6/7 | 117.5 | 231.1 | drop |
| F31 no-op ban | 3 | 34.0% | −0.8 [−5.9, +3.7] | 7/6 | 119.0 | 241.7 | drop |
| F38 critic | 3 | 33.3% | −1.5 [−8.9, +6.4] | 6/8 | 123.1 | 393.7 | drop |
| F28 objectives log | 3 | 32.7% | −2.1 [−11.3, +6.5] | 7/11 | 120.9 | 238.5 | drop |
| F10 budget shown | 3 | 32.0% | −2.8 [−7.1, +1.2] | 3/6 | 119.1 | 234.2 | drop |
| F16 multi-action | 3 | 32.0% | −2.8 [−10.3, +4.4] | 8/11 | 117.4 | 180.2 | drop |
| F29 tips doc | 3 | 32.0% | −2.8 [−11.9, +6.4] | 6/7 | 113.8 | 224.8 | drop |
| F04 on-screen text | 4 | 31.0% | −3.8 [−8.5, +0.2] | 3/8 | 120.8 | 237.6 | drop after retest |
| F54 mode detection | 4 | 31.0% | −3.8 [−8.4, +0.4] | 3/9 | 119.2 | 234.5 | drop after retest |
| F37 vote (k = 3) | 3 | 30.7% | −4.1 [−9.6, +0.9] | 4/8 | 124.9 | 491.8 | drop (2.2× calls) |
| F19 structured output | 3 | 30.7% | −4.1 [−10.3, +1.9] | 3/8 | 122.5 | 241.5 | drop |
| F17 macros | 4 | 29.5% | −5.3 [−13.2, +1.8] | 5/8 | 122.6 | 198.2 | drop after retest |
| F26=15 longer history | 3 | 29.3% | −5.5 [−10.1, −1.5] | 2/8 | 122.2 | 240.5 | drop |
| F25 summarised history | 3 | 29.3% | −5.5 [−13.3, +1.5] | 4/8 | 121.5 | 119.5 | drop |
| F21 plan | 3 | 28.0% | −6.8 [−14.3, −0.0] | 4/14 | 126.7 | 249.8 | drop |
| F30 guide from priors | 2 | 22.0% | −12.8 [−21.2, −5.6] | 1/11 | 132.7 | 261.5 | dropped early (see A.8) |

None of the 17 features was promoted, so there was no combination stage. **The final GameBoy design is the
baseline executor.** On the full benchmark it scores 31.3% [22.2, 40.4] on 99 tasks: 38.0% on the dev half and
24.5% [12.2, 36.7] on the 49 held-out tasks. The held-out half is harder for the baseline itself (129 vs 116 steps
per task).

The dominant failures are loops and screen misreads. Neither loop handling (F31, F32, F38) nor richer screen inputs
(F04, F07, F54) moved them.

## A.7 Interaction with the repository's state-space features

- **Loop detection from a novelty archive.** We asked whether an in-episode novelty archive (image embedding
  similarity to earlier states of the same episode) detects loops better than the exact-frame-hash detector used by
  F32. Offline, on 150 GameBoy baseline episodes (27 loop failures, 73 other failures, 50 successes), it does not.
  - The archive flags every step the hash flags, plus a further 21–45% of steps (sprite and animation changes hide
    near-revisits from the hash).
  - Neither signal separates loop failures from other failures: AUC 0.50–0.58.
  - Over the first 40 steps, both are at chance even against successes. The higher whole-episode AUCs against
    successes (0.81–0.95) only reflect that failures run to the step budget.
  - Revisiting a screen is normal for this agent: F32's warning already fired on 35% of action calls.
  - The text half of the archive could not be tested offline.
- **On-screen text as executor input** (F04) did not help on GameBoy (−3.8 after 4 repeats).
- **Not pursued:** knowledge from exploration runs as executor context. The closest feature (F29, self-written
  tips) was borderline or negative on every benchmark: Web +2.4, Android +2.6, GameBoy −2.8.

## A.8 Methodological notes and things that did not work

- **Dev gains shrink on held-out tasks, and on small benchmarks they can vanish.** The Web design kept about 75% of
  its dev gain. On Android, with 57 dev tasks and a 6-point single-run floor, the selected feature's gain did not
  survive re-running. For small benchmarks, a held-out confirmation with several repeats is needed before claiming
  an improvement. We adopted that for the follow-up stage (A.9).
- **Literature priors transferred poorly.** The top-ranked GameBoy feature (on-screen text) and Android's
  second-ranked (an outcome-classifying reflector) did not help this model in this harness. The three Web features
  that survived all had weak prior evidence.
- **Overlapping gains.** Features that each help alone often add nothing together (Android F19/F21 over F27; Web F11
  vs F10). Greedy combination with a positive margin was necessary.
- **Reasoning add-ons rarely helped.** Plans, notes, structured templates, self-checks, critics, votes and task
  rewrites were flat or negative everywhere except Android notes at screening time. The Web gains all came from
  the harness: a budget counter, removing a trap action, and more visual history.
- **Cost-heavy features were never worth it.** Voting (2–3× calls) and critics (1.5–1.7× calls) gave no gain on any
  benchmark.
- **Added rule (flagged).** Midway through GameBoy screening we added an early-drop rule: drop after 2 of 3 repeats
  if diff ≤ −floor/√2 and McNemar p < 0.05, to free compute. It was applied once (F30, −12.8, p = 0.003). No arm
  above the threshold was affected.
- **Serving several arms from one inference server.** We checked that this does not change results: two packed
  baseline repeats scored 38% and 36% against 34/32/34% unpacked, within the floor.
- **Harness trap found by the failure taxonomy.** The Web `Google` action leads to a CAPTCHA that the agent cannot
  leave. Removing the action (F14=google) was among the largest single gains. An alternative fix (a `GoTo url`
  action, F12) was not completed.
- **Unfinished or untested.** Thinking mode (F24) was deferred. Several shortlisted items with high build cost (tile
  maps and navigation tools on GameBoy, F05/F06/F18) were not built.

## A.9 Follow-up: AndroidWorld with ideas from scaffolds that solve it

Two open-source Android agents report solving AndroidWorld: Minitap mobile-use (116/116) and Google's ARTEMIS
(above 99%, built on mobile-use). Both use frontier models, several model calls per step and, for ARTEMIS, no step
limit. Both numbers are self-reported, and neither repository ships an AndroidWorld harness. We kept the model, the
step budgets and the M3A harness fixed, and asked whether the *executor-side* ideas in these scaffolds transfer.

**What was taken and what was not.** Each scaffold was mapped component by component onto the catalogue:

| Scaffold component | Mapping |
|---|---|
| Minitap scratchpad; ARTEMIS notes | F27 (already tested, A.5) |
| Minitap Planner + Orchestrator; ARTEMIS plan ledger | **F59** subgoals: one planner call writes 3–5 subgoals; the action prompt shows them as a checklist the model ticks on observed evidence; finish only when all are ticked (also F21) |
| Minitap post-validation of typed text (their ablation: −15) | **F56** read-back: after typing, a deterministic note of what the focused field now contains |
| Minitap focus-and-clear typing | **F55** `clear_text` option on the typing action |
| Minitap typing never presses Enter | **F57** no automatic Enter after typing |
| Minitap Cortex rules (meta-cognitive −9, data fidelity −3) | **F58** rules: review history and never repeat a failed action, finish only on observed evidence and check formatting, copy data exactly, fill forms field by field |
| Minitap hybrid perception, resource-id-first targeting (−11) | **F60** resource ids added to the element list |
| Minitap Contextor | **F61** device context: foreground app and full device date and time in each action prompt |
| ARTEMIS operator's state check | **F62** expectation: the reply adds "Expect: \<the immediate screen change you expect\>"; the next prompt shows it and asks the model to compare it with the current screen |
| ARTEMIS list and long-text procedure | **F63** list scan: one direction with an anchor item, one-step recovery, the end proven by an end marker or a scroll that shows nothing new, no item handled twice |
| ARTEMIS controlled scroll | **F64** swipe: scrolls become a drag over 70%→30% of the container in 800 ms instead of AndroidWorld's centre-to-edge fling |
| ARTEMIS stuck hints; checker / outputter | F31, F32; F41 (already tested, A.5) |

Not transferred, because each changes a part we hold fixed rather than the executor: the frontier models and the
grounding model; an unlimited step budget; an external OCR model; ARTEMIS's shell access and its checker's direct
device-database probes (these bypass the UI and read the very state AndroidWorld's evaluator checks); a
coordinate-drift safety net (index targeting has no drift); and bursts of several actions per step (which change
what the step budget means).

**Rules** (written before any run): screening, combination and leave-one-out as in A.2, threshold 4.4 at 2
repeats. Prior ranks for the retest clause follow Minitap's own ablation (F59, F56, F60 top half; F58 next; the rest,
including the ARTEMIS features, at the bottom). These ranks were fixed mid-stage, after the first retest decision;
that decision was consistent with them. In parallel, the **bundle** of all Minitap features (F27 + F55–F61) was run
against the baseline. The final candidate is the better of the greedy design and the bundle on dev. Because a
single repeat could not resolve a dev-sized effect on Android (A.5), the **confirmation uses 3 repeats** of both
the baseline and the candidate on the 58 held-out tasks. Improvement counts as confirmed only if the pooled paired
difference has a bootstrap 95% CI excluding 0.

**Smoke tests caught two bugs.** Reading the smoke records showed that (1) prompt-edit features stacked on one
executor applied only the first edit, because they shared one overridden model call and method resolution ran only
one of them, and (2) the device date was cut to the weekday by shell word-splitting. Both were fixed before
screening. A per-class check of the prompt had passed in both cases; only reading live records caught them. F64's
swipe could not be seen in the smoke (its few scrolls were on short lists), so it was instrumented to log every
controlled swipe; the screening runs logged 53 swipes and no fallbacks.

**Screening** (57 dev tasks, baseline 60.5% over 3 repeats):

| Arm | Repeats | Success | Diff vs base [95% CI] | W/L | Calls vs base | Decision |
|---|---|---|---|---|---|---|
| Bundle F27 + F55–F61 | 3 | 67.8% | +7.3 [−0.6, +15.2] | 11/4 | ×1.02 | final candidate (higher on dev) |
| F61 device context | 3 | 66.1% | +5.6 [+0.6, +11.1] | 9/2 | ×0.99 | **promote**; the greedy design |
| F62 expectation | 2 | 64.0% | +3.5 [−2.6, +10.8] | 8/5 | ×1.01 | drop |
| F63 list scan | 2 | 64.0% | +3.5 [−2.0, +9.4] | 6/3 | ×1.03 | drop |
| F59 subgoals | 3 | 63.7% | +3.2 [−1.8, +8.8] | 6/3 | ×1.06 | drop after retest |
| F57 no automatic Enter | 2 | 63.2% | +2.6 [−2.6, +8.2] | 5/3 | ×1.01 | drop |
| F60 resource ids | 3 | 62.9% | +2.3 [−3.8, +8.5] | 8/5 | ×1.01 | drop after retest |
| F56 read-back | 3 | 61.7% | +1.2 [−3.2, +6.1] | 5/5 | ×1.02 | drop after retest |
| F58 rules | 2 | 61.4% | +0.9 [−2.3, +5.3] | 2/4 | ×0.96 | drop |
| F64 controlled swipe | 2 | 61.4% | +0.9 [−5.8, +7.9] | 7/6 | ×0.95 | drop |
| F55 clear_text | 2 | 59.6% | −0.9 [−6.7, +5.3] | 4/6 | ×1.00 | drop |

**Combination and choice of candidate.** Only F61 was promoted, so the greedy design is F61 alone (its 3-repeat
gain, +5.6, is above the +2.2 keep margin, and its leave-one-out is the same comparison). The bundle scored higher on
dev (67.8% vs 66.1%), so by the pre-registered rule **the bundle is the final candidate**. F61 alone was not run on
held-out tasks.

**Held-out confirmation** (58 tasks, 3 repeats each):

| Arm | Success | Per repeat | Diff [95% CI] | W/L | p | Calls/task |
|---|---|---|---|---|---|---|
| Baseline | 60.3% | 62.9 / 58.6 / 59.5 | — | — | — | 21.2 |
| Bundle F27 + F55–F61 | 68.4% | 65.8 / 67.2 / 71.6 | **+8.0 [+0.9, +15.8]** | 11/3 | 0.006 | 22.6 |

**The improvement is confirmed.** The held-out gain (+8.0) matches the dev gain (+7.3), unlike the single notes
feature in A.5, which went from +5.0 on dev to +0.0 on held-out tasks. The cost is small: 7% more model calls (one planning
call per task) and 13% more input tokens. One held-out episode could not be scored, because
AndroidWorld's own SMS check reported an emulator failure; that task is averaged over its other two repeats. Counting
the episode as a failure gives +7.5 [+0.3, +15.2], so the conclusion does not depend on it. Two caveats apply. The
lower bound is close to zero. And the run does not show which components carry the gain: no single Minitap feature
except device context passed screening, so the gain may come from several small effects adding up. Two of the 11
held-out wins are date-relative tasks, which fits the device-context feature, but that is an observation, not a
test. **The final Android design therefore changes from notes (F27) to the Minitap bundle (F27 + F55–F61).**

**What this says about the scaffolds.** Neither the ARTEMIS mechanisms (expectations, list scanning, controlled
swipes) nor most Minitap components helped alone with this model. What did help is cheap context the agent otherwise
lacks (the device date and foreground app) and, together, a set of small harness and prompt changes. The large
reported gains of these scaffolds are not reproduced at this scale. They rest on parts we held fixed: frontier
models, several calls per step, an unlimited step budget, and device access that bypasses the UI.

**Transfer to GameBoy.** We also asked whether any mechanism these scaffolds rely on is untested on GameBoy. Notes,
plans, stuck hints, critics and richer perception all map onto features already screened there (A.6), all flat or
negative. The device-, typing- and list-specific features have no counterpart in a button-press game. The one
untested mechanism was the expectation line (F62), which targets both main GameBoy failure buckets (loops and screen
misreads) at no extra model call. A smoke run confirmed that every deciding call wrote an expectation and that the
model compared it on the next step. Over 4 repeats on the 50 dev tasks it scored 36.0% against 34.8%: +1.2
[−4.8, +6.9], 11/8, p = 0.70, below the 5.3 threshold. It was dropped, and GameBoy stays at the baseline executor
(18 features tried, none promoted). While this run was in progress, the table at 35 of 50 tasks read +7.4. The
tasks still running were the long ones, which mostly fail, so the partial number was inflated. Decisions are taken
only on complete runs.

**Other things that went wrong.** The inference server's port was derived from the job identifier and, for one job,
landed on a port the Android emulators use. That run failed before any task, and two queued jobs that would have
collided were resubmitted unchanged. No result is affected.
