# Oracle preview executor: versions tried (2026-11-13 / 14)

`cusi/world_action/oracle_preview.py` (spec `--world_action oracle_preview`) wraps an env's base executor:
the base executor proposes an action with its own prompt and memory, the env's exact lookahead (`env.peek`) shows
what it would do, and a check call on (screen now, screen after) keeps it or gives the executor feedback to decide
again (at most 3 proposals; the last one is executed). It reads the env, so it is a diagnostic upper bound, not a
learned world model. Running-log detail for every entry below is in `plans/decisions.md` (same dates).

Model throughout: google/gemma-4-31b-it (executor, check and judge), greedy decoding.

## Final design (committed code)

One model for all envs, check mode only:

1. `policy()`: the base executor's decision. Finishes and invalid replies are returned as is (a finish is never
   previewed: its peek would leak the reward).
2. Peek the action. If the env refuses it, it is rejected without a check. It is also rejected if the screen does
   not change (except in a GameBoy dialogue, menu or battle, where the check is told that repeated A presses are
   normal).
3. Otherwise a check call (`PREVIEW_CHECK`, v4 rubric) runs. It says bad only if the after screen shows one of:
   - a wrong target;
   - an error or refusal;
   - an unrelated place;
   - destroyed progress.

   It never says bad for a guess about later steps or for a different plan.
4. On a rejection, `policy(feedback=...)` gets the previewed outcome and the reason, through the executor's own
   channel:
   - GameBoy: the error block;
   - WebVoyager: the next chat message;
   - M3A: a history line.
5. If the executor re-proposes an action the check rejected, it runs ("insisted"). It doesn't run if the action was
   refused, changed nothing, or was judged by a battle rollout.

Per-env state the oracle reads (never the reward, success or task trackers):

| Env | State used |
|---|---|
| GameBoy (Red) | Emulator peeks; `peek_rollout`, which shows a battle choice's whole turn up to the next choice (press, then B); the controller's "unavailable"; screen-parsed agent state and battle menus; the buttons that would change the screen, named in the no-change feedback. No RAM. |
| Web | A sandbox browser synced to the live state by episode replay and reused between peeks. The check sees the URL / title before and after, the previewed page's text (2500 chars) and today's date. |
| Android | Emulator snapshots. The check sees the app activity before and after. A real long press right after a peek is held 3 s, as in the peek. |

## Results (final)

| Env | Tasks | Preview | Base (same tasks) |
|---|---|---|---|
| GameBoy Red | 50 | v5: **28/50** | 23, 25 (new emulator); 15-22 older reps |
| Web dev3 (60 non-Wolfram) | 60 | v10 check: **36/60** and **27/60**; v11: **34/60** | 28, 26, 27 |
| Web dev3 pooled (all runs, incl. interim tasks) | — | check: 158/295 (54%) | 46% |
| Android dev2 | 20-23 scored | v2: 9/23; v4: **9/20** | 7/23; 8/20 |

- **GameBoy:** v7 is the final code. v5 with battle rollouts and no insisting past a rollout scored 13/20 and 16/19
  over two runs, against base 12/14 on the same tasks.
- **Web:** gains come from tasks base never solves (Coursera--40 5/6, GitHub--19 5/6, Huggingface--38 4/5, GitHub--22
  2/3, Google Map--28 3/5; base 0/3 each). Losses are on tasks base always solves (ESPN--20 2/6, Google Map--5 1/3,
  Coursera--4 / GitHub--9 / GitHub--11 4/6) and Booking--2 (1/6 vs 2/3). Live-site noise per run is large: complete check runs scored 36, 34
  and 27 (base mean 27), and one partial run (32 tasks) came in 1.7 tasks below base.
- **Android:** 6 of 26 tasks were lost to emulator hangs on snapshot loads.

## Versions

### Shared (all envs)

| Version | Change | Why | Outcome |
|---|---|---|---|
| (before) `oracle_lookahead` | The VLM picks among 7 next GameBoy frames. It has no history. | — | 10/27 Red; v5 got 19 on the same tasks. Superseded. |
| v1 | Propose, peek, check, revise. | — | Bugs: the feedback recursed into the world action `decide` (5 proposals, 5 identical checks); `_policy` name clash on GameBoy. Fixed with the `Executor.feedback()` context manager and `_base_policy`. |
| v2 | The check judges the action as one step of the agent's plan. | Android ClockTimerEntry: the right-filling digit pad showed "01m 63s" and was called bad. | Kept. |
| v3 | GameBoy dialogue, menu and battle screens go to the check, with a note, instead of being auto-rejected as no-change. A re-proposed rejected action is kept ("insist"). | v2 stalled in battles (143 rejected A presses in Defeat Blue). | Kept. |
| v4 | Rubric: bad only for a visible mistake (wrong target, error, unrelated place, destroyed progress). Never for guesses about later behaviour. | The ClockTimerEntry check still steered the agent to a wrong theory. | Kept. This is the final rubric. |
| v5 | Insisting overrides only a check verdict. A refused or no-change action stays rejected. The GameBoy no-change feedback names the buttons that would change the screen. | Long free-roam tasks: blocked UP / A re-proposed and executed, 75-103 no-change rejections per task. | Kept. GameBoy 28/50. |

### GameBoy

| Version | Change | Outcome |
|---|---|---|
| v6 | Battle A previewed by `peek_rollout` (press, then B until the base menu or the end of the battle). | Lost 013 / 012: insisting got past a correct rollout rejection, and text-only A presses were judged by an outcome already decided. |
| v7 | Rollouts only for an A that commits a choice (move list, bag, Pokemon menu). A rollout-backed rejection can't be insisted past. | 13/20 and 16/19 vs base 12/14. **Final.** |
| v8 | The no-change feedback adds walk distances per direction, from RAM map / x / y. | 11 vs v7's 10 and 11 on 13 shared tasks: no gain. **Removed** (no RAM). |

### Android

| Version | Change | Outcome |
|---|---|---|
| v2 | (shared v2) | 9/23 vs base 7/23. Lost MarkorDeleteNote: the peek long-pressed for 3 s, but the real press after a snapshot load was 1 s and registered as a tap. |
| v4 | (shared v4) plus a 3 s real long press right after a peek. | 9/20 vs 8/20, no losses. **Final.** |

### Web

| Version | Change | Outcome |
|---|---|---|
| v2 | (shared v2) | dev2 non-Wolfram: 18/26 vs base 13 and 16. Wolfram Alpha blocked the node (each peek replayed the episode in a fresh Chrome). |
| v4 | (shared v4) | 13/26. Same intervention rate as v2, so v2 vs v4 is noise. |
| v5 | The sandbox browser is reused (cookies and origin storage cleared) instead of starting a new Chrome per sync. | 15/26. Previews 25 s vs 99 s. Wolfram still blocks the node after a while, so later dev sets exclude Wolfram. |
| v6 | Select mode: the base decision plus alternatives, all previewed, and one call picks. | No alternatives: eval decoding is greedy. |
| v7 | Alternatives sampled at temperature 1. | Same action reworded 3/3 times. Also fixed: the step owner for fallbacks. |
| v8 | Alternatives asked for explicitly ("a different action, other than X"). | 3 distinct candidates, but the base candidate (always image 2) was always picked. |
| v9 | Candidates shown in hash order. | 8/11 vs base 5 and 8 (run cut at 17/26 by the host's time limit). |
| v10 | The previewed page's whole text (innerText) goes to the check and the select call. New 60-task dev3. Check mode and select mode both run. | Check: 36/60 and 27/60 (two complete runs); 4 runs pooled 102/194 (53%) vs base 46%. Select: 3 runs pooled 56/107 (52%) vs 46% at ~3x the previews. **Select mode dropped.** |
| v11 | Today's date in the check. | GitHub--9: a search filtered on the wrong year was approved. 34/60 (one complete run); 2 runs pooled 56/101 (55%) vs 45%. **Final.** |
| v12 (not evaluated) | An unsynced preview is not judged. | 18% of previews were unsynced and 4% failed (Booking 40%). The check rejected 43% of unsynced previews vs 22% of synced ones, mostly for "progress destroyed" that only the sandbox had lost (dates or filled fields cleared, a menu closed). Runs cancelled at wrap-up, so **not in the code**. This is the clearest next step, along with raising the sync rate. |

## Lessons

- **Web noise:** live sites make single runs noisy (base 30 / 14 / 16 historically on dev2). Use 60+ tasks and
  several reps, and compare on the same tasks.
- **Feedback channel:** feedback through the executor's own memory works. A separate picker without history (old
  `oracle_lookahead`) does not.
- **Strict checks:** checks that are too strict hurt more than they help. The working rubric only flags mistakes
  visible on the after screen.
- **Exact vs inexact previews:** GameBoy peeks are exact and instant. Web and Android previews are inexact (sync,
  emulator hangs), and an inexact preview judged as exact causes false rejections.
- **Live-site load:** previewing on live sites loads them, and Wolfram Alpha blocks a node after a few hundred
  peeks.
