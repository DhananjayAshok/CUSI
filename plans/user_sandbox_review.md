# User sandbox: decisions and issues to review (2026-11-11)

Built per plans/user_sandbox_plan.md. Code: cusi/env_server/ (codec, policy, sessions, recorder, server, sandbox,
remote_env, check), run_env_server.py, run_user_eval.py, user_template/ (envclient.py, envctl, README.md,
guides/ tools/ prompts/), slurm/user_env.sh + scripts/slurm/user_env.sh, cusi/eval/splits.py. Not committed.

## Design decisions made along the way

1. **One server process per env kind.** GameBoyRL's and WebVoyager's `utils` modules cannot share a process, and Web
   and Android must run inside the CUSI container anyway. The client takes a {env: url} map.
2. **Two token levels.** The server token (dev: `storage/env_server/token`, test: `storage/env_server/test_token`,
   separate files) opens sessions; each session also gets its own token that reaches only that session's endpoints.
   The test harness gives the policy only the session token, so at test time the policy cannot list tasks, open
   another test task (and read its goal) or touch other sessions. Checked live.
3. **Test servers: no save/load.** Read "no state at test time" as covering save/load too (restoring a snapshot is a
   way to search with the env's state). Dev allows it.
4. **What a client never gets:** `info["true"]` (true state; only via `state`, dev only), AndroidWorld's
   `probe_success`, GameBoyWorlds' `subgoals` (the benchmark tracker's progress: grader output), and GameBoyRL's
   `transition_states` (per-transition records holding RAM-derived state: memory signals, the test tracker, team
   info). Everything else in obs and info is sent as the env returns it (Android's raw screenshot, Web's URL, GameBoy's
   emulator frames, OCR regions).
5. **Held-out websites/apps on dev (revised 2026-11-11, user's call): undo and invalid.** Every dev session (task or
   free play) builds its env with block-lists (`blocked_packages` on AndroidPlayEnv, `blocked` scopes on
   WebVoyagerPlayEnv, both usable in any mode). A step that lands in a held-out app or on a held-out site is undone
   inside `step()` before anything is observed (Android: Back, then Home if still there, or the scene's app in free
   play; Web: browser Back, else reload the previous URL) and is invalid with "That action is not available here, so
   it was undone. No action was performed." The app/URL it would have reached is kept in `info["blocked"]`, which
   is never sent. The server's old end-the-session check remains only as a safety net for a failed undo. Checked
   live (jobs 294747, 294749): Android open_app VLC from a train task (back to the home screen) and from Clock free play
   (back in Clock); Web a click on a Wolfram Alpha link from an Apple task (stays on apple.com/store).
6. **Held-out lists are constants** in cusi/env_server/policy.py (apps: osmand, chrome, expense, vlc, camera,
   contacts, draw, gallery, clipper; sites: Wolfram Alpha, Google Flights, Google Maps), not derived from the split
   file (the file has no app/site field by design). Keep them in sync if the split changes.
7. **GameBoy free play** starts from the start states of the train games' train and val tasks (55 scenes for
   Red + Link's Awakening). Test tasks' start states never appear (the split keeps a start state in one split).
8. **Web success = WebVoyager's auto_eval judge**, run by the Web server when an episode ends with ANSWER, on the answer
   and the last 15 screenshots (the eval's setting), with `--judge_model` (the E2E uses the agent's own Gemma). No
   judge configured: success is null. A truncated episode scores 0. The judge reads the labelled screenshots the
   agent saw (as run.py saves them).
9. **Android:** one emulator per server, so one session at a time (`--max_sessions 1`); sessions reuse one
   AndroidWorld connection as the eval does. Suite seed 30, instance 0, as the eval.
10. **Step budgets:** GameBoy 175 (eval), Web 15 (eval), Android 10 x task complexity (AndroidWorld). Free play has
    none. GameBoy's `load` resets the step counter (dev only).
11. **Idle sessions close after 30 minutes**; sessions cap per server (default 4; Web 2, Android 1 in the launcher).
12. **Recording is off by default** (`--record_dir` turns it on): see the discussion below.
13. **Sandbox image:** `docker://python:3.12-slim` pulled to `storage/containers/sandbox.sif`; run with
    `--containall --cleanenv`, only the workspace bound at /work, host network. Checked: CUSI, storage, GameBoyRL
    absent; $HOME is an empty stand-in; the internet and the servers reachable.
14. **The client is standard library only** and runs on Python 3.6 (the host's system Python) through 3.12.
15. **RemoteEnv** (CUSI side) lets our own agents run through a server: enums and classes cross as
    `module:name` and are rebuilt; GameBoy's `transition_states` arrive empty (the agent's history policy does not use
    them); WebVoyager's PDF answer cannot read the downloaded PDF (it stays on the server) and falls back to the text
    note.
16. **Two small edits outside the new code:** `cusi/agents/executors/m3a.py` `device_context` asks the server when the
    env is remote (else M3A's prompt would say "unknown" for app and time); `cusi/utils/paths.py` gained
    `benchmark_splits_file`.

## Issues found

- The GameBoyWorlds data for Link's Awakening, Crystal and Oracle of Seasons had not been pulled since the 37b80e3
  update (fixed today).
- Encoding GameBoy info recursed forever on `transition_states` (self-referencing); now dropped, and the encoder has
  a depth limit.

## Recording (for discussion)

Off by default. With `--record_dir`, every reset/step of every session writes a JSON line (action, the full info
including the true state, reward, done, success) plus the obs frame as PNG. Rough size: GameBoy ~10 KB/step,
Web/Android ~0.5-1 MB/step (screenshots). Options to rationalize:
(a) record only test servers (the diagnostic you asked for), (b) record state and actions but no frames,
(c) frames only every N steps, (d) per-session opt-in from the harness.

## End-to-end check (2026-11-11, jobs 294745 dev / 294746 test, Gemma-4-31B on vLLM, all three envs)

CUSI's final agent per env through the servers (cusi/env_server/check.py; GameBoy capped at 60 agent steps):

| server | task | result |
|---|---|---|
| dev | gb.pokemon_red.004 "Read the sign below you" | success 1, 15 steps |
| dev | gb.legend_of_zelda_links_awakening.002 "pick up the sword from the shore" | unsolved at the 60-step cap |
| dev | web.Apple--7, web.ArXiv--0 | success 1 (server judge), 14 and 3 steps |
| dev | aw.ClockStopWatchPausedVerify, aw.TasksDueOnDate | success 1, 3 steps each |
| test | gb.pokemon_red.025 "Use the Super Rod properly" | unsolved at the 60-step cap |
| test | web.Apple--6 | success 1 (server judge), 3 steps |
| test | aw.ClockStopWatchRunning | success 1, 4 steps |

No response carried state or `info["true"]` (state was not requested; on test it cannot be). Sandbox probes (dev):
CUSI, storage and GameBoyRL absent; listing shows train + val only (GameBoy 55 + 15, Web 180 + 45, Android 48 + 13);
a test id and a held-out site scene refused; Web and Android free play work; an Android task session that opens VLC
ends with held_out_reached; Android free play leaving Clock is undone. Harness (test, stand-in random policy, one task
per env incl. a held-out env each): ran in the sandbox, scored, and the policy's session token could not list tasks,
open sessions, save, or see state. No record directory was written (recording off by default).

Smaller observations: Web's server step count is one below the agent's (the agent counts its finishing step);
AndroidWorld answer tasks (TasksDueOnDate) score through the env as in the eval.

## prime-agent in the sandbox (2026-11-11)

- Sandbox image switched to `docker://python:3.12` (Debian 13, glibc 2.41, curl, git): prime-agent's installer
  needs curl and glibc >= 2.35; the slim image has no curl.
- The sandbox now mounts a persistent agent home, `storage/env_server/agent_home`, as `$HOME` (/home/agent):
  prime-agent 0.10.2 is installed there by its official installer (run inside the sandbox), and its login and
  sessions (~/.prime/agent) live there. Nothing of CUSI or the user's home is in it.
- `python -m cusi.env_server.sandbox --workspace <dir> [--servers ... --token_file ...] -- prime-agent` opens the
  sandbox interactively (a new workspace is made from user_template and git-initialised).
- The test harness does NOT mount the agent home: at test time the policy runs without prime-agent or its login.
  Decide whether the policy may call a coding agent at test time.
- Login is still to do (interactive, by the user).

## State (revised 2026-11-11)

`state` (dev, on request) is now the env's `full_state()`, not `true_state()`. GameBoy: the true_state summary
(Pokemon RAM: map, x, y, battle, party, badges, Pokedex, items, money, new event flags) plus GameBoyWorlds' memory
snapshot (`memory`: player, location with x/y/facing, party, bag/items, PC, Pokedex, battle, UI, map objects, warps,
signs, events raised...; for Zelda: player, location with room and x/y, inventory, UI) and `agent_state` (free roam,
dialogue, menu, battle). Size per step: Red ~6 KB, Crystal ~48 KB, Zelda ~0.7 KB; reading it takes ~1 ms. Zelda had
no state before; it now has the memory snapshot. `true_state()` is unchanged, because exploration records it on every
step. Android and Web: full_state = true_state (foreground activity and fragments; URL, title, page skeleton, new JS
functions). The recorder writes full_state per step as well.

## run_task.sh removed (2026-11-11, user's call)

No fixed entry point: at test time the coding agent itself acts in a session the harness opened. run_user_eval.py
takes the command to run in the sandbox (--agent_cmd) until the agent mode (prime-agent with step / time / turn /
token budgets) is built; user_template/run_task.sh and the README's contract are gone.

## Harness with modes, memory and schedules (2026-11-11)

Built per the user's revisions: fixed_repo (policy.json entrypoint) and live_agent (prime-agent) over a schedule,
--memory carry | isolated in both modes, budgets = step budget + wall clock per task in both, only env essentials
passed (no model endpoint). Decisions: the agent home's install (.local, .cache) is bound read-only and only its memory
(.prime, .config) is copied/carried; the carried agent home is not committed (its store holds a Python environment);
a resumed carry schedule rewinds the workspace to the last finished task's commit (an interrupted task's writes are
dropped); envctl use now writes the first observation. Local test (GameBoy test server): carry 0/1/2 notes, isolated
0/0/0, resume sees 2 notes, a different schedule into the same out is refused, a never-ending policy stopped at 25 s
(exit 124, session closed), live_agent launches prime-agent (stops: not logged in).

## Full test (2026-11-11, jobs 294927 dev / 294928 test; Qwen3-VL-8B in fp16 on Quadro RTX 8000, plumbing only)

All passed. Agents through the servers: Android 2/2 solved, Web 1/2 (judge), GameBoy 0/2 at a 40-step cap; no
response carried state or info.true. Sandbox (python:3.12 image + agent home): CUSI, storage, GameBoyRL and ~/.ssh
absent; $HOME holds only the agent home; prime-agent 0.10.2 runs; internet works. Dev over HTTP: train/val listing
only, test task and held-out game refused, Zelda `state` = agent_state + memory (room, x, y), no subgoals / true /
transition records in info, save/load, Android open VLC in a task session undone + invalid with the neutral message
and the blocked app not sent, Web free play. Test: nothing listed, state refused (403). Harness: fixed_repo carry over
6 tasks in 3 envs (group_env) with notes 0..5 and a commit per task; isolated 0/0; only the 4 env variables reach the
policy; session token cannot list or save; a never-ending Web policy stopped at the 30 s limit (exit 124; 42 s
wall time with session setup); live_agent launches prime-agent with the agent home (stops: not logged in).
Recording: dev 69 rows / 6 sessions, test 82 rows / 14 sessions, every row with full_state and a frame (test records
hold the state its responses never carried). Fixed after: log/run folder names now replace spaces in task ids.
Not testable yet: live_agent with a logged-in prime-agent.
