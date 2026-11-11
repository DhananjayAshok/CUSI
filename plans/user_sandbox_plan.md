# The User's sandbox: env server, client CLI, isolated agent repo

Goal: a coding agent (the User in "The Everything Model and Its User") develops a repo-as-policy on the train and val
tasks, through the environments and the Everything Model, without ever being able to see the test tasks, the test
environments, CUSI, or our results. At test time a harness on our side runs that repo on the test tasks. The agent
never gets state at test time, but we record it for ourselves.

Split: `storage/tmp/split_final.json` (2026-11-11) until it moves into the repo (step 1).

## Architecture

```
 ┌──────────────── sandbox (Apptainer, --containall) ───────────────┐
 │  agent workspace (its own git repo)                              │
 │    envctl / envclient.py   tasks.json (train+val only)           │
 │    guides/ tools/ prompts/                                       │
 │  coding agent ──bash──▶ envctl ─────HTTP + token───────┐         │
 │                ─────────────────── HTTP ───────────────│──┐      │
 └────────────────────────────────────────────────────────│──│──────┘
                                                          ▼  ▼
   CUSI (outside the sandbox)                 env server      Everything Model (vLLM)
     cusi/env_server/ ── TextActionEnv (GameBoy / Web / Android)
     recorder ──▶ storage/env_server/<instance>/<session>/   (full obs + true state, never served)
```

- Env server (in CUSI): owns the split, builds envs, serves `reset/step/save/load/close`. One server instance =
  one fixed policy (which tasks are visible, whether state is returned). The agent cannot change the policy: it is a
  launch flag, not a request field.
  - `dev` instance: train + val tasks, free play in train envs, state allowed, success signal returned.
  - `test` instance: test tasks only, started only by our harness, obs + success, never state.
- Client: one Python file plus a CLI (`envctl`), the only CUSI-written code inside the sandbox. It knows nothing
  about tasks beyond what the server lists.
- Sandbox: an Apptainer image with Python, the coding agent's CLI and the client; mounts only the workspace; the
  host home, CUSI, storage, `../GameBoyRL` and HF caches are not mounted.

## Server API (HTTP + JSON, one bearer token per instance)

| Call | Returns |
|---|---|
| `GET /tasks` | the visible tasks: id, env, goal text (dev: train+val with their split label; test: nothing listed, the harness passes ids) |
| `POST /sessions {task_id \| env+"free", state: bool}` | session id; refused if the task or env is not visible, or `state` is not allowed |
| `POST /sessions/<id>/reset` | obs |
| `POST /sessions/<id>/step {action}` | obs, valid, error, done, steps left, `success`; `state` only if the session asked and the instance allows (never on test) |
| `POST /sessions/<id>/save`, `/load {handle}` | an opaque state handle; restoring it |
| `DELETE /sessions/<id>` | closes it |

- obs = frame (PNG), `texts`, `actions`, `goal`, `elements` (Web/Android) — exactly what `TextActionEnv` already returns.
- state = `true_state()` (GameBoy RAM summary; Android activity and fragments; Web URL, title, skeleton), plus for
  GameBoy optionally the raw emulator save state. Android/Web state is only restorable through a handle.
- Task ids are opaque to the agent beyond the train/val list (e.g. `gb-red-004`); test ids never leave our side.
- Transport: stdlib `ThreadingHTTPServer`, one env per session, a per-env session cap (Android emulators, browsers).

## State at test time: tracked by us, never served

The server records every session server-side, whatever the session asked for:

- `storage/env_server/<instance>/<session>/steps.jsonl`: action, validity, obs texts, `true_state`, success, timing,
  per step; frames as PNG; GameBoy emulator snapshots every N steps and at the end.
- Written in the `write_episode` layout so coverage, task-coverage and eval tooling read it unchanged.
- The test instance runs with `--state never`: `true_state` is computed and recorded but stripped from every
  response. Success is returned (as on dev) and recorded.
- Check (step 6): a test-instance response never contains `state` or `true`, asserted in a live test.

So the diagnostic stream is a server-side recorder, decoupled from what the client sees. Same mechanism on dev, where
it also gives us the agent's full exploration history without asking the agent to log anything.

## Leaks that a task list alone does not close

- Android: every app is installed on the one emulator, so free play in train apps can open a held-out app (OsmAnd,
  VLC, ...). The server checks the foreground package after each step (it is in `true_state`); in a dev session a
  held-out app is an invalid step, pressed back and reported as "app not available".
- Web: train sessions may navigate anywhere a link goes. The server blocks the held-out sites' domains (Google Maps,
  Google Flights, Wolfram Alpha) in dev sessions, the same way.
- GameBoy: separate ROMs; nothing to do.
- The goal text of a test task is only ever sent inside a test session's obs, which only the harness opens.
- The Everything Model: must not have been trained on test trajectories (holds for the base model; any model we
  fine-tune for the User must use train data only).
- Our results, decisions, reports: not mounted.

## At test time (cusi/env_server/harness.py, run_user_eval.py)

Evaluation is a schedule: an ordered list of test tasks (--order given | shuffle | group_env, seeded), run one at a
time. For each task the harness opens a session on a test server (the User does not choose it), runs the User in the
sandbox with only the environment essentials (ENVCTL_ENV, ENVCTL_SESSION, ENV_SERVER_<ENV>, ENV_SERVER_TOKEN: that
session's token), scores from the server, and commits the workspace.

- --mode fixed_repo: the workspace's policy.json entrypoint runs (RHO / ASPIRE style: code written in development).
- --mode live_agent: prime-agent runs autonomously with a prompt (cusi/env_server/prompts/live_agent.md), its stop gate
  "the session is done", and the agent home's memory (~/.prime, ~/.config; the install is shared read-only).
- --memory carry: one workspace (and live_agent: one agent memory) through the schedule, so what task k wrote is there
  at task k+1 (the sequential evaluation modes of "The Everything Model and Its User"); isolated: every task from the
  same snapshot. Nothing is discarded: commits per task; a resumed carry schedule rewinds the workspace to the last
  finished task.
- Budgets, both modes: the task's step budget (server) and --wall_clock_seconds per task (the harness ends the
  process). No model is provided: the User brings its own (the sandbox has the internet).

## Steps

1. Split file: `configs/splits/benchmark_splits.json` from `storage/tmp/split_final.json`, plus a loader that
   returns task lists per (env, split) and checks each entry against the live benchmark (task text, start state).
   Check: loader round-trips every task; a reworded task fails loudly.
2. `cusi/env_server/`: sessions over the existing env classes (GameBoy first), visibility and state policy,
   recorder. `run_env_server.py` click command. Check: live GameBoy session on a train task: reset, 20 steps,
   save/load, close; recorded episode reads back with the existing episode reader.
3. Client + `envctl` (`reset`, `step`, `save`, `load`, `tasks`, `close`; writes `obs/<step>.png` and `.json` into
   the workspace). Check: drive a Red train task by hand from the CLI.
4. Web and Android sessions (server inside the CUSI container via `scripts/container.sh`; Android with the emulator
   started, as `env_run.sh` does), the held-out app/domain blocks. Check: live session per env; navigating to a
   held-out app/site is refused.
5. Sandbox image + `slurm/user_dev.sh` / `scripts/slurm/user_dev.sh` pair: starts the dev server and the vLLM
   endpoint, builds the workspace from a template in `user_template/`, launches the coding agent in the sandbox.
   Check from inside the sandbox: `ls` of `/`, `$HOME`, the CUSI path, `../GameBoyRL` shows nothing of ours;
   `envctl tasks` lists train+val only; a test id is refused.
6. Test harness: `run_user_eval.py` (test instance, `--state never`), the agent run per task, scoring from the
   server. Check: a stand-in agent (random actions) runs on 3 test tasks; responses carry no state or
   state; the recorder has full state.
7. A real dev run: the coding agent on Red train+val for a fixed budget, then the harness on Red test.

## Decisions (2026-11-11, user)

- Success is returned on train, val and test sessions. State is never returned on test; it is recorded server-side.
- The sandbox has open outbound internet (API models, Web tasks). The boundary is the filesystem: nothing of CUSI,
  storage, GameBoyRL or the user's home is mounted.
- Free play (no goal) in train environments is allowed on dev.
- Val is "seen": listed on dev with train, success returned.

## Open questions

- Which coding agent runs in the sandbox (Claude Code, an open model through a CLI)?

## Status (2026-11-11)

Steps 1-6 built (not committed): split loader, env server (all three envs), client + envctl, sandbox, launcher
(slurm/user_env.sh), test harness (run_user_eval.py). Recording is an option (`--record_dir`), off by default.
Decisions taken during the build and open issues: plans/user_sandbox_review.md.
