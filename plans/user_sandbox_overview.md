# Environment server / client separation: overview (for porting to another domain)

Goal: a coding agent ("the User") develops a policy, as a repository, on train and val tasks. It must never see test
tasks, test environments, the benchmark code or our results. Later we run its repository on the test tasks ourselves.

## Pieces

- **Split file** (`configs/splits/benchmark_splits.json`): one flat map `{task_id: {"split", "text"}}`. Splits are
  `train`, `val`, `test` (held-out tasks in the training environments) and `test_env` (every task of a held-out
  environment: other games, websites, apps). Task ids are `<benchmark prefix>.<the benchmark's own key>`
  (`gb.pokemon_red.004`, `web.Apple--0`, `aw.MarkorCreateNote`). A small resolver per benchmark turns an id into what
  builds that benchmark's environment; nothing else in the system knows benchmark wiring.
- **Env server** (one process per environment kind, on our side): owns the split and the environments, and serves
  them over HTTP + JSON. Its policy is fixed at launch, never by a request:
  - `dev`: lists and opens train + val tasks; free play (no goal) from start points in the training environments;
    state on request; save/load of snapshots.
  - `test`: opens test + test_env tasks (ids come only from our harness; it lists nothing); no state; no save/load.
- **Client**: one standard-library Python file plus a CLI (`envctl`), the only code of ours the agent gets. Each
  call writes the observation image and a JSON of the rest into the agent's workspace.
- **Sandbox**: an Apptainer container that mounts only the agent's workspace (its own git repo) and a separate
  agent home (for its coding-agent install and login). Our repository, data, results and home are not mounted. It has
  normal internet access (API models, web tasks).
- **Test harness** (ours): for each test task it opens a session on a test server, starts the coding agent in a
  fresh sandbox copy of its workspace with only that session's token, lets it act until the episode ends or its
  budget runs out, then scores from the server.

## Sessions

A session is one live environment instance, opened for one task (with its goal and step budget) or one free-play
start point (no goal, no budget). It holds the env, the step count, whether the episode is over and the success value
once it is. `reset` starts a new episode of the same task; it closes on request or after 30 minutes idle.

Tokens: the server token can list tasks and open sessions; every session also gets its own token that reaches only
that session. At test time the policy gets only the session token, so it cannot list tasks, open other test tasks or
read their goals.

## What a step returns

- `obs`: what the agent is meant to see: the screen (image), screen texts, the action format, the goal.
- `info`: the environment's per-step feedback (action valid or not and why, the parsed action, whether the screen
  changed, environment-specific extras such as the unlabelled screenshot or the URL).
- `reward`, `terminated`, `truncated`, `done`, `steps`, `max_steps`.
- `success`: returned when a task episode ends, on dev and on test (null before; free play never). Each benchmark's
  own check: the game's task tracker, AndroidWorld's `is_successful`, and for the web WebVoyager's judge (a VLM we
  configure on the server, scoring the final answer and the last screenshots). An episode cut off by the budget
  scores 0.
- `state` (dev only, when the session asked): the environment's internal state (game memory readings, the foreground
  app, the page URL and structure).

Never sent to a client: the internal state on test, the benchmark's own progress/grader signals (subgoal trackers,
success probes), and internal records that carry memory-derived state.

## Keeping held-out environments out of development

Tasks in training environments can still reach held-out ones (every Android app is on one phone; links lead to other
websites). On dev, an action that would land in a held-out app or website is undone inside the step and reported as an
invalid action, so the agent never sees a frame of it. Free play is confined to its own app or website the same way.

## Recording (diagnostic, off by default)

The server can record every session on our side (actions, full info including the internal state, success, frames),
whatever the client was sent. Off unless the server is started with a record directory.

## What to replicate for robotics

A split file with one id per task; a resolver from id to simulator setup; a server per simulator holding the split
and the dev/test policy; obs (camera frames) vs state (simulator poses, contacts, ...) kept separate, state on dev
only; the task's success checker evaluated on the server and returned at episode end; per-session tokens for the
test harness; undo-and-invalid for anything that would reach a held-out environment.
