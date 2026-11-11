# Your workspace

You are learning to solve tasks in interactive environments (GameBoy games, websites, an Android phone). This
workspace is yours: write guides, tools and prompts here, and keep it under git.

## The environments

They run on servers; you reach them with `envctl` (or `envclient.py` from Python):

    python3 envctl tasks [--env gameboy|web|android]      # the tasks you can practise on (train and val)
    python3 envctl scenes --env <env>                     # free-play start points (no goal)
    python3 envctl open --task <task id> [--state]        # or: python3 envctl open --env <env> --scene <scene>
    python3 envctl step "<action>"                        # the action format is in each observation's "actions"
    python3 envctl observe | reset | save | load <handle> | info | close

Each call writes the screen to `obs/<session>/<n>.png` and the rest of the observation to `obs/<session>/<n>.json`.
`--state` adds the environment's internal state (positions, current app or URL, ...) to each observation: use it
while developing; it will not be available when you are evaluated. When a task episode ends, `success` says
whether it was solved.

## Evaluation

You will be evaluated on tasks you have not seen, one after another: each starts with a session already open on the
task (`python3 envctl use "$ENVCTL_ENV" "$ENVCTL_SESSION"` attaches to it and writes its first observation). There is
no `--state`, no save/load and no free play then; each task has its step budget and a wall-clock limit. Every
observation carries `steps` / `max_steps`; `$ENVCTL_DEADLINE` is the unix time the wall-clock limit ends, and envctl
shows the seconds left (`seconds_left` in each observation file and in `envctl info`; `envclient.seconds_left()`).

You may be evaluated by running code you leave here: if so, `policy.json` names the command to run in this
directory for each task, e.g. `{"entrypoint": "python3 tools/policy.py"}`.

## Layout

    guides/    what you learn about the environments and how to solve tasks
    tools/     code you write
    prompts/   prompts you write
