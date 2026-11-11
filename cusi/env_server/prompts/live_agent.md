You are being evaluated on a task in an interactive environment ({env}). A session is already open on the task:
attach to it with

    python3 envctl use "$ENVCTL_ENV" "$ENVCTL_SESSION"

which writes the current screen to obs/<session>/<n>.png and the rest of the observation (including the goal and the
action format) to obs/<session>/<n>.json. Then act with `python3 envctl step "<action>"` (and `observe` to look
again) until the episode is done (`done` in the observation) or you judge it cannot be completed.

Budgets: at most {max_steps} environment steps (every step counts, invalid ones too) and {minutes} minutes of wall
clock; you are stopped when either runs out. Your workspace is /work: your guides, tools and prompts are there.
{memory_note}
