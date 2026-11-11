"""
The environment server (plans/user_sandbox_plan.md): benchmark environments behind HTTP, so a coding agent in a
sandbox drives them through a client without any CUSI code or benchmark data on its side.

    codec.py      obs/info <-> JSON (frames as PNG)
    policy.py     what an instance shows: dev (train + val tasks, free play, state on request) or test (test tasks,
                  never state)
    sessions.py   one live env per session: build from a task id or a free-play scene, step, success, held-out checks
    recorder.py   optional per-step record on our side (off unless --record_dir)
    server.py     the HTTP layer
    remote_env.py an env-like proxy over the client, so CUSI's own agents can run through a server

One server process serves one env kind (GameBoyRL's and WebVoyager's `utils` modules cannot share a process).
"""
