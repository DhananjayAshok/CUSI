"""
Check cusi.envs.webvoyager.WebVoyagerPlayEnv in the CUSI container (needs Chromium,
not KVM). From the CUSI root:

    bash scripts/container.sh python tests/webvoyager_play_env_test.py [--live]

A local two-page site gives deterministic checks; --live also runs a real WebVoyager task.
"""
import os
import re
import sys
import time
import numpy as np
from cusi.envs.webvoyager import WebVoyagerPlayEnv, MSG_FORMAT, MSG_NO_ANSWER

PAGE = "file://" + os.path.abspath("tests/data/webvoyager_test_page.html")


def check(*, cond: bool, msg: str) -> None:
    print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        raise SystemExit(1)


def find(*, obs: dict, pattern: str) -> int:
    """Index of the first labelled element whose line matches `pattern`."""
    for line in obs["texts"]["web_elements"].split("\t"):  # WebVoyager joins entries with tabs
        m = re.match(r"\[(\d+)\]", line.strip())
        if m and re.search(pattern, line, flags=re.IGNORECASE):
            return int(m.group(1))
    raise AssertionError(f"no element matching {pattern!r} in:\n{obs['texts']['web_elements']}")


def timed_step(*, env, action):
    t = time.time()
    out = env.step(action)
    return out, time.time() - t


def local_checks(*, fast_waits: bool) -> None:
    print(f"\n=== local site, free_play, fast_waits={fast_waits}", flush=True)
    env = WebVoyagerPlayEnv(url=PAGE, mode="free_play", fast_waits=fast_waits)
    obs, info = env.reset()
    frame0 = obs["frame"]
    check(cond=frame0.dtype == np.uint8 and frame0.ndim == 3 and frame0.shape[2] == 3,
          msg=f"frame is HxWx3 uint8 {frame0.shape}")
    check(cond=set(obs["texts"]) == {"web_elements"} and "Search" in obs["texts"]["web_elements"],
          msg=f"texts['web_elements'] lists the page's elements ({len(obs['texts']['web_elements'].split(chr(9)))} entries)")
    check(cond=obs["goal"] == "" and "ANSWER" not in obs["actions"], msg="no goal, no ANSWER in free_play")

    (obs, r, term, trunc, info), _ = timed_step(env=env, action="please click the link")
    check(cond=not info["valid"] and info["error"] == MSG_FORMAT and info["stale"], msg="unparseable -> format error, stale")
    (obs, r, term, trunc, info), _ = timed_step(env=env, action="Thought: x\nAction: Click [999]")
    check(cond=not info["valid"] and info["stale"] and np.array_equal(obs["frame"], frame0),
          msg=f"Click [999] -> failed, previous observation returned ({info['error'][:50]}...)")
    (obs, r, term, trunc, info), _ = timed_step(env=env, action="Thought: done\nAction: ANSWER; nothing")
    check(cond=not term and not info["valid"] and info["error"] == MSG_NO_ANSWER, msg="ANSWER rejected in free_play")

    link = find(obs=obs, pattern=r"next page")
    (obs, r, term, trunc, info), t_click = timed_step(env=env, action=f"Thought: go\nAction: Click [{link}]")
    check(cond=info["valid"] and info["url"].endswith("webvoyager_test_page2.html"),
          msg=f"Click [{link}] (link) -> page 2 in {t_click:.1f}s")
    (obs, r, term, trunc, info), t_back = timed_step(env=env, action="GoBack")
    check(cond=info["valid"] and info["url"].endswith("webvoyager_test_page.html"), msg=f"GoBack -> page 1 in {t_back:.1f}s")
    box = find(obs=obs, pattern=r"<input")
    (obs, r, term, trunc, info), t_type = timed_step(env=env, action=f"Type [{box}]; lasagna")
    check(cond=info["valid"] and "q=lasagna" in info["url"], msg=f"Type [{box}]; lasagna -> submitted ({info['url'][-30:]}) in {t_type:.1f}s")
    env.step("GoBack")
    (obs, r, term, trunc, info), t_scroll = timed_step(env=env, action="Scroll [WINDOW]; down")
    check(cond=info["valid"] and not np.array_equal(obs["frame"], frame0), msg=f"Scroll -> view changed in {t_scroll:.1f}s")

    obs, info = env.reset()
    check(cond=info["step"] == 0 and info["url"].endswith("webvoyager_test_page.html"), msg="reset -> fresh browser on the start page")
    check(cond=np.array_equal(obs["frame"], frame0), msg="reset -> identical initial frame")
    env.close()


def live_check() -> None:
    print("\n=== live task, test mode (ArXiv--0)", flush=True)
    env = WebVoyagerPlayEnv(task_id="ArXiv--0", mode="test", fast_waits=True)
    obs, info = env.reset()
    check(cond=len(obs["goal"]) > 10 and "ANSWER" in obs["actions"], msg=f"goal shown: {obs['goal'][:70]!r}")
    check(cond=env.max_steps == 15, msg="test step budget 15 (run.sh --max_iter)")
    obs, r, term, trunc, info = env.step("Thought: look\nAction: Scroll [WINDOW]; down")
    check(cond=info["valid"] and not term, msg=f"Scroll on the live site ({info['url']})")
    obs, r, term, trunc, info = env.step("Thought: done\nAction: ANSWER; test answer")
    check(cond=term and r == 0.0 and info["answer"] == "test answer", msg="ANSWER -> terminated, answer in info, reward 0 (judged offline)")
    env.close()


if __name__ == "__main__":
    local_checks(fast_waits=False)
    local_checks(fast_waits=True)
    if "--live" in sys.argv:
        live_check()
    print("\nALL OK")
