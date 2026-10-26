"""
True coverage of a search run: what its exploration reached in the environment's ground-truth state (the envs'
true_state, recorded as each step's info["true"], and for curiosity sessions in curiosity/<expansion>_true.jsonl),
rather than the observation novelty the search itself uses.

The run's env steps are replayed in the order they were taken (an expansion's curiosity-session steps, then its tree
steps; restores are not exploration and are not counted), and each measure is the cumulative count of distinct things
reached, reported at every CHECKPOINT steps and at the end:

- gameboy: maps, tiles (map, x, y), event flags newly set (vs the scene's start), battles entered, and the highest
  Pokedex seen / caught, party size and badges.
- web: URLs, on-site URLs, on-site route templates (path segments with digits, and GitHub's owner/repo, generalised),
  page skeletons, titles, and JavaScript functions run.
- android: packages, activities, screens (activity plus its fragments).

    python -m cusi.search.coverage --env web --run tree_random_v3 [--run another ...]
"""
import json
import os
import re
from typing import Optional
from urllib.parse import urlparse
import click
from cusi.agents.specs import ENV_NAMES, ENV_SPECS
from cusi.search.store import EXPANSIONS_FILE, node_rows, read_rows
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.paths import search_run

COVERAGE_FILE = "coverage.json"
CHECKPOINT = 100
#: GitHub's own top-level pages; any other first path segment is an owner.
GITHUB_ROUTES = {"", "features", "topics", "explore", "pricing", "enterprise", "about", "login", "signup", "sponsors",
                 "marketplace", "orgs", "settings", "collections", "trending", "search", "team", "customer-stories",
                 "solutions", "resources", "security", "readme", "site", "events", "codespaces", "copilot", "premium-support"}
_DIGIT = re.compile(r"\d")
#: Every measure per env, so runs report the same columns (0 when nothing was reached).
MEASURES = {"gameboy": ("maps", "tiles", "flags", "battles", "dex_seen", "dex_caught", "party", "badges"),
            "web": ("urls", "onsite_urls", "onsite_routes", "skeletons", "titles", "js_functions"),
            "android": ("packages", "activities", "screens")}


def route_template(*, url: str) -> str:
    """host + path with every segment holding a digit replaced by <n>, and GitHub's owner/repo by <owner>/<repo>."""
    u = urlparse(url)
    parts = [p for p in u.path.split("/") if p]
    if u.netloc.endswith("github.com") and parts and parts[0] not in GITHUB_ROUTES:
        parts = ["<owner>"] + (["<repo>"] if len(parts) > 1 else []) + parts[2:]
    return u.netloc + "/" + "/".join("<n>" if _DIGIT.search(p) else p for p in parts)


class Tracker:
    """The cumulative measures of one env's true states."""

    def __init__(self, *, env_name: str, sites: dict) -> None:
        self.env_name, self.sites = env_name, sites      # scene -> its site's host (web)
        self.sets: dict = {}
        self.maxes: dict = {}
        self.counts: dict = {}
        self._in_battle = False

    def _add(self, key: str, item) -> None:
        self.sets.setdefault(key, set()).add(item)

    def _max(self, key: str, value) -> None:
        self.maxes[key] = max(self.maxes.get(key, value), value)

    def add(self, *, true: dict, scene: Optional[str]) -> None:
        if self.env_name == "gameboy":
            self._add("maps", true["map"])
            self._add("tiles", (true["map"], true["x"], true["y"]))
            for flag in true["flags_on"]:
                self._add("flags", flag)
            if true["battle"] and not self._in_battle:
                self.counts["battles"] = self.counts.get("battles", 0) + 1
            self._in_battle = bool(true["battle"])
            for key in ("dex_seen", "dex_caught", "party", "badges"):
                self._max(key, true[key])
        elif self.env_name == "web":
            url = (true.get("url") or "").split("#")[0]
            if not url:
                return
            self._add("urls", url)
            site = self.sites.get(scene)
            if site and urlparse(url).netloc.endswith(site):
                self._add("onsite_urls", url)
                self._add("onsite_routes", route_template(url=url))
            for key in ("skeleton", "title"):
                if true.get(key):
                    self._add(key + "s", true[key])
            self.counts["js_functions"] = self.counts.get("js_functions", 0) + true.get("js_new", 0)
        else:
            if not true.get("activity"):
                return
            self._add("packages", true["activity"].split("/")[0])
            self._add("activities", true["activity"])
            self._add("screens", (true["activity"], tuple(true.get("fragments") or ())))

    def summary(self) -> dict:
        found = {**{k: len(v) for k, v in self.sets.items()}, **self.maxes, **self.counts}
        return {k: found.get(k, 0) for k in MEASURES[self.env_name]}


def run_records(*, directory: str):
    """(scene, true state, cumulative seconds or None) per env step, in the order taken."""
    nodes = node_rows(directory=directory)
    seconds = 0.0
    for exp in read_rows(os.path.join(directory, EXPANSIONS_FILE)):
        scene = nodes[exp["node_id"]]["env_key"]
        log = (exp.get("expander_info") or {}).get("true_log")
        if log:
            for row in read_rows(os.path.join(directory, log)):
                yield scene, row["true"], seconds + row["seconds"]
        for i, step in enumerate(exp["steps"]):
            true = (step.get("info") or {}).get("true")
            last = i == len(exp["steps"]) - 1
            if true is not None:
                yield scene, true, seconds + (exp.get("stats") or {}).get("seconds", 0.0) if last else None
        seconds += (exp.get("stats") or {}).get("seconds", 0.0)


def coverage(*, env_name: str, directory: str) -> dict:
    """{"final": measures, "curve": [{"steps", "seconds" (when known), measures}, ...]}; also written to the run's
    coverage.json."""
    sites = {name: urlparse(scene.kwargs["url"]).netloc.removeprefix("www.")
             for name, scene in ENV_SPECS[env_name].scenes.items() if "url" in scene.kwargs}
    tracker, curve, steps, last_seconds = Tracker(env_name=env_name, sites=sites), [], 0, None
    for scene, true, seconds in run_records(directory=directory):
        tracker.add(true=true, scene=scene)
        steps += 1
        last_seconds = seconds if seconds is not None else last_seconds
        if steps % CHECKPOINT == 0:
            curve.append({"steps": steps, "seconds": last_seconds, **tracker.summary()})
    out = {"steps": steps, "seconds": last_seconds, "final": tracker.summary(), "curve": curve}
    with open(os.path.join(directory, COVERAGE_FILE), "w") as f:
        json.dump(out, f, indent=1)
    return out


@click.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--run", "runs", required=True, multiple=True, help="Run name(s) under storage/search/<env>/.")
@click.option("--at_steps", default=None, type=int, help="Also report each run's measures at this many env steps.")
def main(env_name, runs, at_steps):
    parameters = load_parameters()
    for run in runs:
        out = coverage(env_name=env_name, directory=search_run(parameters=parameters, env=env_name, run=run))
        print(f"{run}: {out['steps']} steps, {out['seconds'] or 0:.0f}s: {out['final']}")
        if at_steps:
            point = next((c for c in out["curve"] if c["steps"] >= at_steps), None)
            print(f"  at {at_steps} steps: {point}")


if __name__ == "__main__":
    main()
