"""
State coverage of a search run: the distinct environment states its exploration reached, read from the envs'
ground-truth state (true_state, recorded as each step's info["true"], and for curiosity sessions in
curiosity/<expansion>_true.jsonl), rather than the observation novelty the search itself uses. These are states, not
screens: one state (a URL, a map tile) can show many screens (scrolled, a menu or dialogue open) and one screen can
belong to many states. Distinct screens (frames the agent saw) belong in cusi.search.screen_coverage; distinct tasks
in cusi.eval.task_coverage / task_diversity.

The run's env steps are replayed in the order they were taken (an expansion's curiosity-session steps, then its tree
steps; restores are not exploration and are not counted), and each measure is the cumulative count of distinct things
reached, reported at every CHECKPOINT steps and at the end:

- gameboy: maps, tiles (map, x, y), event flags newly set (vs the scene's start), battles entered, and the highest
  Pokedex seen / caught, party size and badges.
- web: URLs, on-site URLs, on-site route templates (path segments with digits, and GitHub's owner/repo, generalised),
  page skeletons, titles, and JavaScript functions run.
- android: packages, activities, activity states (activity plus its fragments).

    python -m cusi.search.state_coverage counts --env web --run tree_random_v3 [--run another ...]

compare: groups of runs (cusi.search.groups; --runs makes each run its own group) by their sets of distinct states per
set measure (gameboy maps, tiles, flags; web urls, onsite_urls, onsite_routes, skeletons, titles; android packages,
activities, activity_states): each run reduced to its sets, a group's sets the union over its runs, groups compared by
set operations, overall and per scene, optionally within each run's first --at_steps env steps. The other measures
(highest badges / party / Pokedex, battles and JavaScript functions counted) are not sets: they are reported per run
and as each group's mean and sd. The sets are recomputed from the run's steps on each compare (no cache: a run that
grew is never compared on stale sets).

    python -m cusi.search.state_coverage compare --env web --groups "random=sat_random_s0,sat_random_s1;vlm=..."
"""
import json
import os
import re
from typing import Optional
from urllib.parse import urlparse
import click
from cusi.agents.specs import ENV_NAMES, ENV_SPECS
from cusi.search.groups import group_table, mean_sd, parse_groups, public, set_stats
from cusi.search.store import EXPANSIONS_FILE, node_rows, read_rows
from cusi.utils.hash_handling import hash_meta_dict, write_meta
from cusi.utils.log_handling import log_info
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.paths import search_root, search_run

STATE_COVERAGE_FILE = "state_coverage.json"
CHECKPOINT = 100
#: GitHub's own top-level pages; any other first path segment is an owner.
GITHUB_ROUTES = {"", "features", "topics", "explore", "pricing", "enterprise", "about", "login", "signup", "sponsors",
                 "marketplace", "orgs", "settings", "collections", "trending", "search", "team", "customer-stories",
                 "solutions", "resources", "security", "readme", "site", "events", "codespaces", "copilot", "premium-support"}
_DIGIT = re.compile(r"\d")
#: The measures that are sets of states (compared across runs); the rest are maxima or counts.
SET_MEASURES = {"gameboy": ("maps", "tiles", "flags"),
                "web": ("urls", "onsite_urls", "onsite_routes", "skeletons", "titles"),
                "android": ("packages", "activities", "activity_states")}
#: Every measure per env, so runs report the same columns (0 when nothing was reached).
MEASURES = {"gameboy": ("maps", "tiles", "flags", "battles", "dex_seen", "dex_caught", "party", "badges"),
            "web": ("urls", "onsite_urls", "onsite_routes", "skeletons", "titles", "js_functions"),
            "android": ("packages", "activities", "activity_states")}


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
            self._add("activity_states", (true["activity"], tuple(true.get("fragments") or ())))

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


def state_coverage(*, env_name: str, directory: str) -> dict:
    """{"final": measures, "curve": [{"steps", "seconds" (when known), measures}, ...]}; also written to the run's
    state_coverage.json."""
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
    with open(os.path.join(directory, STATE_COVERAGE_FILE), "w") as f:
        json.dump(out, f, indent=1)
    return out


def _sites(*, env_name: str) -> dict:
    return {name: urlparse(scene.kwargs["url"]).netloc.removeprefix("www.")
            for name, scene in ENV_SPECS[env_name].scenes.items() if "url" in scene.kwargs}


def run_sets(*, env_name: str, directory: str, at_steps: Optional[int]) -> dict:
    """A run's states, replayed as state_coverage does (optionally only its first at_steps env steps): {"sets":
    {measure: set}, "by_scene": {scene: {measure: set}}, "values": {non-set measure: value}, "steps": n}."""
    sites = _sites(env_name=env_name)
    tracker, scenes, steps = Tracker(env_name=env_name, sites=sites), {}, 0
    for scene, true, _ in run_records(directory=directory):
        if at_steps is not None and steps >= at_steps:
            break
        tracker.add(true=true, scene=scene)
        scenes.setdefault(scene, Tracker(env_name=env_name, sites=sites)).add(true=true, scene=scene)
        steps += 1
    set_measures = SET_MEASURES[env_name]
    summary = tracker.summary()
    return {"sets": {m: tracker.sets.get(m, set()) for m in set_measures},
            "by_scene": {s: {m: tr.sets.get(m, set()) for m in set_measures} for s, tr in scenes.items()},
            "values": {m: summary[m] for m in MEASURES[env_name] if m not in set_measures}, "steps": steps}


def compare_groups(*, env_name: str, groups: dict, at_steps: Optional[int], parameters: dict) -> dict:
    runs = [r for members in groups.values() for r in members]
    data = {r: run_sets(env_name=env_name, directory=search_run(parameters=parameters, env=env_name, run=r),
                        at_steps=at_steps) for r in runs}
    meta = {"env": env_name, "groups": groups, "at_steps": at_steps, "unit": "distinct states per set measure"}
    out_dir = os.path.join(search_root(parameters=parameters, env=env_name), "_compare", "state_coverage",
                           hash_meta_dict(meta))
    os.makedirs(out_dir, exist_ok=True)
    write_meta(out_dir, dict(meta), parameters)
    scenes = sorted({s for d in data.values() for s in d["by_scene"]})
    summary = {**meta, "runs_detail": {r: {"env_steps": d["steps"], **{m: len(v) for m, v in d["sets"].items()},
                                           **d["values"]} for r, d in data.items()},
               "measures": {}, "values": {}}
    title = f"# State coverage: {env_name}" + (f", first {at_steps} env steps of each run" if at_steps else "")
    lines = [title, "", "Distinct states per measure: each run's own, its group's (the union over the group's runs), "
             "and the group's states no other group reached.", ""]
    for m in SET_MEASURES[env_name]:
        stats = set_stats(sets={r: data[r]["sets"][m] for r in runs}, groups=groups)
        by_scene = {s: public(stats=set_stats(sets={r: data[r]["by_scene"].get(s, {}).get(m, set()) for r in runs},
                                              groups=groups)) for s in scenes}
        summary["measures"][m] = {**public(stats=stats), "by_scene": by_scene}
        lines += [f"## {m}", "", *group_table(stats=stats, groups=groups, unit=m), ""]
        if len(groups) > 1:
            lines += ["| scene | " + " | ".join(f"{g} distinct | {g} only" for g in groups) + " |",
                      "|---|" + "---|---|" * len(groups)]
            for s in scenes:
                gs = by_scene[s]["groups"]
                lines.append(f"| {s} | " + " | ".join(f"{gs[g]['distinct']} | {gs[g]['unique']}" for g in groups)
                             + " |")
            lines.append("")
    values = [m for m in MEASURES[env_name] if m not in SET_MEASURES[env_name]]
    if values:
        lines += ["## Other measures (not sets: per run, and each group's mean ± sd)", "",
                  "| measure | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
        for m in values:
            cells = []
            summary["values"][m] = {}
            for g, members in groups.items():
                vals = [data[r]["values"][m] for r in members]
                mean, sd = mean_sd(values=vals)
                summary["values"][m][g] = {"runs": dict(zip(members, vals)), "mean": mean, "sd": sd}
                cells.append(", ".join(str(v) for v in vals) + (f" (mean {mean:.1f}" + (f" ± {sd:.1f})" if sd is not None
                                                                                       else ")")))
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log_info(f"state_coverage compare -> {out_dir}", parameters=parameters)
    return summary


@click.group()
def main():
    pass


@main.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--run", "runs", required=True, multiple=True, help="Run name(s) under storage/search/<env>/.")
@click.option("--at_steps", default=None, type=int, help="Also report each run's measures at this many env steps.")
def counts(env_name, runs, at_steps):
    """Each run's measures, cumulative over its env steps (state_coverage.json in the run's dir)."""
    parameters = load_parameters()
    for run in runs:
        out = state_coverage(env_name=env_name, directory=search_run(parameters=parameters, env=env_name, run=run))
        print(f"{run}: {out['steps']} steps, {out['seconds'] or 0:.0f}s: {out['final']}")
        if at_steps:
            point = next((c for c in out["curve"] if c["steps"] >= at_steps), None)
            print(f"  at {at_steps} steps: {point}")


@main.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--runs", default=None, help="Comma-separated runs of this env, each its own group.")
@click.option("--groups", default=None, help='Groups of runs: "name=run1,run2;name=run3,run4" (2+ groups).')
@click.option("--at_steps", default=None, type=int, help="Only each run's first N env steps.")
def compare(env_name, runs, groups, at_steps):
    """Groups of runs by their distinct states (cusi.search.groups)."""
    parameters = load_parameters()
    groups = parse_groups(groups=groups, runs=runs)
    for run in (r for members in groups.values() for r in members):
        if not os.path.isdir(search_run(parameters=parameters, env=env_name, run=run)):
            raise click.UsageError(f"{run} is not a {env_name} run (runs compared must all be of --env {env_name})")
    summary = compare_groups(env_name=env_name, groups=groups, at_steps=at_steps, parameters=parameters)
    for m, s in summary["measures"].items():
        print(f"{m}: " + "; ".join(f"{g} {v['distinct']} (runs {list(v['runs'].values())}, not in others {v['unique']})"
                                   for g, v in s["groups"].items()))


if __name__ == "__main__":
    main()
