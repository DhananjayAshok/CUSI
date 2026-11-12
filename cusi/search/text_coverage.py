"""
Text coverage: compare the screen text search runs of one env saw, rather than their states
(cusi.search.state_coverage), screens (cusi.search.screen_coverage) or tasks (cusi.eval.task_coverage / task_diversity).

Every observation a run made is read, in the order seen: each scene's start (root nodes), then every expansion step
(restores are not exploration and are not recorded). Its text items:
- web: each element label of texts["web_elements"] ('[i]: "label"' or '[i]: <tag> "label"'; '"a", "b"' is two);
- android: each element's "text" and "content_description" in texts["ui_elements"] (not resource_id, which is not
  on screen);
- gameboy: each line of text_boxes (dialogue, menus, names, HUD; digits included).
Items are normalised (Unicode NFKC, lowercase, whitespace collapsed, punctuation and symbols trimmed from both ends);
digits are kept (user's call), so counters and amounts count as distinct text. GameBoy text types out: an item that is
a strict prefix of a longer item seen in the same expansion is dropped, so half-typed lines do not count.

Runs are compared in groups (cusi.search.groups; --runs makes each run its own group): each run is reduced to its
distinct items (overall, per scene, optionally within its first --at_steps env steps), a group's distinct items are
the union over its runs, and groups are compared by set operations (two groups: both, only the first, only the
second; 3+: each group's items no other group saw, and the items every group saw), with each group's most frequent
items no other group saw (frequency = observations an item was on, summed over the group's runs). Exact matches only:
no embedding, no threshold.

    python -m cusi.search.text_coverage compare --env web --runs sat_random_s0,sat_vlm_w_typing_s0
    python -m cusi.search.text_coverage compare --env web \\
        --groups "random=sat_random_s0,sat_random_s1,sat_random_s2;vlm=sat_vlm_w_typing_s0,sat_vlm_w_typing_s1,sat_vlm_w_typing_s2"

Output: storage/search/<env>/_compare/text_coverage/<hash>/ (meta_<hash>.yaml, summary.json, summary.md).
"""
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Optional
import click
from cusi.search.groups import group_table, parse_groups, public, set_stats
from cusi.search.store import EXPANSIONS_FILE, node_rows, read_rows
from cusi.utils.hash_handling import hash_meta_dict, write_meta
from cusi.utils.log_handling import log_info
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.paths import search_root, search_run

ENV_NAMES = ("gameboy", "web", "android")
DIR = "text_coverage"
N_EXAMPLES = 30
_WEB_ITEM = re.compile(r'^\[\d+\]:\s*(?:<[^>]*>\s*)?"(.*)";?$', re.DOTALL)
_ANDROID_FIELD = re.compile(r'"(text|content_description)": "((?:[^"\\]|\\.)*)"')
_SPACE = re.compile(r"\s+")


def normalise(*, text: str) -> str:
    """NFKC, lowercase, whitespace collapsed, punctuation, symbols and other non-text characters (icon-font glyphs,
    format and control characters) trimmed from both ends ("" if nothing is left)."""
    t = _SPACE.sub(" ", unicodedata.normalize("NFKC", text).lower()).strip()
    is_edge = lambda ch: unicodedata.category(ch)[0] in "PSZC"
    start, end = 0, len(t)
    while start < end and is_edge(t[start]):
        start += 1
    while end > start and is_edge(t[end - 1]):
        end -= 1
    return t[start:end]


def raw_items(*, env_name: str, texts: dict, text_boxes: list) -> list:
    """The text items of one observation, before normalisation."""
    if env_name == "web":
        out = []
        for part in (texts.get("web_elements") or "").split("\t"):
            m = _WEB_ITEM.match(part.strip())
            if m:     # an element with two labels (its text and its aria label) reads '"a", "b"'
                out += m.group(1).split('", "')
        return out
    if env_name == "android":
        out = []
        for _, value in _ANDROID_FIELD.findall(texts.get("ui_elements") or ""):
            try:
                out.append(json.loads(f'"{value}"'))
            except json.JSONDecodeError:
                out.append(value)
        return out
    return [b.get("text") or "" for b in text_boxes or []]


def drop_prefixes(*, items: set) -> set:
    """The items that are not a strict prefix of another item in the set."""
    ordered = sorted(items)
    return {a for a, b in zip(ordered, ordered[1:] + [""]) if not (b.startswith(a) and b != a)}


def run_texts(*, env_name: str, directory: str, at_steps: Optional[int]) -> dict:
    """{"count": Counter item -> observations it was on, "scenes": item -> set of scenes, "by_scene": scene -> set,
    "observations": n, "steps": env steps read}."""
    nodes = node_rows(directory=directory)
    count, scenes, by_scene = Counter(), defaultdict(set), defaultdict(set)
    n_obs = 0

    def add(*, groups: list, scene: str) -> None:
        # groups: per observation, its normalised items; GameBoy drops prefixes within the whole group set
        keep = None
        if env_name == "gameboy":
            keep = drop_prefixes(items={i for g in groups for i in g})
        for g in groups:
            for item in g:
                if keep is None or item in keep:
                    count[item] += 1
                    scenes[item].add(scene)
                    by_scene[scene].add(item)

    def items_of(*, texts: dict, text_boxes: list) -> set:
        return {x for x in (normalise(text=r) for r in raw_items(env_name=env_name, texts=texts or {},
                                                                 text_boxes=text_boxes)) if x}

    for n in nodes.values():
        if n["parent"] is None:
            add(groups=[items_of(texts=n.get("texts"), text_boxes=n.get("text_boxes"))], scene=n["env_key"])
            n_obs += 1
    steps = 0
    for exp in read_rows(os.path.join(directory, EXPANSIONS_FILE)):
        scene = nodes[exp["node_id"]]["env_key"]
        groups = []
        for step in exp["steps"]:
            if at_steps is not None and steps >= at_steps:
                break
            steps += 1
            groups.append(items_of(texts=step.get("texts"), text_boxes=step.get("text_boxes")))
        add(groups=groups, scene=scene)
        n_obs += len(groups)
        if at_steps is not None and steps >= at_steps:
            break
    return {"count": count, "scenes": scenes, "by_scene": by_scene, "observations": n_obs, "steps": steps}


def compare_groups(*, env_name: str, groups: dict, at_steps: Optional[int], parameters: dict) -> dict:
    """Group statistics (cusi.search.groups) of the runs' distinct items, overall and per scene, with each group's most
    frequent items no other group saw (frequency: observations summed over the group's runs)."""
    runs = [r for members in groups.values() for r in members]
    data = {r: run_texts(env_name=env_name, directory=search_run(parameters=parameters, env=env_name, run=r),
                         at_steps=at_steps) for r in runs}
    meta = {"env": env_name, "groups": groups, "at_steps": at_steps, "unit": "normalised text items, digits kept"}
    out_dir = os.path.join(search_root(parameters=parameters, env=env_name), "_compare", DIR, hash_meta_dict(meta))
    os.makedirs(out_dir, exist_ok=True)
    write_meta(out_dir, dict(meta), parameters)
    stats = set_stats(sets={r: set(d["count"]) for r, d in data.items()}, groups=groups)
    scenes = sorted({s for d in data.values() for s in d["by_scene"]})
    by_scene = {s: public(stats=set_stats(sets={r: data[r]["by_scene"].get(s, set()) for r in runs}, groups=groups))
                for s in scenes}
    summary = {**meta, "runs_detail": {r: {"env_steps": d["steps"], "observations": d["observations"],
                                           "distinct": len(d["count"])} for r, d in data.items()},
               **public(stats=stats), "by_scene": by_scene, "unique_examples": {}}
    title = f"# Text coverage: {env_name}" + (f", first {at_steps} env steps of each run" if at_steps else "")
    lines = [title, "", "Distinct normalised text items: each run's own, its group's (the union over the group's "
             "runs), and the group's items no other group saw.", "",
             *group_table(stats=stats, groups=groups, unit="items")]
    if len(groups) > 1:
        lines += ["", "## By scene", "", "| scene | " + " | ".join(f"{g} distinct | {g} only" for g in groups) + " |",
                  "|---|" + "---|---|" * len(groups)]
        for s in scenes:
            gs = by_scene[s]["groups"]
            lines.append(f"| {s} | " + " | ".join(f"{gs[g]['distinct']} | {gs[g]['unique']}" for g in groups) + " |")
        for g, members in groups.items():
            others = set().union(*(stats["_union"][h] for h in groups if h != g))
            unique = stats["_union"][g] - others
            freq = Counter()
            for r in members:
                freq.update({i: c for i, c in data[r]["count"].items() if i in unique})
            examples = [[i, c] for i, c in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:N_EXAMPLES]]
            summary["unique_examples"][g] = examples
            lines += ["", f"## Most frequent items only {g} saw (observations)", ""]
            lines += [f"- {i!r} ({c})" for i, c in examples]
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log_info(f"text_coverage compare -> {out_dir}", parameters=parameters)
    return summary


@click.group()
def main():
    pass


@main.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--runs", default=None, help="Comma-separated runs of this env, each its own group (one run: its "
                                           "distinct items only).")
@click.option("--groups", default=None, help='Groups of runs: "name=run1,run2;name=run3,run4" (2+ groups).')
@click.option("--at_steps", default=None, type=int, help="Only each run's first N env steps.")
def compare(env_name, runs, groups, at_steps):
    parameters = load_parameters()
    groups = parse_groups(groups=groups, runs=runs)
    for run in (r for members in groups.values() for r in members):
        if not os.path.isdir(search_run(parameters=parameters, env=env_name, run=run)):
            raise click.UsageError(f"{run} is not a {env_name} run (runs compared must all be of --env {env_name})")
    summary = compare_groups(env_name=env_name, groups=groups, at_steps=at_steps, parameters=parameters)
    for g, s in summary["groups"].items():
        print(f"{g}: runs {s['runs']}, group distinct {s['distinct']}"
              + (f", not in other group(s) {s['unique']}" if len(groups) > 1 else ""))
    if "pair" in summary:
        print(summary["pair"])
    elif "shared_by_all" in summary:
        print(f"in every group: {summary['shared_by_all']}")


if __name__ == "__main__":
    main()
