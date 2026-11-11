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

Per run: its distinct items (overall, per scene, optionally within its first --at_steps env steps). For two runs:
items both saw, only the first, only the second; for 3+ runs: items unique to each and items all saw; for each run the
most frequent items no other run saw (frequency = observations an item was on). Exact matches only: no embedding, no
threshold.

    python -m cusi.search.text_coverage compare --env web --runs sat_random_s0,sat_vlm_w_typing_s0

Output: storage/search/<env>/_compare/text_coverage/<hash>/ (meta_<hash>.yaml, summary.json, summary.md).
"""
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Optional
import click
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


def compare_runs(*, env_name: str, runs: list, at_steps: Optional[int], parameters: dict) -> dict:
    data = {r: run_texts(env_name=env_name, directory=search_run(parameters=parameters, env=env_name, run=r),
                         at_steps=at_steps) for r in runs}
    meta = {"env": env_name, "runs": runs, "at_steps": at_steps, "unit": "normalised text items, digits kept"}
    out_dir = os.path.join(search_root(parameters=parameters, env=env_name), "_compare", DIR, hash_meta_dict(meta))
    os.makedirs(out_dir, exist_ok=True)
    write_meta(out_dir, dict(meta), parameters)
    sets = {r: set(d["count"]) for r, d in data.items()}
    summary = {**meta, "runs_detail": {}}
    lines = [f"# Text coverage: {env_name}" + (f", first {at_steps} env steps" if at_steps else ""), "",
             "Distinct normalised text items each run saw, and the items no other run saw.", "",
             "| run | env steps | observations | distinct items | not seen by the other run(s) |",
             "|---|---|---|---|---|"]
    for a in runs:
        others = [b for b in runs if b != a]
        unique = sets[a] - set().union(*(sets[b] for b in others)) if others else set(sets[a])
        detail = {"env_steps": data[a]["steps"], "observations": data[a]["observations"], "distinct": len(sets[a]),
                  "by_scene": {s: len(v) for s, v in sorted(data[a]["by_scene"].items())},
                  "only_vs": {b: len(sets[a] - sets[b]) for b in others},
                  "unique": len(unique),
                  "unique_by_scene": {s: len(v & unique) for s, v in sorted(data[a]["by_scene"].items())},
                  "unique_examples": [[i, data[a]["count"][i]] for i in
                                      sorted(unique, key=lambda i: (-data[a]["count"][i], i))[:N_EXAMPLES]]}
        summary["runs_detail"][a] = detail
        lines.append(f"| {a} | {detail['env_steps']} | {detail['observations']} | {detail['distinct']} | "
                     f"{detail['unique'] if others else '-'} |")
    if len(runs) > 1:
        shared = set.intersection(*sets.values())
        summary["shared_by_all"] = len(shared)
        lines += ["", f"Items all runs saw: {len(shared)}."]
    for a in runs:
        d = summary["runs_detail"][a]
        lines += ["", f"## {a}", "", "| scene | distinct | not seen by the other run(s) |", "|---|---|---|"]
        for s, v in sorted(d["by_scene"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {s} | {v} | {d['unique_by_scene'][s]} |")
        if len(runs) > 1:
            lines += ["", "Most frequent items no other run saw (observations):", ""]
            lines += [f"- {i!r} ({c})" for i, c in d["unique_examples"]]
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
@click.option("--runs", required=True, help="Comma-separated runs of this env (one run: its distinct items only).")
@click.option("--at_steps", default=None, type=int, help="Only each run's first N env steps.")
def compare(env_name, runs, at_steps):
    parameters = load_parameters()
    runs = [r.strip() for r in runs.split(",") if r.strip()]
    for run in runs:
        if not os.path.isdir(search_run(parameters=parameters, env=env_name, run=run)):
            raise click.UsageError(f"{run} is not a {env_name} run (runs compared must all be of --env {env_name})")
    summary = compare_runs(env_name=env_name, runs=runs, at_steps=at_steps, parameters=parameters)
    for run, d in summary["runs_detail"].items():
        print(f"{run}: {d['env_steps']} env steps, {d['distinct']} distinct items"
              + "".join(f", {v} not seen by {b}" for b, v in d["only_vs"].items())
              + (f", {d['unique']} by none of the others" if len(runs) > 2 else ""))
    if "shared_by_all" in summary:
        print(f"shared by all: {summary['shared_by_all']}")


if __name__ == "__main__":
    main()
