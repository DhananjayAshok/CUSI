"""
Groups of search runs for the coverage comparisons (cusi.search.text_coverage, screen_coverage, state_coverage): e.g.
random s0-s2 against VLM s0-s2. Each run is first reduced to its own distinct items (no duplicates within a run); a
group's distinct items are the union over its runs, with duplicates across runs removed; groups are then compared:
for two groups, the items both have, only the first, only the second; for 3+ groups, each group's items no other group
has, and the items every group has. Each run's own distinct count is reported next to its group's (mean and sd over
the group's runs), so seed variance shows beside the union.

    --groups "random=sat_random_s0,sat_random_s1;vlm=sat_vlm_no_typing_s0,sat_vlm_no_typing_s1"
    --runs sat_random_s0,sat_vlm_no_typing_s0        (each run its own group)
"""
import math
from typing import Optional
import click


def parse_groups(*, groups: Optional[str], runs: Optional[str]) -> dict:
    """{group name: [runs]} from --groups "name=r1,r2;name=r3" or --runs "r1,r2" (each run its own group, named after
    it). Exactly one of them; a run may be in one group only."""
    if (groups is None) == (runs is None):
        raise click.UsageError("give --groups or --runs (one of them)")
    if runs is not None:
        names = [r.strip() for r in runs.split(",") if r.strip()]
        out = {r: [r] for r in names}
    else:
        out = {}
        for part in groups.split(";"):
            if not part.strip():
                continue
            if "=" not in part:
                raise click.UsageError(f"--groups: {part!r} is not name=run1,run2")
            name, members = part.split("=", 1)
            name = name.strip()
            if name in out:
                raise click.UsageError(f"--groups: group {name!r} given twice")
            out[name] = [r.strip() for r in members.split(",") if r.strip()]
            if not out[name]:
                raise click.UsageError(f"--groups: group {name!r} has no runs")
        if len(out) < 2:
            raise click.UsageError("--groups needs two or more groups")
    seen = {}
    for name, members in out.items():
        for r in members:
            if r in seen:
                raise click.UsageError(f"run {r} is in groups {seen[r]!r} and {name!r} (or listed twice)")
            seen[r] = name
    return out


def mean_sd(*, values: list) -> tuple:
    """(mean, sample sd; sd None for fewer than two values)."""
    if not values:
        return None, None
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, None
    return m, math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def set_stats(*, sets: dict, groups: dict) -> dict:
    """Group statistics of per-run sets ({run: set}). {"groups": {name: {"runs": {run: n}, "run_mean", "run_sd",
    "distinct", "unique"}}, "shared_by_all"} plus, for two groups, "pair": {"shared", "only_<a>", "only_<b>"}.
    "unique": items no other group has (the group's distinct for a single group)."""
    union = {g: set().union(*(sets[r] for r in members)) for g, members in groups.items()}
    out = {"groups": {}}
    for g, members in groups.items():
        others = [h for h in groups if h != g]
        rest = set().union(*(union[h] for h in others)) if others else set()
        m, sd = mean_sd(values=[len(sets[r]) for r in members])
        out["groups"][g] = {"runs": {r: len(sets[r]) for r in members}, "run_mean": m, "run_sd": sd,
                            "distinct": len(union[g]), "unique": len(union[g] - rest)}
    if len(groups) > 1:
        out["shared_by_all"] = len(set.intersection(*union.values()))
    if len(groups) == 2:
        a, b = list(groups)
        out["pair"] = {"shared": len(union[a] & union[b]), f"only_{a}": len(union[a] - union[b]),
                       f"only_{b}": len(union[b] - union[a])}
    out["_union"] = union
    return out


def fmt_mean_sd(*, m, sd) -> str:
    if m is None:
        return "-"
    return f"{m:.0f}" if sd is None else f"{m:.0f} ± {sd:.0f}"


def group_table(*, stats: dict, groups: dict, unit: str) -> list:
    """Markdown lines: one row per group (its runs' own distinct counts, their mean ± sd, the group's distinct, the
    group's items no other group has), then the shared counts."""
    lines = [f"| group | runs | each run's distinct {unit} | mean ± sd | group distinct | not in other group(s) |",
             "|---|---|---|---|---|---|"]
    for g, s in stats["groups"].items():
        lines.append(f"| {g} | {len(groups[g])} | {', '.join(str(v) for v in s['runs'].values())} | "
                     f"{fmt_mean_sd(m=s['run_mean'], sd=s['run_sd'])} | {s['distinct']} | "
                     f"{s['unique'] if len(groups) > 1 else '-'} |")
    if "pair" in stats:
        lines += ["", ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in stats["pair"].items()) + "."]
    elif "shared_by_all" in stats:
        lines += ["", f"In every group: {stats['shared_by_all']}."]
    return lines


def public(*, stats: dict) -> dict:
    """stats without the sets (JSON-ready)."""
    return {k: v for k, v in stats.items() if not k.startswith("_")}
