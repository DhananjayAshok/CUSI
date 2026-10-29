"""
Tree context for prompts: how a node was reached, what earlier expansions from it already tried, and (explore_hint)
both of those with the scene's discovered set, as the hint the exploring executor is given.
"""
from typing import Optional

MAX_ACTION_CHARS = 160
MAX_DISCOVERED = 40
ONSITE = "Do not leave {site}: stay on this website and do not follow links to other websites (pages there do not count)."
ONAPP = "Do not leave {app}: stay in this app and do not open other apps or the home screen (screens there do not count)."


def action_line(*, action, valid: bool = True) -> str:
    """The action's last line (VLM replies end with it), shortened; failed actions are marked."""
    text = str(action or "").strip()
    line = text.splitlines()[-1][:MAX_ACTION_CHARS] if text else "(empty)"
    return line if valid else f"{line} [failed]"


def path_lines(*, tree, node_id: int, max_actions: int = 20) -> list:
    """The root->node actions, oldest first (the earliest dropped beyond max_actions)."""
    steps = [s for s in tree.path_steps(node_id) if s.action]      # an empty action is a decision that acted not
    lines = [action_line(action=s.action, valid=s.info.get("valid", True)) for s in steps]
    if len(lines) > max_actions:
        lines = [f"({len(lines) - max_actions} earlier actions)"] + lines[-max_actions:]
    return lines


def tried_lines(*, tree, node_id: int, before_expansion: int, max_tries: int = 5, actions_per_try: int = 5) -> list:
    """One line per earlier expansion from node_id (the most recent max_tries): its first actions and the reports of
    the nodes it reached (or that it reached none)."""
    expansions = [e for e in tree.expansions if e.node_id == node_id and e.id < before_expansion and e.steps]
    lines = []
    for e in expansions[-max_tries:]:
        acted = [s for s in e.steps if s.action]
        actions = [action_line(action=s.action, valid=s.info.get("valid", True)) for s in acted[:actions_per_try]]
        if len(acted) > actions_per_try:
            actions.append(f"... ({len(acted) - actions_per_try} more)")
        line = " -> ".join(actions)
        reached = [tree.nodes[c].description for c in e.children if tree.nodes[c].description]
        lines.append(f"{line}  =>  reached: {' / '.join(reached)}" if reached else f"{line}  =>  reached nothing new")
    skipped = len(expansions) - len(lines)
    if skipped > 0:
        lines.insert(0, f"({skipped} older explorations not shown)")
    return lines


def _bullets(lines: list, *, empty: str) -> str:
    return "\n".join(f"- {line}" for line in lines) or empty


def explore_hint(*, tree, node_id: int, before_expansion: int, discovered: list, site: Optional[str] = None,
                 app: Optional[str] = None) -> str:
    """The exploring executor's hint: how the start screen was reached, what earlier explorations from it did and
    reached, what the scene has already discovered (the latest MAX_DISCOVERED), and, on a website or in a scoped app,
    to stay on it."""
    parts = ["How this screen was reached from the start screen (oldest first):\n"
             + _bullets(path_lines(tree=tree, node_id=node_id), empty="(it is the start screen)"),
             "Earlier explorations from this screen (do something different from these):\n"
             + _bullets(tried_lines(tree=tree, node_id=node_id, before_expansion=before_expansion),
                        empty="(none: this is the first)")]
    if discovered:
        parts.append("Already found by earlier explorations (no need to find these again):\n"
                     + _bullets(discovered[-MAX_DISCOVERED:], empty=""))
    if site:
        parts.append(ONSITE.format(site=site))
    if app:
        parts.append(ONAPP.format(app=app))
    return "\n\n".join(parts)
