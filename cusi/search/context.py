"""
Tree context for prompts: how a node was reached, and what earlier expansions from it already tried.
"""
MAX_ACTION_CHARS = 160


def action_line(*, action, valid: bool = True) -> str:
    """The action's last line (VLM replies end with it), shortened; failed actions are marked."""
    text = str(action or "").strip()
    line = text.splitlines()[-1][:MAX_ACTION_CHARS] if text else "(empty)"
    return line if valid else f"{line} [failed]"


def path_lines(*, tree, node_id: int, max_actions: int = 20) -> list:
    """The root->node actions, oldest first (the earliest dropped beyond max_actions)."""
    steps = tree.path_steps(node_id)
    lines = [action_line(action=s.action, valid=s.info.get("valid", True)) for s in steps]
    if len(lines) > max_actions:
        lines = [f"({len(lines) - max_actions} earlier actions)"] + lines[-max_actions:]
    return lines


def tried_lines(*, tree, node_id: int, before_expansion: int, max_tries: int = 5, actions_per_try: int = 5) -> list:
    """One line per earlier expansion from node_id (the most recent max_tries): its first actions and, when it was
    described (TransitionDescriber), what happened."""
    expansions = [e for e in tree.expansions if e.node_id == node_id and e.id < before_expansion and e.steps]
    lines = []
    for e in expansions[-max_tries:]:
        actions = [action_line(action=s.action, valid=s.info.get("valid", True)) for s in e.steps[:actions_per_try]]
        if len(e.steps) > actions_per_try:
            actions.append(f"... ({len(e.steps) - actions_per_try} more)")
        line = " -> ".join(actions)
        lines.append(f"{line}  =>  what happened: {e.description}" if e.description else line)
    skipped = len(expansions) - len(lines)
    if skipped > 0:
        lines.insert(0, f"({skipped} older explorations not shown)")
    return lines
