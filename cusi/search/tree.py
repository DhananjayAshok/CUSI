"""
The search tree: restorable state nodes and the expansions (kept step by step) that created them.
"""
from dataclasses import dataclass, field
from typing import Any, Optional
from cusi.agents.records import EncodedImage

# info keys kept per step (raw frames and region crops are large; the frame is kept separately).
INFO_KEYS = ("valid", "error", "parsed_action", "step", "url", "stale", "action_name", "frame_changed")


def compact_info(info: dict) -> dict:
    return {k: info[k] for k in INFO_KEYS if k in (info or {})}


@dataclass
class Step:
    action: Optional[str]
    frame: EncodedImage
    texts: dict
    info: dict
    value: float = 0.0
    components: dict = field(default_factory=dict)
    cell: Optional[int] = None
    new_cell: bool = False
    node_id: Optional[int] = None          # the node created at this step, if any
    extra: dict = field(default_factory=dict)


@dataclass
class Node:
    id: int
    parent: Optional[int]
    depth: int
    env_key: str
    state_id: Optional[str]
    frame: EncodedImage
    texts: dict
    embedding: Any
    cell: Optional[int]
    segment: Optional[tuple] = None        # (expansion id, first step, last step) that reached it
    n_expanded: int = 0
    yields: list = field(default_factory=list)
    prior: Optional[float] = None
    prior_reason: str = ""
    prior_raw: str = ""
    flags: dict = field(default_factory=dict)   # e.g. unrestorable, challenge_page, prior_parse_failed
    created_at: int = 0                    # iteration that created it (0: roots)

    @property
    def restorable(self) -> bool:
        return self.state_id is not None and not self.flags.get("unrestorable", False)


@dataclass
class Expansion:
    id: int
    node_id: int
    iteration: int
    steps: list = field(default_factory=list)
    children: list = field(default_factory=list)
    yield_value: float = 0.0
    stats: dict = field(default_factory=dict)


class Tree:
    def __init__(self) -> None:
        self.nodes: list = []
        self.expansions: list = []
        self.children: dict = {}

    @property
    def roots(self) -> list:
        return [n for n in self.nodes if n.parent is None]

    def add_node(self, *, parent: Optional[int], env_key: str, state_id: Optional[str], frame, texts: dict,
                 embedding: Any, cell: Optional[int], segment: Optional[tuple] = None, created_at: int = 0) -> Node:
        depth = 0 if parent is None else self.nodes[parent].depth + 1
        node = Node(id=len(self.nodes), parent=parent, depth=depth, env_key=env_key, state_id=state_id,
                    frame=EncodedImage.of(frame), texts=dict(texts), embedding=embedding, cell=cell,
                    segment=segment, created_at=created_at)
        self.nodes.append(node)
        self.children.setdefault(node.id, [])
        if parent is not None:
            self.children[parent].append(node.id)
        return node

    def add_expansion(self, *, node_id: int, iteration: int) -> Expansion:
        exp = Expansion(id=len(self.expansions), node_id=node_id, iteration=iteration)
        self.expansions.append(exp)
        return exp

    def ancestors(self, node_id: int) -> list:
        """node_id, its parent, ..., the root."""
        out = []
        cur: Optional[int] = node_id
        while cur is not None:
            out.append(cur)
            cur = self.nodes[cur].parent
        return out

    def path_steps(self, node_id: int) -> list:
        """Every Step from the root to node_id (the root -> node trajectory)."""
        steps = []
        for nid in reversed(self.ancestors(node_id)):
            seg = self.nodes[nid].segment
            if seg is not None:
                exp_id, first, last = seg
                steps += self.expansions[exp_id].steps[first:last + 1]
        return steps
