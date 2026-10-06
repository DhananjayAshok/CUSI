"""Layer 3 of cusi.state: novelty scorers (curiosity_plan §3.2.3).

    value, components = scorer.score(prev=..., action=..., next=..., archive=archive, add=True)

On a transition prev -(action)-> next (StateRecords with embeddings). Pure with respect to
lifetime: it reads, and with add=True adds to, the archive it is handed; it never resets it.
components (e.g. {"frame": ..., "region": ...}) are for logging.

seed_frame=True (curiosity's first_add, GameBoyRL's EmbedBuffer) means: store next's embedding
and give the frame part 0 without scoring it. An empty archive is always seeded this way.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional
from cusi.state.archive import NoveltyArchive
from cusi.state.state import StateRecord


class Scorer(ABC):
    name: str = "scorer"

    @abstractmethod
    def score(self, *, prev: Optional[StateRecord], action: Any, next: StateRecord, archive: NoveltyArchive,
              add: bool = True, seed_frame: bool = False) -> tuple:
        """(value, components)."""
