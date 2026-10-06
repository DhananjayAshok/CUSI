"""
The novelty scorer interface: score a transition against an archive it reads (and adds to) but never resets.
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
        """(value, components); seed_frame stores next and gives the frame part 0 without scoring it."""
