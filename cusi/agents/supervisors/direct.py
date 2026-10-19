"""
No supervision (the Web and Android designs, decisions.md E8.0): one executor leg with the whole budget; in test
mode its finish goes to the env and is scored.
"""
from typing import Optional
from cusi.agents.supervisors.base import Supervisor


class DirectSupervisor(Supervisor):

    def _evaluate(self) -> Optional[dict]:
        self.call_executor(task=self._task, hint=None, max_steps=self._max_steps)
        return None
