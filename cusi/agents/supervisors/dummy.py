"""
The supervisor that does nothing: one executor leg with the whole budget.
"""
from typing import Optional
from cusi.agents.supervisors.base import Supervisor


class DummySupervisor(Supervisor):

    def _evaluate(self) -> Optional[dict]:
        self.call_executor(self._task)
        return None

    def process_executor_return(self, report):
        return report
