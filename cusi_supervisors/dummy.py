"""The supervisor that does nothing (GameBoyRL execution/supervisors/dummy.py): the `baseline` arm.

One executor leg with the whole budget, no hint, no self-termination; success is the env's verdict.
"""
from typing import Optional
from cusi_supervisors.base import Supervisor


class DummySupervisor(Supervisor):

    def _evaluate(self) -> Optional[dict]:
        self.call_executor(self._task)
        return None

    def process_executor_return(self, report):
        return report
