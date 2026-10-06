"""Test-set evaluation through cusi.envs in test mode (plans/eval_plan.md, option 2).

One runner per benchmark (cusi/eval/{gameboy,android,web}.py) builds the env in mode="test"
per test task, drives it with an agent (a native-replica executor from cusi.practice, with the
native harness's settings), and records the env's verdict in the shared per-task format
(cusi/eval/records.py). Entry point: run_eval.py. Equivalence with each native harness is
checked by tests/eval_parity_<env>.py.
"""
