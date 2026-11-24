"""
Command-line plumbing the eval runners share: the repeat label, the sampling temperature and file-valued --tasks.
"""
import os
from typing import Optional
import click


def seed_option(command):
    """--seed: the repeat number of a run (a label recorded in its config; sampling is unseeded)."""
    return click.option("--seed", default=0, show_default=True, type=int,
                        help="Repeat number of this run (a label recorded in the config).")(command)


def temperature_option(command):
    """--temperature: every agent call (supervisor, executor, frame diff) sampled at this temperature; default greedy.
    The Web judge stays greedy."""
    return click.option("--temperature", default=None, type=float,
                        help="Sample every agent call at this temperature (default: greedy decoding).")(command)


def task_list(*, tasks: Optional[str]) -> Optional[str]:
    """--tasks as given, or, if it names a file, its non-empty lines comma-joined."""
    if tasks and os.path.isfile(tasks):
        with open(tasks) as f:
            return ",".join(line.strip() for line in f if line.strip())
    return tasks
