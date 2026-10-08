"""
Run directories keyed by --run_name that may only be resumed with the config they were started with.
"""
import json
import os
import shutil
from typing import Optional
import click

CONFIG_FILE = "config.json"


def run_options(command):
    """--overwrite and --ignore_config_violation, for every command that writes a run directory."""
    command = click.option("--ignore_config_violation", is_flag=True,
                           help="Resume a run whose stored config differs (kept as stored).")(command)
    return click.option("--overwrite", is_flag=True, help="Delete the run directory first.")(command)


def config_violations(*, stored: dict, new: dict, operational_keys=frozenset(), filled_later_keys=frozenset()) -> list:
    """operational_keys may differ freely; filled_later_keys may go from "none" to a value once."""
    out = []
    for key in sorted(set(stored) | set(new)):
        if key in operational_keys:
            continue
        old, cur = json.loads(json.dumps(stored.get(key), default=str)), json.loads(json.dumps(new.get(key), default=str))
        if old == cur or (key in filled_later_keys and (old in (None, "none") or cur in (None, "none"))):
            continue
        out.append(f"{key}: {old!r} -> {cur!r}")
    return out


def open_run_dir(*, directory: str, config: Optional[dict] = None, overwrite: bool = False,
                 ignore_config_violation: bool = False, operational_keys=frozenset(),
                 filled_later_keys=frozenset()) -> Optional[dict]:
    """Create (or, with overwrite, clear) the directory and check config against config.json; returns the
    config written there. A differing config raises unless ignore_config_violation, which keeps the stored values."""
    if overwrite and os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
    if config is None:
        return None
    config_path = os.path.join(directory, CONFIG_FILE)
    if os.path.exists(config_path):
        with open(config_path) as f:
            stored = json.load(f)
        violations = config_violations(stored=stored, new=config, operational_keys=operational_keys,
                                       filled_later_keys=filled_later_keys)
        if violations and not ignore_config_violation:
            raise click.UsageError(f"{directory} was started with a different config ({'; '.join(violations)}). "
                                   f"Use a new --run_name, --overwrite, or --ignore_config_violation.")
        if violations:
            print(f"WARNING: resuming {directory} despite config differences ({'; '.join(violations)}); "
                  f"config.json keeps the stored values.")
            merged = {**config, **stored}
        else:
            merged = {**stored, **config}
        for key in filled_later_keys:     # filled once, never reset to "none"
            values = [v for v in (stored.get(key), config.get(key)) if v not in (None, "none")]
            if values:
                merged[key] = values[-1] if not violations else values[0]
        config = merged
    with open(config_path, "w") as f:
        json.dump(config, f, indent=1, default=str)
    return config
