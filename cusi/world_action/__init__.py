"""World Action Models: state in, actions out (base.py). load_world_action_model picks the class from the spec:
"oracle" (the emulator lookahead, no directory) or a world model directory / name (siglip_lookahead)."""
import os
from cusi.world_action.base import Plan, WorldActionModel, game_series, read_meta

MODELS = {"siglip_lookahead": "cusi.world_action.siglip_lookahead:SiglipLookaheadModel",
          "oracle_lookahead": "cusi.world_action.oracle_lookahead:OracleLookaheadModel"}
ORACLE = "oracle"


def resolve_directory(*, spec: str, parameters: dict) -> str:
    """A model directory, or a world model's name under storage/explore/<env>/world_model."""
    if os.path.isdir(spec):
        return spec
    from cusi.utils.paths import world_model_dir
    for env in ("gameboy", "android", "web"):
        d = world_model_dir(parameters=parameters, env=env, name=spec)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"no world action model directory {spec!r}")


def load_world_action_model(*, spec: str, device: str, parameters: dict):
    import importlib
    if spec == ORACLE:
        model, directory = "oracle_lookahead", None
    else:
        model, directory = "siglip_lookahead", resolve_directory(spec=spec, parameters=parameters)
    module, cls = MODELS[model].split(":")
    return getattr(importlib.import_module(module), cls).load(directory=directory, device=device,
                                                               parameters=parameters)
