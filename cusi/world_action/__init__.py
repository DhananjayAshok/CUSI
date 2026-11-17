"""World Action Models: state in, actions out (base.py). load_world_action_model picks the class from the spec:
"oracle" (the emulator lookahead, no directory), "oracle_preview" (any env: the executor's proposals previewed with
env.peek and vetted), "oracle_preview_image" (the same, the check shown the preview as images only),
"wm_preview_image:<name or directory>" (oracle_preview_image with a tree world model's predicted screen in place of the
peek; <name> under storage/world_model/<env>/) or a world model directory / name (siglip_lookahead)."""
import os
from cusi.world_action.base import Plan, WorldActionModel, game_series, read_meta

MODELS = {"siglip_lookahead": "cusi.world_action.siglip_lookahead:SiglipLookaheadModel",
          "oracle_lookahead": "cusi.world_action.oracle_lookahead:OracleLookaheadModel",
          "oracle_preview": "cusi.world_action.oracle_preview:OraclePreviewModel",
          "oracle_preview_image": "cusi.world_action.oracle_preview:OraclePreviewImageModel",
          "wm_preview_image": "cusi.world_action.wm_preview:WMPreviewImageModel"}
ORACLE = "oracle"
ORACLE_PREVIEW = "oracle_preview"
ORACLE_PREVIEW_IMAGE = "oracle_preview_image"
WM_PREVIEW_IMAGE = "wm_preview_image"


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


def _is_tree_world_model(*, directory: str) -> bool:
    from cusi.world_model.train import WM_META
    return os.path.isdir(directory) and any(os.path.exists(os.path.join(directory, *sub, WM_META))
                                            for sub in ((), ("wm",)))


def resolve_tree_world_model(*, spec: str, parameters: dict) -> str:
    """A tree world model's directory, or its name under storage/world_model/<env>/."""
    if os.path.isdir(spec):
        return spec
    from cusi.world_model import model_root
    for env in ("gameboy", "android", "web"):
        d = model_root(parameters=parameters, env=env, name=spec)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"no tree world model {spec!r}")


def load_world_action_model(*, spec: str, device: str, parameters: dict):
    import importlib
    if spec == ORACLE:
        model, directory = "oracle_lookahead", None
    elif spec in (ORACLE_PREVIEW, ORACLE_PREVIEW_IMAGE):
        model, directory = spec, None
    elif spec.startswith(WM_PREVIEW_IMAGE + ":"):
        model = WM_PREVIEW_IMAGE
        directory = resolve_tree_world_model(spec=spec.split(":", 1)[1], parameters=parameters)
    elif _is_tree_world_model(directory=spec):     # a loaded wm_preview_image's directory, passed on to eval workers
        model, directory = WM_PREVIEW_IMAGE, spec
    else:
        model, directory = "siglip_lookahead", resolve_directory(spec=spec, parameters=parameters)
    module, cls = MODELS[model].split(":")
    return getattr(importlib.import_module(module), cls).load(directory=directory, device=device,
                                                               parameters=parameters)
