"""
World models trained on search trees: the dataset (dataset.py), the SigLIP embedder (embedder.py), the world model
(model.py, train.py) and its action encoding (actions.py). One model directory per name:
storage/world_model/<env>/<name>/{data, embedder, wm}.
"""
import os


def model_root(*, parameters: dict, env: str, name: str) -> str:
    return os.path.join(parameters["storage_dir"], "world_model", env, name)


def model_dirs(*, parameters: dict, env: str, name: str) -> dict:
    root = model_root(parameters=parameters, env=env, name=name)
    return {"root": root, "data": os.path.join(root, "data"), "embedder": os.path.join(root, "embedder"),
            "wm": os.path.join(root, "wm")}
