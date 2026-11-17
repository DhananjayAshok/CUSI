"""
Seed study: is one tree policy better than another at producing world-model training data, beyond the variation
between tree seeds? One world model per tree (not pooled), each tested on the same real task-attempt transitions
(cusi.world_model.evaluate), then the policies' per-seed scores compared.

Layout (storage/world_model/<env>/seeds/<study>/):
    study.json                      groups ({label: [tree, ...]}, one tree per seed), cap
    <label>_s<k>/data/              the tree's dataset (cusi.world_model.dataset)
    <label>_s<k>_cap/data/          the same, train cut to the study's cap (the smallest tree's train size)
    <dataset>/wm_<embedder>/        a world model on that dataset, in the named model's fine-tuned SigLIP
    <dataset>/test_<embedder>/<tag>.json
    compare_<tag>.json

The embedder is held fixed across a comparison (an existing model's fine-tuned SigLIP and decoder), so the trees
differ only in the transitions the world model is trained on; running the study with each policy's embedder shows
whether the answer depends on it.
"""
import glob
import json
import os
from typing import Any, Optional
import numpy as np
from cusi.utils.log_handling import log_info
from cusi.world_model import model_dirs, model_root

STUDY_FILE = "study.json"
#: metric name -> function of (test json, wm train summary); higher is better unless listed in LOWER_IS_BETTER
METRICS = {
    "cos_margin": lambda t, s: t["one_step"]["all"]["cos"] - t["one_step"]["all"]["copy_cos"],
    "cos_margin_changed": lambda t, s: t["one_step"]["all"]["cos_changed"] - t["one_step"]["all"]["copy_cos_changed"],
    "cos": lambda t, s: t["one_step"]["all"]["cos"],
    "pixel_mae": lambda t, s: t["one_step"]["all"]["pixel_mae"],
    "pixel_mae_vs_copy": lambda t, s: t["one_step"]["all"]["pixel_mae"] - t["one_step"]["all"]["copy_pixel_mae"],
    "rollout3_margin": lambda t, s: t["rollout"]["3"]["cos"] - t["rollout"]["3"]["copy_cos"],
    "rollout5_margin": lambda t, s: t["rollout"]["5"]["cos"] - t["rollout"]["5"]["copy_cos"],
    "tree_val_cos_margin": lambda t, s: s["best"]["val_cos"] - s["best"]["val_copy_cos"],
    "n_train": lambda t, s: s["n_train"],
}
LOWER_IS_BETTER = {"pixel_mae", "pixel_mae_vs_copy"}


def study_root(*, parameters: dict, env: str, study: str) -> str:
    return model_root(parameters=parameters, env=env, name=os.path.join("seeds", study))


def read_study(*, root: str) -> dict:
    with open(os.path.join(root, STUDY_FILE)) as f:
        return json.load(f)


def _datasets(*, root: str) -> list:
    """(dataset name, label, seed index, capped) for every dataset in the study."""
    study = read_study(root=root)
    out = []
    for label, trees in study["groups"].items():
        for k in range(len(trees)):
            for capped in (False, True):
                out.append((f"{label}_s{k}" + ("_cap" if capped else ""), label, k, capped))
    return out


def build_seed_datasets(*, env_name: str, study: str, groups: dict, run_dirs: dict, train_frac: float, seed: int,
                        workers: int, parameters: dict[str, Any]) -> dict:
    """groups: {label: [tree name, ...]}; run_dirs: {tree name: directory}. Builds every tree's dataset, then the
    capped copies at the smallest train size. Existing datasets are kept."""
    from cusi.world_model.dataset import DATASET_FILE, build_dataset
    root = study_root(parameters=parameters, env=env_name, study=study)
    os.makedirs(root, exist_ok=True)
    study_path = os.path.join(root, STUDY_FILE)
    if os.path.exists(study_path) and read_study(root=root)["groups"] != groups:
        raise ValueError(f"{study_path} exists with other groups")
    summaries = {}
    for label, trees in groups.items():
        for k, tree in enumerate(trees):
            out = os.path.join(root, f"{label}_s{k}", "data")
            if not os.path.exists(os.path.join(out, DATASET_FILE)):
                build_dataset(env_name=env_name, run_dirs=[run_dirs[tree]], out_dir=out, train_frac=train_frac,
                              seed=seed, workers=workers, parameters=parameters)
            with open(os.path.join(out, DATASET_FILE)) as f:
                summaries[(label, k)] = json.load(f)
    cap = min(s["n_train"] for s in summaries.values())
    with open(study_path, "w") as f:
        json.dump({"env": env_name, "groups": groups, "cap": cap, "train_frac": train_frac, "seed": seed,
                   "n_train": {f"{label}_s{k}": s["n_train"] for (label, k), s in summaries.items()}}, f, indent=1)
    for label, trees in groups.items():
        for k, tree in enumerate(trees):
            out = os.path.join(root, f"{label}_s{k}_cap", "data")
            if not os.path.exists(os.path.join(out, DATASET_FILE)):
                build_dataset(env_name=env_name, run_dirs=[run_dirs[tree]], out_dir=out, train_frac=train_frac,
                              seed=seed, workers=workers, max_train=cap, parameters=parameters)
    log_info(f"seed study {study} [{env_name}]: {sum(map(len, groups.values()))} trees, cap {cap} train transitions",
             parameters=parameters)
    return read_study(root=root)


def train_and_test_seed_models(*, env_name: str, study: str, embedder: str, eval_run_dirs: list, tag: str,
                               max_epochs: int, patience: int, batch_size: int, lr: float, hidden_dim: int,
                               residual: bool, seed: int, workers: int, max_k: int,
                               parameters: dict[str, Any]) -> None:
    """A world model per study dataset in `embedder`'s (a trained model name) SigLIP, then its test on the eval
    runs' transitions. Finished models and tests are kept."""
    from cusi.world_model.evaluate import evaluate_on_episodes
    from cusi.world_model.train import WM_META, train_world_model
    root = study_root(parameters=parameters, env=env_name, study=study)
    embedder_dir = model_dirs(parameters=parameters, env=env_name, name=embedder)["embedder"]
    for name, _label, _k, _capped in _datasets(root=root):
        wm_dir = os.path.join(root, name, f"wm_{embedder}")
        if not os.path.exists(os.path.join(wm_dir, "train_summary.json")):
            train_world_model(env_name=env_name, dataset_dir=os.path.join(root, name, "data"),
                              embedder_dir=embedder_dir, out_dir=wm_dir, max_epochs=max_epochs, patience=patience,
                              batch_size=batch_size, lr=lr, hidden_dim=hidden_dim, residual=residual, seed=seed,
                              embed_batch_size=128, workers=workers, parameters=parameters)
        out = os.path.join(root, name, f"test_{embedder}", f"{tag}.json")
        if os.path.exists(os.path.join(wm_dir, WM_META)) and not os.path.exists(out):
            evaluate_on_episodes(wm_dir=wm_dir, run_dirs=eval_run_dirs, out_path=out, max_k=max_k,
                                 batch_size=128, workers=workers, parameters=parameters)


def _stats(*, a: list, b: list) -> dict:
    from scipy import stats
    out = {"mean_a": float(np.mean(a)), "sd_a": float(np.std(a, ddof=1)) if len(a) > 1 else None,
           "mean_b": float(np.mean(b)), "sd_b": float(np.std(b, ddof=1)) if len(b) > 1 else None,
           "diff": float(np.mean(a) - np.mean(b)), "n_a": len(a), "n_b": len(b)}
    if len(a) > 1 and len(b) > 1:
        out["welch_p"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        out["mannwhitney_p"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        out["cohens_d"] = float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else None
    return out


def compare_seed_models(*, env_name: str, study: str, tag: str, parameters: dict[str, Any]) -> dict:
    """Per embedder and size condition (full trees / capped), every metric's per-seed values for each label and the
    first two labels compared (Welch t-test, Mann-Whitney U, Cohen's d; a minus b, a = the first label)."""
    root = study_root(parameters=parameters, env=env_name, study=study)
    study_meta = read_study(root=root)
    labels = list(study_meta["groups"])
    values = {}
    for name, label, k, capped in _datasets(root=root):
        for test_path in glob.glob(os.path.join(root, name, "test_*", f"{tag}.json")):
            embedder = os.path.basename(os.path.dirname(test_path))[len("test_"):]
            with open(test_path) as f:
                test = json.load(f)
            with open(os.path.join(root, name, f"wm_{embedder}", "train_summary.json")) as f:
                summary = json.load(f)["summary"]
            cond = "capped" if capped else "full"
            for metric, fn in METRICS.items():
                try:
                    v = fn(test, summary)
                except (KeyError, TypeError):
                    continue
                values.setdefault((embedder, cond, metric), {}).setdefault(label, {})[k] = v
    result = {"env": env_name, "study": study, "tag": tag, "labels": labels, "cap": study_meta["cap"],
              "lower_is_better": sorted(LOWER_IS_BETTER), "comparisons": []}
    for (embedder, cond, metric), by_label in sorted(values.items()):
        row = {"embedder": embedder, "condition": cond, "metric": metric,
               "values": {lab: [by_label.get(lab, {}).get(k) for k in range(len(study_meta["groups"][lab]))]
                          for lab in labels}}
        if len(labels) >= 2 and all(by_label.get(lab) for lab in labels[:2]):
            row.update(_stats(a=list(by_label[labels[0]].values()), b=list(by_label[labels[1]].values())))
        result["comparisons"].append(row)
    with open(os.path.join(root, f"compare_{tag}.json"), "w") as f:
        json.dump(result, f, indent=1)
    for row in result["comparisons"]:
        if "diff" in row and row["metric"] != "n_train":
            log_info(f"seed study {study} [{env_name}] {row['embedder']} {row['condition']} {row['metric']}: "
                     f"{labels[0]} {row['mean_a']:.4f}±{row['sd_a'] or 0:.4f} vs {labels[1]} "
                     f"{row['mean_b']:.4f}±{row['sd_b'] or 0:.4f}, diff {row['diff']:+.4f}, "
                     f"welch p {row.get('welch_p', float('nan')):.3g}, MWU p {row.get('mannwhitney_p', float('nan')):.3g}",
                     parameters=parameters)
    return result
