"""
The run viewer (plans/archive/next_plan.md §D): a static page per search run, <run dir>/viewer/index.html, with
- the tree with node thumbnails;
- per node: the report of the segment that reached it (description, discovered features, window descriptions),
  features, the practice/knowledge notes (screen description, tasks from it, the edge's hindsight task and knowledge), the latest decision score with its verdicts, Q / N / P and flags;
- every expansion stepped through frame by frame (actions, values, texts);
- selection probabilities, decision scores, yields and archive size over the iterations.
Frames are copied out of the run's zips unchanged (WebP); the page needs no server or network.

    python -m cusi.search.viewer --env gameboy --run gemma_vlm_v2
"""
import json
import os
import zipfile
from typing import Optional
import click
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.paths import search_run
from cusi.search.store import EXPANSIONS_FILE, node_rows, read_rows

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.html")
VIEWER_DIR = "viewer"
MAX_STEP_TEXTS = 60


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


class FrameCopier:
    """Copies "frames/<zip>/<member>" references into <out>/frames/<zip stem>_<member>, once each."""

    def __init__(self, *, run_dir: str, out_dir: str) -> None:
        self.run_dir, self.frames_dir = run_dir, os.path.join(out_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self._wanted: dict = {}       # zip path -> {member: file name}

    def __call__(self, ref: Optional[str]) -> Optional[str]:
        if not ref:
            return None
        archive, member = ref.rsplit("/", 1)
        name = f"{os.path.basename(archive)[:-len('.zip')]}_{member}"
        self._wanted.setdefault(os.path.join(self.run_dir, archive), {})[member] = name
        return f"frames/{name}"

    def flush(self) -> int:
        copied = 0
        for archive, members in self._wanted.items():
            todo = {m: n for m, n in members.items() if not os.path.exists(os.path.join(self.frames_dir, n))}
            if not todo or not os.path.exists(archive):
                continue
            with zipfile.ZipFile(archive) as z:
                for member, name in todo.items():
                    with open(os.path.join(self.frames_dir, name), "wb") as f:
                        f.write(z.read(member))
                    copied += 1
        return copied


def _step_texts(step: dict) -> list:
    boxes = step.get("text_boxes") or []
    if boxes:
        return [b.get("text", "") for b in boxes[:MAX_STEP_TEXTS]]
    texts = step.get("texts") or {}
    out = []
    for v in texts.values():
        out.extend(v if isinstance(v, list) else [v])
    return [str(t) for t in out[:MAX_STEP_TEXTS]]


def run_data(*, run_dir: str, frame: FrameCopier) -> dict:
    """The page's data: nodes, expansions (with steps), selection records, config and summary."""
    nodes, order = {}, []
    for row in node_rows(directory=run_dir).values():
        nodes[row["id"]] = {"id": row["id"], "parent": row["parent"], "depth": row["depth"],
                            "scene": row["env_key"], "frame": frame(row["frame"]), "value": row["value"],
                            "segment": row["segment"], "features": row.get("features"), "flags": row.get("flags") or {},
                            "description": row.get("description"), "discovered": row.get("discovered"),
                            "windows": row.get("window_descriptions"),
                            "screen_description": row.get("screen_description"), "tasks": row.get("tasks"),
                            "edge_task": row.get("edge_task"), "knowledge": row.get("knowledge"),
                            "created_at": row["created_at"], "n_expanded": 0, "expansions": []}
        order.append(row["id"])
    expansions, records = [], []
    for row in read_rows(os.path.join(run_dir, EXPANSIONS_FILE)):
        steps = [{"action": s["action"], "frame": frame(s["frame"]), "value": s["value"],
                  "components": s.get("components") or {}, "node_id": s.get("node_id"),
                  "valid": (s.get("info") or {}).get("valid", True), "error": (s.get("info") or {}).get("error"),
                  "texts": _step_texts(s), "changed": (s.get("extra") or {}).get("changed"),
                  "explorer": (s.get("extra") or {}).get(row.get("expander")) or {}} for s in row["steps"]]
        expansions.append({"id": row["id"], "node_id": row["node_id"], "iteration": row["iteration"],
                           "children": row["children"], "yield": row["yield"], "stats": row.get("stats") or {},
                           "expander": row.get("expander"), "expander_info": row.get("expander_info") or {},
                           "steps": steps})
        if row["node_id"] in nodes:
            nodes[row["node_id"]]["n_expanded"] += 1
            nodes[row["node_id"]]["expansions"].append(row["id"])
        rec = row.get("record")
        if rec is not None:
            records.append({k: v for k, v in rec.items() if k != "scores"} | {
                "scores": [{"node_id": e["node_id"], "score": e["score"], "verdicts": e.get("verdicts") or []}
                           for e in rec.get("scores") or []]})
    # the latest selection statistics and decision score each node had
    for rec in records:
        for i, nid in enumerate(rec.get("candidates") or []):
            if nid in nodes:
                nodes[nid]["sel"] = {"iteration": rec["iteration"], "q": rec["q"][i], "n": rec["n"][i],
                                     "p": rec["p"][i], "prob": rec["probs"][i], "energy": rec["energies"][i]}
        for e in rec["scores"]:
            if e["node_id"] in nodes:
                nodes[e["node_id"]]["score"] = {"iteration": rec["iteration"], "score": e["score"],
                                                "verdicts": e["verdicts"]}
    return {"nodes": [nodes[i] for i in order], "expansions": expansions, "records": records,
            "config": _load_json(os.path.join(run_dir, "config.json")) or {},
            "summary": _load_json(os.path.join(run_dir, "summary.json"))}


def build_viewer(*, run_dir: str, title: Optional[str] = None, out_dir: Optional[str] = None) -> str:
    """Write (or refresh) the viewer of the run in run_dir -> the path of its index.html."""
    out = out_dir or os.path.join(run_dir, VIEWER_DIR)
    os.makedirs(out, exist_ok=True)
    frame = FrameCopier(run_dir=run_dir, out_dir=out)
    data = run_data(run_dir=run_dir, frame=frame)
    data["title"] = title or os.path.basename(os.path.normpath(run_dir))
    frame.flush()
    tmp = os.path.join(out, "data.js.tmp")
    with open(tmp, "w") as f:
        f.write("window.RUN = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    os.replace(tmp, os.path.join(out, "data.js"))
    with open(TEMPLATE) as f:
        page = f.read()
    with open(os.path.join(out, "index.html"), "w") as f:
        f.write(page)
    return os.path.join(out, "index.html")


@click.command()
@click.option("--env", "env_name", required=True)
@click.option("--run", "run_name", required=True)
@click.option("--out_dir", default=None, help="Default: <run dir>/viewer.")
def main(env_name, run_name, out_dir):
    parameters = load_parameters()
    path = build_viewer(run_dir=search_run(parameters=parameters, env=env_name, run=run_name),
                        title=f"{env_name} / {run_name}", out_dir=out_dir)
    print(path)


if __name__ == "__main__":
    main()
