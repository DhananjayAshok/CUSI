"""
Annotate a finished search run, every node and edge, with:
- what a VLM-explorer search writes as it goes: each node's features (VLMFeatures) and the report of the segment that
  reached it (TransitionDescriber: its parent's screen, then every step to it); random and curiosity trees run
  without a model, so this gives them the same notes. Decision scores are not added: they are made at selection
  time, which is over.
- the notes the practice and knowledge pipelines read (cusi/search/notes.py): each node's screen description and the
  tasks that could be done from it; each edge's hindsight task and knowledge. A node's `tasks` lists the tasks that
  start from it, each with its source: {"task", "source": "node"} from its own screen, {"task", "source": "edge",
  "child", "segment"} from the transition to that child (whose `edge_task` and `knowledge` hold the edge's notes).

The notes are appended to nodes.jsonl as update rows, node by node, so an interrupted run continues where it stopped;
annotations.json records the model. The run's viewer is rebuilt at the end.

    python -m cusi.search.annotate --env web --run tree_random_v3 --model_name <model> --vllm_base_url <url>
"""
import json
import os
from typing import Any
import click
from cusi.agents.frames import FrameRef
from cusi.agents.specs import ENV_NAMES, ENV_SPECS
from cusi.search.store import EXPANSIONS_FILE, append_node_updates, node_rows, read_rows, step_from_row
from cusi.utils.log_handling import log_info
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.paths import search_run

ANNOTATIONS_FILE = "annotations.json"


def annotate_run(*, directory: str, env_name: str, vlm, model_name: str, max_features: int,
                 parameters: dict[str, Any]) -> dict:
    from cusi.search.describe import TransitionDescriber
    from cusi.search.features import VLMFeatures
    from cusi.search.notes import TASK_EXAMPLES, EdgeNotes, NodeNotes
    meta_path = os.path.join(directory, ANNOTATIONS_FILE)
    meta = {"model_name": model_name, "max_features": max_features}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            done = json.load(f)
        if {k: done.get(k) for k in meta} != meta:
            raise click.UsageError(f"{directory} was annotated with {done}; not mixing in {meta}")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    env_description = ENV_SPECS[env_name].env_description
    features = VLMFeatures(vlm=vlm, max_features=max_features)
    describer = TransitionDescriber(vlm=vlm, parameters=parameters)
    node_notes, edge_notes = (NodeNotes(vlm=vlm, task_examples=TASK_EXAMPLES[env_name]),
                              EdgeNotes(vlm=vlm, task_examples=TASK_EXAMPLES[env_name]))
    nodes = node_rows(directory=directory)
    expansions = {e["id"]: e for e in read_rows(os.path.join(directory, EXPANSIONS_FILE))}
    counts = {"features": 0, "reports": 0, "node_notes": 0, "edge_notes": 0}
    for i, nid in enumerate(sorted(nodes)):
        node = nodes[nid]
        if (node.get("flags") or {}).get("challenge_page"):
            continue
        update, parent_update = {}, {}
        frame = FrameRef(path=os.path.join(directory, node["frame"])).pil()
        if node.get("features") is None:
            f = features(env_description=env_description, frame=frame, text_boxes=node.get("text_boxes") or [])
            if not f.pop("parsed"):
                update["flags"] = {**(node.get("flags") or {}), "features_parse_failed": True}
            update.update(f)
            counts["features"] += 1
        if node.get("node_notes_raw") is None:
            notes = node_notes(env_description=env_description, frame=frame, text_boxes=node.get("text_boxes") or [])
            edge_tasks = [t for t in node.get("tasks") or [] if t["source"] == "edge"]
            update.update(notes, tasks=[{"task": t, "source": "node"} for t in notes["node_tasks"]] + edge_tasks)
            counts["node_notes"] += 1
        if node["parent"] is not None and (node.get("description") is None or node.get("edge_notes_raw") is None):
            exp_id, first, last = node["segment"]
            steps = [step_from_row(directory=directory, row=r) for r in expansions[exp_id]["steps"][first:last + 1]]
            parent = nodes[node["parent"]]
            start_frame = FrameRef(path=os.path.join(directory, parent["frame"]))
            if node.get("description") is None:
                update.update(describer(env_description=env_description, steps=steps, start_frame=start_frame))
                counts["reports"] += 1
            if node.get("edge_notes_raw") is None:
                notes = edge_notes(env_description=env_description, start_frame=start_frame.pil(), end_frame=frame,
                                   steps=steps, description=update.get("description", node.get("description")))
                update.update(notes)
                if notes["edge_task"]:
                    parent_update["tasks"] = (parent.get("tasks") or []) + [
                        {"task": notes["edge_task"], "source": "edge", "child": nid, "segment": node["segment"]}]
                counts["edge_notes"] += 1
        if update:
            rows = [{"id": nid, **update}] + ([{"id": node["parent"], **parent_update}] if parent_update else [])
            append_node_updates(directory=directory, rows=rows)
            node.update(update)
            if parent_update:
                nodes[node["parent"]].update(parent_update)
        if (i + 1) % 20 == 0:
            log_info(f"annotate {directory}: {i + 1}/{len(nodes)} nodes, {counts}", parameters=parameters)
    return counts


@click.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--run", "run_name", required=True)
@click.option("--model_name", required=True)
@click.option("--model_backend", default="vllm", show_default=True)
@click.option("--vllm_base_url", default=None)
@click.option("--max_features", default=8, show_default=True, help="Features listed per node.")
def main(env_name, run_name, model_name, model_backend, vllm_base_url, max_features):
    from cusi.agents.vlm import AgentVLM
    from cusi.search.viewer import build_viewer
    parameters = load_parameters()
    directory = search_run(parameters=parameters, env=env_name, run=run_name)
    vlm = AgentVLM(model_name=model_name, model_backend=model_backend, vllm_base_url=vllm_base_url,
                   parameters=parameters)
    counts = annotate_run(directory=directory, env_name=env_name, vlm=vlm, model_name=model_name,
                          max_features=max_features, parameters=parameters)
    viewer = build_viewer(run_dir=directory, title=f"{env_name} / {run_name}")
    log_info(f"annotated {directory}: {counts}; viewer {viewer}", parameters=parameters)


if __name__ == "__main__":
    main()
