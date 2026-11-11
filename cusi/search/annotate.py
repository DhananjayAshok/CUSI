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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import click
from cusi.agents.frames import FrameRef
from cusi.agents.specs import ENV_NAMES, ENV_SPECS
from cusi.search.store import EXPANSIONS_FILE, append_node_updates, node_rows, read_rows, step_from_row
from cusi.utils.compute import tally
from cusi.utils.log_handling import log_info
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.paths import search_run

ANNOTATIONS_FILE = "annotations.json"


def node_app(*, env_name: str, node: dict, expansions: dict):
    """Android: the app on a node's screen, named so its tasks name it right (the annotator, seeing only the screen,
    called other apps' settings "the Settings app" and many apps "the app"): the foreground package recorded on the
    step that reached the node (the env's true state), else, for a root, its scene's app. None elsewhere."""
    if env_name != "android":
        return None
    from cusi.agents.specs import ENV_SPECS, android_app_label
    if node.get("segment"):
        exp_id, _, last = node["segment"]
        true = ((expansions[exp_id]["steps"][last].get("info") or {}).get("true") or {})
        if true.get("activity"):
            return android_app_label(package=true["activity"].split("/")[0])
    scene = ENV_SPECS["android"].scenes.get(node.get("env_key"))
    return scene.kwargs.get("app_label") if scene else None


def unannotated(*, nodes: dict) -> list:
    """Ids of the nodes annotate_run would still annotate (it skips challenge pages)."""
    return [nid for nid, n in nodes.items() if not (n.get("flags") or {}).get("challenge_page") and (
        n.get("features") is None or n.get("node_notes_raw") is None
        or (n["parent"] is not None and (n.get("description") is None or n.get("edge_notes_raw") is None)))]


def annotate_run(*, directory: str, env_name: str, vlm, model_name: str, max_features: int, workers: int,
                 parameters: dict[str, Any]) -> dict:
    """workers: nodes annotated at a time (their VLM calls run concurrently; their updates are written one by one)."""
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

    def calls(nid: int) -> tuple:
        """One node's annotation calls (on a worker thread) -> (its new fields, the counts they add). Reads only the
        node's own fields and its parent's frame, which no other node's update changes."""
        node = nodes[nid]
        update, added = {}, []
        frame = FrameRef(path=os.path.join(directory, node["frame"])).pil()
        with tally() as compute:     # this node's annotation calls
            if node.get("features") is None:
                f = features(env_description=env_description, frame=frame, text_boxes=node.get("text_boxes") or [])
                if not f.pop("parsed"):
                    update["flags"] = {**(node.get("flags") or {}), "features_parse_failed": True}
                update.update(f)
                added.append("features")
            app = node_app(env_name=env_name, node=node, expansions=expansions)
            if node.get("node_notes_raw") is None:
                update.update(node_notes(env_description=env_description, frame=frame,
                                         text_boxes=node.get("text_boxes") or [], app=app))
                added.append("node_notes")
            if node["parent"] is not None and (node.get("description") is None or node.get("edge_notes_raw") is None):
                exp_id, first, last = node["segment"]
                steps = [step_from_row(directory=directory, row=r) for r in expansions[exp_id]["steps"][first:last + 1]]
                start_frame = FrameRef(path=os.path.join(directory, nodes[node["parent"]]["frame"]))
                if node.get("description") is None:
                    update.update(describer(env_description=env_description, steps=steps, start_frame=start_frame))
                    added.append("reports")
                if node.get("edge_notes_raw") is None:
                    update.update(edge_notes(env_description=env_description, start_frame=start_frame.pil(),
                                             end_frame=frame, steps=steps,
                                             description=update.get("description", node.get("description")),
                                             app=app))
                    added.append("edge_notes")
        if update:
            update["annotate_compute"] = compute.snapshot()
        return update, added

    def apply(nid: int, update: dict) -> None:
        """Write one node's fields (on this thread only). Task lists are merged here, from their current values: a
        node's own tasks keep the edge tasks its children already added, and an edge task joins its parent's list."""
        node = nodes[nid]
        if "node_tasks" in update:
            edge_tasks = [t for t in node.get("tasks") or [] if t["source"] == "edge"]
            update["tasks"] = [{"task": t, "source": "node"} for t in update["node_tasks"]] + edge_tasks
        rows = [{"id": nid, **update}]
        parent_update = {}
        if "edge_notes_raw" in update and update.get("edge_task"):
            parent = nodes[node["parent"]]
            parent_update["tasks"] = (parent.get("tasks") or []) + [
                {"task": update["edge_task"], "source": "edge", "child": nid, "segment": node["segment"]}]
            rows.append({"id": node["parent"], **parent_update})
        append_node_updates(directory=directory, rows=rows)
        node.update(update)
        if parent_update:
            nodes[node["parent"]].update(parent_update)

    todo = [nid for nid in sorted(nodes) if not (nodes[nid].get("flags") or {}).get("challenge_page")]
    log_info(f"annotate {directory}: {len(todo)} nodes, {workers} at a time", parameters=parameters)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(calls, nid): nid for nid in todo}
        for i, future in enumerate(as_completed(futures)):
            update, added = future.result()
            if update:
                apply(futures[future], update)
            for k in added:
                counts[k] += 1
            if (i + 1) % 20 == 0:
                log_info(f"annotate {directory}: {i + 1}/{len(todo)} nodes, {counts}", parameters=parameters)
    return counts


@click.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--run", "run_name", required=True)
@click.option("--model_name", required=True)
@click.option("--model_backend", default="vllm", show_default=True)
@click.option("--vllm_base_url", default=None)
@click.option("--max_features", default=8, show_default=True, help="Features listed per node.")
@click.option("--workers", default=8, show_default=True, help="Nodes annotated at a time.")
def main(env_name, run_name, model_name, model_backend, vllm_base_url, max_features, workers):
    from cusi.agents.vlm import AgentVLM
    from cusi.search.viewer import build_viewer
    parameters = load_parameters()
    directory = search_run(parameters=parameters, env=env_name, run=run_name)
    vlm = AgentVLM(model_name=model_name, model_backend=model_backend, vllm_base_url=vllm_base_url,
                   parameters=parameters)
    counts = annotate_run(directory=directory, env_name=env_name, vlm=vlm, model_name=model_name,
                          max_features=max_features, workers=workers, parameters=parameters)
    viewer = build_viewer(run_dir=directory, title=f"{env_name} / {run_name}")
    log_info(f"annotated {directory}: {counts}; viewer {viewer}", parameters=parameters)


if __name__ == "__main__":
    main()
