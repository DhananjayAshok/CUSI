"""
A search run directory, written as the search goes and read back to resume or extend it:

    config.json, nodes.jsonl, expansions.jsonl, embeddings/<key>.pt, frames/<key>.zip, states/, archive/

<key> is "roots" or an expansion id. The tree files are append-only; the archive save, which records how many
expansions and nodes it includes (commit.json), is the commit point. On load, anything past the last commit is
removed, so the tree and the archive always agree and a crash loses at most the iteration in progress.
"""
import json
import os
import shutil
from typing import Optional
import torch
from cusi.agents.frames import FrameRef, FrameStore
from cusi.search.tree import Step
from cusi.utils.run_dir import open_run_dir

NODES_FILE = "nodes.jsonl"
EXPANSIONS_FILE = "expansions.jsonl"
COMMIT_FILE = "commit.json"
ARCHIVE_DIR = "archive"
ROOTS = "roots"
#: Config keys that may differ between launches of one run: max_expansions grows to extend a search.
OPERATIONAL_KEYS = {"max_expansions", "vllm_base_url"}


def _jsonable(value):
    if hasattr(value, "item"):       # numpy / torch scalars
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _rows(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class SearchStore:
    def __init__(self, *, directory: str, config: Optional[dict] = None, overwrite: bool = False,
                 ignore_config_violation: bool = False) -> None:
        """A run directory may only be resumed with the config it was started with (cusi.utils.run_dir)."""
        self.directory = directory
        open_run_dir(directory=directory, config=config, overwrite=overwrite,
                     ignore_config_violation=ignore_config_violation, operational_keys=OPERATIONAL_KEYS)
        for sub in ("frames", "embeddings", "states"):
            os.makedirs(self._path(sub), exist_ok=True)
        self._n_nodes = 0           # nodes already written

    def _path(self, *parts) -> str:
        return os.path.join(self.directory, *parts)

    def _append(self, *, name: str, rows: list) -> None:
        with open(self._path(name), "a") as f:
            for row in rows:
                f.write(json.dumps(row, default=_jsonable) + "\n")

    # ------------------------------------------------------------ writing

    def commit(self, *, search, expansion=None, record: Optional[dict] = None) -> None:
        """Write the nodes made since the last commit with their frames, embeddings and states, then the
        expansion (None: the roots), then the archive, which commits them."""
        key = ROOTS if expansion is None else str(expansion.id)
        nodes = search.tree.nodes[self._n_nodes:]
        frames = FrameStore(directory=self._path("frames"), name=f"{key}.zip")
        node_frames = {n.id: f"frames/{frames(n.frame)}" for n in nodes}
        step_frames = [f"frames/{frames(s.frame)}" for s in (expansion.steps if expansion is not None else [])]
        frames.close()
        torch.save({n.id: n.embedding for n in nodes}, self._path("embeddings", f"{key}.pt"))
        rows = []
        for n in nodes:
            state = None
            if n.state_id is not None:
                state = search.envs[n.env_key].export_state(state_id=n.state_id, directory=self._path("states"))
            rows.append({"id": n.id, "parent": n.parent, "depth": n.depth, "env_key": n.env_key, "state": state,
                         "embedding": f"embeddings/{key}.pt#{n.id}", "frame": node_frames[n.id], "texts": n.texts,
                         "value": n.value, "segment": n.segment, "prior": n.prior, "prior_reason": n.prior_reason,
                         "prior_raw": n.prior_raw, "flags": n.flags, "created_at": n.created_at, "replay": n.replay})
        self._append(name=NODES_FILE, rows=rows)
        if expansion is not None:
            steps = [{"action": s.action, "frame": ref, "texts": s.texts, "info": s.info, "value": s.value,
                      "components": s.components, "node_id": s.node_id, "extra": s.extra}
                     for s, ref in zip(expansion.steps, step_frames)]
            self._append(name=EXPANSIONS_FILE, rows=[{
                "id": expansion.id, "node_id": expansion.node_id, "iteration": expansion.iteration,
                "children": expansion.children, "yield": expansion.yield_value, "stats": expansion.stats,
                "record": record, "steps": steps}])
        self._save_archive(search=search)
        self._n_nodes = len(search.tree.nodes)

    def write_flags(self, *, node) -> None:
        """Flags set after a node was written (e.g. unrestorable); the latest row for a node wins."""
        self._append(name=NODES_FILE, rows=[{"id": node.id, "flags": node.flags}])

    def _save_archive(self, *, search) -> None:
        new, old, cur = self._path(ARCHIVE_DIR + ".new"), self._path(ARCHIVE_DIR + ".old"), self._path(ARCHIVE_DIR)
        for d in (new, old):
            if os.path.exists(d):
                shutil.rmtree(d)
        search.archive.save(directory=new)
        with open(os.path.join(new, COMMIT_FILE), "w") as f:
            json.dump({"n_expansions": len(search.tree.expansions), "n_nodes": len(search.tree.nodes),
                       "iteration": search.iteration, "total_steps": search.total_steps}, f)
        if os.path.exists(cur):
            os.rename(cur, old)
        os.rename(new, cur)
        if os.path.exists(old):
            shutil.rmtree(old)

    # ------------------------------------------------------------ reading

    def _latest_commit(self) -> Optional[tuple]:
        """(directory, commit) of the most recent complete archive save, or None."""
        best = None
        for name in (ARCHIVE_DIR, ARCHIVE_DIR + ".new", ARCHIVE_DIR + ".old"):
            path = self._path(name, COMMIT_FILE)
            if os.path.exists(path):
                with open(path) as f:
                    commit = json.load(f)
                if best is None or commit["n_expansions"] > best[1]["n_expansions"]:
                    best = (self._path(name), commit)
        return best

    def _truncate(self, *, n_nodes: int, n_expansions: int) -> None:
        """Drop everything written after the commit, so the next writes continue from it."""
        nodes = [r for r in _rows(self._path(NODES_FILE)) if r["id"] < n_nodes]
        exps = [r for r in _rows(self._path(EXPANSIONS_FILE)) if r["id"] < n_expansions]
        for name, rows in ((NODES_FILE, nodes), (EXPANSIONS_FILE, exps)):
            if not os.path.exists(self._path(name)):
                continue
            with open(self._path(name), "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
        keys = {ROOTS} | {str(i) for i in range(n_expansions)} if n_nodes else set()
        for sub, ext in (("frames", ".zip"), ("embeddings", ".pt")):
            for name in os.listdir(self._path(sub)):
                if not (name.endswith(ext) and name[:-len(ext)] in keys):
                    os.remove(self._path(sub, name))
        states = {r["state"] for r in nodes if r.get("state")}
        for name in os.listdir(self._path("states")):
            if name not in states:
                os.remove(self._path("states", name))

    def load(self, *, search) -> bool:
        """Rebuild search (tree, stats, archive, counters, env states) from the last commit; False if there is none."""
        latest = self._latest_commit()
        if latest is None:
            self._truncate(n_nodes=0, n_expansions=0)
            return False
        archive_dir, commit = latest
        if archive_dir != self._path(ARCHIVE_DIR):
            if os.path.exists(self._path(ARCHIVE_DIR)):
                shutil.rmtree(self._path(ARCHIVE_DIR))
            os.rename(archive_dir, self._path(ARCHIVE_DIR))
        for name in (ARCHIVE_DIR + ".new", ARCHIVE_DIR + ".old"):
            if os.path.exists(self._path(name)):
                shutil.rmtree(self._path(name))
        n_nodes, n_exps = commit["n_nodes"], commit["n_expansions"]
        self._truncate(n_nodes=n_nodes, n_expansions=n_exps)

        tree, embeddings, state_files = search.tree, {}, {}
        for row in _rows(self._path(NODES_FILE)):
            if "parent" not in row:                      # a flags update
                tree.nodes[row["id"]].flags = dict(row["flags"])
                continue
            file, nid = row["embedding"].split("#")
            if file not in embeddings:
                embeddings[file] = torch.load(self._path(file), weights_only=False)
            node = tree.add_node(parent=row["parent"], env_key=row["env_key"], state_id=None,
                                 frame=FrameRef(path=self._path(row["frame"])), texts=row["texts"],
                                 embedding=embeddings[file][int(nid)], value=row["value"],
                                 segment=None if row["segment"] is None else tuple(row["segment"]),
                                 created_at=row["created_at"], replay=row["replay"])
            node.prior, node.prior_reason, node.prior_raw = row["prior"], row["prior_reason"], row["prior_raw"]
            node.flags = dict(row["flags"])
            if row["state"] is not None:
                state_files[node.id] = row["state"]
        for row in _rows(self._path(EXPANSIONS_FILE)):
            exp = tree.add_expansion(node_id=row["node_id"], iteration=row["iteration"])
            exp.steps = [Step(action=s["action"], frame=FrameRef(path=self._path(s["frame"])), texts=s["texts"],
                              info=s["info"], value=s["value"], components=s["components"], node_id=s["node_id"],
                              extra=s["extra"]) for s in row["steps"]]
            exp.children, exp.yield_value, exp.stats = list(row["children"]), row["yield"], row["stats"]
            search.stats.record_yield(tree=tree, node_id=exp.node_id, value=exp.yield_value)
            tree.nodes[exp.node_id].n_expanded += 1
            if row["record"] is not None:
                search.records.append(row["record"])
        search.archive.load_from(directory=self._path(ARCHIVE_DIR))
        search.iteration, search.total_steps = commit["iteration"], commit["total_steps"]

        for nid, state_file in state_files.items():
            node, path = tree.nodes[nid], self._path("states", state_file)
            if os.path.exists(path):
                node.state_id = search.envs[node.env_key].import_state(path=path)
            else:
                node.flags["unrestorable"] = True
        self._n_nodes = len(tree.nodes)
        return True
