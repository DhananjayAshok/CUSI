# TOY (skill_discovery.md §1.5): throwaway first-build scaffolding, not part of the kept code.
# Strip it by deleting cusi_search/toy/, run_preexplore_toy.py, slurm/preexplore_toy.sh,
# scripts/slurm/preexplore_toy.sh, tests/*toy* and the TOY block in skill_discovery.md §1.5.
# Nothing outside those files imports this.
"""Toy threshold calibration: a short random run from each root, then
    cell_threshold = the cell_percentile-th percentile of each state's similarity to its nearest
                     other state in the run (50: about half the states would share a cell with
                     their nearest neighbour)
    node_threshold = the node_percentile-th percentile of the scorer values over the run
                     (90: about 1 in 10 steps becomes a node, plus segment ends)
The scorer runs against a scratch archive, so the search's archive is untouched.
"""
import numpy as np
from cusi_state import StateRecord, build_archive


def calibrate(*, envs: dict, encoder, scorer, steps_per_root: int, cell_percentile: float = 50.0,
              node_percentile: float = 90.0) -> dict:
    archive = build_archive(encoder=encoder)
    embs, values = [], []
    for env in envs.values():
        obs, info = env.reset()
        prev = StateRecord(obs=obs, info=info, embedding=encoder.encode_one(obs=obs, info=info))
        archive.add(prev.embedding)
        embs.append(prev.embedding)
        for _ in range(steps_per_root):
            obs, _r, term, trunc, info = env.step(env.sample_action())
            nxt = StateRecord(obs=obs, info=info, embedding=encoder.encode_one(obs=obs, info=info),
                              prev_image=prev.embedding.image)
            values.append(scorer.score(prev=prev, action=None, next=nxt, archive=archive, add=True)[0])
            embs.append(nxt.embedding)
            prev = nxt
            if term or trunc:
                break
    unique = []          # exact repeats (no-op steps) would pull the percentile to 1.0
    for e in embs:
        if all(encoder.similarity(a=e, b=u) < 0.9999 for u in unique):
            unique.append(e)
    embs = unique
    nn = []
    for i, a in enumerate(embs):
        sims = [encoder.similarity(a=a, b=b) for j, b in enumerate(embs) if j != i]
        if sims:
            nn.append(max(sims))
    out = {"cell_threshold": float(np.percentile(nn, cell_percentile)),
           "node_threshold": float(np.percentile(values, node_percentile)),
           "n_unique_states": len(embs), "nn_similarity": {p: float(np.percentile(nn, p)) for p in (10, 50, 90)},
           "values": {p: float(np.percentile(values, p)) for p in (10, 50, 90, 99)},
           "cell_percentile": cell_percentile, "node_percentile": node_percentile}
    return out
