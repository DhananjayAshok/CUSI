"""
Offline calibration of the selection (plans/tree_search_plan.md, plans/next_plan.md §B step 6): every saved decision
record holds each candidate's Q, N and P, so the selection distribution can be recomputed for other c, tau and P
normalisations without rerunning the search; step values give node_threshold quantiles.

    python -m cusi.search.calibrate --env gameboy --run <run>
"""
import json
import os
import click
import numpy as np
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.paths import search_run


def decision_stats(*, records: list, c: float, tau: float, normalise_p: bool) -> dict:
    """Mean over decisions: entropy, effective number of candidates (exp entropy), probability on unexpanded
    candidates and on the single top candidate."""
    ent, eff, fresh, top = [], [], [], []
    for r in records:
        q, n, p = np.array(r["q"]), np.array(r["n"]), np.array(r["p"], dtype=float)
        if normalise_p:
            p = p / max(p.sum(), 1e-12)
        e = q + c * p * np.sqrt(r["n_total"]) / (1 + n)
        z = e / tau
        pr = np.exp(z - z.max())
        pr /= pr.sum()
        h = float(-(pr * np.log(pr + 1e-12)).sum())
        ent.append(h)
        eff.append(np.exp(h))
        fresh.append(float(pr[n == 0].sum()))
        top.append(float(pr.max()))
    return {"entropy": round(float(np.mean(ent)), 3), "effective_candidates": round(float(np.mean(eff)), 2),
            "p_unexpanded": round(float(np.mean(fresh)), 3), "p_top": round(float(np.mean(top)), 3)}


@click.command()
@click.option("--env", "env_name", required=True)
@click.option("--run", "run_name", required=True)
def main(env_name, run_name):
    parameters = load_parameters()
    directory = search_run(parameters=parameters, env=env_name, run=run_name)
    rows = [json.loads(l) for l in open(os.path.join(directory, "expansions.jsonl")) if l.strip()]
    records = [r["record"] for r in rows if r.get("record")]
    values = np.array([s["value"] for r in rows for s in r["steps"]])
    qs = [0.5, 0.75, 0.9, 0.95, 0.99]
    print(f"{len(rows)} expansions, {len(values)} steps; step value quantiles "
          f"{dict(zip(qs, np.round(np.quantile(values, qs), 4).tolist()))}")
    for t in (0.05, 0.1, 0.2):
        print(f"  node_threshold {t}: {float((values >= t).mean()):.3f} of steps become nodes")
    print(f"recorded decisions: {len(records)}; mean candidates {np.mean([len(r['q']) for r in records]):.1f}")
    for normalise_p in (False, True):
        for c in (0.25, 1.0, 2.0):
            for tau in (0.02, 0.05, 0.1, 0.3, 1.0):
                s = decision_stats(records=records, c=c, tau=tau, normalise_p=normalise_p)
                print(f"  P {'normalised' if normalise_p else 'raw':10} c={c:<4} tau={tau:<5} {s}")


if __name__ == "__main__":
    main()
