"""
Live check that curiosity candidates (plans/next_plan.md §E.2) map to actions the env accepts: on each scene's start
screen, a sample of candidates of every type is executed from a saved state (Android: from the reset scene).
    bash scripts/container.sh python -m tests.curiosity_candidates_live --env web
"""
import collections
import click
import numpy as np
from cusi.utils.parameter_handling import load_parameters
from cusi.agents.specs import ENV_NAMES, ENV_SPECS
from cusi.search.curiosity.candidates import candidates_for


@click.command()
@click.option("--env", "env_name", required=True, type=click.Choice(ENV_NAMES))
@click.option("--per_type", default=3, show_default=True)
@click.option("--seed", default=0, show_default=True)
def main(env_name, per_type, seed):
    parameters = load_parameters()
    spec = ENV_SPECS[env_name]
    rng = np.random.default_rng(seed)
    failures = 0
    for scene in spec.scenes:
        env = spec.make_env(scene=spec.scenes[scene], worker=0, parameters=parameters)
        try:
            obs, info = env.reset()
            elements, cands = candidates_for(env_name=env_name, obs=obs, scene=scene)
            by_type = collections.defaultdict(list)
            for c in cands:
                by_type[c.type].append(c)
            print(f"== {scene}: {len(elements)} elements, {len(cands)} candidates: "
                  f"{ {t: len(v) for t, v in by_type.items()} }")
            sid = env.save_state() if env_name != "android" else None
            for t, cs in by_type.items():
                for k in rng.choice(len(cs), min(per_type, len(cs)), replace=False):
                    c = cs[int(k)]
                    if sid is not None:
                        env.load_state(state_id=sid)
                    else:
                        env.reset()
                    _o, _r, _te, _tr, inf = env.step(c.action)
                    ok = bool(inf.get("valid", True))
                    failures += not ok
                    target = elements[c.target].text[:30] if c.target is not None else ""
                    print(f"   {'ok  ' if ok else 'FAIL'} {t:12} {target!r:34} {c.argument or ''!s:20} "
                          f"{'' if ok else inf.get('error')}")
        finally:
            env.close()
    print(f"failures: {failures}")


if __name__ == "__main__":
    main()
