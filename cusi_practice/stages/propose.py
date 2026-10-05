"""Stage 1: propose tasks from each scene's first frame (GameBoyRL propose_tasks_zeroshot).

Output: proposals.jsonl, one {"scene", "tasks", "all_tasks", "response"} line per scene.
`tasks` is the first n_tasks of the parsed list (the scale decision: 3 per scene).
"""
import json
import os
from cusi_utils.log_handling import log_info, log_warn
from cusi_practice.envs import EnvPool, proposal_context
from cusi_practice.parsing import parse_list
from cusi_practice.prompts import PROPOSE_PROMPT, PROPOSE_TEXTS_BLOCK, fill
from cusi_practice.stages.common import PracticePaths, run_jobs


def propose(*, spec, pool: EnvPool, vlm, paths: PracticePaths, n_tasks: int, max_new_tokens: int,
            overwrite: bool, parameters: dict) -> None:
    out_path = paths.proposals
    done = {}
    if os.path.exists(out_path) and not overwrite:
        with open(out_path) as f:
            done = {r["scene"]: r for r in map(json.loads, f)}
    jobs = [s for s in spec.scenes if s not in done]
    if not jobs:
        log_info(f"propose: all {len(done)} scenes done ({out_path}).", parameters=parameters)
        return

    def one(scene: str, *, worker: int) -> dict:
        env = pool.get(worker=worker, scene=scene)
        obs, info = env.reset()
        actions, texts = proposal_context(env=env, obs=obs)
        texts_block = fill(PROPOSE_TEXTS_BLOCK, domain=spec.domain, TEXTS=texts) if texts else ""
        prompt = fill(PROPOSE_PROMPT, domain=spec.domain, ACTION_SPACE=actions, TEXTS_BLOCK=texts_block)
        frame = info.get("raw_frame", obs["frame"])
        output = vlm.infer(texts=prompt, images=[frame], max_new_tokens=max_new_tokens)["output"]
        tasks = parse_list(output, "Tasks")
        if not tasks:
            log_warn(f"propose: no tasks parsed for scene {scene}:\n{output}", parameters=parameters)
        return {"scene": scene, "tasks": tasks[:n_tasks], "all_tasks": tasks, "response": output}

    def save(scene, record):
        done[scene] = record
        with open(out_path + ".tmp", "w") as f:
            for s in spec.scenes:
                if s in done:
                    f.write(json.dumps(done[s]) + "\n")
        os.replace(out_path + ".tmp", out_path)
        log_info(f"propose [{scene}]: {record['tasks']}", parameters=parameters)

    run_jobs(jobs=jobs, fn=one, n_workers=min(spec.max_workers, len(jobs)), desc="propose", on_result=save,
             parameters=parameters)
