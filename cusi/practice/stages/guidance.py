"""Stage 3: distil guidance from each successful attempt (GameBoyRL infer_guidance).

8-frame slices of the successful leg (initial frame + the frame after each step) become a
summary, a goal condition and steps, then one consolidation call.

Output: guidance.json, {group: {task, scene, goal_condition, guidance: {summary, steps}}}.
"""
import os
import pickle
from cusi.utils.log_handling import log_info, log_warn
from cusi.practice.judging import infer_guidance
from cusi.practice.stages.common import PracticePaths, atomic_json, load_json, run_jobs


def guidance(*, spec, vlm, paths: PracticePaths, max_new_tokens: int, max_obs_at_once: int, overwrite: bool,
             parameters: dict) -> None:
    out_path = paths.guidance
    if os.path.exists(out_path) and not overwrite:
        log_info(f"guidance: already done ({out_path}).", parameters=parameters)
        return
    success = load_json(os.path.join(paths.attempts_dir, "success.json"), None)
    if success is None:
        log_warn("guidance: no attempts/success.json; run the attempt stage first.", parameters=parameters)
        return
    if not success:
        log_warn(f"guidance: no successful attempts for {spec.name}; nothing to practise.", parameters=parameters)
        atomic_json({}, out_path)
        return
    with open(os.path.join(paths.attempts_dir, "legs.pkl"), "rb") as f:
        legs = pickle.load(f)
    ckpt = out_path.replace(".json", "_checkpoint.json")
    out = {} if overwrite else load_json(ckpt, {})
    jobs = [g for g in success if g not in out]

    def one(group, *, worker: int):
        frames = legs[group].frames()
        return infer_guidance(frames=frames, task=success[group]["task"], vlm=vlm, domain=spec.domain,
                              max_new_tokens=max_new_tokens, max_obs_at_once=max_obs_at_once)

    def save(group, parsed):
        if parsed is None:
            log_warn(f"guidance: could not generate guidance for {group}.", parameters=parameters)
            return
        goal_condition = parsed.pop("goal_condition", "")
        out[group] = {"task": success[group]["task"], "scene": success[group]["scene"],
                      "goal_condition": goal_condition, "guidance": parsed}
        atomic_json(out, ckpt)

    run_jobs(jobs=jobs, fn=one, n_workers=8, desc="guidance", on_result=save, parameters=parameters)
    atomic_json(out, out_path)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    log_info(f"guidance: {len(out)}/{len(success)} groups -> {out_path}", parameters=parameters)
