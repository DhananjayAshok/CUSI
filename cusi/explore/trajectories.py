"""
Curiosity tasks: high-novelty replay trajectories, grouped and turned into a practice attempts directory.
"""
import os
from typing import Optional
import numpy as np
from cusi.utils.parsing import parse_key_value
from cusi.utils.log_handling import log_info, log_warn
from cusi.agents.records import CallRecord, EncodedImage, LegReport, StepRecord
from cusi.practice.prompts import Domain, fill
from cusi.explore.replay import load_episodes

SIMILARITY_THRESHOLD = 0.05


# --------------------------------------------------------------------------- outliers

def rescore(*, episodes: list, text_alpha: float, has_text: bool) -> list:
    """Recompute a replay's intrinsic rewards in place with the current curiosity code."""
    import torch
    from cusi.explore.curiosity import CuriosityModule
    from cusi.state import NoveltyArchive, StateEmbedding, StateRecord, build_scorer
    scorer = build_scorer(novelty_scorer="combination", env_name="web" if has_text else "gameboy",
                          region_alpha=text_alpha if has_text else 0.0)
    module = CuriosityModule(scorer=scorer, archive=NoveltyArchive(), invalid_action_penalty=0.0)
    for ep in episodes:
        module.reset()
        for t, step in enumerate(ep):
            emb = torch.nn.functional.normalize(torch.tensor(np.asarray(step["embedding"], dtype=np.float32)), dim=-1)
            record = StateRecord(obs={"texts": step["texts"]}, info={},
                                 embedding=StateEmbedding(image=emb, text=None, image_name="stored", text_name="none"))
            r = module.get_reward(prev=None, action=None, next=record)
            if t > 0:
                step["reward_frame"], step["reward_text"] = r["frame"], r["region"]
                step["reward"] = step.get("reward_ext", 0.0) + r["total"]
    return episodes


def high_novelty_trajectories(*, episodes: list, outlier_threshold: float = 2.5,
                              max_trajectory_length: int = 30) -> list:
    """Back-traced trajectories ending at each transition whose reward z-score exceeds outlier_threshold."""
    transitions = [(e, t) for e, ep in enumerate(episodes) for t in range(1, len(ep))]
    if not transitions:
        return []
    rewards = np.array([episodes[e][t]["reward"] for e, t in transitions], dtype=np.float64)
    z = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    out = []
    for (e, t), zi in zip(transitions, z):
        if zi <= outlier_threshold:
            continue
        start = max(0, t - max_trajectory_length)
        out.append({"steps": episodes[e][start:t + 1], "z": float(zi), "episode": e, "end_step": t})
    return out


# --------------------------------------------------------------------------- grouping

def _emb(step) -> np.ndarray:
    v = np.asarray(step["embedding"], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _similar(a: np.ndarray, b: np.ndarray) -> bool:
    return float(a @ b) >= 1 - SIMILARITY_THRESHOLD


def _overlap(frames_1: list, frames_2: list) -> bool:
    return any(_similar(a, b) for a in frames_1 for b in frames_2)


def _check_frames(final_frames: list, n_sample: int, rng) -> list:
    if len(final_frames) > n_sample:
        return [final_frames[i] for i in rng.choice(len(final_frames), size=n_sample, replace=False)]
    return final_frames


def _merge_groups(group_1: list, group_2: list, rng, n_sample: int = 5) -> list:
    for first in group_1:
        if not group_2:
            return group_1
        check_1 = _check_frames(first["final_frames"], n_sample, rng)
        rest = []
        for second in group_2:
            if _overlap(check_1, _check_frames(second["final_frames"], n_sample, rng)):
                first["indexes"].extend(second["indexes"])
                first["final_frames"].extend(second["final_frames"])
            else:
                rest.append(second)
        group_2 = rest
    return group_1 + group_2


def _infer_groups(trajectories: list, indices: list, rng) -> list:
    if len(indices) == 1:
        return [{"indexes": [indices[0]], "final_frames": [_emb(trajectories[indices[0]]["steps"][-1])]}]
    mid = len(indices) // 2
    return _merge_groups(_infer_groups(trajectories, indices[:mid], rng),
                         _infer_groups(trajectories, indices[mid:], rng), rng)


def snip_cycles(*, steps: list, protected_lookback: int = 3) -> list:
    """Remove revisits (a frame equal to an earlier one) outside the protected tail."""
    n = len(steps)
    if n <= protected_lookback:
        return steps
    working, protected = list(steps[:n - protected_lookback]), list(steps[n - protected_lookback:])
    while True:
        seen, cycle = [], None
        for i, s in enumerate(working):
            e = np.asarray(s["embedding"], dtype=np.float32)
            for j, f in seen:
                if np.abs(e - f).max() < 1e-3:
                    cycle = (j, i)
                    break
            if cycle:
                break
            seen.append((i, e))
        if cycle is None:
            return working + protected
        start, end = cycle
        # Keep the first visit (with the action that reached it) and drop the loop after it;
        # the step after the loop was reached from frame `end`, which equals frame `start`.
        working = working[:start + 1] + working[end + 1:]


def group_trajectories(*, trajectories: list, z_min: float = 3.0, protected_lookback: int = 3,
                       seed: int = 0) -> list:
    """Groups (lists) of snipped trajectories with z >= z_min."""
    selected = [t for t in trajectories if z_min is None or t["z"] >= z_min]
    if not selected:
        return []
    rng = np.random.default_rng(seed)
    groups = _infer_groups(selected, list(range(len(selected))), rng)
    return [[{**selected[i], "steps": snip_cycles(steps=selected[i]["steps"], protected_lookback=protected_lookback)}
             for i in g["indexes"]] for g in groups]


# --------------------------------------------------------------------------- infer tasks

INFER_PROMPT = """You are analysing multiple screenshots in sequence from [DOMAIN].

Over the course of some of these [SCREEN]s, a single primary task may have been performed by the [ACTOR], with the task being completed either at the very end or in some [SCREEN] close to the end.
Describe, with a single phrase, the action or task the [ACTOR] performed over the course these [SCREEN]s? Do not use conjunctions like "and" or "while" in your description. If there are multiple distinct tasks that seem to be happening, try to describe the whole subtrajectory wholistically and omit the less important subtasks. If there is no clear task, say "NO TASK".
Be specific but concise, each task should be a single, specific and meaningful action and not trivial. Describe only what is clearly supported by the evidence above. Make the task description as unambiguous as possible — include enough distinguishing detail that it cannot be confused with other similar tasks that could occur in the same [DOMAIN_SHORT].

Always try to pick the longest horizon, most multistep version of the task that is present in the trajectory. If there is no clear task, respond with "NO TASK"
Otherwise, respond in exactly this format:
Visual Description: <a description of the individual [SCREEN]s and changes that occur from leftmost [SCREEN] to rightmost [SCREEN]>
Reasoning: <one single, short sentence describing your thinking. Reference visual evidence of the key [SCREEN]s and overall actions that led you to infer this task.>
Task: <one or two sentence description of what the [ACTOR] did or is doing>
Start: <integer index of starting [SCREEN]> this is to indicate when the [ACTOR] seems to be acting with the intent to perform the task, not simply the step right before the task is executed. This may be several [SCREEN]s before the task is completed, and will depend on the specific task and context.
End: <integer index of [SCREEN] where task is performed or executed or completed or first detected>
[STOP]"""

REFINE_PROMPT = """You are given a description of what a [ACTOR] did over the course of some [SCREEN]s:
"[CANDIDATE_TASK]"

First judge whether this is a well-formed task. A VALID task is SPECIFIC and CONCRETE — a
clearly-defined action with an unambiguous completion state that could be checked from the
screen (e.g. "select the diamond from the inventory", "open the cellar door", "exit the taxi").
Judge it INVALID if it is too generic or vague to complete exactly — e.g. "navigate the
environment", "interact with an object", "explore the area", "manage inventory", "pick up an
object" — cases where many different behaviours would all satisfy it.

If VALID, rewrite it as a concise imperative instruction:
- Use second-person imperative tone (no subject).
- Keep it short (under 10 words if possible).
- Do not add any detail that was not in the original description.

Respond in exactly this format:
Verdict: <VALID or INVALID>
Task: <imperative task string if VALID, or NONE if INVALID>
[STOP]"""

DISTILL_PROMPT = """You are given several candidate descriptions of a task performed on [DOMAIN], all inferred from similar states:

[CANDIDATE_LIST]

Generate a single, unifying task string that captures the core commonality between all of these candidates, while omitting any extraneous detail or noise. Use imperative tone.

The distilled task must stay SPECIFIC, CONCRETE and UNAMBIGUOUS — a clearly-defined action with a checkable completion state (e.g. "select the diamond from the inventory", "open the cellar door"). Do not over-generalise into a vague or generic instruction (e.g. "interact with an object", "navigate the environment", "manage inventory") that many different behaviours would satisfy. Keep the most specific meaning shared by the candidates.

Respond in exactly this format:
Reasoning: <one single, short sentence describing your thinking. Reference the commonalities between the candidates that led you to infer this distilled task.>
Task: <single distilled imperative task string>
[STOP]
"""


def _got_bigger(subject: str, refined: str, multiplier: float = 1.5) -> bool:
    return refined.count(" ") > multiplier * subject.count(" ")


def infer_task(*, steps: list, vlm, domain: Domain, max_new_tokens: int, lookback: int = 8) -> Optional[str]:
    """INFER on the last `lookback` frames, then REFINE. None for NO TASK / invalid."""
    frames = [s["frame"] for s in steps][-lookback:]
    output = vlm.infer(texts=fill(INFER_PROMPT, domain=domain, DOMAIN_SHORT=domain.domain.split()[-1]),
                       images=frames, max_new_tokens=max_new_tokens)["output"].lower()
    if "no task" in output:
        return None
    task = parse_key_value(output, "Task")
    if task is None or "no task" in task:
        return None
    refine = vlm.infer(texts=fill(REFINE_PROMPT, domain=domain, CANDIDATE_TASK=task),
                       max_new_tokens=max_new_tokens)["output"].lower()
    verdict = parse_key_value(refine, "Verdict") or ""
    if "invalid" in verdict or "valid" not in verdict:
        return None
    refined = parse_key_value(refine, "Task")
    if refined is None or refined.strip() in ("", "none"):
        return None
    return task if _got_bigger(task, refined) else refined


def distill(*, tasks: list, vlm, domain: Domain, max_new_tokens: int) -> str:
    output = vlm.infer(texts=fill(DISTILL_PROMPT, domain=domain, CANDIDATE_LIST="\n".join(f"- {t}" for t in tasks)),
                       max_new_tokens=max_new_tokens)["output"].lower()
    return parse_key_value(output, "Task") or tasks[0]


def canonical(task: str) -> str:
    return " ".join(task.lower().split()).strip(" .!?\"'")


def trajectory_to_leg(*, steps: list, task: str, env_name: str) -> LegReport:
    """A LegReport holding the trajectory frames, as the guidance stage reads them."""
    report = LegReport(env_name=env_name, task=task, hint=None, max_steps=len(steps) - 1,
                       initial_frame=EncodedImage.of(steps[0]["frame"]), termination_reason="curiosity")
    for prev, cur in zip(steps[:-1], steps[1:]):
        call = CallRecord(tag="action", images=[], response=cur.get("generated") or "", prompt="", decision="step")
        call.steps.append(StepRecord(frame_before=EncodedImage.of(prev["frame"]),
                                     frame_after=EncodedImage.of(cur["frame"]), action_text=cur.get("generated") or "",
                                     parsed_action=cur.get("parsed_action"), valid=bool(cur.get("valid", True)),
                                     error=None, reward=float(cur.get("reward", 0.0))))
        report.calls.append(call)
    return report


def curiosity_tasks(*, replay_dir: str, scene: str, env_name: str, vlm, domain: Domain, out_attempts_dir: str,
                    max_new_tokens: int = 2000, outlier_threshold: float = 2.5, z_min: float = 3.0,
                    max_trajectories_per_group: int = 3, lookback: int = 8, max_groups: Optional[int] = None,
                    rescore_text_alpha: Optional[float] = None, seed: int = 0, parameters: dict = None) -> dict:
    """Replay -> outlier trajectories -> groups -> tasks, written as an attempts dir. Returns stats."""
    from cusi.practice.stages.common import atomic_json, atomic_pickle
    episodes = load_episodes(directory=replay_dir)
    if rescore_text_alpha is not None:
        has_text = any(v for ep in episodes for st in ep[:1] for v in st["texts"].values())
        rescore(episodes=episodes, text_alpha=rescore_text_alpha, has_text=has_text)
    trajectories = high_novelty_trajectories(episodes=episodes, outlier_threshold=outlier_threshold)
    groups = group_trajectories(trajectories=trajectories, z_min=z_min, seed=seed)
    groups.sort(key=lambda g: -max(t["z"] for t in g))
    if max_groups is not None:
        groups = groups[:max_groups]
    log_info(f"curiosity: {len(episodes)} episodes, {len(trajectories)} outlier trajectories, {len(groups)} groups",
             parameters=parameters)
    rng = np.random.default_rng(seed)
    success, legs, seen = {}, {}, {}
    for gi, group in enumerate(groups):
        chosen = [group[i] for i in rng.choice(len(group), size=min(len(group), max_trajectories_per_group),
                                               replace=False)]
        tasks, used = [], []
        for traj in chosen:
            task = infer_task(steps=traj["steps"], vlm=vlm, domain=domain, max_new_tokens=max_new_tokens,
                              lookback=lookback)
            if task:
                tasks.append(task)
                used.append(traj)
        if not tasks:
            log_warn(f"curiosity: no task for group {gi}", parameters=parameters)
            continue
        task = distill(tasks=tasks, vlm=vlm, domain=domain, max_new_tokens=max_new_tokens)
        key = canonical(task)
        if key in seen:      # identical tasks share one group
            continue
        group_idx = f"cur{gi}"
        seen[key] = group_idx
        success[group_idx] = {"task": task, "scene": scene, "candidates": tasks, "n_trajectories": len(group)}
        legs[group_idx] = trajectory_to_leg(steps=used[0]["steps"], task=task, env_name=env_name)
        log_info(f"curiosity task [{group_idx}]: {task}  (from {tasks})", parameters=parameters)
    os.makedirs(out_attempts_dir, exist_ok=True)
    atomic_json(success, os.path.join(out_attempts_dir, "success.json"))
    atomic_pickle(legs, os.path.join(out_attempts_dir, "legs.pkl"))
    stats = {"episodes": len(episodes), "outlier_trajectories": len(trajectories), "groups": len(groups),
             "tasks": len(success)}
    atomic_json(stats, os.path.join(out_attempts_dir, "curiosity_stats.json"))
    return stats
