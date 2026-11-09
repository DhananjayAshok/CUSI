"""
The generic World Action executor: an env's own executor class with its decisions replaced by a WorldActionModel's
plan(). Everything else of the env's executor stays (history policy, done check, invalid handling), so the only
difference between a base run and a world-action run is who decides. plan() returns actions; the executor executes
them one env step at a time, replanning when the plan is used up, and reports every step back through observe().

    cls = world_action_executor_class(base=GameBoyExecutor, model=model)
    executor = cls(env=env, vlm=vlm, previous_history=..., parameters=parameters)
"""
from typing import Any, Optional
from cusi.agents.executors.base import Decision
from cusi.world_action.base import Plan, WorldActionModel


def world_action_executor_class(*, base: type, model: WorldActionModel, shadow_base: bool = True) -> type:
    """shadow_base: before each plan, the base executor's own policy also decides on the same screen (a recorded
    call, tag "shadow_<tag>", never executed), so the record shows where the world-action policy diverges from it."""
    class WorldActionExecutor(base):
        name = f"{base.name}+{model.name}"
        world_action = model
        SHADOW_BASE = shadow_base

        def _shadow_decision(self, *, obs: dict, info: dict, error: Optional[str], hint: Optional[str]) -> dict:
            """The base policy's decision from this state, recorded as an auxiliary call and not executed."""
            decision = super().decide(obs=obs, info=info, error=error, hint=hint)
            call = self._current_call
            call.tag, call.decision = f"shadow_{call.tag}", None
            call.extra["shadow_base"] = True
            return {"base_action": decision.action_text, "base_invalid": decision.invalid,
                    "base_reasoning": getattr(self, "_last_reasoning", None)}

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self._plan: Optional[Plan] = None
            self._queued: list = []

        def begin(self, *, task: str, hint: Optional[str], max_steps: int, obs: dict, info: dict) -> None:
            super().begin(task=task, hint=hint, max_steps=max_steps, obs=obs, info=info)
            self._plan, self._queued = None, []
            model.attach_env(self.env)
            model.begin(task=task, hint=hint, obs=obs, info=info)

        def history(self) -> Any:
            return {"base": super().history(), "world_action": model.history()}

        def restore_history(self, history: Any) -> None:
            if isinstance(history, dict) and "world_action" in history:
                super().restore_history(history["base"])
                model.restore_history(history["world_action"])
            else:
                super().restore_history(history)

        def decide(self, *, obs: dict, info: dict, error: Optional[str], hint: Optional[str]) -> Decision:
            if not self._queued:
                shadow = self._shadow_decision(obs=obs, info=info, error=error, hint=hint) if self.SHADOW_BASE else {}
                plan = model.plan(obs=obs, info=info, error=error, call=self.call)
                plan.record.update(shadow)
                if shadow:
                    plan.record["diverged"] = None if plan.invalid is not None or shadow["base_action"] is None \
                        else shadow["base_action"] != (plan.actions or [None])[0]
                if self._current_call is not None:
                    self._current_call.extra["world_action"] = plan.record
                # the done check reads the last reasoning: the executed decision's, not the shadow's
                if plan.record.get("reasoning"):
                    self._last_reasoning = plan.record["reasoning"]
                if plan.invalid is not None:
                    return Decision(invalid=plan.invalid, error=plan.error)
                if plan.finish:
                    return Decision(finish=True, answer=plan.answer)
                self._plan, self._queued = plan, list(plan.actions)
            return Decision(action_text=self._queued.pop(0))

        def after_step(self, *, decision, obs_before, info_before, obs_after, info_after, step) -> None:
            model.observe(action_text=decision.action_text, obs_before=obs_before, info_before=info_before,
                          obs_after=obs_after, info_after=info_after, valid=bool(step.valid))
            if not step.valid:
                self._queued = []          # a rejected action invalidates the rest of the plan
            super().after_step(decision=decision, obs_before=obs_before, info_before=info_before,
                               obs_after=obs_after, info_after=info_after, step=step)

        def after_invalid(self, *, decision, obs, info) -> None:
            self._queued = []
            super().after_invalid(decision=decision, obs=obs, info=info)

    WorldActionExecutor.__name__ = f"WorldAction{base.__name__}"
    return WorldActionExecutor


def episode_summary(*, legs: list) -> dict:
    """Per-episode diagnostics over the world-action decisions of a run's legs (for EvalRow.extra)."""
    recs = [c.extra["world_action"] for leg in legs for c in leg.calls
            if getattr(c, "extra", None) and "world_action" in c.extra]
    scored = [r for r in recs if r.get("predicted_pixel_error") is not None]
    changed = [r for r in scored if r.get("frame_changed")]
    import numpy as np
    def mean(key, rows):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None
    return {"wa_decisions": len(recs), "wa_invalid_choices": sum(r.get("chosen") is None for r in recs),
            "wa_scored": len(scored), "wa_changed": len(changed),
            "wa_pred_pixel_error": mean("predicted_pixel_error", changed),
            "wa_copy_pixel_error": mean("copy_pixel_error", changed),
            "wa_recon_pixel_error": mean("reconstruction_pixel_error", changed),
            "wa_pred_beats_copy": mean("_beats", [{"_beats": float(r["predicted_pixel_error"] < r["copy_pixel_error"])}
                                                   for r in changed]),
            "wa_chosen_is_hindsight_best": mean("_h", [{"_h": float(r["chosen_is_hindsight_best"])} for r in scored]),
            "wa_candidate_spread": mean("candidate_spread", recs),
            "wa_chosen_beats_copy": mean("_b", [{"_b": float(r["chosen_beats_copy"])} for r in changed
                                                 if r.get("chosen_beats_copy") is not None]),
            "wa_diverged": mean("_d", [{"_d": float(r["diverged"])} for r in recs if r.get("diverged") is not None]),
            "wa_base_invalid": sum(bool(r.get("base_invalid")) for r in recs),
            "wa_base_action_pred_sim": mean("base_action_predicted_similarity", scored),
            "wa_chosen_pred_sim": mean("chosen_predicted_similarity", scored),
            "wa_choice_counts": dict(zip(*np.unique([r["candidates"][r["chosen"]] for r in recs if r.get("chosen") is not None],
                                                    return_counts=True))) if any(r.get("chosen") is not None for r in recs) else {}}
