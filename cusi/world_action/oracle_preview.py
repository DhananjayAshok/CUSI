"""
The oracle preview: the base executor proposes, the env's exact lookahead (TextActionEnv.peek) shows what the
proposal would do, and a check call on (screen now, screen after) keeps it or sends the executor feedback to decide
again. Any env: GameBoy (GameBoyWorlds' emulator peek), Web (a sandbox browser), Android (emulator snapshots).

Per decision, up to MAX_PROPOSALS proposals:
  1. policy(): the executor's own decision on the current state (its full prompt and memory).
  2. finish / invalid: returned as is (a finish is never previewed; a peek of it would leak the env's reward).
  3. peek the action: the next screen and the env's valid / error / changed. The reward is never read.
  4. kept without a preview if the executor proposes again an action the check rejected for this decision (it insists
     after the feedback; an action the env refuses or that changes nothing stays rejected, and on GameBoy the feedback
     then names the buttons that would change the screen); rejected without a check call if the env would refuse it or the screen would not change
     (except in a GameBoy dialogue, menu or battle, where a press often shows nothing while text and animations play:
     those go to the check, told so); otherwise the check call (PREVIEW_CHECK) judges whether the previewed outcome is a
     step toward the task.
  5. rejected: policy(feedback=...) with what the preview showed and why it was rejected; the last proposal is executed
     whatever its verdict (the executor's memory, WebVoyager's chat, ends with it).
A preview that cannot be made (Web's peek_failed, an error) keeps the proposal.

Per env, the check also sees: Web, the previewed page's URL / title and text (page_text) and today's date; Android,
the app screen (activity) before and after; GameBoy, a battle choice's whole turn (peek_rollout: the press, then B
through the battle text) instead of one screen, and a no-change rejection's feedback names the buttons that would
change the screen. A rejection backed by a battle rollout cannot be overridden by insisting.
The versions tried and their results: plans/oracle_preview_versions.md.

A diagnostic model: it reads the env (attach_env), which no learned world model may.
"""
import re
import time
from typing import Callable, Optional
import numpy as np
from cusi.world_action.base import Plan, WorldActionModel

MAX_PROPOSALS = 3
CHECK_MAX_NEW_TOKENS = 400
PREVIEW_CHECK = """
Task: [TASK]
[TODAY]
You are checking one proposed action of an agent operating [ENV]. The action has NOT been taken yet.

Image 1 is the screen NOW. [AFTER_IMAGES]

Proposed action: [ACTION]
The agent's reasoning: [REASONING]
[FACTS][PAGE_TEXT][ENV_NOTE]
Is taking this action a good next step for the task? Most tasks take several actions, so judge the action as one
step of the agent's plan (its reasoning above), not by whether image 2 already looks finished. It is good if it does
what the agent intended and that intention is reasonable for the task: making progress, or a sensible step on the
way (opening a menu or page to look for something, scrolling to see more, entering one character or field of
several, moving toward a target). An intermediate screen that looks incomplete or odd because the plan is not
finished yet is still good. It is bad only if image 2 itself shows one of these mistakes:
- the action hit a different target than the agent intended (another element, link, item, date, page or app);
- an error message, or the environment refusing the action;
- it lands somewhere unrelated to the task;
- it destroys progress already made (deletes, clears or undoes something the task needs).
Do not call it bad because of your own guess about how the app or site will behave on later steps, or because you
would have planned differently: an app may work differently from what you expect (a number pad may fill from the
right, a form may update after the next field), and the agent will see the result and can adapt.

Respond in exactly this format:
Outcome: <one sentence: what the action did, as seen in image 2>
Verdict: <good or bad>
Reason: <one sentence: why; if bad, what the agent should do instead>
[STOP]"""
ONE_AFTER = "Image 2 is the screen the proposed action leads to: an exact preview, not a guess."
ROLLOUT_AFTER = ("Images 2 to {last} are an exact preview, not a guess, of what follows in order: the proposed press, "
                 "then the battle text played out (pressing B, which never chooses anything) up to the next choice. "
                 "Judge the action by where this leads: the move's effect, damage, fainting, the battle ending.")
#: A battle rollout shows at most this many screens after the press (the last one always).
ROLLOUT_IMAGES = 5
#: Shown to the check on a GameBoy screen that is not free roaming.
GAMEBOY_NOTE = ("\nIn a game's dialogue, menus and battles, the game often needs several presses of A while text and "
                "animations play: a press that shows no change yet is still good when pressing A (to continue or wait) "
                "is the right way to go on.\n")
#: Characters of a previewed page's text shown to the check (Web).
PAGE_TEXT_CHECK = 2500
FEEDBACK = ("[Preview] Before acting, your proposed action was tried in a simulator. Proposed action: {action}. "
            "What it would do: {outcome} It does not help: {reason} Do not choose that action again here; choose a "
            "different action.")


def _rgb(frame) -> np.ndarray:
    x = np.asarray(frame, dtype=np.uint8)
    if x.ndim == 2:
        x = x[:, :, None]
    return np.repeat(x, 3, axis=2) if x.shape[2] == 1 else x[..., :3]


def _field(*, name: str, text: str) -> Optional[str]:
    m = re.search(rf"{name}\s*:\s*\**\s*(.+)", text or "", re.I)
    return m.group(1).strip().strip("*").strip() if m else None


class OraclePreviewModel(WorldActionModel):
    name = "oracle_preview"
    uses_policy = True

    def __init__(self, *, parameters: dict) -> None:
        super().__init__(directory=None, meta={"env": None, "game": None, "game_series": None, "oracle": True})
        self._parameters = parameters
        self._env = None
        self.task = None

    @classmethod
    def load(cls, *, directory: Optional[str], device: str, parameters: dict) -> "OraclePreviewModel":
        return cls(parameters=parameters)

    def attach_env(self, env) -> None:
        self._env = env
        self.meta["game"] = getattr(env, "game", None)

    def begin(self, *, task: str, hint: Optional[str], obs: dict, info: dict) -> None:
        self.task = task

    # ------------------------------------------------------------ the preview

    def _preview(self, *, action: str) -> dict:
        """The peek of one action: frame, valid, error, changed, facts, seconds; {"failed": reason} if none."""
        t0 = time.time()
        try:
            obs, _reward, _terminated, _truncated, info = self._env.peek(actions=[action])[0]
        except Exception as e:      # an Android emulator restart, a dead sandbox browser
            return {"failed": f"{type(e).__name__}: {str(e)[:200]}", "seconds": time.time() - t0}
        if info.get("peek_failed"):
            return {"failed": str(info["peek_failed"])[:200], "seconds": time.time() - t0}
        true = info.get("true") or {}
        frames = [info.get("raw_frame", obs["frame"])]
        if self._battle_commit(action=action, info=info):
            rolled = self._env.peek_rollout(action=action)
            if len(rolled) > 1:     # the press, then the turn's text: the screens that carry it, the last one kept
                keep = np.unique(np.linspace(0, len(rolled) - 1, min(ROLLOUT_IMAGES, len(rolled))).round().astype(int))
                frames = [rolled[i] for i in keep]
        return {"frame": frames[0], "frames": frames, "text": info.get("page_text") or "",
                "valid": bool(info.get("valid", True)),
                "error": info.get("error"), "changed": bool(info.get("changed", True)) or len(frames) > 1,
                "true": true, "synced": info.get("peek_synced"), "agent_state": info.get("agent_state"),
                "seconds": time.time() - t0}

    def _battle_commit(self, *, action: str, info: dict) -> bool:
        """GameBoy (Pokemon) A that commits a battle choice (in the move list, the battle bag or the Pokemon menu), whose
        effect shows only after several screens (peek_rollout). Not an A that only goes on with battle text: there is
        nothing to choose, and judging it by the turn it plays out rejected presses whose outcome was already decided."""
        if not hasattr(self._env, "peek_rollout") or info.get("agent_state") != "IN_BATTLE" or _short(action) != "A":
            return False
        parser = self._env._env._emulator.state_parser
        frame = self._env._env.get_info()["core"]["current_frame"]
        return any(f(frame) for f in (parser.is_in_fight_options_menu, parser.is_in_fight_bag,
                                      parser.is_in_pokemon_menu))

    def _changing_buttons(self) -> str:
        """GameBoy only (a peek is exact and costs no time there): which buttons would change the screen now."""
        if getattr(self._env, "game", None) is None:
            return ""
        from cusi.envs.action_vocab import GAMEBOY_BUTTONS
        results = self._env.peek(actions=list(GAMEBOY_BUTTONS))
        moving = [b for b, r in zip(GAMEBOY_BUTTONS, results) if r[4].get("valid") and r[4].get("changed")]
        return (f" Buttons that would change the screen here: {', '.join(moving)}." if moving else
                " No button would change the screen here.")

    def _facts(self, *, info: dict, preview: dict) -> str:
        """What the env says about the outcome, beyond the screen."""
        now, after = info.get("true") or {}, preview["true"]
        lines = []
        if "url" in after:
            lines.append(f"Page now: {now.get('url')} ({now.get('title')!r}). Page after: {after.get('url')} "
                         f"({after.get('title')!r}).")
        if "activity" in after:
            lines.append(f"App screen now: {now.get('activity')}. After: {after.get('activity')}.")
        return "\n".join(lines) + ("\n" if lines else "")

    def _check(self, *, call: Callable, obs: dict, info: dict, action: str, reasoning: Optional[str],
               preview: dict) -> dict:
        prompt = (PREVIEW_CHECK.replace("[TASK]", self.task).replace("[TODAY]", _today() if hasattr(self._env, "start_url") else "").replace("[ENV]", self._env.env_description)
                  .replace("[ACTION]", _short(action)).replace("[REASONING]", (reasoning or "(none)")[:1500])
                  .replace("[FACTS]", self._facts(info=info, preview=preview))
                  .replace("[PAGE_TEXT]", _page_text_block(preview=preview, chars=PAGE_TEXT_CHECK))
                  .replace("[ENV_NOTE]", GAMEBOY_NOTE if _in_game_ui(preview) else "")
                  .replace("[AFTER_IMAGES]", ONE_AFTER if len(preview["frames"]) == 1 else
                           ROLLOUT_AFTER.format(last=len(preview["frames"]) + 1)))
        response = call(tag="preview_check", prompt=prompt, images=[_rgb(obs["frame"])] + [_rgb(f) for f in preview["frames"]],
                        max_new_tokens=CHECK_MAX_NEW_TOKENS)
        verdict = (_field(name="Verdict", text=response) or "").lower()
        return {"good": not verdict.startswith("bad"), "verdict": verdict or None,
                "outcome": _field(name="Outcome", text=response), "reason": _field(name="Reason", text=response)}

    # ------------------------------------------------------------ the contract

    def plan(self, *, obs: dict, info: dict, error: Optional[str], call: Callable,
             policy: Optional[Callable] = None) -> Plan:
        if policy is None:
            raise RuntimeError("oracle_preview needs the executor's policy (uses_policy)")
        if self._env is None:
            raise RuntimeError("oracle_preview: attach_env was not called")
        proposals, feedback = [], None
        for k in range(MAX_PROPOSALS):
            out = policy(feedback=feedback)
            d = out["decision"]
            p = {"action": d.action_text, "finish": d.finish, "invalid": d.invalid}
            proposals.append(p)
            record = {"model": self.name, "proposals": proposals, "owner": k,
                      "n_proposals": len(proposals), "base_action": proposals[0]["action"],
                      "reasoning": out.get("reasoning")}
            if d.invalid is not None:
                return Plan(invalid=d.invalid, error=d.error, record=record)
            if d.finish:
                return Plan(finish=True, answer=d.answer, finish_text=d.action_text, record=record)
            same = [q for q in proposals[:-1] if q["action"] == d.action_text]
            # insisting overrides a check's judgement of one screen only; not one backed by a played-out battle turn
            if any(q.get("verdict") and not q.get("rollout") for q in same):
                p["kept"] = "insisted"
                break
            if same:      # again an action refused, changing nothing, or judged by its rollout: known, not previewed
                p.update({k: same[0].get(k) for k in ("valid", "changed", "outcome", "reason", "rollout")}, kept=False,
                         repeated=True)
                if k == MAX_PROPOSALS - 1:
                    break
                feedback = FEEDBACK.format(action=_short(d.action_text), outcome=_sentence(p["outcome"]),
                                           reason=_sentence(p["reason"]))
                continue
            preview = self._preview(action=d.action_text)
            p["preview_seconds"] = preview["seconds"]
            if "failed" in preview:
                p.update(preview_failed=preview["failed"], kept="no preview")
                break
            p.update(valid=preview["valid"], changed=preview["changed"], synced=preview["synced"],
                     rollout=len(preview["frames"]) > 1)
            if not preview["valid"]:
                p.update(kept=False, outcome=f"The environment refuses it: {preview['error']}",
                         reason="it cannot be performed here.")
            elif not preview["changed"] and not _in_game_ui(preview):
                p.update(kept=False, outcome="Nothing on the screen changes.",
                         reason="it has no effect here." + self._changing_buttons())
            else:
                check = self._check(call=call, obs=obs, info=info, action=d.action_text,
                                    reasoning=out.get("reasoning"), preview=preview)
                p.update(kept=check["good"], verdict=check["verdict"], outcome=check["outcome"],
                         reason=check["reason"])
            if p["kept"] or k == MAX_PROPOSALS - 1:
                break
            feedback = FEEDBACK.format(action=_short(d.action_text), outcome=_sentence(p["outcome"]),
                                       reason=_sentence(p["reason"]))
        record["revised"] = len(proposals) > 1
        record["diverged"] = proposals[-1]["action"] != proposals[0]["action"]
        return Plan(actions=[proposals[-1]["action"]], record=record)


def _today() -> str:
    """Today's date, for a live web site's "last week", "this year" and search filters (Web only: an Android device
    keeps its own clock)."""
    return time.strftime("Today's date: %A, %Y-%m-%d.\n")


def _page_text_block(*, preview: dict, chars: int) -> str:
    text = preview.get("text") or ""
    return f"Text of the page the action leads to (start, an exact preview): {text[:chars]}\n" if text else ""


def _in_game_ui(preview: dict) -> bool:
    """A GameBoy screen in dialogue, a menu or a battle (not free roaming)."""
    return preview.get("agent_state") not in (None, "FREE_ROAM")


def _short(action: Optional[str]) -> str:
    """The action part of an executor's reply (M3A's whole reply is its action text)."""
    a = str(action or "")
    m = re.search(r"Action\s*:\s*(.*)", a, re.S)
    return " ".join((m.group(1) if m else a).split())[:300]


def _sentence(text: Optional[str]) -> str:
    t = (text or "").strip()
    return (t if t.endswith((".", "!", "?")) else t + ".") if t else "(no detail)."
