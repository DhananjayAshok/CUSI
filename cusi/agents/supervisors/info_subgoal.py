"""
Subgoal planning informed by the relevant, distilled insights of info documents.
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional
from cusi.utils.log_handling import log_error, log_warn
from cusi.utils.parsing import parse_key_value, parse_list, parse_yes_no
from cusi.agents.supervisors.subgoal import SubgoalSupervisor


def load_documents(*, env: str, mode: str, info_docs: Optional[str], game: str = "", knowledge_vlm=None,
                   parameters: dict) -> list:
    """The info documents for an info_subgoal arm; mode is "retrieval" or "parametric"."""
    if env != "gameboy":
        log_error(f"info_subgoal_{mode}: no info documents exist for {env} yet.",
                  parameters=parameters)
    from benchmark_scripts import common    # GameBoyRL, on sys.path in the GameBoy process
    if mode == "retrieval":
        if not info_docs:
            log_error("info_subgoal_retrieval needs --info_docs (comma-separated document paths).",
                      parameters=parameters)
        return common.load_documents(info_docs, parameters)
    from execution.parametric_doc import load_or_generate_parametric_document
    # Generated once from the supervisor model's priors and cached per model.
    from cusi.utils.paths import parametric_doc_file
    path = parametric_doc_file(parameters=parameters, game=game, model_name=knowledge_vlm.model_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return [load_or_generate_parametric_document(game=game, path=path, vlm=knowledge_vlm, parameters=parameters)]


class InfoSubgoalSupervisor(SubgoalSupervisor):
    """The subgoal supervisor with knowledge from info documents (GameBoyRL's InfoDocument interface)."""

    def __init__(self, *, documents: Optional[List[Any]] = None, max_concurrency: int = 8, **kwargs) -> None:
        self._documents = documents or []
        self._max_concurrency = max_concurrency
        self.selection_log: List[dict] = []
        self.selected_ids: List[str] = []
        self.insights_block: str = ""
        self.n_insights_candidate = 0
        self.n_insights_kept = 0
        self.n_insights_distilled = 0
        super().__init__(**kwargs)
        _ = self._vlm

    def _run_config(self) -> dict:
        return {**super()._run_config(), "max_concurrency": self._max_concurrency,
                "n_documents": len(self._documents)}

    def _knowledge(self) -> str:
        return self.insights_block

    def _resolve_targets(self) -> List[Optional[str]]:
        self.selection_log = []
        self.selected_ids = []
        self.insights_block = ""
        screen = self.current_frame()

        selected = self._select_entries(self._candidates(), screen)
        self.selected_ids = [self.entry_id(entry) for entry in selected]
        if not selected:
            log_warn("[info] nothing selected from the documents; planning without insights (equivalent to the "
                     "subgoal arm for this episode).", parameters=self._parameters)
        else:
            self.insights_block = self.filter_insights(selected, screen)
        return super()._resolve_targets()

    def _extras(self) -> dict:
        return {**super()._extras(), "selected_ids": list(self.selected_ids), "insights_block": self.insights_block,
                "n_insights_candidate": self.n_insights_candidate, "n_insights_kept": self.n_insights_kept,
                "n_insights_distilled": self.n_insights_distilled}

    # -- Knowledge selection --------------------------------------------------

    @staticmethod
    def entry_id(entry) -> str:
        return f"{entry.source}#{entry.category}" if entry.source else entry.category

    @staticmethod
    def _entry_frame(entry):
        from PIL import Image
        path = entry.resolved_frame
        if not path or not os.path.exists(path):
            return None
        return Image.open(path).convert("RGB")

    def _judge_relevance(self, entry, kind: str, screen) -> tuple:
        images = [screen]
        entry_frame = self._entry_frame(entry)
        if entry_frame is not None:
            images.append(entry_frame)
        if entry_frame is not None:
            frame_note = self._domain.relevance_frame_note
            evidence_note = "The frames are your primary evidence: compare what is actually visible in them."
        else:
            frame_note = self._domain.relevance_no_frame_note
            evidence_note = ("The current screen is your primary evidence: the entry's description must fit what "
                             "is actually visible in it.")

        prompt = (
            self._prompts["RELEVANCE_PROMPT"]
            .replace("[GAME]", self._game)
            .replace("[TASK]", self._task)
            .replace("[KIND]", kind)
            .replace("[ENTRY]", entry.evidence_block())
            .replace("[FRAME_NOTE]", frame_note)
            .replace("[EVIDENCE_NOTE]", evidence_note)
        )
        output, records = self._vlm_infer("filter", texts=prompt, images=images)
        verdict = parse_yes_no(output, "Relevant")
        reason = (parse_key_value(output, "Reasoning") or "").strip()
        return verdict is True, reason, records

    def _select_entries(self, candidates, screen) -> List[Any]:
        if not candidates:
            return []
        with ThreadPoolExecutor(max_workers=self._max_concurrency) as pool:
            futures = [pool.submit(self._judge_relevance, entry, kind, screen) for entry, kind in candidates]
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as error:  # a failed judgement must not sink the episode
                    results.append(error)

        selected = []
        for (entry, kind), result in zip(candidates, results):
            if isinstance(result, Exception):
                log_warn(f"relevance call failed for '{entry.category}': {result}", parameters=self._parameters)
                continue
            relevant, reason, records = result
            self.report.event_log.extend(records)
            self.selection_log.append({"category": entry.category, "kind": kind, "source": entry.source,
                                       "relevant": relevant, "reason": reason})
            if relevant:
                selected.append(entry)
        return selected

    def _candidates(self):
        from execution.info_doc import IMAGE_SECTION, TASK_SECTION
        candidates = []
        for document in self._documents:
            candidates += [(e, "task") for e in document.entries(TASK_SECTION)]
            candidates += [(e, "kind of screen") for e in document.entries(IMAGE_SECTION)]
        return candidates

    def filter_insights(self, selected, screen) -> str:
        pairs = [(entry, insight) for entry in selected for insight in entry.insights]
        self.n_insights_candidate = len(pairs)
        self.n_insights_kept = len(pairs)
        if not pairs:
            return "(no insights recorded)"

        numbered = "\n".join(f"  {i + 1}. [{entry.source or 'unknown'} / {entry.category}] {insight}"
                             for i, (entry, insight) in enumerate(pairs))
        prompt = (
            self._prompts["FILTER_INSIGHTS_PROMPT"]
            .replace("[GAME]", self._game)
            .replace("[TASK]", self._task)
            .replace("[CANDIDATES]", numbered)
        )
        output = self._vlm_call("filter_insights", texts=prompt, images=[screen])
        answer = (parse_key_value(output, "Keep") or "").strip()

        if not answer or answer.upper().startswith("ALL"):
            kept = pairs
        else:
            wanted = {int(n) for n in re.findall(r"\d+", answer)}
            kept = [pair for i, pair in enumerate(pairs) if i + 1 in wanted]
            if not kept:
                log_warn(f"[plan] insight filter kept nothing from {len(pairs)} candidates (reply: {answer[:80]!r}); "
                         f"keeping all.", parameters=self._parameters)
                kept = pairs

        self.n_insights_kept = len(kept)
        self._say(f"  insight filter: kept {len(kept)}/{len(pairs)}")

        grouped: dict = {}
        for entry, insight in kept:
            grouped.setdefault((entry.category, entry.source), []).append(insight)
        blocks = []
        for (category, source), insights in grouped.items():
            label = f" (source: {source})" if source else ""
            body = "\n".join(f"- {i}" for i in insights)
            blocks.append(f"From '{category}'{label}:\n{body}")
        grouped_block = "\n\n".join(blocks)

        return self.distil_insights(kept, grouped_block, screen)

    def distil_insights(self, kept: list, grouped_block: str, screen) -> str:
        if not kept:
            return grouped_block
        prompt = (
            self._prompts["DISTILL_INSIGHTS_PROMPT"]
            .replace("[GAME]", self._game)
            .replace("[TASK]", self._task)
            .replace("[CANDIDATES]", grouped_block)
        )
        output = self._vlm_call("distil_insights", texts=prompt, images=[screen])
        lines = parse_list(output, "Insights")
        if not lines:
            log_warn(f"[plan] insight distillation produced nothing from {len(kept)} insights; keeping the "
                     f"unconsolidated block.", parameters=self._parameters)
            self.n_insights_distilled = len(kept)
            return grouped_block
        self.n_insights_distilled = len(lines)
        self._say(f"  insight distillation: {len(kept)} -> {len(lines)} statements")
        return "\n".join(f"- {line}" for line in lines)
