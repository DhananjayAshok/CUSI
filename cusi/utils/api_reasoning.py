"""
A thinking model served by an API (e.g. Gemini via OpenRouter) returns a reasoning summary beside its answer, and the
API client (cusi.utils.lm_inference.OpenAIAPIModel.get_output_texts) writes the reply as
"Reasoning: <summary>\nResponse: <answer>". strip_api_reasoning gives the answer alone.
"""
import re

API_REASONING_PREFIX, API_RESPONSE_SEP = "Reasoning: ", "\nResponse: "


def strip_api_reasoning(text: str, *, api: bool = True) -> str:
    """A reply without the API's reasoning summary; a reply that is only reasoning (no answer: the token budget ran
    out while thinking) becomes empty. Other replies are returned unchanged. api=False (a vLLM-served model): a reply
    without the Response separator is the model's own answer, even when its format starts with "Reasoning: " (Gemma's
    judge and agent replies do), so it is kept whole."""
    if not text or not text.startswith(API_REASONING_PREFIX):
        return text
    _, sep, answer = text.partition(API_RESPONSE_SEP)
    return answer.strip() if sep else "" if api else text


# A reply with its reasoning summary that was later pasted into another prompt's text (an M3A step summary in the
# history: "Step k- Action selected: {...}. Reasoning: <summary>\nResponse: <answer>").
# The summary may span lines but never another history step (a reasoning-only reply has no Response to end at).
_EMBEDDED_REASONING = re.compile(r"Reasoning: (?:(?!\nStep \d+- ).)*?\nResponse:[ \t]*", re.DOTALL)


def strip_embedded_reasoning(text: str) -> str:
    """Prompt text with every embedded "Reasoning: ...\\nResponse: " cut, so only the answers stay where the replies
    were pasted. Text without one is returned unchanged."""
    if not text or "\nResponse:" not in text:
        return text
    return _EMBEDDED_REASONING.sub("", text)
