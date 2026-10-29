"""
One model handle for every role, accepting any image type and the neutral chat format.
"""
import base64
import time
from typing import Any, Optional, Union
from cusi.utils.compute import record_llm
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.model_factory import build_model
from cusi.agents.records import EncodedImage, to_pil


def _convert(images):
    if images is None:
        return None
    out = []
    for item in images:
        out.append(_convert(item) if isinstance(item, list) else to_pil(item))
    return out


def _data_url(image) -> str:
    png = EncodedImage.of(image).png
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def to_openai_messages(*, messages: list, images: list) -> list:
    """Neutral chat, whose image parts index `images`, to OpenAI messages."""
    out = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
            continue
        parts = []
        for part in content:
            if part["type"] == "text":
                parts.append({"type": "text", "text": part["text"]})
            else:
                parts.append({"type": "image_url", "image_url": {"url": _data_url(images[part["image"]])}})
        out.append({"role": msg["role"], "content": parts})
    return out


def _per_record(value, n: int) -> list:
    """A meta count as one entry per record (a single prompt's can be a scalar)."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] if n == 1 else [None] * n


class AgentVLM:
    """The pipeline's model, for single-turn prompts and chats; every call is recorded (cusi.utils.compute)."""

    def __init__(self, *, model_name: str, model_backend: str = "vllm", vllm_base_url: Optional[str] = None,
                 parameters: dict[str, Any] = None, greedy: bool = False) -> None:
        """greedy: every call at temperature 0, whatever the caller asks (test-set evals)."""
        self._parameters = load_parameters(parameters)
        self.model_name, self.greedy = model_name, greedy
        self._model = build_model(model_name=model_name, model_backend=model_backend,
                                  vllm_base_url=vllm_base_url, parameters=self._parameters)

    def infer(self, *, texts: Union[str, list], max_new_tokens: int, images=None,
              temperature: Optional[float] = None) -> dict:
        """One prompt with a flat image list, or a list of prompts with a list of image lists."""
        temperature = 0.0 if self.greedy else temperature
        images = _convert(images)
        if images is not None and len(images) == 0:
            images = None
        if isinstance(texts, list) and images is not None and all(len(i) == 0 for i in images):
            images = None
        t0 = time.perf_counter()
        out = self._model.infer(texts=texts, images=images, max_new_tokens=max_new_tokens,
                                temperature=temperature)
        n = len(texts) if isinstance(texts, list) else 1
        n_images = [0] * n if images is None else [len(images)] if n == 1 and not isinstance(texts, list) \
            else [len(i) for i in images]
        self._record(out=out, latency=time.perf_counter() - t0, n=n, images=n_images)
        return out

    def chat(self, *, messages: list, images: list, max_new_tokens: int,
             temperature: Optional[float] = None) -> dict:
        """`messages` in the neutral format of CallRecord.messages."""
        temperature = 0.0 if self.greedy else temperature
        t0 = time.perf_counter()
        out = self._model.infer_messages(messages=to_openai_messages(messages=messages, images=images),
                                         max_new_tokens=max_new_tokens, temperature=temperature)
        n_images = sum(1 for m in messages if not isinstance(m["content"], str)
                       for part in m["content"] if part["type"] != "text")
        self._record(out=out, latency=time.perf_counter() - t0, n=1, images=[n_images])
        return out

    def _record(self, *, out: dict, latency: float, n: int, images: list) -> None:
        meta = out.get("meta") or {}
        record_llm(model_name=self.model_name, latency=latency,
                   prompt_tokens=_per_record(meta.get("input_tokens"), n),
                   completion_tokens=_per_record(meta.get("output_tokens"), n), images=images)
