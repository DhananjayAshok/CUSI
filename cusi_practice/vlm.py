"""One model handle for every role (executor, judge, proposer, guidance, cleaning).

A thin layer over cusi_utils' InferenceModel that accepts any image type the pipeline
holds (numpy, PIL, EncodedImage), converted to PIL RGB (colour kept), and that sends the
pipeline's neutral chat format (records.CallRecord.messages) as OpenAI messages.
"""
import base64
from typing import Any, Optional, Union
from cusi_utils.parameter_handling import load_parameters
from cusi_utils.model_factory import build_model
from cusi_practice.records import EncodedImage, to_pil


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
    """Neutral chat (content parts {"type": "image", "image": i}) -> OpenAI format."""
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


class PracticeVLM:
    """The pipeline's model. infer() mirrors GameBoyRL's VLM.infer; chat() sends a chat."""

    def __init__(self, *, model_name: str, model_backend: str = "vllm", vllm_base_url: Optional[str] = None,
                 parameters: dict[str, Any] = None) -> None:
        self._parameters = load_parameters(parameters)
        self.model_name = model_name
        self._model = build_model(model_name=model_name, model_backend=model_backend,
                                  vllm_base_url=vllm_base_url, parameters=self._parameters)

    def infer(self, *, texts: Union[str, list], max_new_tokens: int, images=None,
              temperature: Optional[float] = None) -> dict:
        """As InferenceModel.infer: a str prompt with a flat image list, or a list of
        prompts with a list of image lists. Returns {"output", "meta"}."""
        images = _convert(images)
        if images is not None and len(images) == 0:
            images = None
        if isinstance(texts, list) and images is not None and all(len(i) == 0 for i in images):
            images = None
        return self._model.infer(texts=texts, images=images, max_new_tokens=max_new_tokens,
                                 temperature=temperature)

    def chat(self, *, messages: list, images: list, max_new_tokens: int,
             temperature: Optional[float] = None) -> dict:
        """messages in the neutral format (see records.CallRecord). Returns {"output", "meta"}."""
        return self._model.infer_messages(messages=to_openai_messages(messages=messages, images=images),
                                          max_new_tokens=max_new_tokens, temperature=temperature)
