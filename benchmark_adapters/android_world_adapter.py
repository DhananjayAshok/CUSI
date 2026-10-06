from typing import Any, Optional
import numpy as np
from PIL import Image
from android_world.agents import infer
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.lm_inference import InferenceModel
from cusi.utils.model_factory import build_model


class InferenceModelWrapper(infer.LlmWrapper, infer.MultimodalLlmWrapper):
    """Lets AndroidWorld agents (M3A, T3A) use an InferenceModel in place of ``infer.Gpt4Wrapper``."""

    def __init__(
        self,
        *,
        model: InferenceModel,
        max_new_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

    # predict/predict_mm keep AndroidWorld's positional signatures: the agents call them positionally.
    def predict(self, text_prompt: str) -> tuple[str, Optional[bool], Any]:
        return self.predict_mm(text_prompt, [])

    def predict_mm(
        self, text_prompt: str, images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:
        out = self._model.infer(
            texts=text_prompt,
            images=[Image.fromarray(image) for image in images] or None,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
        )
        # is_safe is None: only Gemini reports safety blocks.
        return out["output"], None, out


def make_llm_wrapper(
    *,
    model_name: str,
    model_backend: str,
    vllm_base_url: Optional[str] = None,
    parameters: dict[str, Any] = None,
) -> InferenceModelWrapper:
    """Build the wrapper AndroidWorld's run.py hands to M3A/T3A."""
    parameters = load_parameters(parameters)
    model = build_model(
        model_name=model_name,
        model_backend=model_backend,
        vllm_base_url=vllm_base_url,
        parameters=parameters,
    )
    return InferenceModelWrapper(model=model)
