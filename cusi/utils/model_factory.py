import os
from typing import Any, Optional
from cusi.utils.parameter_handling import load_parameters
from cusi.utils.log_handling import log_info, log_warn, log_error
from cusi.utils.lm_inference import (
    InferenceModel,
    OpenAIModel,
    AnthropicModel,
    OpenRouterModel,
    vLLMModel,
)

MODEL_BACKENDS = ("vllm", "openai", "anthropic", "openrouter", "huggingface")


def build_model(
    *,
    model_name: str,
    model_backend: str,
    vllm_base_url: Optional[str] = None,
    parameters: dict[str, Any] = None,
) -> InferenceModel:
    """Build an InferenceModel for one of ``MODEL_BACKENDS``; vllm_base_url is vllm-only (None uses the config's)."""
    parameters = load_parameters(parameters)
    if not model_name:
        log_error("model_name must be given", parameters=parameters)
    if model_backend not in MODEL_BACKENDS:
        log_error(
            f"Unknown model backend {model_backend}. Must be one of {MODEL_BACKENDS}",
            parameters=parameters,
        )
    if vllm_base_url is not None and model_backend != "vllm":
        log_warn(
            f"vllm_base_url={vllm_base_url} is ignored for backend {model_backend}",
            parameters=parameters,
        )
    log_info(f"Building {model_backend} model {model_name}", parameters=parameters)
    if model_backend == "vllm":
        # The OpenAI client refuses to start without a key; vLLM ignores it unless started with --api-key.
        return vLLMModel(
            model=model_name,
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=vllm_base_url,  # None -> parameters["vLLM_base_url"]
            parameters=parameters,
        )
    if model_backend == "openai":
        return OpenAIModel(model=model_name, parameters=parameters)
    if model_backend == "anthropic":
        return AnthropicModel(model=model_name, parameters=parameters)
    if model_backend == "openrouter":
        return OpenRouterModel(model=model_name, parameters=parameters)
    from cusi.utils.huggingface_inference import HuggingFaceModel

    return HuggingFaceModel(model=model_name, parameters=parameters)
