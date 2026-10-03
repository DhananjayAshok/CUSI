import os
from typing import Any, Optional
from cusi_utils.parameter_handling import load_parameters
from cusi_utils.log_handling import log_info, log_warn, log_error
from cusi_utils.lm_inference import (
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
    """
    Build an InferenceModel for the given backend.

    :param model_name: Model identifier passed to the backend.
    :type model_name: str
    :param model_backend: One of ``MODEL_BACKENDS``.
    :type model_backend: str
    :param vllm_base_url: Base url of the vLLM server. Only used by the vllm backend.
        None uses ``parameters["vLLM_base_url"]``.
    :type vllm_base_url: str or None
    :param parameters: Loaded parameters dict. If None, loads from config.
    :type parameters: dict[str, Any] or None
    :return: The constructed model.
    :rtype: InferenceModel
    """
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
        # vLLM ignores the key unless the server was started with --api-key, but the
        # OpenAI client refuses to start without one.
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
    from cusi_utils.huggingface_inference import HuggingFaceModel

    return HuggingFaceModel(model=model_name, parameters=parameters)
