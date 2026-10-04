import argparse
from typing import Any, Optional
from pypdf import PdfReader
from cusi_utils.parameter_handling import load_parameters
from cusi_utils.log_handling import log_info
from cusi_utils.lm_inference import InferenceModel
from cusi_utils.model_factory import build_model

# WebVoyager builds OpenAI-format chat messages (a system role, image_url parts),
# which only the OpenAI-compatible backends accept as is.
WEBVOYAGER_BACKENDS = ("vllm", "openai", "openrouter")

# Characters of PDF text sent to the model; keeps long PDFs inside the context window.
MAX_PDF_CHARS = 30000


def add_model_args(*, parser: argparse.ArgumentParser) -> None:
    """
    Add --model_name, --model_backend and --vllm_base_url to a WebVoyager argparse parser.
    """
    parser.add_argument("--model_name", type=str, required=True, help="Model id.")
    parser.add_argument("--model_backend", type=str, required=True, choices=WEBVOYAGER_BACKENDS,
                        help="Model backend.")
    parser.add_argument("--vllm_base_url", type=str, default=None,
                        help="vllm backend: base url of the vLLM server. Defaults to vLLM_base_url in the CUSI config.")


def build_model_from_args(*, args: argparse.Namespace, parameters: dict[str, Any] = None) -> InferenceModel:
    """
    Build the model named by the flags that ``add_model_args`` added.
    """
    parameters = load_parameters(parameters)
    return build_model(
        model_name=args.model_name,
        model_backend=args.model_backend,
        vllm_base_url=args.vllm_base_url,
        parameters=parameters,
    )


def chat(
    *,
    model: InferenceModel,
    messages: list[dict],
    max_new_tokens: int,
    temperature: Optional[float] = None,
) -> tuple[str, int, int]:
    """
    Send OpenAI-format chat messages and return the reply with its token counts.

    :param model: The model to query.
    :type model: InferenceModel
    :param messages: OpenAI-format chat messages.
    :type messages: list[dict]
    :param max_new_tokens: Maximum number of tokens to generate.
    :type max_new_tokens: int
    :param temperature: Sampling temperature. None means the backend default.
    :type temperature: float or None
    :return: ``(text, input_tokens, output_tokens)``. Counts the backend does not report are 0.
    :rtype: tuple[str, int, int]
    """
    out = model.infer_messages(messages=messages, max_new_tokens=max_new_tokens, temperature=temperature)
    return out["output"], out["meta"]["input_tokens"] or 0, out["meta"]["output_tokens"] or 0


def answer_from_pdf(*, model: InferenceModel, pdf_path: str, question: str, parameters: dict[str, Any] = None) -> str:
    """
    Answer ``question`` from a downloaded PDF by extracting its text and asking ``model``.

    Replaces WebVoyager's OpenAI Assistants API retrieval.
    """
    parameters = load_parameters(parameters)
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    log_info(f"Read {len(reader.pages)} pages ({len(text)} chars) from {pdf_path}", parameters=parameters)
    if len(text) > MAX_PDF_CHARS:
        text = text[:MAX_PDF_CHARS] + "\n[... truncated ...]"
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant that can analyze the content of a PDF file and give an answer "
            "that matches the given task, or retrieve relevant content that matches the task."
        )},
        {"role": "user", "content": f"PDF content:\n{text}\n\nTask: {question}"},
    ]
    answer, _, _ = chat(model=model, messages=messages, max_new_tokens=1000)
    return answer
