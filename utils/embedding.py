from typing import Optional, Any, Union
from abc import ABC, abstractmethod
import asyncio
import random
import torch
import torch.nn.functional as F
import os
from PIL import Image
import transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, AutoProcessor, GenerationMixin
from utils.log_handling import log_info, log_warn, log_error
from utils.lm_inference import (
    RateLimitedAPIBase,
    OpenAICompatibleAPIBase,
)
from utils.huggingface_inference import (
    HuggingFaceModelBase,
    HuggingFaceModelStore,
    HUGGINGFACE_MODEL_MAPPING,
    remove_from_model_store,
    clear_model_store,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool token embeddings weighted by the attention mask."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def maybe_normalize(embeddings: torch.Tensor, normalize: bool) -> torch.Tensor:
    if normalize:
        return F.normalize(embeddings, p=2, dim=-1)
    return embeddings


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool by selecting the last non-padding token. Handles both left- and right-padded inputs."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base classes
# ─────────────────────────────────────────────────────────────────────────────


class TextEmbeddingModel(ABC):
    """Abstract base for models that embed text into dense vectors."""

    @abstractmethod
    def do_embed_text(self, *, texts: list[str], normalize: bool) -> torch.Tensor:
        """
        Embed a validated, non-empty list of strings.

        :param texts: Pre-validated list of strings.
        :param normalize: If True, L2-normalise each embedding.
        :return: Float tensor of shape (N, dim).
        :rtype: torch.Tensor
        """
        pass

    def embed(
        self,
        *,
        texts: Union[str, list[str]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Embed one or more texts.

        :param texts: A single string or a list of strings.
        :param normalize: If True, L2-normalise the output embeddings. Default True.
        :return: Shape (dim,) for a single string input, (N, dim) for a list.
        :rtype: torch.Tensor
        """
        passed_str = isinstance(texts, str)
        if passed_str:
            if texts.strip() == "":
                log_error("texts cannot be empty.")
            texts = [texts]
        else:
            if not isinstance(texts, list):
                log_error(
                    f"texts must be a string or list of strings. Got {type(texts)}"
                )
            if len(texts) == 0:
                log_error("texts cannot be empty.")
            for i, item in enumerate(texts):
                if not isinstance(item, str):
                    log_error(f"Got {type(item)}:{item} instead of str at texts[{i}].")
        result = self.do_embed_text(texts=texts, normalize=normalize)
        return result[0] if passed_str else result


class ImageEmbeddingModel(ABC):
    """Abstract base for models that embed images into dense vectors."""

    @abstractmethod
    def do_embed_image(
        self, *, images: list[Image.Image], normalize: bool
    ) -> torch.Tensor:
        """
        Embed a validated, non-empty list of PIL images.

        :param images: Pre-validated list of PIL Images.
        :param normalize: If True, L2-normalise each embedding.
        :return: Float tensor of shape (N, dim).
        :rtype: torch.Tensor
        """
        pass

    def embed(
        self,
        *,
        images: Union[Image.Image, list[Image.Image]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Embed one or more images.

        :param images: A single PIL Image or a list of PIL Images.
        :param normalize: If True, L2-normalise the output embeddings. Default True.
        :return: Shape (dim,) for a single image input, (N, dim) for a list.
        :rtype: torch.Tensor
        """
        passed_single = isinstance(images, Image.Image)
        if passed_single:
            images = [images]
        else:
            if not isinstance(images, list):
                log_error(
                    f"images must be a PIL Image or list of PIL Images. Got {type(images)}"
                )
            if len(images) == 0:
                log_error("images cannot be empty.")
            for i, item in enumerate(images):
                if not isinstance(item, Image.Image):
                    log_error(f"Got {type(item)} instead of PIL Image at images[{i}].")
        result = self.do_embed_image(images=images, normalize=normalize)
        return result[0] if passed_single else result


class ImageTextEmbeddingModel(ABC):
    """Abstract base for models that embed (text, image) pairs into dense vectors."""

    @abstractmethod
    def do_embed_image_text(
        self,
        *,
        texts: list[str],
        images: list[Image.Image],
        normalize: bool,
    ) -> torch.Tensor:
        """
        Embed validated, paired (text, image) inputs.

        :param texts: Pre-validated list of strings, one per pair.
        :param images: Pre-validated list of PIL Images, one per pair.
        :param normalize: If True, L2-normalise each embedding.
        :return: Float tensor of shape (N, dim).
        :rtype: torch.Tensor
        """
        pass

    def embed(
        self,
        *,
        texts: Union[str, list[str]],
        images: Union[Image.Image, list[Image.Image]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Embed one or more (text, image) pairs.

        Both ``texts`` and ``images`` must be the same kind (single item or list).

        :param texts: A single string or a list of strings.
        :param images: A single PIL Image or a list of PIL Images.
        :param normalize: If True, L2-normalise the output embeddings. Default True.
        :return: Shape (dim,) for a single pair, (N, dim) for a list.
        :rtype: torch.Tensor
        """
        passed_str = isinstance(texts, str)
        passed_single_image = isinstance(images, Image.Image)
        if passed_str != passed_single_image:
            log_error(
                "texts and images must both be single items or both be lists. "
                f"Got {'str' if passed_str else 'list'} for texts and "
                f"{'Image' if passed_single_image else 'list'} for images."
            )
        if passed_str:
            if texts.strip() == "":
                log_error("texts cannot be empty.")
            texts = [texts]
            images = [images]
        else:
            if not isinstance(texts, list):
                log_error(
                    f"texts must be a string or list of strings. Got {type(texts)}"
                )
            if not isinstance(images, list):
                log_error(
                    f"images must be a PIL Image or list of PIL Images. Got {type(images)}"
                )
            if len(texts) == 0:
                log_error("texts cannot be empty.")
            if len(images) == 0:
                log_error("images cannot be empty.")
            if len(texts) != len(images):
                log_error(
                    f"texts and images must have the same length. "
                    f"Got {len(texts)} texts and {len(images)} images."
                )
            for i, item in enumerate(texts):
                if not isinstance(item, str):
                    log_error(f"Got {type(item)}:{item} instead of str at texts[{i}].")
            for i, item in enumerate(images):
                if not isinstance(item, Image.Image):
                    log_error(f"Got {type(item)} instead of PIL Image at images[{i}].")
        result = self.do_embed_image_text(
            texts=texts, images=images, normalize=normalize
        )
        return result[0] if passed_str else result


# ─────────────────────────────────────────────────────────────────────────────
# API text embedding
# ─────────────────────────────────────────────────────────────────────────────


class APITextEmbeddingModel(RateLimitedAPIBase, TextEmbeddingModel, ABC):
    """
    Abstract base for API-backed text embedding models.

    Handles rate limiting. Subclasses implement ``query_embedding_client``.
    """

    def __init__(
        self,
        *,
        model: str,
        max_queries_per_minute: int = 60,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        :param model: Model identifier string.
        :param max_queries_per_minute: Maximum requests per minute. Must be >= 1.
        :param parameters: Loaded parameters dict. If None, loads from config.
        """
        super().__init__(
            model=model,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )

    @abstractmethod
    async def query_client(self, *, client: Any, texts: list[str]) -> list[list[float]]:
        """
        Send texts to the embedding API and return raw embedding vectors (asynchronously).

        :param texts: List of strings to embed.
        :return: List of embedding vectors (each a list of floats), in input order.
        :rtype: list[list[float]]
        """
        pass

    async def _embed_async(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """
        Issue one query per batch of ``batch_size`` texts concurrently.

        A single ``self.wait()`` paces the start of this call relative to the last
        request issued; all batches are then fired concurrently. Per-request
        rate-limit errors are handled by ``query_client``'s retry/backoff.
        """
        self.wait()
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        async with self._make_async_client() as client:
            results = await asyncio.gather(*(self.query_client(client=client, texts=batch) for batch in batches))
        flat: list[list[float]] = []
        for batch_result in results:
            flat.extend(batch_result)
        return flat

    def do_embed_text(self, *, texts: list[str], normalize: bool) -> torch.Tensor:
        batch_size = self.parameters["max_batch_size_api"]
        raw = asyncio.run(self._embed_async(texts, batch_size))
        embeddings = torch.tensor(raw, dtype=torch.float32)
        return maybe_normalize(embeddings, normalize)


class OpenAIAPITextEmbeddingModel(OpenAICompatibleAPIBase, APITextEmbeddingModel):
    """
    Text embedding backed by an OpenAI-compatible ``/v1/embeddings`` endpoint.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: Optional[str],
        api_key: Optional[str] = None,
        max_queries_per_minute: int = 60,
        parameters: dict[str, Any] = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )

    async def query_client(self, *, client: Any, texts: list[str]) -> list[list[float]]:
        max_tries = 3
        last_error = None
        for attempt in range(max_tries):
            try:
                response = await client.embeddings.create(model=self.model, input=texts, encoding_format="float")
                if response is None or not getattr(response, "data", None):
                    raise ValueError(f"API returned an invalid response (None/missing/empty data): {response}")
                # Sort by index to preserve input order regardless of API response ordering.
                return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            except Exception as e:
                last_error = e
                log_warn(f"OpenAI embeddings API call failed on attempt {attempt+1}/{max_tries} with error: {e}")
                if attempt < max_tries - 1:
                    # exponential backoff with time_to_wait between attempts, plus jitter so
                    # concurrent coroutines retrying after the same failure don't all collide
                    backoff_time = self.seconds_to_wait * (2 ** attempt) * random.uniform(1.0, 1.5)
                    log_info(f"Waiting for {backoff_time:.2f} seconds before retrying...")
                    await asyncio.sleep(backoff_time)
        raise RuntimeError(f"OpenAI embeddings API call failed after {max_tries} attempts. Last error: {last_error}") from last_error


class OpenAITextEmbeddingModel(OpenAIAPITextEmbeddingModel):
    """
    Text embedding using the official OpenAI embeddings API (e.g. ``text-embedding-3-small``).
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        max_queries_per_minute: int = 60,
        parameters: dict[str, Any] = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=None,
            api_key=api_key,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )


class OpenRouterTextEmbeddingModel(OpenAIAPITextEmbeddingModel):
    """
    Text embedding via OpenRouter's OpenAI-compatible embeddings endpoint.

    Reads ``OPENROUTER_API_KEY`` from the environment (same as ``OpenRouterModel``).

    # Not all models listed on OpenRouter support the /v1/embeddings
    # endpoint. Verify the specific model you pass supports embeddings before use.
    # See the "API" tab on the model's OpenRouter page.
    """

    def __init__(
        self,
        *,
        model: str,
        max_queries_per_minute: int = 60,
        parameters: dict[str, Any] = None,
    ) -> None:
        api_key = os.environ["OPENROUTER_API_KEY"]
        super().__init__(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace text embedding
# ─────────────────────────────────────────────────────────────────────────────


def load_text_embedding_into_store(*, model_name: str, model_kwargs: dict) -> None:
    remove_from_model_store(model_name, verbose=False)
    log_info(
        f"Loading text embedding model {model_name} into store with kwargs {model_kwargs}"
    )
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    is_generation_model = False
    for arch_name in (config.architectures or []):
        arch_cls = getattr(transformers, arch_name, None)
        if arch_cls is not None and issubclass(arch_cls, GenerationMixin):
            is_generation_model = True
            break
    if is_generation_model:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True, **model_kwargs)
    else:
        model = AutoModel.from_pretrained(model_name, device_map="auto", trust_remote_code=True, **model_kwargs)
    processor = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
    HUGGINGFACE_MODEL_MAPPING[model_name] = HuggingFaceModelStore(
        model=model, processor=processor, model_kwargs=model_kwargs
    )


class HuggingFaceTextEmbeddingModel(HuggingFaceModelBase, TextEmbeddingModel):
    """
    HuggingFace text embedding model.

    Supports encoder-based and decoder-based models via a configurable ``pooling_strategy``:
    ``"last_token"`` (default), ``"mean"``, or ``"cls"``. Tokenizer always loads with
    ``padding_side="left"`` to make last-token pooling correct regardless of batch size.
    Suitable for models such as ``Qwen/Qwen3-Embedding-0.6B``, ``BAAI/bge-*``,
    ``intfloat/e5-*``, etc.
    """

    def __init__(
        self,
        *,
        model: str,
        pooling_strategy: str = "last_token",
        parameters: dict[str, Any] = None,
        **model_kwargs,
    ) -> None:
        """
        :param model: HuggingFace model identifier.
        :type model: str
        :param pooling_strategy: How to pool token embeddings. One of ``"last_token"``,
            ``"mean"``, or ``"cls"``. Default ``"last_token"``.
        :type pooling_strategy: str
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict[str, Any] or None
        :param model_kwargs: Extra kwargs forwarded to ``AutoModel.from_pretrained``.
        """
        self._init_store(
            model=model,
            parameters=parameters,
            model_kwargs=model_kwargs,
            load_fn=load_text_embedding_into_store,
        )
        if pooling_strategy not in ("last_token", "mean", "cls"):
            log_error(
                f"pooling_strategy must be one of 'last_token', 'mean', 'cls'. Got '{pooling_strategy}'.",
                parameters=self.parameters,
            )
        self.pooling_strategy = pooling_strategy

    def do_embed_text(self, *, texts: list[str], normalize: bool) -> torch.Tensor:
        """
        Embed a list of strings using the configured pooling strategy.

        :param texts: Pre-validated list of strings.
        :type texts: list[str]
        :param normalize: If True, L2-normalise each embedding.
        :type normalize: bool
        :return: Float tensor of shape (N, dim).
        :rtype: torch.Tensor
        """
        if self.is_defunct:
            log_error("Cannot embed with a defunct model.", parameters=self.parameters)
        store = HUGGINGFACE_MODEL_MAPPING[self.model]
        inputs = store.processor(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(store.model.device)
        with torch.no_grad():
            outputs = store.model(**inputs, output_hidden_states=True)
        # Encoder models expose last_hidden_state directly; CausalLM models require hidden_states[-1]
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs.hidden_states[-1]
        if self.pooling_strategy == "last_token":
            embeddings = last_token_pool(hidden, inputs["attention_mask"])
        elif self.pooling_strategy == "cls":
            embeddings = hidden[:, 0]
        else:
            embeddings = mean_pool(hidden, inputs["attention_mask"])
        return maybe_normalize(embeddings, normalize)


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace image embedding
# ─────────────────────────────────────────────────────────────────────────────


def load_image_embedding_into_store(*, model_name: str, model_kwargs: dict) -> None:
    remove_from_model_store(model_name, verbose=False)
    log_info(
        f"Loading image embedding model {model_name} into store with kwargs {model_kwargs}"
    )
    config = AutoConfig.from_pretrained(model_name)
    is_generation_model = False
    for arch_name in (config.architectures or []):
        arch_cls = getattr(transformers, arch_name, None)
        if arch_cls is not None and issubclass(arch_cls, GenerationMixin):
            is_generation_model = True
            break
    if is_generation_model:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, device_map="auto", **model_kwargs
        )
    else:
        model = AutoModel.from_pretrained(model_name, device_map="auto", **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_name)
    HUGGINGFACE_MODEL_MAPPING[model_name] = HuggingFaceModelStore(
        model=model, processor=processor, model_kwargs=model_kwargs
    )


class HuggingFaceImageEmbeddingModel(HuggingFaceModelBase, ImageEmbeddingModel):
    """
    HuggingFace image embedding model.

    Uses ``get_image_features()`` when available (CLIP and CLIP-like models).
    For VLM-style or causal LM-based models (e.g. Qwen3-VL-Embedding), loads with
    ``AutoModelForImageTextToText`` to match the checkpoint architecture and extracts
    the last token of the final hidden state as the embedding.
    """

    def __init__(
        self,
        *,
        model: str,
        parameters: dict[str, Any] = None,
        **model_kwargs,
    ) -> None:
        """
        :param model: HuggingFace model identifier.
        :type model: str
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict[str, Any] or None
        :param model_kwargs: Extra kwargs forwarded to ``AutoModel.from_pretrained``.
        """
        self._init_store(
            model=model,
            parameters=parameters,
            model_kwargs=model_kwargs,
            load_fn=load_image_embedding_into_store,
        )

    def do_embed_image(
        self, *, images: list[Image.Image], normalize: bool
    ) -> torch.Tensor:
        """
        Embed a list of images using ``get_image_features`` when available,
        otherwise runs a full forward pass with ``output_hidden_states=True`` and
        selects from ``pooler_output``, ``last_hidden_state[:, -1]``, or
        ``hidden_states[-1][:, -1]`` in priority order.

        :param images: Pre-validated list of PIL Images.
        :type images: list[Image.Image]
        :param normalize: If True, L2-normalise each embedding.
        :type normalize: bool
        :return: Float tensor of shape (N, dim).
        :rtype: torch.Tensor
        """
        if self.is_defunct:
            log_error("Cannot embed with a defunct model.", parameters=self.parameters)
        store = HUGGINGFACE_MODEL_MAPPING[self.model]
        # VLM-style processors (e.g. Qwen3-VL) require text alongside images.
        # Detect by checking for apply_chat_template and build one template per image.
        vlm_style = hasattr(store.processor, "apply_chat_template")
        if vlm_style:
            texts = []
            for img in images:
                msg = [{"role": "user", "content": [{"type": "image", "image": img}]}]
                text = store.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
            inputs = store.processor(
                text=texts, images=images, return_tensors="pt", padding=True
            ).to(store.model.device)
        else:
            inputs = store.processor(
                images=images, return_tensors="pt", padding=True
            ).to(store.model.device)
        with torch.no_grad():
            if not vlm_style and hasattr(store.model, "get_image_features"):
                embeddings = store.model.get_image_features(**inputs)
            else:
                outputs = store.model(**inputs, output_hidden_states=True)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                    embeddings = outputs.last_hidden_state[:, -1]
                else:
                    # Causal LM-based embedding models: last token of the final hidden state.
                    embeddings = outputs.hidden_states[-1][:, -1]
        return maybe_normalize(embeddings, normalize)


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace image+text embedding
# ─────────────────────────────────────────────────────────────────────────────


def load_image_text_embedding_into_store(
    *, model_name: str, model_kwargs: dict
) -> None:
    remove_from_model_store(model_name, verbose=False)
    log_info(
        f"Loading image+text embedding model {model_name} into store with kwargs {model_kwargs}"
    )
    model = AutoModel.from_pretrained(model_name, device_map="auto", **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_name)
    HUGGINGFACE_MODEL_MAPPING[model_name] = HuggingFaceModelStore(
        model=model, processor=processor, model_kwargs=model_kwargs
    )


class HuggingFaceImageTextEmbeddingModel(HuggingFaceModelBase, ImageTextEmbeddingModel):
    """
    HuggingFace multimodal (image + text) embedding model.

    Produces one joint embedding per (text, image) pair. Errors out if the model
    produces separate ``image_embeds`` (not a true joint representation). For models
    that expose a custom ``.encode()`` API (e.g. ``jinaai/jina-embeddings-v4``),
    subclass this and override ``do_embed_image_text``. Those models also typically
    require ``trust_remote_code=True`` passed as a ``model_kwarg``.
    """

    def __init__(
        self,
        *,
        model: str,
        parameters: dict[str, Any] = None,
        **model_kwargs,
    ) -> None:
        """
        :param model: HuggingFace model identifier.
        :type model: str
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict[str, Any] or None
        :param model_kwargs: Extra kwargs forwarded to ``AutoModel.from_pretrained``.
        """
        self._init_store(
            model=model,
            parameters=parameters,
            model_kwargs=model_kwargs,
            load_fn=load_image_text_embedding_into_store,
        )

    def do_embed_image_text(
        self,
        *,
        texts: list[str],
        images: list[Image.Image],
        normalize: bool,
    ) -> torch.Tensor:
        """
        Embed paired (text, image) inputs into a joint representation.

        Output selection: ``pooler_output`` if present, else CLS token of ``last_hidden_state``.
        Errors out if the model produces ``image_embeds`` (not a joint embedding).

        :param texts: Pre-validated list of strings, one per pair.
        :type texts: list[str]
        :param images: Pre-validated list of PIL Images, one per pair.
        :type images: list[Image.Image]
        :param normalize: If True, L2-normalise each embedding.
        :type normalize: bool
        :return: Float tensor of shape (N, dim).
        :rtype: torch.Tensor
        """
        if self.is_defunct:
            log_error("Cannot embed with a defunct model.", parameters=self.parameters)
        store = HUGGINGFACE_MODEL_MAPPING[self.model]
        inputs = store.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(store.model.device)
        with torch.no_grad():
            outputs = store.model(**inputs)
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            log_error(
                f"Model {self.model} produces separate image_embeds — this is not a joint "
                "image+text embedding. Use HuggingFaceImageEmbeddingModel for image-only "
                "embeddings, or subclass and override do_embed_image_text for models with "
                "a true joint representation.",
                parameters=self.parameters,
            )
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs.last_hidden_state[:, 0]
        return maybe_normalize(embeddings, normalize)


# ─────────────────────────────────────────────────────────────────────────────
# Jina Embeddings V4 (custom encode API)
# ─────────────────────────────────────────────────────────────────────────────


class JinaV4TextEmbeddingModel(HuggingFaceTextEmbeddingModel):
    """
    Text embedding for ``jinaai/jina-embeddings-v4``.

    Overrides ``do_embed_text`` to use the model's custom ``encode_text()`` API
    instead of a raw forward pass. Always loads with ``trust_remote_code=True``.

    :param task: Adapter task. One of ``"retrieval"``, ``"text-matching"``, ``"code"``.
    :param prompt_name: Prompt role. ``"query"`` or ``"passage"``.
    """

    def __init__(
        self,
        *,
        model: str = "jinaai/jina-embeddings-v4",
        task: str = "retrieval",
        prompt_name: str = "query",
        parameters=None,
        **model_kwargs,
    ) -> None:
        # pooling_strategy is irrelevant (encode_text handles pooling internally)
        super().__init__(model=model, parameters=parameters, **model_kwargs)
        self.task = task
        self.prompt_name = prompt_name

    def do_embed_text(self, *, texts: list[str], normalize: bool) -> torch.Tensor:
        if self.is_defunct:
            log_error("Cannot embed with a defunct model.", parameters=self.parameters)
        store = HUGGINGFACE_MODEL_MAPPING[self.model]
        with torch.no_grad():
            embeddings = store.model.encode_text(
                texts=texts, task=self.task, prompt_name=self.prompt_name
            )
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)
        return maybe_normalize(embeddings, normalize)


class JinaV4ImageEmbeddingModel(HuggingFaceImageEmbeddingModel):
    """
    Image embedding for ``jinaai/jina-embeddings-v4``.

    Overrides ``do_embed_image`` to use the model's custom ``encode_image()`` API
    instead of a raw forward pass. Always loads with ``trust_remote_code=True``.

    :param task: Adapter task. One of ``"retrieval"``, ``"text-matching"``, ``"code"``.
    """

    def __init__(
        self,
        *,
        model: str = "jinaai/jina-embeddings-v4",
        task: str = "retrieval",
        parameters=None,
        **model_kwargs,
    ) -> None:
        model_kwargs["trust_remote_code"] = True
        super().__init__(model=model, parameters=parameters, **model_kwargs)
        self.task = task

    def do_embed_image(self, *, images: list[Image.Image], normalize: bool) -> torch.Tensor:
        if self.is_defunct:
            log_error("Cannot embed with a defunct model.", parameters=self.parameters)
        store = HUGGINGFACE_MODEL_MAPPING[self.model]
        with torch.no_grad():
            embeddings = store.model.encode_image(images=images, task=self.task)
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)
        return maybe_normalize(embeddings, normalize)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding utilities
# ─────────────────────────────────────────────────────────────────────────────


def cosine_similarity(*, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between two sets of embeddings.

    :param a: Tensor of shape (M, dim) or (dim,).
    :type a: torch.Tensor
    :param b: Tensor of shape (N, dim) or (dim,).
    :type b: torch.Tensor
    :return: Similarity matrix of shape (M, N). If both inputs are 1-D, returns a scalar tensor.
    :rtype: torch.Tensor
    """
    squeeze = a.dim() == 1 and b.dim() == 1
    a = F.normalize(a.unsqueeze(0) if a.dim() == 1 else a, p=2, dim=-1)
    b = F.normalize(b.unsqueeze(0) if b.dim() == 1 else b, p=2, dim=-1)
    sim = a @ b.T
    return sim.squeeze() if squeeze else sim


def get_top_k_similars(
    *,
    query: torch.Tensor,
    corpus: torch.Tensor,
    k: int,
    largest: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return the top-k cosine similarity scores and their corpus indices for each query.

    :param query: Tensor of shape (Q, dim) or (dim,) for a single query.
    :param corpus: Tensor of shape (N, dim).
    :param k: Number of top results to return per query.
    :param largest: If True (default), return highest-similarity items.
    :return: Tuple of (scores, indices), each of shape (Q, k). Shape is (k,) when query is 1-D.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    squeeze = query.dim() == 1
    if squeeze:
        query = query.unsqueeze(0)
    if k > corpus.shape[0]:
        log_error(
            f"k={k} is larger than corpus size {corpus.shape[0]}. "
            "k must be <= number of corpus embeddings."
        )
    sim = cosine_similarity(a=query, b=corpus)  # (Q, N)
    result = torch.topk(sim, k=k, dim=-1, largest=largest)
    scores, indices = result.values, result.indices
    if squeeze:
        return scores.squeeze(0), indices.squeeze(0)
    return scores, indices
