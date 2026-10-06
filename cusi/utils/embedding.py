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
from cusi.utils.log_handling import log_info, log_warn, log_error
from cusi.utils.lm_inference import (
    RateLimitedAPIBase,
    OpenAICompatibleAPIBase,
)
from cusi.utils.huggingface_inference import (
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
        """(N, dim) embeddings of a validated, non-empty list."""
        pass

    def embed(
        self,
        *,
        texts: Union[str, list[str]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """Shape (dim,) for a single string, (N, dim) for a list."""
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
        """(N, dim) embeddings of a validated, non-empty list."""
        pass

    def embed(
        self,
        *,
        images: Union[Image.Image, list[Image.Image]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """Shape (dim,) for a single image, (N, dim) for a list."""
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
        """(N, dim) embeddings of validated (text, image) pairs."""
        pass

    def embed(
        self,
        *,
        texts: Union[str, list[str]],
        images: Union[Image.Image, list[Image.Image]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """Both single items -> (dim,), both lists -> (N, dim)."""
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
    """Rate-limited base for API-backed text embedding models."""

    def __init__(
        self,
        *,
        model: str,
        max_queries_per_minute: int = 60,
        parameters: dict[str, Any] = None,
    ) -> None:
        super().__init__(
            model=model,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )

    @abstractmethod
    async def query_client(self, *, client: Any, texts: list[str]) -> list[list[float]]:
        """Raw embedding vectors in input order."""
        pass

    async def _embed_async(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """One wait(), then every batch fires concurrently."""
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
    """Text embedding via an OpenAI-compatible ``/v1/embeddings`` endpoint."""

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
                # The API may return items out of order.
                return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            except Exception as e:
                last_error = e
                log_warn(f"OpenAI embeddings API call failed on attempt {attempt+1}/{max_tries} with error: {e}")
                if attempt < max_tries - 1:
                    # jittered so concurrent retries don't collide
                    backoff_time = self.seconds_to_wait * (2 ** attempt) * random.uniform(1.0, 1.5)
                    log_info(f"Waiting for {backoff_time:.2f} seconds before retrying...")
                    await asyncio.sleep(backoff_time)
        raise RuntimeError(f"OpenAI embeddings API call failed after {max_tries} attempts. Last error: {last_error}") from last_error


class OpenAITextEmbeddingModel(OpenAIAPITextEmbeddingModel):
    """Text embedding via the official OpenAI embeddings API."""

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
    """Text embedding via OpenRouter (OPENROUTER_API_KEY); not every OpenRouter model supports embeddings."""

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
    """HuggingFace encoder or decoder text embedding, pooled by "last_token", "mean" or "cls"."""

    def __init__(
        self,
        *,
        model: str,
        pooling_strategy: str = "last_token",
        parameters: dict[str, Any] = None,
        **model_kwargs,
    ) -> None:
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
    """HuggingFace image embedding: get_image_features for CLIP-like models, else the last token's hidden state."""

    def __init__(
        self,
        *,
        model: str,
        parameters: dict[str, Any] = None,
        **model_kwargs,
    ) -> None:
        self._init_store(
            model=model,
            parameters=parameters,
            model_kwargs=model_kwargs,
            load_fn=load_image_embedding_into_store,
        )

    def do_embed_image(
        self, *, images: list[Image.Image], normalize: bool
    ) -> torch.Tensor:
        if self.is_defunct:
            log_error("Cannot embed with a defunct model.", parameters=self.parameters)
        store = HUGGINGFACE_MODEL_MAPPING[self.model]
        # VLM-style processors (e.g. Qwen3-VL) require a chat-templated text per image.
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
    """HuggingFace joint embedding per (text, image) pair; models with a custom encode API need a subclass."""

    def __init__(
        self,
        *,
        model: str,
        parameters: dict[str, Any] = None,
        **model_kwargs,
    ) -> None:
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
        """pooler_output, else the CLS token; errors if the model gives separate image_embeds."""
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
    """jina-embeddings-v4 text embedding via its encode_text API; task in retrieval/text-matching/code."""

    def __init__(
        self,
        *,
        model: str = "jinaai/jina-embeddings-v4",
        task: str = "retrieval",
        prompt_name: str = "query",
        parameters=None,
        **model_kwargs,
    ) -> None:
        # encode_text pools internally, so pooling_strategy is irrelevant
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
    """jina-embeddings-v4 image embedding via its encode_image API."""

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
    """(M, N) pairwise cosine similarity; a scalar if both inputs are 1-D."""
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
    """Top-k cosine (scores, indices) per query, shaped (Q, k), or (k,) for a 1-D query."""
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
