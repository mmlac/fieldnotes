from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass
class CompletionRequest:
    """Normalised input for any text completion call."""

    system: str
    messages: list[dict[str, Any]]  # [{"role": "user"|"assistant", "content": "..."}]
    tools: list[dict] | None = None  # for function calling / tool use
    max_tokens: int = 4096
    temperature: float = 0.0  # 0.0 = deterministic; extraction always uses 0.0
    timeout: float | None = (
        None  # per-request timeout in seconds; None = provider default
    )


@dataclass
class CompletionResponse:
    """Normalised output from any text completion call."""

    text: str
    tool_calls: list[dict] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0  # prompt cache hits (Anthropic / OpenAI)


@dataclass
class StreamChunk:
    """A single chunk from a streaming completion."""

    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    done: bool = False


@dataclass
class EmbedRequest:
    texts: list[str]
    timeout: float | None = (
        None  # per-request timeout in seconds; None = provider default
    )


@dataclass
class EmbedResponse:
    vectors: list[list[float]]
    model: str
    input_tokens: int = 0


class ModelProvider(ABC):
    """Base class for all model provider backends."""

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Stable identifier — matches [modelproviders.<name>] type field."""
        ...

    @abstractmethod
    def configure(self, cfg: dict[str, Any]) -> None:
        """Receive the provider's config section on startup."""
        ...

    @abstractmethod
    def complete(self, model: str, req: CompletionRequest) -> CompletionResponse:
        """Run a chat completion. model is the raw model string (e.g. 'qwen3.5:27b')."""
        ...

    def stream_complete(
        self, model: str, req: CompletionRequest
    ) -> Iterator[StreamChunk]:
        """Stream a chat completion token-by-token.

        Default implementation falls back to non-streaming :meth:`complete`.
        Providers override this to yield incremental tokens.
        """
        resp = self.complete(model, req)
        yield StreamChunk(
            text=resp.text,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            done=True,
        )

    def embed(self, model: str, req: EmbedRequest) -> EmbedResponse:
        """Run a batch embedding call. Not all providers support this."""
        raise NotImplementedError(f"{self.provider_type} does not support embedding")

    def rerank(
        self, model: str, query: str, passages: list[str]
    ) -> list[float]:
        """Score (query, passage) pairs and return one float per passage.

        Higher = more relevant.  Used by the cross-encoder reranker
        backend; LLM/embedding providers do not implement this.
        """
        raise NotImplementedError(f"{self.provider_type} does not support reranking")
