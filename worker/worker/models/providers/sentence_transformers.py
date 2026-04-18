"""SentenceTransformersProvider — local cross-encoder rerank model.

Loads a HuggingFace cross-encoder (e.g. ``BAAI/bge-reranker-v2-m3``) via
the ``sentence-transformers`` library and exposes it through the
:meth:`ModelProvider.rerank` interface.  No completion, no embedding —
this provider is reranker-only.

The model is **lazy-loaded on first request** so a daemon that never
issues a query never pays the multi-second import + weight load cost.
The same instance is cached for the lifetime of the process.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from ..base import CompletionRequest, CompletionResponse, ModelProvider
from ..registry import register

logger = logging.getLogger(__name__)


_SUPPORTED_DEVICES = {"auto", "cpu", "cuda", "mps"}


def _resolve_device(device: str) -> str:
    """Translate the configured device string into a torch device label.

    ``auto`` picks ``cuda`` if available, then ``mps`` (Apple Silicon),
    falling back to ``cpu``.  This matches what every other ML library
    in this stack does and keeps the config readable.
    """
    if device != "auto":
        return device
    try:
        import torch  # imported lazily so non-rerank installs don't pay
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@register
class SentenceTransformersProvider(ModelProvider):
    """Provider for in-process HuggingFace cross-encoders.

    Currently only used for reranking.  ``configure`` records device and
    cache settings; the actual ``CrossEncoder`` is built inside
    :meth:`rerank` on first call (and reused thereafter).
    """

    @property
    def provider_type(self) -> str:
        return "sentence_transformers"

    def __init__(self) -> None:
        self._device: str = "auto"
        self._cache_dir: Path | None = None
        self._models: dict[str, Any] = {}
        self._load_lock = threading.Lock()

    def configure(self, cfg: dict[str, Any]) -> None:
        device = str(cfg.get("device", "auto")).lower()
        if device not in _SUPPORTED_DEVICES:
            raise ValueError(
                f"[modelproviders.<name>] device={device!r} is not supported. "
                f"Use one of: {sorted(_SUPPORTED_DEVICES)}"
            )
        self._device = device

        cache_dir_raw = cfg.get("cache_dir")
        if cache_dir_raw:
            self._cache_dir = Path(str(cache_dir_raw)).expanduser()
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "sentence_transformers provider configured (device=%s, cache_dir=%s)",
            self._device,
            self._cache_dir,
        )

    def complete(
        self, model: str, req: CompletionRequest
    ) -> CompletionResponse:
        raise NotImplementedError(
            "sentence_transformers provider is reranker-only; "
            "do not bind it to chat/extract/query roles"
        )

    # embed() inherits the NotImplementedError default from ModelProvider.

    def _load_model(self, model: str) -> Any:
        cached = self._models.get(model)
        if cached is not None:
            return cached
        with self._load_lock:
            cached = self._models.get(model)
            if cached is not None:
                return cached
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Run `pip install sentence-transformers` (already a "
                    "dependency of fieldnotes; this means the install is "
                    "incomplete)."
                ) from exc

            device = _resolve_device(self._device)
            cache_folder = str(self._cache_dir) if self._cache_dir else None
            logger.info(
                "Loading cross-encoder %s on device=%s (this may take a moment on first use)",
                model,
                device,
            )
            instance = CrossEncoder(
                model,
                device=device,
                cache_folder=cache_folder,
            )
            self._models[model] = instance
            return instance

    def rerank(
        self, model: str, query: str, passages: list[str]
    ) -> list[float]:
        if not passages:
            return []
        ce = self._load_model(model)
        pairs = [(query, p) for p in passages]
        scores = ce.predict(pairs, convert_to_numpy=True)
        # CrossEncoder returns numpy.ndarray; normalize to plain floats.
        return [float(s) for s in scores]
