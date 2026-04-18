"""Model providers — import all to trigger @register decorators."""

from . import anthropic  # noqa: F401
from . import ollama  # noqa: F401
from . import openai  # noqa: F401
from . import sentence_transformers  # noqa: F401

__all__ = ["anthropic", "ollama", "openai", "sentence_transformers"]
