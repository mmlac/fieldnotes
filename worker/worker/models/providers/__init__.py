"""Model providers — import all to trigger @register decorators."""

from . import ollama  # noqa: F401
from . import openai  # noqa: F401

__all__ = ["ollama", "openai"]
