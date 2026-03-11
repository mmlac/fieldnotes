from typing import Type

from .base import BaseParser

_registry: dict[str, Type[BaseParser]] = {}


def register(cls: Type[BaseParser]) -> Type[BaseParser]:
    """Class decorator that registers a parser. Use on every BaseParser subclass."""
    instance = cls()
    _registry[instance.source_type] = cls
    return cls


def get(source_type: str) -> BaseParser:
    """Return a new instance of the parser registered for the given source_type."""
    cls = _registry.get(source_type)
    if cls is None:
        raise ValueError(
            f"No parser registered for source_type={source_type!r}. "
            f"Known types: {sorted(_registry)}"
        )
    return cls()
