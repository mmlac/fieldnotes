from typing import Type

from .base import ModelProvider

_registry: dict[str, Type[ModelProvider]] = {}


def register(cls: Type[ModelProvider]) -> Type[ModelProvider]:
    instance = cls()
    _registry[instance.provider_type] = cls
    return cls


def get_provider(provider_type: str) -> Type[ModelProvider]:
    cls = _registry.get(provider_type)
    if cls is None:
        raise ValueError(
            f"No model provider registered for type={provider_type!r}. "
            f"Known types: {sorted(_registry)}"
        )
    return cls
