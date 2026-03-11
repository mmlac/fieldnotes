"""ModelRegistry — three-layer config resolution (providers → models → roles).

Single entry point for the pipeline to access any model. Resolution layers:
  1. [modelproviders.*] → configured provider instances
  2. [models.*] → alias → ResolvedModel (skipping 'roles' key)
  3. [models.roles] → role → ResolvedModel
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import (
    CompletionRequest,
    CompletionResponse,
    EmbedRequest,
    EmbedResponse,
    ModelProvider,
)
from .registry import get_provider
from ..config import Config


@dataclass
class ResolvedModel:
    """A fully resolved model: alias + model string + configured provider instance."""

    alias: str
    model: str
    provider: ModelProvider

    def complete(self, req: CompletionRequest) -> CompletionResponse:
        """Passthrough to the underlying provider's complete()."""
        return self.provider.complete(self.model, req)

    def embed(self, req: EmbedRequest) -> EmbedResponse:
        """Passthrough to the underlying provider's embed()."""
        return self.provider.embed(self.model, req)


class ModelRegistry:
    """Resolves parsed config into ready-to-use model handles.

    Three-layer resolution:
      1. [modelproviders.*] → provider instances (configured via .configure())
      2. [models.*] → alias → ResolvedModel
      3. [models.roles] → role → ResolvedModel
    """

    def __init__(self, cfg: Config) -> None:
        # Layer 1: build configured provider instances
        providers: dict[str, ModelProvider] = {}
        for name, pcfg in cfg.providers.items():
            cls = get_provider(pcfg.type)
            instance = cls()
            instance.configure(pcfg.settings)
            providers[name] = instance

        # Layer 2: alias → ResolvedModel
        self._by_alias: dict[str, ResolvedModel] = {}
        for alias, mcfg in cfg.models.items():
            provider = providers.get(mcfg.provider)
            if provider is None:
                raise ValueError(
                    f"Model {alias!r} references provider {mcfg.provider!r}, "
                    f"but no such provider is configured. "
                    f"Known providers: {sorted(providers)}"
                )
            self._by_alias[alias] = ResolvedModel(
                alias=alias,
                model=mcfg.model,
                provider=provider,
            )

        # Layer 3: role → ResolvedModel
        self._by_role: dict[str, ResolvedModel] = {}
        for role, alias in cfg.roles.mapping.items():
            resolved = self._by_alias.get(alias)
            if resolved is None:
                raise ValueError(
                    f"Role {role!r} maps to alias {alias!r}, "
                    f"but no such model is defined. "
                    f"Known aliases: {sorted(self._by_alias)}"
                )
            self._by_role[role] = resolved

    def for_role(self, role: str) -> ResolvedModel:
        """Look up a model by pipeline role (e.g. 'extraction', 'embedding')."""
        resolved = self._by_role.get(role)
        if resolved is None:
            raise KeyError(
                f"No model configured for role {role!r}. "
                f"Known roles: {sorted(self._by_role)}"
            )
        return resolved

    def by_alias(self, alias: str) -> ResolvedModel:
        """Look up a model by its config alias."""
        resolved = self._by_alias.get(alias)
        if resolved is None:
            raise KeyError(
                f"No model configured with alias {alias!r}. "
                f"Known aliases: {sorted(self._by_alias)}"
            )
        return resolved
