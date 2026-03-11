"""Tests for ModelRegistry three-layer config resolution."""

import pytest

from worker.config import Config, ModelConfig, ProviderConfig, RolesConfig
from worker.models.resolver import ModelRegistry, ResolvedModel

# Ensure provider decorators run
import worker.models.providers  # noqa: F401


def _make_config(
    providers: dict | None = None,
    models: dict | None = None,
    roles: dict | None = None,
) -> Config:
    """Build a Config with only the model-related fields populated."""
    cfg = Config()
    if providers:
        for name, pcfg in providers.items():
            cfg.providers[name] = ProviderConfig(
                name=name,
                type=pcfg["type"],
                settings={k: v for k, v in pcfg.items() if k != "type"},
            )
    if models:
        for alias, mcfg in models.items():
            cfg.models[alias] = ModelConfig(
                alias=alias,
                provider=mcfg["provider"],
                model=mcfg["model"],
            )
    if roles:
        cfg.roles = RolesConfig(mapping=roles)
    return cfg


@pytest.fixture()
def basic_config() -> Config:
    return _make_config(
        providers={"local": {"type": "ollama"}},
        models={
            "llm": {"provider": "local", "model": "qwen3.5:27b"},
            "embedder": {"provider": "local", "model": "nomic-embed-text"},
        },
        roles={
            "extraction": "llm",
            "embedding": "embedder",
        },
    )


class TestModelRegistryInit:
    def test_builds_from_config(self, basic_config: Config) -> None:
        reg = ModelRegistry(basic_config)
        assert reg.by_alias("llm").model == "qwen3.5:27b"
        assert reg.by_alias("embedder").model == "nomic-embed-text"

    def test_unknown_provider_type_raises(self) -> None:
        cfg = _make_config(
            providers={"x": {"type": "nonexistent"}},
            models={"m": {"provider": "x", "model": "foo"}},
        )
        with pytest.raises(ValueError, match="No model provider registered"):
            ModelRegistry(cfg)

    def test_model_references_missing_provider_raises(self) -> None:
        cfg = _make_config(
            providers={},
            models={"m": {"provider": "missing", "model": "foo"}},
        )
        with pytest.raises(ValueError, match="references provider 'missing'"):
            ModelRegistry(cfg)

    def test_role_references_missing_alias_raises(self) -> None:
        cfg = _make_config(
            providers={"local": {"type": "ollama"}},
            models={"llm": {"provider": "local", "model": "x"}},
            roles={"extraction": "nonexistent"},
        )
        with pytest.raises(ValueError, match="maps to alias 'nonexistent'"):
            ModelRegistry(cfg)


class TestForRole:
    def test_returns_resolved_model(self, basic_config: Config) -> None:
        reg = ModelRegistry(basic_config)
        m = reg.for_role("extraction")
        assert isinstance(m, ResolvedModel)
        assert m.alias == "llm"
        assert m.model == "qwen3.5:27b"

    def test_unknown_role_raises(self, basic_config: Config) -> None:
        reg = ModelRegistry(basic_config)
        with pytest.raises(KeyError, match="No model configured for role 'summarize'"):
            reg.for_role("summarize")


class TestByAlias:
    def test_returns_resolved_model(self, basic_config: Config) -> None:
        reg = ModelRegistry(basic_config)
        m = reg.by_alias("embedder")
        assert isinstance(m, ResolvedModel)
        assert m.alias == "embedder"

    def test_unknown_alias_raises(self, basic_config: Config) -> None:
        reg = ModelRegistry(basic_config)
        with pytest.raises(KeyError, match="No model configured with alias 'nope'"):
            reg.by_alias("nope")


class TestResolvedModelPassthrough:
    def test_provider_is_set(self, basic_config: Config) -> None:
        reg = ModelRegistry(basic_config)
        m = reg.by_alias("llm")
        assert m.provider.provider_type == "ollama"

    def test_provider_shared_across_aliases(self, basic_config: Config) -> None:
        """Two models on the same provider share the same instance."""
        reg = ModelRegistry(basic_config)
        assert reg.by_alias("llm").provider is reg.by_alias("embedder").provider


class TestEmptyConfig:
    def test_empty_config_creates_empty_registry(self) -> None:
        reg = ModelRegistry(Config())
        with pytest.raises(KeyError):
            reg.by_alias("anything")
        with pytest.raises(KeyError):
            reg.for_role("anything")
