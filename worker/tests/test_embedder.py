"""Tests for the embedder pipeline stage."""

import pytest

from worker.config import Config, ModelConfig, ProviderConfig, RolesConfig
from worker.models.base import EmbedRequest, EmbedResponse, ModelProvider
from worker.models.resolver import ModelRegistry
from worker.pipeline.embedder import embed_chunks

# Ensure provider decorators run
import worker.models.providers  # noqa: F401


VECTOR_DIM = 768


def _make_config(role_name: str = "embed") -> Config:
    cfg = Config()
    cfg.providers["local"] = ProviderConfig(name="local", type="ollama", settings={})
    cfg.models["embedder"] = ModelConfig(
        alias="embedder", provider="local", model="nomic-embed-text"
    )
    cfg.roles = RolesConfig(mapping={role_name: "embedder"})
    return cfg


def _fake_vector(text: str) -> list[float]:
    """Deterministic fake vector derived from text length."""
    val = len(text) / 1000.0
    return [val] * VECTOR_DIM


class FakeProvider(ModelProvider):
    """Test double that returns deterministic vectors without network calls."""

    def __init__(self) -> None:
        self.embed_calls: list[EmbedRequest] = []

    @property
    def provider_type(self) -> str:
        return "ollama"

    def configure(self, cfg: dict) -> None:
        pass

    def complete(self, model, req):
        raise NotImplementedError

    def embed(self, model: str, req: EmbedRequest) -> EmbedResponse:
        self.embed_calls.append(req)
        vectors = [_fake_vector(t) for t in req.texts]
        return EmbedResponse(
            vectors=vectors, model=model, input_tokens=len(req.texts) * 10
        )


@pytest.fixture()
def fake_registry() -> tuple[ModelRegistry, FakeProvider]:
    """Registry with a FakeProvider swapped in."""
    cfg = _make_config()
    reg = ModelRegistry(cfg)
    fake = FakeProvider()
    # Swap in fake provider
    resolved = reg.for_role("embed")
    resolved.provider = fake
    return reg, fake


class TestEmbedChunks:
    def test_empty_input(self, fake_registry: tuple) -> None:
        reg, _ = fake_registry
        assert embed_chunks([], reg) == []

    def test_single_chunk(self, fake_registry: tuple) -> None:
        reg, fake = fake_registry
        results = embed_chunks(["hello world"], reg)
        assert len(results) == 1
        text, vec = results[0]
        assert text == "hello world"
        assert len(vec) == VECTOR_DIM
        assert vec == _fake_vector("hello world")
        assert len(fake.embed_calls) == 1

    def test_multiple_chunks(self, fake_registry: tuple) -> None:
        reg, fake = fake_registry
        chunks = ["chunk one", "chunk two", "chunk three"]
        results = embed_chunks(chunks, reg)
        assert len(results) == 3
        assert [r[0] for r in results] == chunks
        for text, vec in results:
            assert len(vec) == VECTOR_DIM
            assert vec == _fake_vector(text)

    def test_preserves_order(self, fake_registry: tuple) -> None:
        reg, _ = fake_registry
        chunks = [f"chunk-{i}" for i in range(10)]
        results = embed_chunks(chunks, reg)
        assert [r[0] for r in results] == chunks

    def test_batching(self, fake_registry: tuple) -> None:
        reg, fake = fake_registry
        chunks = [f"text-{i}" for i in range(5)]
        results = embed_chunks(chunks, reg, batch_size=2)
        assert len(results) == 5
        # 5 chunks with batch_size=2 → 3 API calls (2+2+1)
        assert len(fake.embed_calls) == 3
        assert len(fake.embed_calls[0].texts) == 2
        assert len(fake.embed_calls[1].texts) == 2
        assert len(fake.embed_calls[2].texts) == 1

    def test_custom_role(self) -> None:
        cfg = _make_config(role_name="embedding")
        reg = ModelRegistry(cfg)
        fake = FakeProvider()
        reg.for_role("embedding").provider = fake
        results = embed_chunks(["test"], reg, role="embedding")
        assert len(results) == 1

    def test_vector_mismatch_raises(self, fake_registry: tuple) -> None:
        reg, fake = fake_registry

        # Monkey-patch to return wrong number of vectors
        original_embed = fake.embed

        def bad_embed(model, req):
            resp = original_embed(model, req)
            resp.vectors = resp.vectors[:-1]  # drop one
            return resp

        fake.embed = bad_embed
        with pytest.raises(ValueError, match="Expected 2 vectors, got 1"):
            embed_chunks(["a", "b"], reg)

    def test_unknown_role_raises(self, fake_registry: tuple) -> None:
        reg, _ = fake_registry
        with pytest.raises(KeyError, match="No model configured for role"):
            embed_chunks(["test"], reg, role="nonexistent")
