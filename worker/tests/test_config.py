"""Tests for config.py — load_config and _parse."""

import pytest

from worker.config import (
    Config,
    CoreConfig,
    ClusteringConfig,
    McpConfig,
    QdrantConfig,
    RolesConfig,
    VisionConfig,
    load_config,
    _parse,
)


class TestParseDefaults:
    """_parse with empty input returns all defaults."""

    def test_empty_dict_returns_defaults(self) -> None:
        cfg = _parse({})
        assert cfg.core == CoreConfig()
        assert cfg.qdrant == QdrantConfig()
        assert cfg.providers == {}
        assert cfg.models == {}
        assert cfg.roles == RolesConfig()
        assert cfg.sources == {}
        assert cfg.clustering == ClusteringConfig()
        assert cfg.mcp == McpConfig()

    def test_neo4j_validate_raises_without_password(self) -> None:
        cfg = _parse({})
        with pytest.raises(ValueError, match="Neo4j password must be set"):
            cfg.neo4j.validate()


class TestParseCore:
    def test_overrides_data_dir(self) -> None:
        cfg = _parse({"core": {"data_dir": "/tmp/fn"}})
        assert cfg.core.data_dir == "/tmp/fn"
        assert cfg.core.log_level == "info"  # default preserved

    def test_overrides_log_level(self) -> None:
        cfg = _parse({"core": {"log_level": "debug"}})
        assert cfg.core.log_level == "debug"


class TestParseNeo4j:
    def test_overrides_all_fields(self) -> None:
        cfg = _parse({"neo4j": {"uri": "bolt://db:7687", "user": "admin", "password": "secret"}})
        assert cfg.neo4j.uri == "bolt://db:7687"
        assert cfg.neo4j.user == "admin"
        assert cfg.neo4j.password == "secret"

    def test_partial_override(self) -> None:
        cfg = _parse({"neo4j": {"password": "new"}})
        assert cfg.neo4j.password == "new"
        assert cfg.neo4j.uri == "bolt://localhost:7687"  # default


class TestParseQdrant:
    def test_overrides_port(self) -> None:
        cfg = _parse({"qdrant": {"port": 6334}})
        assert cfg.qdrant.port == 6334
        assert cfg.qdrant.host == "localhost"

    def test_overrides_collection(self) -> None:
        cfg = _parse({"qdrant": {"collection": "test"}})
        assert cfg.qdrant.collection == "test"


class TestParseProviders:
    def test_single_provider(self) -> None:
        raw = {
            "modelproviders": {
                "local": {"type": "ollama", "base_url": "http://localhost:11434"}
            }
        }
        cfg = _parse(raw)
        assert "local" in cfg.providers
        p = cfg.providers["local"]
        assert p.name == "local"
        assert p.type == "ollama"
        assert p.settings == {"base_url": "http://localhost:11434"}

    def test_multiple_providers(self) -> None:
        raw = {
            "modelproviders": {
                "a": {"type": "ollama"},
                "b": {"type": "openai", "api_key": "sk-xxx"},
            }
        }
        cfg = _parse(raw)
        assert len(cfg.providers) == 2
        assert cfg.providers["b"].settings == {"api_key": "sk-xxx"}


class TestParseModels:
    def test_model_definition(self) -> None:
        raw = {
            "models": {
                "llm": {"provider": "local", "model": "qwen3.5:27b"},
            }
        }
        cfg = _parse(raw)
        assert "llm" in cfg.models
        m = cfg.models["llm"]
        assert m.alias == "llm"
        assert m.provider == "local"
        assert m.model == "qwen3.5:27b"

    def test_roles_section(self) -> None:
        raw = {
            "models": {
                "roles": {"extraction": "llm", "embedding": "emb"},
            }
        }
        cfg = _parse(raw)
        assert cfg.roles.get("extraction") == "llm"
        assert cfg.roles.get("embedding") == "emb"
        assert cfg.roles.get("missing") is None

    def test_models_and_roles_together(self) -> None:
        raw = {
            "models": {
                "llm": {"provider": "local", "model": "qwen3.5:27b"},
                "roles": {"extraction": "llm"},
            }
        }
        cfg = _parse(raw)
        assert "llm" in cfg.models
        assert "roles" not in cfg.models  # roles is not a model
        assert cfg.roles.get("extraction") == "llm"


class TestParseSources:
    def test_source_config(self) -> None:
        raw = {
            "sources": {
                "vault": {"root": "/home/user/notes", "extensions": [".md"]},
            }
        }
        cfg = _parse(raw)
        assert "vault" in cfg.sources
        s = cfg.sources["vault"]
        assert s.name == "vault"
        assert s.settings == {"root": "/home/user/notes", "extensions": [".md"]}


class TestParseVision:
    def test_defaults(self) -> None:
        cfg = _parse({})
        assert cfg.vision == VisionConfig()
        assert cfg.vision.enabled is True
        assert cfg.vision.concurrency == 2
        assert cfg.vision.min_file_size_kb == 1
        assert cfg.vision.max_file_size_mb == 20
        assert cfg.vision.queue_size == 256
        assert "icon" in cfg.vision.skip_patterns

    def test_overrides_all_fields(self) -> None:
        cfg = _parse({"vision": {
            "enabled": False,
            "concurrency": 8,
            "min_file_size_kb": 5,
            "max_file_size_mb": 50,
            "skip_patterns": ["thumb", "banner"],
            "queue_size": 128,
        }})
        assert cfg.vision.enabled is False
        assert cfg.vision.concurrency == 8
        assert cfg.vision.min_file_size_kb == 5
        assert cfg.vision.max_file_size_mb == 50
        assert cfg.vision.skip_patterns == ["thumb", "banner"]
        assert cfg.vision.queue_size == 128

    def test_partial_override_preserves_defaults(self) -> None:
        cfg = _parse({"vision": {"concurrency": 4}})
        assert cfg.vision.concurrency == 4
        assert cfg.vision.enabled is True  # default preserved
        assert cfg.vision.queue_size == 256  # default preserved

    def test_min_file_size_kb_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[vision\] min_file_size_kb: expected int"):
            _parse({"vision": {"min_file_size_kb": "small"}})

    def test_max_file_size_mb_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[vision\] max_file_size_mb: expected int"):
            _parse({"vision": {"max_file_size_mb": 10.5}})


class TestParseClustering:
    def test_overrides(self) -> None:
        cfg = _parse({"clustering": {"enabled": False, "cron": "0 0 * * *"}})
        assert cfg.clustering.enabled is False
        assert cfg.clustering.cron == "0 0 * * *"


class TestParseMcp:
    def test_overrides(self) -> None:
        cfg = _parse({"mcp": {"enabled": False, "port": 9999}})
        assert cfg.mcp.enabled is False
        assert cfg.mcp.port == 9999


class TestLoadConfig:
    def test_loads_from_toml_file(self, tmp_path) -> None:
        toml_content = """\
[core]
data_dir = "/data"
log_level = "warn"

[neo4j]
uri = "bolt://db:7687"
password = "testpass"

[modelproviders.local]
type = "ollama"

[models.llm]
provider = "local"
model = "qwen3.5:27b"

[models.roles]
extraction = "llm"

[sources.vault]
root = "/notes"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        assert cfg.core.data_dir == "/data"
        assert cfg.core.log_level == "warn"
        assert cfg.neo4j.uri == "bolt://db:7687"
        assert "local" in cfg.providers
        assert cfg.models["llm"].model == "qwen3.5:27b"
        assert cfg.roles.get("extraction") == "llm"
        assert cfg.sources["vault"].settings["root"] == "/notes"

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.toml")

    def test_raises_without_neo4j_password(self, tmp_path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text("[core]\nlog_level = 'info'\n")
        with pytest.raises(ValueError, match="Neo4j password must be set"):
            load_config(config_file)


class TestFullRoundtrip:
    """Parse a complete config matching a real-world layout."""

    def test_complete_config(self) -> None:
        raw = {
            "core": {"data_dir": "~/.fieldnotes/data", "log_level": "info"},
            "neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "pw"},
            "qdrant": {"host": "localhost", "port": 6333, "collection": "fn"},
            "modelproviders": {
                "local": {"type": "ollama", "base_url": "http://localhost:11434"},
            },
            "models": {
                "llm": {"provider": "local", "model": "qwen3.5:27b"},
                "emb": {"provider": "local", "model": "nomic-embed-text"},
                "roles": {"extraction": "llm", "embedding": "emb"},
            },
            "sources": {
                "vault": {"root": "/notes"},
            },
            "clustering": {"enabled": True, "cron": "0 3 * * 0"},
            "mcp": {"enabled": True, "port": 3456},
        }

        cfg = _parse(raw)
        assert isinstance(cfg, Config)
        assert len(cfg.providers) == 1
        assert len(cfg.models) == 2
        assert cfg.roles.get("extraction") == "llm"
        assert cfg.roles.get("embedding") == "emb"


class TestTypeValidation:
    """_parse rejects values with wrong types."""

    def test_core_data_dir_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[core\] data_dir: expected str, got int"):
            _parse({"core": {"data_dir": 123}})

    def test_core_log_level_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[core\] log_level: expected str"):
            _parse({"core": {"log_level": True}})

    def test_neo4j_uri_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[neo4j\] uri: expected str"):
            _parse({"neo4j": {"uri": 42}})

    def test_qdrant_port_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[qdrant\] port: expected int, got str"):
            _parse({"qdrant": {"port": "not-a-number"}})

    def test_qdrant_vector_size_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[qdrant\] vector_size: expected int"):
            _parse({"qdrant": {"vector_size": "big"}})

    def test_qdrant_host_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[qdrant\] host: expected str"):
            _parse({"qdrant": {"host": 999}})

    def test_vision_enabled_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[vision\] enabled: expected bool"):
            _parse({"vision": {"enabled": "yes"}})

    def test_vision_concurrency_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[vision\] concurrency: expected int"):
            _parse({"vision": {"concurrency": "fast"}})

    def test_vision_queue_size_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[vision\] queue_size: expected int"):
            _parse({"vision": {"queue_size": 25.5}})

    def test_vision_skip_patterns_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[vision\] skip_patterns: expected list"):
            _parse({"vision": {"skip_patterns": "icon,avatar"}})

    def test_vision_skip_patterns_bad_item(self) -> None:
        with pytest.raises(TypeError, match=r"\[vision\] skip_patterns\[1\]: expected str"):
            _parse({"vision": {"skip_patterns": ["icon", 42]}})

    def test_clustering_enabled_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[clustering\] enabled: expected bool"):
            _parse({"clustering": {"enabled": 1}})

    def test_clustering_min_corpus_size_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[clustering\] min_corpus_size: expected int"):
            _parse({"clustering": {"min_corpus_size": "many"}})

    def test_clustering_cron_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[clustering\] cron: expected str"):
            _parse({"clustering": {"cron": 12345}})

    def test_clustering_min_interval_seconds_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[clustering\] min_interval_seconds: expected float"):
            _parse({"clustering": {"min_interval_seconds": "fast"}})

    def test_clustering_min_interval_seconds_accepts_int(self) -> None:
        cfg = _parse({"clustering": {"min_interval_seconds": 30}})
        assert cfg.clustering.min_interval_seconds == 30

    def test_clustering_min_interval_seconds_rejects_below_10(self) -> None:
        with pytest.raises(ValueError, match=r"min_interval_seconds must be >= 10\.0"):
            _parse({"clustering": {"min_interval_seconds": 5}})

    def test_clustering_min_interval_seconds_accepts_10(self) -> None:
        cfg = _parse({"clustering": {"min_interval_seconds": 10}})
        assert cfg.clustering.min_interval_seconds == 10

    def test_mcp_port_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[mcp\] port: expected int"):
            _parse({"mcp": {"port": "3456"}})

    def test_mcp_enabled_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[mcp\] enabled: expected bool"):
            _parse({"mcp": {"enabled": "true"}})

    def test_mcp_auth_token_parsed(self) -> None:
        cfg = _parse({"mcp": {"auth_token": "secret-abc"}})
        assert cfg.mcp.auth_token == "secret-abc"

    def test_mcp_auth_token_default_none(self) -> None:
        cfg = _parse({"mcp": {}})
        assert cfg.mcp.auth_token is None

    def test_mcp_auth_token_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[mcp\] auth_token: expected str"):
            _parse({"mcp": {"auth_token": 12345}})

    def test_valid_types_still_accepted(self) -> None:
        """Ensure valid configs still parse without errors."""
        cfg = _parse({
            "qdrant": {"port": 6334, "host": "db", "vector_size": 1024},
            "vision": {
                "enabled": False,
                "concurrency": 4,
                "queue_size": 512,
                "skip_patterns": ["icon", "thumb"],
            },
            "clustering": {"enabled": True, "cron": "0 0 * * *", "min_corpus_size": 50},
            "mcp": {"enabled": False, "port": 9999},
        })
        assert cfg.qdrant.port == 6334
        assert cfg.vision.concurrency == 4
        assert cfg.vision.skip_patterns == ["icon", "thumb"]
        assert cfg.clustering.min_corpus_size == 50
        assert cfg.mcp.port == 9999


class TestConfigValidate:
    """Config.validate() — cron, vector_size, skip_patterns, boundaries."""

    def _make_config(self, **overrides) -> Config:
        """Build a Config with sensible defaults, applying overrides."""
        cfg = _parse({})
        for key, value in overrides.items():
            parts = key.split(".")
            obj = cfg
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        return cfg

    # -- Cron validation --

    def test_valid_cron_passes(self) -> None:
        cfg = self._make_config()
        warnings = cfg.validate()  # default "0 3 * * 0" is valid
        assert isinstance(warnings, list)

    def test_invalid_cron_raises(self) -> None:
        cfg = self._make_config(**{"clustering.cron": "not a cron"})
        with pytest.raises(ValueError, match=r"\[clustering\] cron.*not a valid cron"):
            cfg.validate()

    def test_cron_too_many_fields_raises(self) -> None:
        cfg = self._make_config(**{"clustering.cron": "* * * * * * *"})
        with pytest.raises(ValueError, match=r"\[clustering\] cron"):
            cfg.validate()

    # -- Vector size warnings --

    def test_default_vector_size_no_warning(self) -> None:
        cfg = self._make_config()
        warnings = cfg.validate()
        assert not any("vector_size" in w for w in warnings)

    def test_mismatched_vector_size_warns(self) -> None:
        cfg = self._make_config(**{"qdrant.vector_size": 1536})
        warnings = cfg.validate()
        assert any("vector_size is 1536" in w for w in warnings)

    # -- Skip patterns (regex) validation --

    def test_valid_skip_patterns_pass(self) -> None:
        cfg = self._make_config(**{"vision.skip_patterns": ["icon", "thumb.*", r"\d+"]})
        warnings = cfg.validate()
        assert not any("skip_patterns" in w for w in warnings)

    def test_invalid_regex_skip_pattern_raises(self) -> None:
        cfg = self._make_config(**{"vision.skip_patterns": ["icon", "[invalid"]})
        with pytest.raises(ValueError, match=r"\[vision\] skip_patterns\[1\].*not a valid regex"):
            cfg.validate()

    # -- min_interval_seconds boundary warnings --

    def test_min_interval_at_minimum_warns(self) -> None:
        cfg = self._make_config(**{"clustering.min_interval_seconds": 10.0})
        warnings = cfg.validate()
        assert any("at the minimum" in w for w in warnings)

    def test_min_interval_at_maximum_warns(self) -> None:
        cfg = self._make_config(**{"clustering.min_interval_seconds": 86_400.0})
        warnings = cfg.validate()
        assert any("at the maximum" in w for w in warnings)

    def test_min_interval_normal_no_warning(self) -> None:
        cfg = self._make_config(**{"clustering.min_interval_seconds": 60.0})
        warnings = cfg.validate()
        assert not any("min_interval" in w for w in warnings)
