"""Tests for config.py — load_config and _parse."""

from pathlib import Path

import pytest

from worker.config import (
    CalendarAccountConfig,
    Config,
    CoreConfig,
    ClusteringConfig,
    GmailAccountConfig,
    McpConfig,
    MeConfig,
    MigrationRequiredError,
    QdrantConfig,
    RolesConfig,
    SlackSourceConfig,
    VisionConfig,
    load_config,
    _parse,
)
from worker.parsers.attachments import DEFAULT_INDEXABLE_MIMETYPES


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
        cfg = _parse(
            {"neo4j": {"uri": "bolt://db:7687", "user": "admin", "password": "secret"}}
        )
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
        cfg = _parse(
            {
                "vision": {
                    "enabled": False,
                    "concurrency": 8,
                    "min_file_size_kb": 5,
                    "max_file_size_mb": 50,
                    "skip_patterns": ["thumb", "banner"],
                    "queue_size": 128,
                }
            }
        )
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
        with pytest.raises(
            TypeError, match=r"\[vision\] min_file_size_kb: expected int"
        ):
            _parse({"vision": {"min_file_size_kb": "small"}})

    def test_max_file_size_mb_wrong_type(self) -> None:
        with pytest.raises(
            TypeError, match=r"\[vision\] max_file_size_mb: expected int"
        ):
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
            "neo4j": {
                "uri": "bolt://localhost:7687",
                "user": "neo4j",
                "password": "pw",
            },
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
        with pytest.raises(
            TypeError, match=r"\[core\] data_dir: expected str, got int"
        ):
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
        with pytest.raises(
            TypeError, match=r"\[vision\] skip_patterns\[1\]: expected str"
        ):
            _parse({"vision": {"skip_patterns": ["icon", 42]}})

    def test_clustering_enabled_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[clustering\] enabled: expected bool"):
            _parse({"clustering": {"enabled": 1}})

    def test_clustering_min_corpus_size_wrong_type(self) -> None:
        with pytest.raises(
            TypeError, match=r"\[clustering\] min_corpus_size: expected int"
        ):
            _parse({"clustering": {"min_corpus_size": "many"}})

    def test_clustering_cron_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[clustering\] cron: expected str"):
            _parse({"clustering": {"cron": 12345}})

    def test_clustering_min_interval_seconds_wrong_type(self) -> None:
        with pytest.raises(
            TypeError, match=r"\[clustering\] min_interval_seconds: expected float"
        ):
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

    def test_mcp_auth_token_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIELDNOTES_MCP_AUTH_TOKEN", "env-token-xyz")
        cfg = _parse({"mcp": {}})
        assert cfg.mcp.auth_token == "env-token-xyz"

    def test_mcp_auth_token_config_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FIELDNOTES_MCP_AUTH_TOKEN", "env-token-xyz")
        cfg = _parse({"mcp": {"auth_token": "config-token"}})
        assert cfg.mcp.auth_token == "config-token"

    def test_mcp_auth_token_empty_env_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FIELDNOTES_MCP_AUTH_TOKEN", "")
        cfg = _parse({"mcp": {}})
        assert cfg.mcp.auth_token is None

    def test_valid_types_still_accepted(self) -> None:
        """Ensure valid configs still parse without errors."""
        cfg = _parse(
            {
                "qdrant": {"port": 6334, "host": "db", "vector_size": 1024},
                "vision": {
                    "enabled": False,
                    "concurrency": 4,
                    "queue_size": 512,
                    "skip_patterns": ["icon", "thumb"],
                },
                "clustering": {
                    "enabled": True,
                    "cron": "0 0 * * *",
                    "min_corpus_size": 50,
                },
                "mcp": {"enabled": False, "port": 9999},
            }
        )
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
        with pytest.raises(
            ValueError, match=r"\[vision\] skip_patterns\[1\].*not a valid regex"
        ):
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


class TestParseSlackSource:
    """[sources.slack] parses into a typed SlackSourceConfig."""

    def test_defaults(self) -> None:
        cfg = _parse({})
        assert cfg.slack == SlackSourceConfig()
        assert cfg.slack.enabled is False
        assert cfg.slack.poll_interval_seconds == 300
        assert cfg.slack.max_initial_days == 90
        assert cfg.slack.include_channels == []
        assert cfg.slack.exclude_channels == []
        assert cfg.slack.include_dms is True
        assert cfg.slack.include_archived is False
        assert cfg.slack.window_max_tokens == 512
        assert cfg.slack.window_gap_seconds == 1800
        assert cfg.slack.window_overlap_messages == 3
        assert cfg.slack.download_attachments is False
        # When disabled, no [sources.slack] entry leaks into cfg.sources.
        assert "slack" not in cfg.sources

    def test_partial_override_disabled(self) -> None:
        cfg = _parse(
            {
                "sources": {
                    "slack": {
                        "poll_interval_seconds": 600,
                        "include_dms": False,
                        "window_max_tokens": 1024,
                    }
                }
            }
        )
        assert cfg.slack.enabled is False
        assert cfg.slack.poll_interval_seconds == 600
        assert cfg.slack.include_dms is False
        assert cfg.slack.window_max_tokens == 1024
        # Defaults preserved.
        assert cfg.slack.window_gap_seconds == 1800

    def test_enabled_with_existing_secrets(self, tmp_path) -> None:
        secrets = tmp_path / "slack.json"
        secrets.write_text("{}")
        cfg = _parse(
            {
                "sources": {
                    "slack": {
                        "enabled": True,
                        "client_secrets_path": str(secrets),
                    }
                }
            }
        )
        assert cfg.slack.enabled is True
        assert cfg.slack.client_secrets_path == str(secrets)

    def test_enabled_without_secrets_raises(self, tmp_path) -> None:
        missing = tmp_path / "does-not-exist.json"
        with pytest.raises(
            ValueError, match=r"\[sources.slack\] enabled=true but client_secrets_path"
        ):
            _parse(
                {
                    "sources": {
                        "slack": {
                            "enabled": True,
                            "client_secrets_path": str(missing),
                        }
                    }
                }
            )

    def test_disabled_with_missing_secrets_does_not_raise(self, tmp_path) -> None:
        missing = tmp_path / "does-not-exist.json"
        cfg = _parse(
            {
                "sources": {
                    "slack": {
                        "enabled": False,
                        "client_secrets_path": str(missing),
                    }
                }
            }
        )
        assert cfg.slack.enabled is False

    # -- Window bounds --

    def test_window_max_tokens_below_min_raises(self) -> None:
        with pytest.raises(
            ValueError, match=r"window_max_tokens must be in \[128, 4096\]"
        ):
            _parse({"sources": {"slack": {"window_max_tokens": 64}}})

    def test_window_max_tokens_above_max_raises(self) -> None:
        with pytest.raises(
            ValueError, match=r"window_max_tokens must be in \[128, 4096\]"
        ):
            _parse({"sources": {"slack": {"window_max_tokens": 8192}}})

    def test_window_max_tokens_at_bounds(self) -> None:
        cfg = _parse({"sources": {"slack": {"window_max_tokens": 128}}})
        assert cfg.slack.window_max_tokens == 128
        cfg = _parse({"sources": {"slack": {"window_max_tokens": 4096}}})
        assert cfg.slack.window_max_tokens == 4096

    def test_window_gap_seconds_out_of_range_raises(self) -> None:
        with pytest.raises(
            ValueError, match=r"window_gap_seconds must be in \[60, 86400\]"
        ):
            _parse({"sources": {"slack": {"window_gap_seconds": 30}}})
        with pytest.raises(
            ValueError, match=r"window_gap_seconds must be in \[60, 86400\]"
        ):
            _parse({"sources": {"slack": {"window_gap_seconds": 100_000}}})

    def test_window_overlap_messages_out_of_range_raises(self) -> None:
        with pytest.raises(
            ValueError, match=r"window_overlap_messages must be in \[0, 10\]"
        ):
            _parse({"sources": {"slack": {"window_overlap_messages": -1}}})
        with pytest.raises(
            ValueError, match=r"window_overlap_messages must be in \[0, 10\]"
        ):
            _parse({"sources": {"slack": {"window_overlap_messages": 11}}})

    def test_window_overlap_messages_at_bounds(self) -> None:
        cfg = _parse({"sources": {"slack": {"window_overlap_messages": 0}}})
        assert cfg.slack.window_overlap_messages == 0
        cfg = _parse({"sources": {"slack": {"window_overlap_messages": 10}}})
        assert cfg.slack.window_overlap_messages == 10

    # -- Type validation --

    def test_enabled_wrong_type(self) -> None:
        with pytest.raises(
            TypeError, match=r"\[sources.slack\] enabled: expected bool"
        ):
            _parse({"sources": {"slack": {"enabled": "yes"}}})

    def test_client_secrets_path_wrong_type(self) -> None:
        with pytest.raises(
            TypeError, match=r"\[sources.slack\] client_secrets_path: expected str"
        ):
            _parse({"sources": {"slack": {"client_secrets_path": 42}}})

    def test_include_channels_wrong_item_type(self) -> None:
        with pytest.raises(
            TypeError, match=r"\[sources.slack\] include_channels\[0\]: expected str"
        ):
            _parse({"sources": {"slack": {"include_channels": [123]}}})


class TestSlackValidateWarnings:
    """Config.validate() — Slack include/exclude mutual-exclusivity warning."""

    def test_both_include_and_exclude_warns(self) -> None:
        cfg = _parse(
            {
                "sources": {
                    "slack": {
                        "include_channels": ["#general"],
                        "exclude_channels": ["#noise"],
                    }
                }
            }
        )
        warnings = cfg.validate()
        assert any(
            "include_channels and exclude_channels are both" in w for w in warnings
        )

    def test_only_include_no_warning(self) -> None:
        cfg = _parse({"sources": {"slack": {"include_channels": ["#general"]}}})
        warnings = cfg.validate()
        assert not any("include_channels and exclude_channels" in w for w in warnings)

    def test_only_exclude_no_warning(self) -> None:
        cfg = _parse({"sources": {"slack": {"exclude_channels": ["#noise"]}}})
        warnings = cfg.validate()
        assert not any("include_channels and exclude_channels" in w for w in warnings)

    def test_neither_no_warning(self) -> None:
        cfg = _parse({})
        warnings = cfg.validate()
        assert not any("include_channels and exclude_channels" in w for w in warnings)


def _install_fake_slack_auth(monkeypatch, check_fn):
    """Install a fake worker.sources.slack_auth module exporting check_slack_auth."""
    import sys
    import types

    sources_mod = sys.modules.get("worker.sources")
    if sources_mod is None:
        sources_mod = types.ModuleType("worker.sources")
        monkeypatch.setitem(sys.modules, "worker.sources", sources_mod)

    fake = types.ModuleType("worker.sources.slack_auth")
    fake.check_slack_auth = check_fn  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "worker.sources.slack_auth", fake)
    monkeypatch.setattr(sources_mod, "slack_auth", fake, raising=False)


class TestDoctorSlack:
    """doctor.check_slack — disabled vs enabled paths."""

    def test_disabled_returns_zero(self, capsys) -> None:
        from worker.doctor import check_slack

        cfg = SlackSourceConfig(enabled=False)
        errors = check_slack(cfg)
        out = capsys.readouterr().out
        assert errors == 0
        assert "Slack disabled" in out

    def test_enabled_missing_secrets_file_fails(self, tmp_path, capsys) -> None:
        from worker.doctor import check_slack

        missing = tmp_path / "absent.json"
        cfg = SlackSourceConfig(enabled=True, client_secrets_path=str(missing))
        errors = check_slack(cfg)
        out = capsys.readouterr().out
        assert errors == 1
        assert "client_secrets_path missing" in out

    def test_enabled_with_secrets_no_auth_module(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        """If the slack_auth sister module isn't available, warn and pass."""
        import sys

        from worker import doctor

        secrets = tmp_path / "slack.json"
        secrets.write_text("{}")
        cfg = SlackSourceConfig(enabled=True, client_secrets_path=str(secrets))

        # Make any future import of worker.sources.slack_auth raise ImportError.
        monkeypatch.setitem(sys.modules, "worker.sources.slack_auth", None)
        errors = doctor.check_slack(cfg)
        out = capsys.readouterr().out
        assert errors == 0
        assert "Slack client secrets present" in out
        assert "auth module not available" in out

    def test_enabled_auth_ok(self, tmp_path, capsys, monkeypatch) -> None:
        from worker import doctor

        secrets = tmp_path / "slack.json"
        secrets.write_text("{}")
        cfg = SlackSourceConfig(enabled=True, client_secrets_path=str(secrets))

        _install_fake_slack_auth(monkeypatch, lambda *a, **k: 0)
        monkeypatch.setattr(doctor, "check_slack_auth", lambda *a, **k: 0)
        errors = doctor.check_slack(cfg)
        out = capsys.readouterr().out
        assert errors == 0
        assert "Slack client secrets present" in out

    def test_enabled_auth_fails(self, tmp_path, capsys, monkeypatch) -> None:
        from worker import doctor

        secrets = tmp_path / "slack.json"
        secrets.write_text("{}")
        cfg = SlackSourceConfig(enabled=True, client_secrets_path=str(secrets))

        _install_fake_slack_auth(monkeypatch, lambda *a, **k: 1)
        monkeypatch.setattr(doctor, "check_slack_auth", lambda *a, **k: 1)
        errors = doctor.check_slack(cfg)
        assert errors == 1


class TestParseGmailMultiAccount:
    """[sources.gmail.<account>] keyed table parses into cfg.gmail."""

    def test_no_gmail_section_yields_empty_dict(self) -> None:
        cfg = _parse({})
        assert cfg.gmail == {}

    def test_single_account(self) -> None:
        raw = {
            "sources": {
                "gmail": {
                    "personal": {
                        "client_secrets_path": "/secrets/gmail.json",
                        "poll_interval_seconds": 600,
                        "max_initial_threads": 200,
                        "label_filter": "INBOX",
                    }
                }
            }
        }
        cfg = _parse(raw)
        assert "personal" in cfg.gmail
        acct = cfg.gmail["personal"]
        assert acct.name == "personal"
        assert acct.enabled is True  # presence implies enabled
        assert acct.client_secrets_path == "/secrets/gmail.json"
        assert acct.poll_interval_seconds == 600
        assert acct.max_initial_threads == 200
        assert acct.label_filter == "INBOX"
        # gmail accounts must NOT leak into the generic sources dict.
        assert "gmail" not in cfg.sources

    def test_two_accounts(self) -> None:
        raw = {
            "sources": {
                "gmail": {
                    "personal": {"client_secrets_path": "/s/a.json"},
                    "work": {"client_secrets_path": "/s/b.json"},
                }
            }
        }
        cfg = _parse(raw)
        assert set(cfg.gmail.keys()) == {"personal", "work"}
        assert cfg.gmail["personal"].client_secrets_path == "/s/a.json"
        assert cfg.gmail["work"].client_secrets_path == "/s/b.json"

    def test_shared_client_secrets_path(self) -> None:
        """Two accounts can point at the same OAuth client file."""
        shared = "/secrets/shared.json"
        raw = {
            "sources": {
                "gmail": {
                    "personal": {"client_secrets_path": shared},
                    "work": {"client_secrets_path": shared},
                }
            }
        }
        cfg = _parse(raw)
        assert cfg.gmail["personal"].client_secrets_path == shared
        assert cfg.gmail["work"].client_secrets_path == shared

    def test_enabled_false_keeps_account(self) -> None:
        cfg = _parse(
            {
                "sources": {
                    "gmail": {
                        "personal": {
                            "enabled": False,
                            "client_secrets_path": "/s.json",
                        }
                    }
                }
            }
        )
        assert cfg.gmail["personal"].enabled is False

    def test_enabled_false_skips_required_secrets_path(self) -> None:
        """A disabled account need not define client_secrets_path."""
        cfg = _parse({"sources": {"gmail": {"placeholder": {"enabled": False}}}})
        acct = cfg.gmail["placeholder"]
        assert acct.enabled is False
        # default path still populated
        assert acct.client_secrets_path.endswith(".json")

    def test_empty_account_section_raises(self) -> None:
        with pytest.raises(
            ValueError, match=r"sources\.gmail\.foo.*must define client_secrets_path"
        ):
            _parse({"sources": {"gmail": {"foo": {}}}})

    # -- Account name validation --

    def test_rejects_uppercase_name(self) -> None:
        with pytest.raises(
            ValueError, match=r"sources\.gmail.*account name 'Personal' is invalid"
        ):
            _parse(
                {"sources": {"gmail": {"Personal": {"client_secrets_path": "/s.json"}}}}
            )

    def test_rejects_leading_digit_name(self) -> None:
        with pytest.raises(
            ValueError, match=r"sources\.gmail.*account name '1abc' is invalid"
        ):
            _parse({"sources": {"gmail": {"1abc": {"client_secrets_path": "/s.json"}}}})

    def test_rejects_too_long_name(self) -> None:
        too_long = "a-very-long-name-that-exceeds-30-chars"
        with pytest.raises(
            ValueError, match=r"sources\.gmail.*account name.*is invalid"
        ):
            _parse(
                {"sources": {"gmail": {too_long: {"client_secrets_path": "/s.json"}}}}
            )

    def test_accepts_underscore_and_hyphen(self) -> None:
        cfg = _parse(
            {
                "sources": {
                    "gmail": {
                        "work_2024": {"client_secrets_path": "/a.json"},
                        "side-proj": {"client_secrets_path": "/b.json"},
                    }
                }
            }
        )
        assert "work_2024" in cfg.gmail
        assert "side-proj" in cfg.gmail

    # -- Old-shape detection --

    def test_old_shape_raises_migration_error(self) -> None:
        with pytest.raises(MigrationRequiredError) as exc_info:
            _parse(
                {
                    "sources": {
                        "gmail": {
                            "client_secrets_path": "/secrets/gmail.json",
                            "poll_interval_seconds": 300,
                        }
                    }
                }
            )
        msg = str(exc_info.value)
        assert "Multi-account schema is required" in msg
        assert "fieldnotes migrate gmail-multiaccount" in msg
        assert "[sources.gmail.<account>]" in msg
        assert "client_secrets_path" in msg
        assert "poll_interval_seconds" in msg

    def test_detect_old_shape_pure_old(self) -> None:
        """Only [sources.gmail] with leaf keys -> raises pure-old message."""
        with pytest.raises(MigrationRequiredError) as exc_info:
            _parse(
                {
                    "sources": {
                        "gmail": {
                            "client_secrets_path": "/s.json",
                            "label_filter": "INBOX",
                        }
                    }
                }
            )
        msg = str(exc_info.value)
        assert "legacy single-account shape" in msg
        assert "client_secrets_path" in msg
        assert "label_filter" in msg

    def test_detect_old_shape_pure_new(self) -> None:
        """Only [sources.gmail.<account>.*] sub-tables -> no raise."""
        cfg = _parse(
            {
                "sources": {
                    "gmail": {
                        "personal": {"client_secrets_path": "/p.json"},
                        "work": {"client_secrets_path": "/w.json"},
                    }
                }
            }
        )
        assert set(cfg.gmail.keys()) == {"personal", "work"}

    def test_detect_old_shape_mixed(self) -> None:
        """Both shapes present -> raises ambiguous message naming triggers."""
        with pytest.raises(MigrationRequiredError) as exc_info:
            _parse(
                {
                    "sources": {
                        "gmail": {
                            "personal": {"client_secrets_path": "/p.json"},
                            "client_secrets_path": "/stray.json",
                            "label_filter": "INBOX",
                        }
                    }
                }
            )
        msg = str(exc_info.value)
        assert "ambiguous" in msg
        assert "client_secrets_path" in msg
        assert "label_filter" in msg
        assert "[sources.gmail.<account>]" in msg

    def test_detect_old_shape_empty_table(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Empty [sources.gmail] -> warn, do not raise; treated as absent."""
        with caplog.at_level("WARNING", logger="worker.config"):
            cfg = _parse({"sources": {"gmail": {}}})
        assert cfg.gmail == {}
        assert any("[sources.gmail] is empty" in rec.message for rec in caplog.records)

    def test_no_false_positive_from_comments(self, tmp_path: Path) -> None:
        """tomllib strips comments; '# [sources.gmail]' must not trigger."""
        cfg_text = (
            "[neo4j]\n"
            'password = "x"\n'
            "\n"
            "# Old shape (commented out for reference):\n"
            "# [sources.gmail]\n"
            '# client_secrets_path = "/old.json"\n'
            "\n"
            "[sources.gmail.personal]\n"
            'client_secrets_path = "/new.json"\n'
        )
        path = tmp_path / "config.toml"
        path.write_text(cfg_text)
        cfg = load_config(path)
        assert "personal" in cfg.gmail

    # -- Type validation per-account --

    def test_account_enabled_wrong_type(self) -> None:
        with pytest.raises(
            TypeError, match=r"sources\.gmail\.foo\] enabled: expected bool"
        ):
            _parse(
                {
                    "sources": {
                        "gmail": {
                            "foo": {
                                "enabled": "yes",
                                "client_secrets_path": "/s.json",
                            }
                        }
                    }
                }
            )

    def test_account_label_filter_wrong_type(self) -> None:
        with pytest.raises(
            TypeError, match=r"sources\.gmail\.foo\] label_filter: expected str"
        ):
            _parse(
                {
                    "sources": {
                        "gmail": {
                            "foo": {
                                "client_secrets_path": "/s.json",
                                "label_filter": ["INBOX"],
                            }
                        }
                    }
                }
            )


class TestParseCalendarMultiAccount:
    """[sources.google_calendar.<account>] parses into cfg.google_calendar."""

    def test_no_section_yields_empty_dict(self) -> None:
        cfg = _parse({})
        assert cfg.google_calendar == {}

    def test_single_account_defaults(self) -> None:
        cfg = _parse(
            {
                "sources": {
                    "google_calendar": {"personal": {"client_secrets_path": "/s.json"}}
                }
            }
        )
        acct = cfg.google_calendar["personal"]
        assert acct.name == "personal"
        assert acct.enabled is True
        assert acct.poll_interval_seconds == 300
        assert acct.max_initial_days == 90
        assert acct.calendar_ids == ["primary"]

    def test_two_accounts_with_overrides(self) -> None:
        raw = {
            "sources": {
                "google_calendar": {
                    "personal": {
                        "client_secrets_path": "/s/a.json",
                        "calendar_ids": ["primary", "family@group.calendar"],
                    },
                    "work": {
                        "client_secrets_path": "/s/b.json",
                        "max_initial_days": 30,
                        "calendar_ids": ["primary"],
                    },
                }
            }
        }
        cfg = _parse(raw)
        assert set(cfg.google_calendar.keys()) == {"personal", "work"}
        assert cfg.google_calendar["personal"].calendar_ids == [
            "primary",
            "family@group.calendar",
        ]
        assert cfg.google_calendar["work"].max_initial_days == 30
        # calendar accounts do NOT leak into the generic sources dict.
        assert "google_calendar" not in cfg.sources

    def test_old_shape_raises_migration_error(self) -> None:
        with pytest.raises(MigrationRequiredError) as exc_info:
            _parse(
                {
                    "sources": {
                        "google_calendar": {
                            "client_secrets_path": "/secrets/g.json",
                            "calendar_ids": ["primary"],
                        }
                    }
                }
            )
        msg = str(exc_info.value)
        assert "fieldnotes migrate gmail-multiaccount" in msg
        assert "[sources.google_calendar.<account>]" in msg

    def test_rejects_invalid_account_name(self) -> None:
        with pytest.raises(
            ValueError, match=r"sources\.google_calendar.*account name '1abc'"
        ):
            _parse(
                {
                    "sources": {
                        "google_calendar": {"1abc": {"client_secrets_path": "/s.json"}}
                    }
                }
            )

    def test_empty_account_section_raises(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"sources\.google_calendar\.foo.*must define client_secrets_path",
        ):
            _parse({"sources": {"google_calendar": {"foo": {}}}})

    def test_calendar_ids_wrong_item_type(self) -> None:
        with pytest.raises(
            TypeError,
            match=r"sources\.google_calendar\.foo\] calendar_ids\[0\]: expected str",
        ):
            _parse(
                {
                    "sources": {
                        "google_calendar": {
                            "foo": {
                                "client_secrets_path": "/s.json",
                                "calendar_ids": [123],
                            }
                        }
                    }
                }
            )


class TestAccountNameReservedWords:
    """Account names that match a source word produce ambiguous cursor keys
    like ``gmail:gmail`` and must be rejected at config parse time."""

    @pytest.mark.parametrize(
        "source,reserved",
        [
            ("gmail", "gmail"),
            ("gmail", "calendar"),
            ("gmail", "google_calendar"),
            ("gmail", "slack"),
            ("google_calendar", "gmail"),
            ("google_calendar", "calendar"),
            ("google_calendar", "google_calendar"),
            ("google_calendar", "slack"),
        ],
    )
    def test_reserved_account_name_rejected(self, source: str, reserved: str) -> None:
        with pytest.raises(
            ValueError,
            match=rf"sources\.{source}.*account name {reserved!r} is reserved",
        ):
            _parse(
                {
                    "sources": {
                        source: {reserved: {"client_secrets_path": "/s.json"}}
                    }
                }
            )

    @pytest.mark.parametrize(
        "name", ["gmail-work", "gmail2", "mygmail", "calendar-personal", "myslack"]
    )
    def test_adjacent_names_allowed(self, name: str) -> None:
        cfg = _parse(
            {"sources": {"gmail": {name: {"client_secrets_path": "/s.json"}}}}
        )
        assert name in cfg.gmail

    def test_default_account_name_allowed(self) -> None:
        """``default`` is the migrate fallback name and remains valid."""
        cfg = _parse(
            {"sources": {"gmail": {"default": {"client_secrets_path": "/s.json"}}}}
        )
        assert "default" in cfg.gmail


class TestParseMeConfig:
    """[me] block parses into cfg.me with email canonicalization."""

    def test_absent_me_block_is_none(self) -> None:
        cfg = _parse({})
        assert cfg.me is None

    def test_basic_me_block(self) -> None:
        cfg = _parse(
            {
                "me": {
                    "emails": ["alice@personal.com", "alice@work.com"],
                    "name": "Alice Example",
                }
            }
        )
        assert cfg.me is not None
        assert cfg.me.emails == ["alice@personal.com", "alice@work.com"]
        assert cfg.me.name == "Alice Example"

    def test_name_optional(self) -> None:
        cfg = _parse({"me": {"emails": ["me@example.com"]}})
        assert cfg.me is not None
        assert cfg.me.name is None

    def test_emails_canonicalized(self) -> None:
        """@googlemail.com rewrites to @gmail.com; case is normalized."""
        cfg = _parse(
            {
                "me": {
                    "emails": ["Alice@GoogleMail.com", "BOB@gmail.com"],
                }
            }
        )
        assert cfg.me is not None
        assert cfg.me.emails == ["alice@gmail.com", "bob@gmail.com"]

    def test_missing_emails_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[me\] emails.*required"):
            _parse({"me": {"name": "Solo"}})

    def test_empty_emails_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[me\] emails.*non-empty list"):
            _parse({"me": {"emails": []}})

    def test_emails_non_list_raises(self) -> None:
        with pytest.raises(TypeError, match=r"\[me\] emails: expected list"):
            _parse({"me": {"emails": "me@example.com"}})

    def test_emails_non_string_item_raises(self) -> None:
        with pytest.raises(TypeError, match=r"\[me\] emails\[0\]: expected str"):
            _parse({"me": {"emails": [42]}})

    def test_name_wrong_type(self) -> None:
        with pytest.raises(TypeError, match=r"\[me\] name: expected str"):
            _parse({"me": {"emails": ["x@y.com"], "name": 123}})


class TestMultiAccountDataclassDefaults:
    """Sanity-check dataclass field defaults match the bead's spec."""

    def test_gmail_account_defaults(self) -> None:
        acct = GmailAccountConfig(name="x")
        assert acct.enabled is True
        assert acct.poll_interval_seconds == 300
        assert acct.max_initial_threads == 500
        assert acct.label_filter == "INBOX"

    def test_calendar_account_defaults(self) -> None:
        acct = CalendarAccountConfig(name="x")
        assert acct.enabled is True
        assert acct.poll_interval_seconds == 300
        assert acct.max_initial_days == 90
        assert acct.calendar_ids == ["primary"]

    def test_me_config_defaults(self) -> None:
        me = MeConfig()
        assert me.emails == []
        assert me.name is None


class TestDoctorGmailAccounts:
    """doctor.check_gmail_accounts — per-account auth status."""

    def test_no_accounts_reports_disabled(self, capsys) -> None:
        from worker.doctor import check_gmail_accounts

        errors = check_gmail_accounts({})
        out = capsys.readouterr().out
        assert errors == 0
        assert "Gmail disabled" in out

    def test_disabled_account_reports_and_skips_probe(self, capsys) -> None:
        from worker.doctor import check_gmail_accounts

        accts = {
            "personal": GmailAccountConfig(
                name="personal",
                enabled=False,
                client_secrets_path="/does/not/exist.json",
            )
        }
        errors = check_gmail_accounts(accts)
        out = capsys.readouterr().out
        assert errors == 0
        assert "Gmail [personal]: disabled" in out

    def test_single_account_valid_token_reports_ok(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        """1 gmail account, valid token: 'Gmail [personal]: OK'."""
        import json
        from unittest.mock import MagicMock

        from google.oauth2.credentials import Credentials

        from worker import doctor

        # Sandbox Path.home() so token paths land in tmp_path.
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

        secrets = tmp_path / "gmail.json"
        secrets.write_text("{}")

        from worker.sources.gmail_auth import token_path_for_account

        token = token_path_for_account("personal")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(json.dumps({"token": "t"}))

        creds = MagicMock(spec=Credentials)
        creds.valid = True
        creds.expired = False
        monkeypatch.setattr(
            Credentials,
            "from_authorized_user_file",
            classmethod(lambda cls, *a, **k: creds),
        )

        accts = {
            "personal": GmailAccountConfig(
                name="personal", client_secrets_path=str(secrets)
            )
        }
        errors = doctor.check_gmail_accounts(accts)
        out = capsys.readouterr().out
        assert errors == 0
        assert "Gmail [personal]: OK" in out

    def test_two_accounts_one_missing_secrets_reports_both(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        """2 gmail accounts, one missing secrets: doctor reports both, exit non-zero."""
        import json
        from unittest.mock import MagicMock

        from google.oauth2.credentials import Credentials

        from worker import doctor

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

        # Account A: secrets present, token valid → OK.
        secrets_a = tmp_path / "a.json"
        secrets_a.write_text("{}")
        from worker.sources.gmail_auth import token_path_for_account

        token_a = token_path_for_account("personal")
        token_a.parent.mkdir(parents=True, exist_ok=True)
        token_a.write_text(json.dumps({"token": "t"}))

        creds = MagicMock(spec=Credentials)
        creds.valid = True
        creds.expired = False
        monkeypatch.setattr(
            Credentials,
            "from_authorized_user_file",
            classmethod(lambda cls, *a, **k: creds),
        )

        # Account B: secrets missing → fail.
        missing = tmp_path / "missing.json"

        accts = {
            "personal": GmailAccountConfig(
                name="personal", client_secrets_path=str(secrets_a)
            ),
            "work": GmailAccountConfig(name="work", client_secrets_path=str(missing)),
        }
        errors = doctor.check_gmail_accounts(accts)
        out = capsys.readouterr().out
        assert errors == 1
        # Both accounts get reported; loop does NOT short-circuit.
        assert "Gmail [personal]: OK" in out
        assert "Gmail [work]: client_secrets file missing" in out


class TestDoctorCalendarAccounts:
    """doctor.check_calendar_accounts — per-account auth status."""

    def test_no_accounts_reports_disabled(self, capsys) -> None:
        from worker.doctor import check_calendar_accounts

        errors = check_calendar_accounts({})
        out = capsys.readouterr().out
        assert errors == 0
        assert "Calendar disabled" in out

    def test_missing_secrets_fails(self, tmp_path, capsys) -> None:
        from worker.doctor import check_calendar_accounts

        missing = tmp_path / "absent.json"
        accts = {
            "personal": CalendarAccountConfig(
                name="personal", client_secrets_path=str(missing)
            )
        }
        errors = check_calendar_accounts(accts)
        out = capsys.readouterr().out
        assert errors == 1
        assert "Calendar [personal]: client_secrets file missing" in out

    def test_drive_scope_reported_per_account(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        """When download_attachments=True, doctor reports the drive scope status."""
        import json

        from worker.doctor import check_calendar_accounts
        from worker.sources.calendar_auth import (
            CALENDAR_SCOPE,
            DRIVE_SCOPE,
            token_path_for_account,
        )

        monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path))

        secrets = tmp_path / "secrets.json"
        secrets.write_text("{}")

        # Account A: drive scope granted.
        token_a = token_path_for_account("personal")
        token_a.parent.mkdir(parents=True, exist_ok=True)
        token_a.write_text(json.dumps({"scopes": [CALENDAR_SCOPE, DRIVE_SCOPE]}))
        token_a.chmod(0o600)

        # Account B: drive scope missing.
        token_b = token_path_for_account("work")
        token_b.write_text(json.dumps({"scopes": [CALENDAR_SCOPE]}))
        token_b.chmod(0o600)

        accts = {
            "personal": CalendarAccountConfig(
                name="personal",
                client_secrets_path=str(secrets),
                download_attachments=True,
            ),
            "work": CalendarAccountConfig(
                name="work",
                client_secrets_path=str(secrets),
                download_attachments=True,
            ),
        }

        check_calendar_accounts(accts)
        out = capsys.readouterr().out
        assert "Calendar [personal]: drive scope granted" in out
        assert "Calendar [work]: drive scope missing" in out


class TestDoctorMe:
    """doctor.check_me — [me] block status reporting."""

    def test_absent_prints_not_configured(self, capsys) -> None:
        from worker.doctor import check_me

        errors = check_me(None)
        out = capsys.readouterr().out
        assert errors == 0
        assert "[me]: not configured" in out

    def test_present_with_two_emails_prints_canonicalized(self, capsys) -> None:
        """[me] with 2 emails: prints both canonicalized."""
        from worker.doctor import check_me

        me = MeConfig(emails=["alice@personal.com", "alice@work.com"])
        errors = check_me(me)
        out = capsys.readouterr().out
        assert errors == 0
        assert "[me]: 2 email(s)" in out
        assert "alice@personal.com" in out
        assert "alice@work.com" in out

    def test_googlemail_alias_prints_canonical_form(self, capsys) -> None:
        """[me] with @googlemail.com: prints @gmail.com (canonicalized)."""
        from worker.doctor import check_me

        # MeConfig built via _parse canonicalizes; simulate that here.
        me = _parse({"me": {"emails": ["Alice@GoogleMail.com"]}}).me
        assert me is not None
        errors = check_me(me)
        out = capsys.readouterr().out
        assert errors == 0
        assert "alice@gmail.com" in out
        assert "googlemail.com" not in out


# --------------------------------------------------------------------- #
# Attachment knobs (download_attachments / attachment_indexable_mimetypes
# / attachment_max_size_mb) on Gmail / Calendar / Slack source sections.
# --------------------------------------------------------------------- #


class TestAttachmentKnobsDefaults:
    """All three sources default to download_attachments=False with the
    DEFAULT_INDEXABLE_MIMETYPES allowlist and a 25 MB cap."""

    def test_slack_defaults(self) -> None:
        cfg = _parse({})
        assert cfg.slack.download_attachments is False
        assert cfg.slack.attachment_indexable_mimetypes == DEFAULT_INDEXABLE_MIMETYPES
        assert cfg.slack.attachment_max_size_mb == 25

    def test_gmail_account_defaults(self) -> None:
        cfg = _parse(
            {"sources": {"gmail": {"personal": {"client_secrets_path": "/s.json"}}}}
        )
        acct = cfg.gmail["personal"]
        assert acct.download_attachments is False
        assert acct.attachment_indexable_mimetypes == DEFAULT_INDEXABLE_MIMETYPES
        assert acct.attachment_max_size_mb == 25

    def test_calendar_account_defaults(self) -> None:
        cfg = _parse(
            {
                "sources": {
                    "google_calendar": {"personal": {"client_secrets_path": "/s.json"}}
                }
            }
        )
        acct = cfg.google_calendar["personal"]
        assert acct.download_attachments is False
        assert acct.attachment_indexable_mimetypes == DEFAULT_INDEXABLE_MIMETYPES
        assert acct.attachment_max_size_mb == 25

    def test_indexable_default_is_a_copy_per_account(self) -> None:
        """Mutating one account's allowlist must not affect another."""
        cfg = _parse(
            {
                "sources": {
                    "gmail": {
                        "a": {"client_secrets_path": "/a.json"},
                        "b": {"client_secrets_path": "/b.json"},
                    }
                }
            }
        )
        cfg.gmail["a"].attachment_indexable_mimetypes.append("application/zip")
        assert "application/zip" not in cfg.gmail["b"].attachment_indexable_mimetypes


class TestAttachmentKnobsOverrides:
    """Per-source overrides land in the parsed dataclass."""

    def test_slack_override_all_three_knobs(self, tmp_path) -> None:
        secrets = tmp_path / "slack.json"
        secrets.write_text("{}")
        cfg = _parse(
            {
                "sources": {
                    "slack": {
                        "enabled": True,
                        "client_secrets_path": str(secrets),
                        "download_attachments": True,
                        "attachment_indexable_mimetypes": [
                            "application/pdf",
                            "image/png",
                        ],
                        "attachment_max_size_mb": 50,
                    }
                }
            }
        )
        assert cfg.slack.download_attachments is True
        assert cfg.slack.attachment_indexable_mimetypes == [
            "application/pdf",
            "image/png",
        ]
        assert cfg.slack.attachment_max_size_mb == 50

    def test_gmail_account_override(self) -> None:
        cfg = _parse(
            {
                "sources": {
                    "gmail": {
                        "personal": {
                            "client_secrets_path": "/s.json",
                            "download_attachments": True,
                            "attachment_indexable_mimetypes": ["application/pdf"],
                            "attachment_max_size_mb": 10,
                        }
                    }
                }
            }
        )
        acct = cfg.gmail["personal"]
        assert acct.download_attachments is True
        assert acct.attachment_indexable_mimetypes == ["application/pdf"]
        assert acct.attachment_max_size_mb == 10

    def test_calendar_account_override(self) -> None:
        cfg = _parse(
            {
                "sources": {
                    "google_calendar": {
                        "personal": {
                            "client_secrets_path": "/s.json",
                            "download_attachments": True,
                            "attachment_indexable_mimetypes": ["text/plain"],
                            "attachment_max_size_mb": 100,
                        }
                    }
                }
            }
        )
        acct = cfg.google_calendar["personal"]
        assert acct.download_attachments is True
        assert acct.attachment_indexable_mimetypes == ["text/plain"]
        assert acct.attachment_max_size_mb == 100


class TestAttachmentMaxSizeBounds:
    """attachment_max_size_mb is enforced to lie in [1, 200] inclusive."""

    @pytest.mark.parametrize("v", [0, -1, 201, 1000])
    def test_out_of_range_raises_for_slack(self, v: int) -> None:
        with pytest.raises(
            ValueError,
            match=r"\[sources\.slack\] attachment_max_size_mb must be in \[1, 200\]",
        ):
            _parse({"sources": {"slack": {"attachment_max_size_mb": v}}})

    @pytest.mark.parametrize("v", [0, 500])
    def test_out_of_range_raises_for_gmail(self, v: int) -> None:
        with pytest.raises(
            ValueError,
            match=r"sources\.gmail\.personal.*attachment_max_size_mb must be in \[1, 200\]",
        ):
            _parse(
                {
                    "sources": {
                        "gmail": {
                            "personal": {
                                "client_secrets_path": "/s.json",
                                "attachment_max_size_mb": v,
                            }
                        }
                    }
                }
            )

    def test_out_of_range_raises_for_calendar(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"sources\.google_calendar\.personal.*attachment_max_size_mb must be in",
        ):
            _parse(
                {
                    "sources": {
                        "google_calendar": {
                            "personal": {
                                "client_secrets_path": "/s.json",
                                "attachment_max_size_mb": 0,
                            }
                        }
                    }
                }
            )

    @pytest.mark.parametrize("v", [1, 25, 200])
    def test_inclusive_bounds_accepted(self, v: int) -> None:
        cfg = _parse({"sources": {"slack": {"attachment_max_size_mb": v}}})
        assert cfg.slack.attachment_max_size_mb == v


class TestAttachmentKnobsTypeValidation:
    def test_indexable_must_be_list(self) -> None:
        with pytest.raises(
            TypeError,
            match=r"\[sources\.slack\] attachment_indexable_mimetypes: expected list",
        ):
            _parse(
                {
                    "sources": {
                        "slack": {"attachment_indexable_mimetypes": "application/pdf"}
                    }
                }
            )

    def test_indexable_items_must_be_strings(self) -> None:
        with pytest.raises(
            TypeError,
            match=r"\[sources\.slack\] attachment_indexable_mimetypes\[1\]: expected str",
        ):
            _parse(
                {
                    "sources": {
                        "slack": {
                            "attachment_indexable_mimetypes": [
                                "application/pdf",
                                42,
                            ]
                        }
                    }
                }
            )

    def test_max_size_must_be_int(self) -> None:
        with pytest.raises(
            TypeError,
            match=r"\[sources\.slack\] attachment_max_size_mb: expected int",
        ):
            _parse({"sources": {"slack": {"attachment_max_size_mb": "25"}}})

    def test_download_attachments_must_be_bool(self) -> None:
        with pytest.raises(
            TypeError,
            match=r"\[sources\.slack\] download_attachments: expected bool",
        ):
            _parse({"sources": {"slack": {"download_attachments": "yes"}}})

    def test_gmail_account_indexable_wrong_item_type(self) -> None:
        with pytest.raises(
            TypeError,
            match=r"sources\.gmail\.personal.*attachment_indexable_mimetypes\[0\]",
        ):
            _parse(
                {
                    "sources": {
                        "gmail": {
                            "personal": {
                                "client_secrets_path": "/s.json",
                                "attachment_indexable_mimetypes": [123],
                            }
                        }
                    }
                }
            )


class TestSlackDownloadFilesAlias:
    """[sources.slack].download_files is the legacy spelling of
    download_attachments.  Alone it's silently promoted; together with
    download_attachments it warns and download_attachments wins."""

    def test_legacy_key_alone_is_silently_aliased(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="worker.config"):
            cfg = _parse({"sources": {"slack": {"download_files": True}}})
        assert cfg.slack.download_attachments is True
        assert not any(
            "download_attachments" in r.getMessage()
            and "download_files" in r.getMessage()
            for r in caplog.records
        )

    def test_legacy_key_alone_false(self) -> None:
        cfg = _parse({"sources": {"slack": {"download_files": False}}})
        assert cfg.slack.download_attachments is False

    def test_both_keys_warns_and_new_key_wins(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="worker.config"):
            cfg = _parse(
                {
                    "sources": {
                        "slack": {
                            "download_files": True,
                            "download_attachments": False,
                        }
                    }
                }
            )
        # download_attachments wins, even though it's "less permissive".
        assert cfg.slack.download_attachments is False
        assert any(
            "download_attachments" in r.getMessage()
            and "download_files" in r.getMessage()
            for r in caplog.records
        )

    def test_both_keys_warns_when_new_key_true(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="worker.config"):
            cfg = _parse(
                {
                    "sources": {
                        "slack": {
                            "download_files": False,
                            "download_attachments": True,
                        }
                    }
                }
            )
        assert cfg.slack.download_attachments is True
        assert any("download_attachments" in r.getMessage() for r in caplog.records)

    def test_legacy_key_type_error_still_caught(self) -> None:
        with pytest.raises(
            TypeError, match=r"\[sources\.slack\] download_files: expected bool"
        ):
            _parse({"sources": {"slack": {"download_files": "yes"}}})


class TestSlackSourceConfigDataclassDefaults:
    """SlackSourceConfig now exposes download_attachments instead of the
    legacy download_files field."""

    def test_dataclass_default_field_present(self) -> None:
        cfg = SlackSourceConfig()
        assert cfg.download_attachments is False
        assert cfg.attachment_max_size_mb == 25
        assert cfg.attachment_indexable_mimetypes == DEFAULT_INDEXABLE_MIMETYPES
        assert not hasattr(cfg, "download_files")


class TestConfigTomlExampleAttachmentKnobs:
    """The shipped config.toml.example demonstrates the new knobs in
    each of the three source sections."""

    def test_example_mentions_attachment_keys_in_each_section(self) -> None:
        from pathlib import Path as _P

        example = (
            _P(__file__).resolve().parent.parent / "worker" / "config.toml.example"
        )
        text = example.read_text()

        # Slack section: knobs live below the [sources.slack] header.
        slack_idx = text.index("[sources.slack]")
        slack_block = text[slack_idx:]
        assert "download_attachments" in slack_block
        assert "attachment_max_size_mb" in slack_block
        assert "attachment_indexable_mimetypes" in slack_block

        # Gmail per-account stub block.
        assert "[sources.gmail.personal]" in text
        gmail_idx = text.index("[sources.gmail.personal]")
        gmail_block = text[gmail_idx : gmail_idx + 1500]
        assert "download_attachments" in gmail_block
        assert "attachment_max_size_mb" in gmail_block
        assert "attachment_indexable_mimetypes" in gmail_block

        # Calendar per-account stub block.
        assert "[sources.google_calendar.personal]" in text
        cal_idx = text.index("[sources.google_calendar.personal]")
        cal_block = text[cal_idx : cal_idx + 1500]
        assert "download_attachments" in cal_block
        assert "attachment_max_size_mb" in cal_block
        assert "attachment_indexable_mimetypes" in cal_block


def _stub_valid_google_creds(monkeypatch) -> None:
    """Make ``Credentials.from_authorized_user_file`` return a valid creds mock.

    Used by the doctor attachment tests where the actual Google auth probe
    is not the focus — we just need ``_check_google_auth`` to return 0.
    """
    from unittest.mock import MagicMock

    from google.oauth2.credentials import Credentials

    creds = MagicMock(spec=Credentials)
    creds.valid = True
    creds.expired = False
    monkeypatch.setattr(
        Credentials,
        "from_authorized_user_file",
        classmethod(lambda cls, *a, **k: creds),
    )


class TestDoctorAttachmentStatus:
    """Per-source attachment-indexing status lines (fn-47w)."""

    def test_gmail_off_prints_off_line(self, tmp_path, capsys, monkeypatch) -> None:
        import json

        from worker import doctor
        from worker.sources.gmail_auth import token_path_for_account

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        _stub_valid_google_creds(monkeypatch)

        secrets = tmp_path / "g.json"
        secrets.write_text("{}")
        token = token_path_for_account("personal")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(json.dumps({"token": "t"}))

        accts = {
            "personal": GmailAccountConfig(
                name="personal",
                client_secrets_path=str(secrets),
                download_attachments=False,
            )
        }
        errors = doctor.check_gmail_accounts(accts)
        out = capsys.readouterr().out
        assert errors == 0
        assert "Gmail [personal] attachments: OFF" in out
        assert "filenames embedded in body" in out

    def test_gmail_on_prints_on_line_with_allowlist_and_max(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        import json

        from worker import doctor
        from worker.sources.gmail_auth import token_path_for_account

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        _stub_valid_google_creds(monkeypatch)

        secrets = tmp_path / "g.json"
        secrets.write_text("{}")
        token = token_path_for_account("personal")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(json.dumps({"token": "t"}))

        accts = {
            "personal": GmailAccountConfig(
                name="personal",
                client_secrets_path=str(secrets),
                download_attachments=True,
                attachment_indexable_mimetypes=["application/pdf", "image/png"],
                attachment_max_size_mb=10,
            )
        }
        errors = doctor.check_gmail_accounts(accts)
        out = capsys.readouterr().out
        assert errors == 0
        assert "Gmail [personal] attachments: ON" in out
        assert "allowlist: 2 MIMEs" in out
        assert "max 10 MB" in out

    def test_gmail_multi_account_per_account_status(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        """Two accounts with different download_attachments → reported separately."""
        import json

        from worker import doctor
        from worker.sources.gmail_auth import token_path_for_account

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        _stub_valid_google_creds(monkeypatch)

        secrets = tmp_path / "g.json"
        secrets.write_text("{}")
        for name in ("personal", "work"):
            token = token_path_for_account(name)
            token.parent.mkdir(parents=True, exist_ok=True)
            token.write_text(json.dumps({"token": "t"}))

        accts = {
            "personal": GmailAccountConfig(
                name="personal",
                client_secrets_path=str(secrets),
                download_attachments=True,
            ),
            "work": GmailAccountConfig(
                name="work",
                client_secrets_path=str(secrets),
                download_attachments=False,
            ),
        }
        doctor.check_gmail_accounts(accts)
        out = capsys.readouterr().out
        assert "Gmail [personal] attachments: ON" in out
        assert "Gmail [work] attachments: OFF" in out

    def test_calendar_on_drive_scope_granted_ok(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        """download_attachments=True + drive.readonly granted → OK + ON line."""
        import json

        from worker import doctor
        from worker.sources.calendar_auth import (
            CALENDAR_SCOPE,
            DRIVE_SCOPE,
            token_path_for_account,
        )

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        _stub_valid_google_creds(monkeypatch)

        secrets = tmp_path / "c.json"
        secrets.write_text("{}")
        token = token_path_for_account("personal")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(
            json.dumps({"token": "t", "scopes": [CALENDAR_SCOPE, DRIVE_SCOPE]})
        )

        accts = {
            "personal": CalendarAccountConfig(
                name="personal",
                client_secrets_path=str(secrets),
                download_attachments=True,
            )
        }
        errors = doctor.check_calendar_accounts(accts)
        out = capsys.readouterr().out
        assert errors == 0
        assert "Calendar [personal]: drive scope granted" in out
        assert "Calendar [personal] attachments: ON" in out

    def test_calendar_on_drive_scope_missing_warns_and_fails(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        """download_attachments=True + drive scope missing → non-zero + ON line."""
        import json

        from worker import doctor
        from worker.sources.calendar_auth import (
            CALENDAR_SCOPE,
            token_path_for_account,
        )

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        _stub_valid_google_creds(monkeypatch)

        secrets = tmp_path / "c.json"
        secrets.write_text("{}")
        token = token_path_for_account("personal")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(json.dumps({"token": "t", "scopes": [CALENDAR_SCOPE]}))

        accts = {
            "personal": CalendarAccountConfig(
                name="personal",
                client_secrets_path=str(secrets),
                download_attachments=True,
            )
        }
        errors = doctor.check_calendar_accounts(accts)
        out = capsys.readouterr().out
        assert errors >= 1
        assert "drive scope missing" in out
        assert "Calendar [personal] attachments: ON" in out

    def test_calendar_off_no_drive_check_no_failure(
        self, tmp_path, capsys, monkeypatch
    ) -> None:
        """download_attachments=False → no drive-scope assertion, exit clean."""
        import json

        from worker import doctor
        from worker.sources.calendar_auth import (
            CALENDAR_SCOPE,
            token_path_for_account,
        )

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        _stub_valid_google_creds(monkeypatch)

        secrets = tmp_path / "c.json"
        secrets.write_text("{}")
        token = token_path_for_account("personal")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(json.dumps({"token": "t", "scopes": [CALENDAR_SCOPE]}))

        accts = {
            "personal": CalendarAccountConfig(
                name="personal",
                client_secrets_path=str(secrets),
                download_attachments=False,
            )
        }
        errors = doctor.check_calendar_accounts(accts)
        out = capsys.readouterr().out
        assert errors == 0
        assert "Calendar [personal] attachments: OFF" in out
        assert "drive scope" not in out

    def test_slack_on_prints_on_line(self, tmp_path, capsys, monkeypatch) -> None:
        from worker import doctor

        secrets = tmp_path / "slack.json"
        secrets.write_text("{}")
        cfg = SlackSourceConfig(
            enabled=True,
            client_secrets_path=str(secrets),
            download_attachments=True,
            attachment_indexable_mimetypes=["application/pdf"],
            attachment_max_size_mb=15,
        )

        _install_fake_slack_auth(monkeypatch, lambda *a, **k: 0)
        monkeypatch.setattr(doctor, "check_slack_auth", lambda *a, **k: 0)

        errors = doctor.check_slack(cfg)
        out = capsys.readouterr().out
        assert errors == 0
        assert "Slack attachments: ON" in out
        assert "allowlist: 1 MIMEs" in out
        assert "max 15 MB" in out

    def test_slack_off_prints_off_line(self, tmp_path, capsys, monkeypatch) -> None:
        from worker import doctor

        secrets = tmp_path / "slack.json"
        secrets.write_text("{}")
        cfg = SlackSourceConfig(
            enabled=True,
            client_secrets_path=str(secrets),
            download_attachments=False,
        )

        _install_fake_slack_auth(monkeypatch, lambda *a, **k: 0)
        monkeypatch.setattr(doctor, "check_slack_auth", lambda *a, **k: 0)

        errors = doctor.check_slack(cfg)
        out = capsys.readouterr().out
        assert errors == 0
        assert "Slack attachments: OFF" in out


class TestDoctorAttachmentFailureCounter:
    """worker_attachment_fetch_failures counter surfacing in doctor."""

    @pytest.fixture(autouse=True)
    def _reset_counter(self):
        from worker.metrics import WORKER_ATTACHMENT_FETCH_FAILURES

        WORKER_ATTACHMENT_FETCH_FAILURES.clear()
        yield
        WORKER_ATTACHMENT_FETCH_FAILURES.clear()

    def test_counter_exists_with_expected_labels(self) -> None:
        from worker.metrics import WORKER_ATTACHMENT_FETCH_FAILURES

        # Sample increment with labels — must not raise.
        WORKER_ATTACHMENT_FETCH_FAILURES.labels(
            source_type="gmail",
            account="personal",
            error_kind="download",
        ).inc(0)

    def test_doctor_silent_when_no_failures(self, capsys) -> None:
        from worker.doctor import check_attachment_failures

        check_attachment_failures()
        out = capsys.readouterr().out
        assert "Attachment failures" not in out

    def test_doctor_reports_per_source_when_nonzero(self, capsys) -> None:
        from worker.doctor import check_attachment_failures
        from worker.metrics import WORKER_ATTACHMENT_FETCH_FAILURES

        WORKER_ATTACHMENT_FETCH_FAILURES.labels(
            source_type="gmail",
            account="personal",
            error_kind="download",
        ).inc(2)
        WORKER_ATTACHMENT_FETCH_FAILURES.labels(
            source_type="slack",
            account="-",
            error_kind="parse",
        ).inc(1)

        check_attachment_failures()
        out = capsys.readouterr().out
        assert "Attachment failures (last 24h)" in out
        assert "gmail: 2 attachment fetch failure(s)" in out
        assert "slack: 1 attachment fetch failure(s)" in out
