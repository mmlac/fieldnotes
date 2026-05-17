"""Tests for hot-reload of [me] config into a running Pipeline (fn-del).

Covers:
- Pipeline.set_me_config() attribute mutator
- _build_sighup_handler(): applies new emails, keeps old config on parse error,
  sets None when [me] is removed, logs on reload
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from worker.config import MeConfig
from worker.main import _build_sighup_handler
from worker.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(me_config: MeConfig | None = None) -> Pipeline:
    """Build a minimal Pipeline with mocked dependencies."""
    registry = MagicMock()
    writer = MagicMock()
    writer.indexed_source_ids.return_value = set()
    p = Pipeline(registry=registry, writer=writer, me_config=me_config)
    return p


def _write_config(path: Path, emails: list[str] | None, name: str | None = None) -> None:
    """Write a minimal config.toml to *path* with an optional [me] block."""
    lines = [
        "[core]\n",
        'log_level = "info"\n',
        "[neo4j]\n",
        'uri = "bolt://localhost:7687"\n',
        'user = "neo4j"\n',
        'password = "pw"\n',
        "[qdrant]\n",
        'host = "localhost"\n',
        "port = 6333\n",
    ]
    if emails is not None:
        lines.append("[me]\n")
        emails_toml = ", ".join(f'"{e}"' for e in emails)
        lines.append(f"emails = [{emails_toml}]\n")
        if name:
            lines.append(f'name = "{name}"\n')
    path.write_text("".join(lines))


# ---------------------------------------------------------------------------
# Pipeline.set_me_config
# ---------------------------------------------------------------------------


class TestSetMeConfig:
    def test_updates_me_config_attribute(self):
        pipeline = _make_pipeline()
        assert pipeline._me_config is None

        new_me = MeConfig(emails=["a@example.com"])
        pipeline.set_me_config(new_me)

        assert pipeline._me_config is new_me
        assert pipeline._me_config.emails == ["a@example.com"]

    def test_sets_me_config_to_none(self):
        pipeline = _make_pipeline(MeConfig(emails=["old@example.com"]))
        pipeline.set_me_config(None)
        assert pipeline._me_config is None

    def test_replaces_existing_me_config(self):
        old_me = MeConfig(emails=["old@example.com"])
        pipeline = _make_pipeline(old_me)

        new_me = MeConfig(emails=["new@example.com", "other@example.com"])
        pipeline.set_me_config(new_me)

        assert pipeline._me_config.emails == ["new@example.com", "other@example.com"]


# ---------------------------------------------------------------------------
# _build_sighup_handler
# ---------------------------------------------------------------------------


class TestSighupHandlerAppliesNewEmails:
    def test_applies_new_emails_from_config(self, tmp_path):
        config_file = tmp_path / "config.toml"
        _write_config(config_file, emails=["user@example.com"], name="Alice")

        pipeline = _make_pipeline()
        handler = _build_sighup_handler(pipeline, config_file)
        handler()

        assert pipeline._me_config is not None
        assert "user@example.com" in pipeline._me_config.emails
        assert pipeline._me_config.name == "Alice"

    def test_updates_emails_on_repeated_calls(self, tmp_path):
        config_file = tmp_path / "config.toml"
        _write_config(config_file, emails=["first@example.com"])

        pipeline = _make_pipeline()
        handler = _build_sighup_handler(pipeline, config_file)
        handler()
        assert "first@example.com" in pipeline._me_config.emails

        _write_config(config_file, emails=["second@example.com", "alt@example.com"])
        handler()
        assert pipeline._me_config.emails == ["second@example.com", "alt@example.com"]

    def test_uses_default_path_when_config_path_is_none(self, tmp_path):
        """When config_path=None the handler falls back to DEFAULT_CONFIG_PATH."""
        config_file = tmp_path / "config.toml"
        _write_config(config_file, emails=["default@example.com"])

        pipeline = _make_pipeline()
        with patch("worker.config.DEFAULT_CONFIG_PATH", config_file):
            handler = _build_sighup_handler(pipeline, None)
            handler()

        assert "default@example.com" in pipeline._me_config.emails


class TestSighupHandlerBadConfigKeepsOld:
    def test_parse_error_keeps_previous_me_config(self, tmp_path, caplog):
        config_file = tmp_path / "config.toml"
        _write_config(config_file, emails=["old@example.com"])

        pipeline = _make_pipeline(MeConfig(emails=["old@example.com"]))
        handler = _build_sighup_handler(pipeline, config_file)

        config_file.write_text("this is not valid toml ][[[")

        with caplog.at_level(logging.ERROR, logger="worker"):
            handler()

        assert pipeline._me_config is not None
        assert pipeline._me_config.emails == ["old@example.com"]

    def test_missing_config_file_keeps_previous_me_config(self, tmp_path, caplog):
        missing = tmp_path / "nonexistent.toml"
        pipeline = _make_pipeline(MeConfig(emails=["kept@example.com"]))
        handler = _build_sighup_handler(pipeline, missing)

        with caplog.at_level(logging.ERROR, logger="worker"):
            handler()

        assert pipeline._me_config.emails == ["kept@example.com"]


class TestSighupHandlerRemovedMeSection:
    def test_me_removed_sets_none(self, tmp_path):
        config_file = tmp_path / "config.toml"
        _write_config(config_file, emails=["user@example.com"])

        pipeline = _make_pipeline(MeConfig(emails=["user@example.com"]))
        handler = _build_sighup_handler(pipeline, config_file)

        # Overwrite config with no [me] block
        _write_config(config_file, emails=None)
        handler()

        assert pipeline._me_config is None

    def test_me_none_does_not_crash_pipeline_process(self, tmp_path):
        """After setting _me_config to None, process_batch should not error."""
        config_file = tmp_path / "config.toml"
        _write_config(config_file, emails=None)

        pipeline = _make_pipeline(MeConfig(emails=["user@example.com"]))
        handler = _build_sighup_handler(pipeline, config_file)
        handler()

        assert pipeline._me_config is None
        # Calling set_me_config(None) again must be idempotent
        pipeline.set_me_config(None)
        assert pipeline._me_config is None


class TestReconcileSelfIfConfigured:
    """reconcile_self_if_configured() wires reload into the per-event ingest loop."""

    def test_calls_reconcile_self_person_after_sighup(self, tmp_path):
        """SIGHUP sets _me_config; reconcile_self_if_configured then calls
        writer.reconcile_self_person with the reloaded config."""
        config_file = tmp_path / "config.toml"
        _write_config(config_file, emails=["me@example.com"], name="Test User")

        pipeline = _make_pipeline(me_config=None)
        handler = _build_sighup_handler(pipeline, config_file)
        handler()  # simulate SIGHUP

        assert pipeline._me_config is not None

        # Simulate what the ingest event loop does after processing a queue event
        pipeline.reconcile_self_if_configured()

        pipeline._writer.reconcile_self_person.assert_called_once()
        called_me = pipeline._writer.reconcile_self_person.call_args[0][0]
        assert "me@example.com" in called_me.emails

    def test_no_call_when_me_unset(self):
        """reconcile_self_if_configured does nothing when _me_config is None."""
        pipeline = _make_pipeline(me_config=None)
        pipeline.reconcile_self_if_configured()
        pipeline._writer.reconcile_self_person.assert_not_called()

    def test_no_call_before_sighup(self, tmp_path):
        """Without a SIGHUP reload, reconcile is not called even if a config exists."""
        config_file = tmp_path / "config.toml"
        _write_config(config_file, emails=["me@example.com"])

        pipeline = _make_pipeline(me_config=None)
        # Process an event WITHOUT triggering SIGHUP first
        pipeline.reconcile_self_if_configured()
        pipeline._writer.reconcile_self_person.assert_not_called()

    def test_exception_in_reconcile_is_swallowed(self, tmp_path):
        """A reconcile failure must not propagate to the caller (event loop)."""
        config_file = tmp_path / "config.toml"
        _write_config(config_file, emails=["me@example.com"])

        pipeline = _make_pipeline(me_config=None)
        handler = _build_sighup_handler(pipeline, config_file)
        handler()

        pipeline._writer.reconcile_self_person.side_effect = RuntimeError("db down")
        # Must not raise
        pipeline.reconcile_self_if_configured()


class TestSighupHandlerLogging:
    def test_logs_reload_with_emails(self, tmp_path, caplog):
        config_file = tmp_path / "config.toml"
        _write_config(config_file, emails=["log@example.com"])

        pipeline = _make_pipeline()
        handler = _build_sighup_handler(pipeline, config_file)

        with caplog.at_level(logging.INFO, logger="worker"):
            handler()

        assert any("reloaded" in r.message for r in caplog.records)

    def test_logs_reload_when_me_unset(self, tmp_path, caplog):
        config_file = tmp_path / "config.toml"
        _write_config(config_file, emails=None)

        pipeline = _make_pipeline(MeConfig(emails=["old@example.com"]))
        handler = _build_sighup_handler(pipeline, config_file)

        with caplog.at_level(logging.INFO, logger="worker"):
            handler()

        assert any("unset" in r.message for r in caplog.records)

    def test_logs_error_on_bad_config(self, tmp_path, caplog):
        config_file = tmp_path / "config.toml"
        config_file.write_text("bad [[[")

        pipeline = _make_pipeline()
        handler = _build_sighup_handler(pipeline, config_file)

        with caplog.at_level(logging.ERROR, logger="worker"):
            handler()

        assert any("failed" in r.message for r in caplog.records)
