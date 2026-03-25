"""Tests for the ``fieldnotes queue`` CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from worker.cli import _build_parser
from worker.queue import PersistentQueue


# ------------------------------------------------------------------
# Parser tests
# ------------------------------------------------------------------


class TestQueueParser:
    def test_queue_no_subcommand(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["queue"])
        assert args.command == "queue"
        assert args.queue_command is None

    def test_queue_top_default(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["queue", "top"])
        assert args.queue_command == "top"
        assert args.n == 20

    def test_queue_top_custom_n(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["queue", "top", "5"])
        assert args.n == 5

    def test_queue_top_with_status_filter(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["queue", "top", "--status", "failed"])
        assert args.status == "failed"

    def test_queue_top_with_source_filter(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["queue", "top", "--source", "files"])
        assert args.source_type == "files"

    def test_queue_tail_default(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["queue", "tail"])
        assert args.queue_command == "tail"
        assert args.n == 20

    def test_queue_tail_custom_n(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["queue", "tail", "10"])
        assert args.n == 10

    def test_queue_retry(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["queue", "retry"])
        assert args.queue_command == "retry"

    def test_queue_purge_default(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["queue", "purge"])
        assert args.queue_command == "purge"
        assert args.status == "failed"

    def test_queue_purge_custom_status(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["queue", "purge", "--status", "pending"])
        assert args.status == "pending"

    def test_queue_migrate(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["queue", "migrate"])
        assert args.queue_command == "migrate"

    def test_queue_json_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["queue", "--json"])
        assert args.json_output is True


# ------------------------------------------------------------------
# Functional tests (with a real temp queue DB)
# ------------------------------------------------------------------


def _make_config(data_dir: Path):
    """Create a mock config whose core.data_dir points to data_dir."""
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.core.data_dir = str(data_dir)
    return cfg


def _seed_queue(q: PersistentQueue, n: int = 3) -> None:
    """Insert n items with distinct source types."""
    types = ["files", "obsidian", "gmail", "calendar", "repositories"]
    for i in range(n):
        q.enqueue(
            {
                "id": f"item-{i:03d}",
                "source_type": types[i % len(types)],
                "source_id": f"/test/path-{i}",
                "operation": "created",
            }
        )


@pytest.fixture
def queue_dir(tmp_path: Path) -> Path:
    """Create a data dir with a seeded queue DB."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    q = PersistentQueue(data_dir / "queue.db", max_retries=1)
    _seed_queue(q, 5)
    # Fail one item for retry/purge tests: claim sets attempts=1,
    # fail with max_retries=1 marks it as 'failed' immediately.
    item = q.claim()
    assert item is not None
    q.fail(item["_queue_id"], "test error")
    q.close()
    return data_dir


class TestQueueSummary:
    def test_summary_human(self, queue_dir: Path, capsys) -> None:
        cfg = _make_config(queue_dir)
        with patch("worker.cli.queue.load_config", return_value=cfg):
            from worker.cli.queue import run_queue_summary

            rc = run_queue_summary(config_path=None)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Ingestion Queue" in out
        assert "pending" in out

    def test_summary_json(self, queue_dir: Path, capsys) -> None:
        cfg = _make_config(queue_dir)
        with patch("worker.cli.queue.load_config", return_value=cfg):
            from worker.cli.queue import run_queue_summary

            rc = run_queue_summary(config_path=None, json_output=True)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "summary" in data
        assert "by_source" in data

    def test_summary_empty_queue(self, tmp_path: Path, capsys) -> None:
        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        q = PersistentQueue(data_dir / "queue.db")
        q.close()

        cfg = _make_config(data_dir)
        with patch("worker.cli.queue.load_config", return_value=cfg):
            from worker.cli.queue import run_queue_summary

            rc = run_queue_summary(config_path=None)
        assert rc == 0
        assert "empty" in capsys.readouterr().out.lower()


class TestQueueList:
    def test_top_default(self, queue_dir: Path, capsys) -> None:
        cfg = _make_config(queue_dir)
        with patch("worker.cli.queue.load_config", return_value=cfg):
            from worker.cli.queue import run_queue_list

            rc = run_queue_list(config_path=None)
        assert rc == 0
        out = capsys.readouterr().out
        assert "oldest" in out

    def test_tail(self, queue_dir: Path, capsys) -> None:
        cfg = _make_config(queue_dir)
        with patch("worker.cli.queue.load_config", return_value=cfg):
            from worker.cli.queue import run_queue_list

            rc = run_queue_list(order="desc", config_path=None)
        assert rc == 0
        out = capsys.readouterr().out
        assert "newest" in out

    def test_list_json(self, queue_dir: Path, capsys) -> None:
        cfg = _make_config(queue_dir)
        with patch("worker.cli.queue.load_config", return_value=cfg):
            from worker.cli.queue import run_queue_list

            rc = run_queue_list(config_path=None, json_output=True)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "source_type" in data[0]

    def test_list_with_status_filter(self, queue_dir: Path, capsys) -> None:
        cfg = _make_config(queue_dir)
        with patch("worker.cli.queue.load_config", return_value=cfg):
            from worker.cli.queue import run_queue_list

            rc = run_queue_list(status="failed", config_path=None, json_output=True)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        for item in data:
            assert item["status"] == "failed"

    def test_list_empty(self, tmp_path: Path, capsys) -> None:
        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        q = PersistentQueue(data_dir / "queue.db")
        q.close()

        cfg = _make_config(data_dir)
        with patch("worker.cli.queue.load_config", return_value=cfg):
            from worker.cli.queue import run_queue_list

            rc = run_queue_list(config_path=None)
        assert rc == 0
        assert "No items" in capsys.readouterr().out


class TestQueueRetry:
    def test_retry_resets_failed(self, queue_dir: Path, capsys) -> None:
        cfg = _make_config(queue_dir)
        with patch("worker.cli.queue.load_config", return_value=cfg):
            from worker.cli.queue import run_queue_retry

            rc = run_queue_retry(config_path=None)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Reset" in out
        assert "1" in out  # 1 failed item

    def test_retry_none_failed(self, tmp_path: Path, capsys) -> None:
        data_dir = tmp_path / "nofail"
        data_dir.mkdir()
        q = PersistentQueue(data_dir / "queue.db")
        q.enqueue(
            {"id": "ok", "source_type": "files", "source_id": "x", "operation": "c"}
        )
        q.close()

        cfg = _make_config(data_dir)
        with patch("worker.cli.queue.load_config", return_value=cfg):
            from worker.cli.queue import run_queue_retry

            rc = run_queue_retry(config_path=None)
        assert rc == 0
        assert "No failed" in capsys.readouterr().out


class TestQueuePurge:
    def test_purge_failed(self, queue_dir: Path, capsys) -> None:
        cfg = _make_config(queue_dir)
        with patch("worker.cli.queue.load_config", return_value=cfg):
            from worker.cli.queue import run_queue_purge

            rc = run_queue_purge(config_path=None)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Purged" in out
        assert "1" in out


class TestQueueMigrate:
    def test_migrate_imports_cursors(self, tmp_path: Path, capsys) -> None:
        data_dir = tmp_path / "migrate"
        data_dir.mkdir()
        # Create old cursor files.
        (data_dir / "file_cursor.json").write_text('{"a.txt": {"sha256": "abc"}}')
        (data_dir / "gmail_cursor.json").write_text('{"last_id": "123"}')

        cfg = _make_config(data_dir)
        with patch("worker.cli.queue.load_config", return_value=cfg):
            from worker.cli.queue import run_queue_migrate

            rc = run_queue_migrate(config_path=None)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Migrated 2" in out

        # Verify they were actually imported.
        q = PersistentQueue(data_dir / "queue.db")
        assert q.load_cursor("files") is not None
        assert q.load_cursor("gmail") is not None
        q.close()

    def test_migrate_no_files(self, tmp_path: Path, capsys) -> None:
        data_dir = tmp_path / "empty"
        data_dir.mkdir()

        cfg = _make_config(data_dir)
        with patch("worker.cli.queue.load_config", return_value=cfg):
            from worker.cli.queue import run_queue_migrate

            rc = run_queue_migrate(config_path=None)
        assert rc == 0
        assert "No cursor files" in capsys.readouterr().out

    def test_migrate_idempotent(self, tmp_path: Path, capsys) -> None:
        data_dir = tmp_path / "idem"
        data_dir.mkdir()
        (data_dir / "file_cursor.json").write_text('{"x": 1}')

        cfg = _make_config(data_dir)
        with patch("worker.cli.queue.load_config", return_value=cfg):
            from worker.cli.queue import run_queue_migrate

            run_queue_migrate(config_path=None)
            capsys.readouterr()  # clear
            rc = run_queue_migrate(config_path=None)
        assert rc == 0
        assert "No cursor files" in capsys.readouterr().out
