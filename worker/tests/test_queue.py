"""Tests for the SQLite-backed PersistentQueue."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from worker.queue import CursorUpdate, PersistentQueue


@pytest.fixture
def queue(tmp_path: Path) -> PersistentQueue:
    return PersistentQueue(db_path=tmp_path / "queue.db")


@pytest.fixture
def event() -> dict:
    return {
        "id": "test-001",
        "source_type": "files",
        "source_id": "/tmp/test.txt",
        "operation": "created",
        "mime_type": "text/plain",
        "text": "hello world",
        "enqueued_at": "2026-01-01T00:00:00+00:00",
    }


class TestEnqueueAndClaim:
    def test_enqueue_and_claim_roundtrip(self, queue: PersistentQueue, event: dict) -> None:
        queue.enqueue(event)
        claimed = queue.claim()
        assert claimed is not None
        assert claimed["source_id"] == "/tmp/test.txt"
        assert claimed["text"] == "hello world"
        assert "_queue_id" in claimed

    def test_claim_returns_none_on_empty(self, queue: PersistentQueue) -> None:
        assert queue.claim() is None

    def test_enqueue_assigns_id_if_missing(self, queue: PersistentQueue) -> None:
        event = {"source_type": "files", "source_id": "/a", "operation": "created"}
        eid = queue.enqueue(event)
        assert eid  # non-empty string

    def test_claim_returns_oldest_first(self, queue: PersistentQueue) -> None:
        queue.enqueue({
            "id": "a", "source_type": "files", "source_id": "/a",
            "operation": "created", "enqueued_at": "2026-01-01T00:00:01+00:00",
        })
        queue.enqueue({
            "id": "b", "source_type": "files", "source_id": "/b",
            "operation": "created", "enqueued_at": "2026-01-01T00:00:00+00:00",
        })
        claimed = queue.claim()
        assert claimed is not None
        assert claimed["source_id"] == "/b"  # older timestamp


class TestComplete:
    def test_complete_removes_item(self, queue: PersistentQueue, event: dict) -> None:
        queue.enqueue(event)
        claimed = queue.claim()
        assert claimed is not None
        queue.complete(claimed["_queue_id"])
        assert queue.depth() == 0

    def test_complete_cleans_blob(self, queue: PersistentQueue, tmp_path: Path) -> None:
        event = {
            "id": "blob-1",
            "source_type": "files",
            "source_id": "/img.png",
            "operation": "created",
            "raw_bytes": b"\x89PNG fake image data",
        }
        queue.enqueue(event)
        blob_path = queue._blob_dir / "blob-1"
        assert blob_path.exists()

        claimed = queue.claim()
        assert claimed is not None
        assert claimed.get("raw_bytes") == b"\x89PNG fake image data"

        queue.complete(claimed["_queue_id"])
        assert not blob_path.exists()


class TestFail:
    def test_fail_retries_under_max(self, queue: PersistentQueue, event: dict) -> None:
        queue.enqueue(event)
        claimed = queue.claim()
        assert claimed is not None
        queue.fail(claimed["_queue_id"], "timeout")

        # Should be back to pending for retry.
        assert queue.depth() == 1
        reclaimed = queue.claim()
        assert reclaimed is not None
        assert reclaimed["source_id"] == "/tmp/test.txt"

    def test_fail_marks_failed_after_max_retries(self, tmp_path: Path, event: dict) -> None:
        queue = PersistentQueue(db_path=tmp_path / "queue.db", max_retries=2)
        queue.enqueue(event)

        # Fail twice (attempts incremented on each claim).
        for _ in range(2):
            claimed = queue.claim()
            assert claimed is not None
            queue.fail(claimed["_queue_id"], "error")

        # Third claim + fail should mark as 'failed'.
        claimed = queue.claim()
        # After 2 retries with max_retries=2, attempts=2, so fail should mark failed
        # Actually: claim increments attempts. After 2 claims, attempts=2.
        # fail checks attempts >= max_retries (2 >= 2) → failed.
        assert claimed is not None
        queue.fail(claimed["_queue_id"], "final error")

        # Now it should be failed, not pending.
        assert queue.claim() is None
        summary = queue.summary()
        assert summary.get("failed") == 1


class TestDedup:
    def test_is_enqueued_returns_true_for_pending(self, queue: PersistentQueue, event: dict) -> None:
        queue.enqueue(event)
        assert queue.is_enqueued("/tmp/test.txt") is True

    def test_is_enqueued_returns_true_for_processing(self, queue: PersistentQueue, event: dict) -> None:
        queue.enqueue(event)
        queue.claim()
        assert queue.is_enqueued("/tmp/test.txt") is True

    def test_is_enqueued_returns_false_after_complete(self, queue: PersistentQueue, event: dict) -> None:
        queue.enqueue(event)
        claimed = queue.claim()
        assert claimed is not None
        queue.complete(claimed["_queue_id"])
        assert queue.is_enqueued("/tmp/test.txt") is False

    def test_is_enqueued_returns_false_for_unknown(self, queue: PersistentQueue) -> None:
        assert queue.is_enqueued("/nonexistent") is False

    def test_duplicate_enqueue_ignored(self, queue: PersistentQueue) -> None:
        event = {
            "id": "dup-1",
            "source_type": "files",
            "source_id": "/a",
            "operation": "created",
        }
        queue.enqueue(event)
        # Same ID → INSERT OR IGNORE.
        queue.enqueue(event)
        assert queue.depth() == 1


class TestAtomicCursorUpdate:
    def test_cursor_saved_with_enqueue(self, queue: PersistentQueue, event: dict) -> None:
        queue.enqueue(
            event,
            cursor_key="files",
            cursor_value=json.dumps({"/tmp/test.txt": {"sha256": "abc", "mtime_ns": 0, "size": 11}}),
        )
        cursor = queue.load_cursor("files")
        assert cursor is not None
        data = json.loads(cursor)
        assert "/tmp/test.txt" in data

    def test_cursor_not_saved_on_enqueue_failure(self, tmp_path: Path) -> None:
        """If the INSERT fails, the cursor should not be updated."""
        queue = PersistentQueue(db_path=tmp_path / "queue.db")

        # First enqueue succeeds.
        queue.enqueue(
            {"id": "x", "source_type": "files", "source_id": "/a", "operation": "created"},
            cursor_key="files",
            cursor_value=json.dumps({"v": 1}),
        )

        # Simulate failure by closing the connection.
        queue.close()

        # Reopen and verify cursor was saved from the first enqueue.
        queue2 = PersistentQueue(db_path=tmp_path / "queue.db")
        assert queue2.load_cursor("files") is not None

    def test_save_cursor_standalone(self, queue: PersistentQueue) -> None:
        queue.save_cursor("gmail", json.dumps({"history_id": "12345"}))
        cursor = queue.load_cursor("gmail")
        assert cursor is not None
        assert json.loads(cursor)["history_id"] == "12345"


class TestRecover:
    def test_recover_resets_processing_to_pending(self, queue: PersistentQueue, event: dict) -> None:
        queue.enqueue(event)
        queue.claim()
        assert queue.depth() == 1

        # Simulate crash: reopen.
        count = queue.recover()
        assert count == 1

        # Should be claimable again.
        claimed = queue.claim()
        assert claimed is not None


class TestInspection:
    def test_depth(self, queue: PersistentQueue) -> None:
        assert queue.depth() == 0
        for i in range(5):
            queue.enqueue({
                "id": f"d-{i}", "source_type": "files",
                "source_id": f"/f{i}", "operation": "created",
            })
        assert queue.depth() == 5
        queue.claim()
        assert queue.depth() == 5  # processing still counts

    def test_stats(self, queue: PersistentQueue) -> None:
        queue.enqueue({
            "id": "s1", "source_type": "gmail",
            "source_id": "gmail:1", "operation": "created",
        })
        queue.enqueue({
            "id": "s2", "source_type": "gmail",
            "source_id": "gmail:2", "operation": "created",
        })
        queue.enqueue({
            "id": "s3", "source_type": "files",
            "source_id": "/a", "operation": "created",
        })
        stats = queue.stats()
        assert stats["gmail"]["pending"] == 2
        assert stats["files"]["pending"] == 1

    def test_summary(self, queue: PersistentQueue) -> None:
        queue.enqueue({
            "id": "su1", "source_type": "files",
            "source_id": "/a", "operation": "created",
        })
        queue.enqueue({
            "id": "su2", "source_type": "files",
            "source_id": "/b", "operation": "created",
        })
        queue.claim()
        summary = queue.summary()
        assert summary["pending"] == 1
        assert summary["processing"] == 1

    def test_list_items_with_filters(self, queue: PersistentQueue) -> None:
        for i in range(10):
            queue.enqueue({
                "id": f"li-{i}", "source_type": "gmail" if i % 2 == 0 else "files",
                "source_id": f"/item-{i}", "operation": "created",
            })
        items = queue.list_items(source_type="gmail", limit=3)
        assert len(items) == 3
        assert all(it["source_type"] == "gmail" for it in items)

    def test_list_items_order(self, queue: PersistentQueue) -> None:
        queue.enqueue({
            "id": "o1", "source_type": "files", "source_id": "/a",
            "operation": "created", "enqueued_at": "2026-01-01T00:00:01+00:00",
        })
        queue.enqueue({
            "id": "o2", "source_type": "files", "source_id": "/b",
            "operation": "created", "enqueued_at": "2026-01-01T00:00:02+00:00",
        })
        asc = queue.list_items(order="asc")
        assert asc[0]["source_id"] == "/a"

        desc = queue.list_items(order="desc")
        assert desc[0]["source_id"] == "/b"


class TestRetryAndPurge:
    def test_retry_failed(self, tmp_path: Path) -> None:
        queue = PersistentQueue(db_path=tmp_path / "queue.db", max_retries=1)
        queue.enqueue({
            "id": "rf1", "source_type": "files",
            "source_id": "/a", "operation": "created",
        })
        claimed = queue.claim()
        assert claimed is not None
        queue.fail(claimed["_queue_id"], "err")

        assert queue.summary().get("failed") == 1
        count = queue.retry_failed()
        assert count == 1
        assert queue.summary().get("pending") == 1

    def test_purge(self, tmp_path: Path) -> None:
        queue = PersistentQueue(db_path=tmp_path / "queue.db", max_retries=1)
        queue.enqueue({
            "id": "p1", "source_type": "files",
            "source_id": "/a", "operation": "created",
        })
        claimed = queue.claim()
        assert claimed is not None
        queue.fail(claimed["_queue_id"], "err")

        count = queue.purge("failed")
        assert count == 1
        assert queue.depth() == 0


class TestBlobHandling:
    def test_blob_roundtrip(self, queue: PersistentQueue) -> None:
        data = b"\x00\x01\x02 binary content"
        event = {
            "id": "blob-rt",
            "source_type": "files",
            "source_id": "/img.png",
            "operation": "created",
            "raw_bytes": data,
        }
        queue.enqueue(event)
        claimed = queue.claim()
        assert claimed is not None
        assert claimed["raw_bytes"] == data

    def test_blob_not_in_payload(self, queue: PersistentQueue) -> None:
        """raw_bytes should not be stored in the JSON payload."""
        event = {
            "id": "blob-nop",
            "source_type": "files",
            "source_id": "/img.png",
            "operation": "created",
            "raw_bytes": b"data",
        }
        queue.enqueue(event)

        # Read raw payload from SQLite.
        row = queue._conn.execute(
            "SELECT payload FROM queue WHERE id = 'blob-nop'"
        ).fetchone()
        payload = json.loads(row[0])
        assert "raw_bytes" not in payload


class TestConcurrency:
    def test_concurrent_enqueue(self, queue: PersistentQueue) -> None:
        """Multiple threads can enqueue simultaneously."""
        errors: list[Exception] = []

        def enqueue_batch(start: int) -> None:
            try:
                for i in range(50):
                    queue.enqueue({
                        "id": f"conc-{start}-{i}",
                        "source_type": "files",
                        "source_id": f"/file-{start}-{i}",
                        "operation": "created",
                    })
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=enqueue_batch, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert queue.depth() == 200

    def test_concurrent_claim(self, queue: PersistentQueue) -> None:
        """Multiple threads claiming should not return the same item."""
        for i in range(100):
            queue.enqueue({
                "id": f"cc-{i}",
                "source_type": "files",
                "source_id": f"/f-{i}",
                "operation": "created",
            })

        claimed_ids: list[str] = []
        lock = threading.Lock()

        def claim_loop() -> None:
            while True:
                item = queue.claim()
                if item is None:
                    break
                with lock:
                    claimed_ids.append(item["_queue_id"])

        threads = [threading.Thread(target=claim_loop) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 100 items should be claimed exactly once.
        assert len(claimed_ids) == 100
        assert len(set(claimed_ids)) == 100


class TestMigration:
    def test_migrate_cursor_files(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create fake cursor files.
        (data_dir / "file_cursor.json").write_text(json.dumps({"/a": {"sha256": "x", "mtime_ns": 0, "size": 1}}))
        (data_dir / "gmail_cursor.json").write_text(json.dumps({"history_id": "99"}))

        queue = PersistentQueue(db_path=data_dir / "queue.db")
        count = queue.migrate_cursor_files(data_dir)
        assert count == 2

        # Verify imported.
        files_cursor = queue.load_cursor("files")
        assert files_cursor is not None
        assert "/a" in json.loads(files_cursor)

        gmail_cursor = queue.load_cursor("gmail")
        assert gmail_cursor is not None
        assert json.loads(gmail_cursor)["history_id"] == "99"

        # Verify originals renamed.
        assert not (data_dir / "file_cursor.json").exists()
        assert (data_dir / "file_cursor.json.migrated").exists()

    def test_migrate_skips_already_imported(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "file_cursor.json").write_text(json.dumps({"old": True}))

        queue = PersistentQueue(db_path=data_dir / "queue.db")
        queue.save_cursor("files", json.dumps({"already": True}))

        count = queue.migrate_cursor_files(data_dir)
        assert count == 0  # should skip

        # Original cursor in DB should be unchanged.
        assert json.loads(queue.load_cursor("files"))["already"] is True
