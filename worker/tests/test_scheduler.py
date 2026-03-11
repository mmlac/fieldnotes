"""Tests for clustering/scheduler.py — cron-based clustering scheduler."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from worker.clustering.scheduler import (
    _corpus_size,
    _seconds_until_next,
    clustering_loop,
    run_clustering_pipeline,
)
from worker.config import ClusteringConfig, Neo4jConfig, QdrantConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clustering_cfg(
    enabled: bool = True,
    cron: str = "0 3 * * 0",
    min_corpus_size: int = 100,
) -> ClusteringConfig:
    return ClusteringConfig(enabled=enabled, cron=cron, min_corpus_size=min_corpus_size)


def _mock_registry() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# _corpus_size
# ---------------------------------------------------------------------------


class TestCorpusSize:
    def test_returns_point_count(self) -> None:
        with patch("worker.clustering.scheduler.QdrantClient") as MockClient:
            client = MockClient.return_value
            info = MagicMock()
            info.points_count = 250
            client.get_collection.return_value = info

            result = _corpus_size(QdrantConfig())

        assert result == 250

    def test_returns_zero_for_none_count(self) -> None:
        with patch("worker.clustering.scheduler.QdrantClient") as MockClient:
            client = MockClient.return_value
            info = MagicMock()
            info.points_count = None
            client.get_collection.return_value = info

            result = _corpus_size(QdrantConfig())

        assert result == 0

    def test_closes_client(self) -> None:
        with patch("worker.clustering.scheduler.QdrantClient") as MockClient:
            client = MockClient.return_value
            info = MagicMock()
            info.points_count = 10
            client.get_collection.return_value = info

            _corpus_size(QdrantConfig())

            client.close.assert_called_once()

    def test_closes_client_on_error(self) -> None:
        with patch("worker.clustering.scheduler.QdrantClient") as MockClient:
            client = MockClient.return_value
            client.get_collection.side_effect = RuntimeError("fail")

            with pytest.raises(RuntimeError):
                _corpus_size(QdrantConfig())

            client.close.assert_called_once()


# ---------------------------------------------------------------------------
# _seconds_until_next
# ---------------------------------------------------------------------------


class TestSecondsUntilNext:
    def test_returns_positive_float(self) -> None:
        # Every minute — next trigger is at most 60s away
        result = _seconds_until_next("* * * * *")
        assert 0.0 <= result <= 60.0

    def test_weekly_cron_returns_large_value(self) -> None:
        # Weekly at 3am Sunday — should be > 0 and <= 7 days
        result = _seconds_until_next("0 3 * * 0")
        assert 0.0 <= result <= 7 * 24 * 3600


# ---------------------------------------------------------------------------
# run_clustering_pipeline
# ---------------------------------------------------------------------------


class TestRunClusteringPipeline:
    def test_skips_when_corpus_too_small(self) -> None:
        with patch("worker.clustering.scheduler._corpus_size", return_value=50):
            result = run_clustering_pipeline(
                _mock_registry(),
                _clustering_cfg(min_corpus_size=100),
                QdrantConfig(),
                Neo4jConfig(),
            )

        assert result is False

    def test_runs_full_pipeline(self) -> None:
        mock_clusters = [MagicMock()]
        mock_labeled = [MagicMock()]

        with (
            patch("worker.clustering.scheduler._corpus_size", return_value=200),
            patch("worker.clustering.scheduler.cluster_embeddings", return_value=mock_clusters) as mock_ce,
            patch("worker.clustering.scheduler.label_clusters", return_value=mock_labeled) as mock_lc,
            patch("worker.clustering.scheduler.write_clusters") as mock_wc,
        ):
            result = run_clustering_pipeline(
                _mock_registry(),
                _clustering_cfg(min_corpus_size=100),
                QdrantConfig(),
                Neo4jConfig(),
            )

        assert result is True
        mock_ce.assert_called_once()
        mock_lc.assert_called_once()
        mock_wc.assert_called_once()

    def test_skips_write_when_all_noise(self) -> None:
        with (
            patch("worker.clustering.scheduler._corpus_size", return_value=200),
            patch("worker.clustering.scheduler.cluster_embeddings", return_value=[]),
            patch("worker.clustering.scheduler.label_clusters") as mock_lc,
            patch("worker.clustering.scheduler.write_clusters") as mock_wc,
        ):
            result = run_clustering_pipeline(
                _mock_registry(),
                _clustering_cfg(min_corpus_size=100),
                QdrantConfig(),
                Neo4jConfig(),
            )

        assert result is True
        mock_lc.assert_not_called()
        mock_wc.assert_not_called()

    def test_handles_corpus_shrink_race(self) -> None:
        from worker.clustering.cluster import CorpusTooSmallError

        with (
            patch("worker.clustering.scheduler._corpus_size", return_value=200),
            patch(
                "worker.clustering.scheduler.cluster_embeddings",
                side_effect=CorpusTooSmallError("shrank"),
            ),
        ):
            result = run_clustering_pipeline(
                _mock_registry(),
                _clustering_cfg(min_corpus_size=100),
                QdrantConfig(),
                Neo4jConfig(),
            )

        assert result is False

    def test_passes_min_corpus_size_to_cluster(self) -> None:
        with (
            patch("worker.clustering.scheduler._corpus_size", return_value=200),
            patch("worker.clustering.scheduler.cluster_embeddings", return_value=[]) as mock_ce,
        ):
            run_clustering_pipeline(
                _mock_registry(),
                _clustering_cfg(min_corpus_size=150),
                QdrantConfig(),
                Neo4jConfig(),
            )

        _, kwargs = mock_ce.call_args
        assert kwargs["min_corpus_size"] == 150


# ---------------------------------------------------------------------------
# clustering_loop (async integration)
# ---------------------------------------------------------------------------


class TestClusteringLoop:
    @pytest.mark.asyncio
    async def test_runs_pipeline_and_cancels_cleanly(self) -> None:
        """Verify the loop calls the pipeline and exits on cancellation."""
        call_count = 0

        def fake_pipeline(*args: object) -> bool:
            nonlocal call_count
            call_count += 1
            return True

        cfg = _clustering_cfg()
        cfg.min_interval_seconds = 0.0  # allow fast iteration in test

        with (
            patch("worker.clustering.scheduler._seconds_until_next", return_value=0.0),
            patch("worker.clustering.scheduler.run_clustering_pipeline", side_effect=fake_pipeline),
        ):
            task = asyncio.create_task(
                clustering_loop(
                    _mock_registry(),
                    cfg,
                    QdrantConfig(),
                    Neo4jConfig(),
                )
            )
            # Let it run a couple of iterations
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert call_count >= 1

    @pytest.mark.asyncio
    async def test_continues_after_pipeline_error(self) -> None:
        """Pipeline errors should be logged but not crash the loop."""
        calls: list[int] = []

        def failing_pipeline(*args: object) -> bool:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("transient failure")
            return True

        cfg = _clustering_cfg()
        cfg.min_interval_seconds = 0.0  # allow fast iteration in test

        with (
            patch("worker.clustering.scheduler._seconds_until_next", return_value=0.0),
            patch("worker.clustering.scheduler.run_clustering_pipeline", side_effect=failing_pipeline),
        ):
            task = asyncio.create_task(
                clustering_loop(
                    _mock_registry(),
                    cfg,
                    QdrantConfig(),
                    Neo4jConfig(),
                )
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        # Should have called at least twice (first fails, second succeeds)
        assert len(calls) >= 2

    @pytest.mark.asyncio
    async def test_min_interval_prevents_tight_loop(self) -> None:
        """When cron returns 0, min_interval_seconds prevents tight-looping."""
        call_times: list[float] = []

        def timed_pipeline(*args: object) -> bool:
            call_times.append(asyncio.get_event_loop().time())
            return True

        cfg = _clustering_cfg()
        cfg.min_interval_seconds = 0.05  # 50ms minimum

        with (
            patch("worker.clustering.scheduler._seconds_until_next", return_value=0.0),
            patch("worker.clustering.scheduler.run_clustering_pipeline", side_effect=timed_pipeline),
        ):
            task = asyncio.create_task(
                clustering_loop(
                    _mock_registry(),
                    cfg,
                    QdrantConfig(),
                    Neo4jConfig(),
                )
            )
            await asyncio.sleep(0.2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        # With 50ms min interval and 200ms runtime, we should get at most ~4 calls
        # Without the fix, we'd get hundreds
        assert len(call_times) <= 6
