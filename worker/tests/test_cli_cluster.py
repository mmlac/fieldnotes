"""Tests for the ``fieldnotes cluster`` CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from worker.cli import main, _build_parser
from worker.clustering.cluster import ClusterResult


# ------------------------------------------------------------------
# Parser tests
# ------------------------------------------------------------------


class TestClusterParser:
    def test_cluster_basic(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["cluster"])
        assert args.command == "cluster"
        assert args.min_cluster_size is None
        assert args.force is False

    def test_cluster_min_cluster_size(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["cluster", "--min-cluster-size", "5"])
        assert args.min_cluster_size == 5

    def test_cluster_force(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["cluster", "--force"])
        assert args.force is True

    def test_cluster_all_flags(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["cluster", "--min-cluster-size", "3", "--force"])
        assert args.min_cluster_size == 3
        assert args.force is True


# ------------------------------------------------------------------
# run_cluster tests (mocked backends)
# ------------------------------------------------------------------


def _make_config_mock(min_corpus_size: int = 100) -> MagicMock:
    cfg = MagicMock()
    cfg.clustering.min_corpus_size = min_corpus_size
    cfg.clustering.max_vectors = 500_000
    return cfg


def _make_cluster_results(n: int = 2) -> list[ClusterResult]:
    results = []
    for i in range(n):
        results.append(
            ClusterResult(
                cluster_id=i,
                chunk_ids=[f"chunk_{i}_0", f"chunk_{i}_1"],
                centroid=[0.1] * 32,
            )
        )
    return results


class TestRunCluster:
    @patch("worker.cli.cluster.link_apps_to_topics")
    @patch("worker.cli.cluster.write_clusters")
    @patch("worker.cli.cluster.label_clusters")
    @patch("worker.cli.cluster.cluster_embeddings")
    @patch("worker.cli.cluster._corpus_size", return_value=200)
    @patch("worker.cli.cluster.ModelRegistry")
    @patch("worker.cli.cluster.load_config")
    def test_full_pipeline(
        self,
        mock_load: MagicMock,
        mock_reg: MagicMock,
        mock_corpus: MagicMock,
        mock_cluster: MagicMock,
        mock_label: MagicMock,
        mock_write: MagicMock,
        mock_link: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_load.return_value = _make_config_mock()
        clusters = _make_cluster_results(3)
        mock_cluster.return_value = clusters
        labeled = [MagicMock() for _ in range(3)]
        mock_label.return_value = labeled

        from worker.cli.cluster import run_cluster

        rc = run_cluster(config_path=None)

        assert rc == 0
        mock_cluster.assert_called_once()
        mock_label.assert_called_once()
        mock_write.assert_called_once()
        mock_link.assert_called_once()
        out = capsys.readouterr().out
        assert "Fetching vectors from Qdrant" in out
        assert "3 topics discovered" in out

    @patch("worker.cli.cluster._corpus_size", return_value=10)
    @patch("worker.cli.cluster.ModelRegistry")
    @patch("worker.cli.cluster.load_config")
    def test_corpus_too_small_exits_cleanly(
        self,
        mock_load: MagicMock,
        mock_reg: MagicMock,
        mock_corpus: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_load.return_value = _make_config_mock(min_corpus_size=100)

        from worker.cli.cluster import run_cluster

        rc = run_cluster(config_path=None)

        assert rc == 0
        out = capsys.readouterr().out
        assert "need at least 100" in out
        assert "--force" in out

    @patch("worker.cli.cluster.link_apps_to_topics")
    @patch("worker.cli.cluster.write_clusters")
    @patch("worker.cli.cluster.label_clusters")
    @patch("worker.cli.cluster.cluster_embeddings")
    @patch("worker.cli.cluster._corpus_size", return_value=10)
    @patch("worker.cli.cluster.ModelRegistry")
    @patch("worker.cli.cluster.load_config")
    def test_force_runs_with_small_corpus(
        self,
        mock_load: MagicMock,
        mock_reg: MagicMock,
        mock_corpus: MagicMock,
        mock_cluster: MagicMock,
        mock_label: MagicMock,
        mock_write: MagicMock,
        mock_link: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_load.return_value = _make_config_mock(min_corpus_size=100)
        clusters = _make_cluster_results(1)
        mock_cluster.return_value = clusters
        mock_label.return_value = [MagicMock()]

        from worker.cli.cluster import run_cluster

        rc = run_cluster(config_path=None, force=True)

        assert rc == 0
        mock_cluster.assert_called_once()
        # Verify min_corpus_size was set to 0 for force mode
        call_kwargs = mock_cluster.call_args
        assert call_kwargs[1]["min_corpus_size"] == 0

    @patch("worker.cli.cluster.cluster_embeddings")
    @patch("worker.cli.cluster._corpus_size", return_value=200)
    @patch("worker.cli.cluster.ModelRegistry")
    @patch("worker.cli.cluster.load_config")
    def test_no_clusters_found(
        self,
        mock_load: MagicMock,
        mock_reg: MagicMock,
        mock_corpus: MagicMock,
        mock_cluster: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_load.return_value = _make_config_mock()
        mock_cluster.return_value = []

        from worker.cli.cluster import run_cluster

        rc = run_cluster(config_path=None)

        assert rc == 0
        assert "No clusters found" in capsys.readouterr().out

    @patch("worker.cli.cluster.cluster_embeddings")
    @patch("worker.cli.cluster._corpus_size", return_value=200)
    @patch("worker.cli.cluster.ModelRegistry")
    @patch("worker.cli.cluster.load_config")
    def test_min_cluster_size_override(
        self,
        mock_load: MagicMock,
        mock_reg: MagicMock,
        mock_corpus: MagicMock,
        mock_cluster: MagicMock,
    ) -> None:
        mock_load.return_value = _make_config_mock()
        mock_cluster.return_value = []

        from worker.cli.cluster import run_cluster

        run_cluster(config_path=None, min_cluster_size=5)

        call_kwargs = mock_cluster.call_args
        assert call_kwargs[1]["min_cluster_size"] == 5


class TestMainCluster:
    @patch("worker.cli.cluster.link_apps_to_topics")
    @patch("worker.cli.cluster.write_clusters")
    @patch("worker.cli.cluster.label_clusters")
    @patch("worker.cli.cluster.cluster_embeddings")
    @patch("worker.cli.cluster._corpus_size", return_value=200)
    @patch("worker.cli.cluster.ModelRegistry")
    @patch("worker.cli.cluster.load_config")
    def test_main_cluster_integration(
        self,
        mock_load: MagicMock,
        mock_reg: MagicMock,
        mock_corpus: MagicMock,
        mock_cluster: MagicMock,
        mock_label: MagicMock,
        mock_write: MagicMock,
        mock_link: MagicMock,
    ) -> None:
        mock_load.return_value = _make_config_mock()
        mock_cluster.return_value = _make_cluster_results(2)
        mock_label.return_value = [MagicMock(), MagicMock()]

        rc = main(["cluster"])
        assert rc == 0

    @patch("worker.cli.cluster.load_config", side_effect=FileNotFoundError("no config"))
    def test_main_cluster_exception(
        self,
        mock_load: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(["cluster"])
        assert rc == 1
        assert "no config" in capsys.readouterr().err
