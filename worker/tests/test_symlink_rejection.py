"""Tests for symlink rejection across all source adapters.

Verifies that symlinked paths are skipped with warnings to prevent
symlink-following attacks that could index arbitrary files.
"""

from __future__ import annotations

from pathlib import Path


from worker.sources.files import FileSource
from worker.sources.homebrew import _formula_binaries
from worker.sources.macos_apps import _discover_apps
from worker.sources.obsidian import ObsidianSource, discover_vaults


# ── FileSource symlink rejection ─────────────────────────────────


def test_file_source_configure_skips_symlinked_watch_path(tmp_path: Path):
    """Symlinked watch_paths should be filtered out during configure."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real_dir)

    fs = FileSource()
    fs.configure({"watch_paths": [str(link), str(real_dir)]})

    # Only the real directory should remain
    assert len(fs._watch_paths) == 1
    assert fs._watch_paths[0] == real_dir


def test_file_source_configure_keeps_non_symlinked_paths(tmp_path: Path):
    """Non-symlinked watch_paths should be kept."""
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    d1.mkdir()
    d2.mkdir()

    fs = FileSource()
    fs.configure({"watch_paths": [str(d1), str(d2)]})

    assert len(fs._watch_paths) == 2


# ── ObsidianSource symlink rejection ─────────────────────────────


def test_obsidian_configure_skips_symlinked_vault_path(tmp_path: Path):
    """Symlinked vault_paths should be filtered out during configure."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real_dir)

    src = ObsidianSource()
    src.configure({"vault_paths": [str(link), str(real_dir)]})

    assert len(src._vault_paths) == 1
    assert src._vault_paths[0] == real_dir


def test_discover_vaults_skips_symlinked_base(tmp_path: Path):
    """discover_vaults should skip symlinked search base directories."""
    real_vault = tmp_path / "real_vault"
    real_vault.mkdir()
    (real_vault / ".obsidian").mkdir()

    link = tmp_path / "link_vault"
    link.symlink_to(real_vault)

    # Only the real path should discover the vault
    vaults = discover_vaults([real_vault, link])
    assert len(vaults) == 1
    assert vaults[0] == real_vault


def test_discover_vaults_skips_symlinked_child(tmp_path: Path):
    """discover_vaults should skip symlinked children during one-level scan."""
    base = tmp_path / "base"
    base.mkdir()

    real_vault = tmp_path / "real_vault"
    real_vault.mkdir()
    (real_vault / ".obsidian").mkdir()

    # Symlink inside the base directory
    (base / "linked_vault").symlink_to(real_vault)

    # Also add a real vault inside base for comparison
    real_child = base / "real_child"
    real_child.mkdir()
    (real_child / ".obsidian").mkdir()

    vaults = discover_vaults([base])
    assert len(vaults) == 1
    assert vaults[0] == real_child


# ── macOS apps symlink rejection ─────────────────────────────────


def test_discover_apps_skips_symlinked_app(tmp_path: Path):
    """Symlinked .app bundles should be skipped."""
    real_app = tmp_path / "Real.app"
    real_app.mkdir()

    link_app = tmp_path / "Fake.app"
    link_app.symlink_to(real_app)

    apps = _discover_apps([tmp_path])
    assert len(apps) == 1
    assert apps[0] == real_app


def test_discover_apps_skips_symlinked_subdir(tmp_path: Path):
    """Symlinked subdirectories should be skipped during one-level recursion."""
    real_subdir = tmp_path / "Utilities"
    real_subdir.mkdir()
    real_app = real_subdir / "Terminal.app"
    real_app.mkdir()

    linked_subdir = tmp_path / "LinkedUtils"
    linked_subdir.symlink_to(real_subdir)

    apps = _discover_apps([tmp_path])
    assert len(apps) == 1
    assert apps[0] == real_app


def test_discover_apps_skips_symlinked_grandchild(tmp_path: Path):
    """Symlinked .app bundles inside subdirs should be skipped."""
    subdir = tmp_path / "Utilities"
    subdir.mkdir()

    real_app = tmp_path / "Real.app"
    real_app.mkdir()

    link_app = subdir / "Fake.app"
    link_app.symlink_to(real_app)

    # Also place a real app in subdir
    real_grandchild = subdir / "Genuine.app"
    real_grandchild.mkdir()

    apps = _discover_apps([tmp_path])
    names = [a.name for a in apps]
    assert "Real.app" in names
    assert "Genuine.app" in names
    assert "Fake.app" not in names


# ── Homebrew binary symlink verification ─────────────────────────


def test_formula_binaries_only_follows_symlinks(tmp_path: Path):
    """_formula_binaries should only consider symlinks in bin_dir."""
    prefix = tmp_path / "prefix"
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True)
    cellar = prefix / "Cellar" / "myformula" / "1.0" / "bin"
    cellar.mkdir(parents=True)

    # Create a real binary in cellar
    real_bin = cellar / "mytool"
    real_bin.write_text("#!/bin/sh\necho hi")

    # Create a symlink in bin_dir pointing to cellar binary
    (bin_dir / "mytool").symlink_to(real_bin)

    # Create a non-symlink regular file in bin_dir (should be ignored)
    regular = bin_dir / "regular"
    regular.write_text("#!/bin/sh\necho regular")

    bins = _formula_binaries(str(prefix), "myformula")
    assert "mytool" in bins
    assert "regular" not in bins


def test_formula_binaries_skips_dangling_symlinks(tmp_path: Path):
    """_formula_binaries should handle dangling symlinks gracefully."""
    prefix = tmp_path / "prefix"
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True)
    cellar_dir = prefix / "Cellar" / "myformula"
    cellar_dir.mkdir(parents=True)

    # Create a dangling symlink
    (bin_dir / "broken").symlink_to(tmp_path / "nonexistent")

    bins = _formula_binaries(str(prefix), "myformula")
    assert bins == []
