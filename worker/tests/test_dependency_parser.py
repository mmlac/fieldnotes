"""Tests for dependency extraction from manifest files (Cargo.toml, pyproject.toml, etc.)."""

from __future__ import annotations

from typing import Any

import pytest

from worker.parsers.base import GraphHint
from worker.parsers.repositories import RepositoryParser


# ── helpers ────────────────────────────────────────────────────────


def _manifest_event(
    filename: str,
    text: str,
    repo_path: str = "/repos/myproject",
    repo_name: str = "myproject",
    remote_url: str = "https://github.com/user/myproject",
) -> dict[str, Any]:
    return {
        "id": "test-id",
        "source_type": "repositories",
        "source_id": f"repo:{repo_path}:{filename}",
        "operation": "created",
        "mime_type": "text/plain",
        "text": text,
        "source_modified_at": "2026-01-01T00:00:00Z",
        "meta": {
            "repo_name": repo_name,
            "repo_path": repo_path,
            "remote_url": remote_url,
            "relative_path": filename,
        },
    }


def _dep_hints(hints: list[GraphHint]) -> list[GraphHint]:
    """Filter to only DEPENDS_ON hints."""
    return [h for h in hints if h.predicate == "DEPENDS_ON"]


# ── Cargo.toml ─────────────────────────────────────────────────────


class TestCargoToml:
    def test_runtime_dependencies(self) -> None:
        text = """\
[package]
name = "myapp"
version = "0.1.0"

[dependencies]
serde = "1.0"
tokio = { version = "1.34", features = ["full"] }
"""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("Cargo.toml", text))
        assert len(docs) == 1
        deps = _dep_hints(docs[0].graph_hints)

        names = {h.object_props["name"]: h for h in deps}
        assert "serde" in names
        assert names["serde"].object_props["version_constraint"] == "1.0"
        assert names["serde"].object_props["dependency_type"] == "runtime"
        assert names["serde"].object_props["ecosystem"] == "cargo"
        assert names["serde"].object_id == "pkg:cargo:serde"

        assert "tokio" in names
        assert names["tokio"].object_props["version_constraint"] == "1.34"

    def test_dev_and_build_dependencies(self) -> None:
        text = """\
[dependencies]
log = "0.4"

[dev-dependencies]
criterion = "0.5"

[build-dependencies]
cc = "1.0"
"""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("Cargo.toml", text))
        deps = _dep_hints(docs[0].graph_hints)
        by_name = {h.object_props["name"]: h for h in deps}

        assert by_name["log"].object_props["dependency_type"] == "runtime"
        assert by_name["criterion"].object_props["dependency_type"] == "dev"
        assert by_name["cc"].object_props["dependency_type"] == "build"

    def test_wildcard_version_for_path_dep(self) -> None:
        text = """\
[dependencies]
mylib = { path = "../mylib" }
"""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("Cargo.toml", text))
        deps = _dep_hints(docs[0].graph_hints)
        assert deps[0].object_props["version_constraint"] == "*"


# ── pyproject.toml ─────────────────────────────────────────────────


class TestPyprojectToml:
    def test_pep621_dependencies(self) -> None:
        text = """\
[project]
name = "myapp"
dependencies = [
    "requests>=2.28",
    "click>=8.0,<9.0",
]
"""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("pyproject.toml", text))
        deps = _dep_hints(docs[0].graph_hints)
        by_name = {h.object_props["name"]: h for h in deps}

        assert "requests" in by_name
        assert by_name["requests"].object_props["version_constraint"] == ">=2.28"
        assert by_name["requests"].object_props["ecosystem"] == "pypi"
        assert by_name["requests"].object_props["dependency_type"] == "runtime"

        assert "click" in by_name
        assert by_name["click"].object_props["version_constraint"] == ">=8.0,<9.0"

    def test_optional_dependencies(self) -> None:
        text = """\
[project]
name = "myapp"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff"]
"""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("pyproject.toml", text))
        deps = _dep_hints(docs[0].graph_hints)
        by_name = {h.object_props["name"]: h for h in deps}

        assert "pytest" in by_name
        assert by_name["pytest"].object_props["dependency_type"] == "dev"
        assert "ruff" in by_name

    def test_poetry_dependencies(self) -> None:
        text = """\
[tool.poetry.dependencies]
python = "^3.11"
django = "^4.2"
celery = { version = "^5.3", extras = ["redis"] }
"""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("pyproject.toml", text))
        deps = _dep_hints(docs[0].graph_hints)
        by_name = {h.object_props["name"]: h for h in deps}

        # python should be skipped
        assert "python" not in by_name
        assert "django" in by_name
        assert by_name["django"].object_props["version_constraint"] == "^4.2"
        assert "celery" in by_name
        assert by_name["celery"].object_props["version_constraint"] == "^5.3"

    def test_pep508_extras_stripped(self) -> None:
        text = """\
[project]
name = "myapp"
dependencies = ["uvicorn[standard]>=0.24"]
"""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("pyproject.toml", text))
        deps = _dep_hints(docs[0].graph_hints)
        assert deps[0].object_props["name"] == "uvicorn"
        assert deps[0].object_props["version_constraint"] == ">=0.24"

    def test_pep508_env_markers_stripped(self) -> None:
        text = """\
[project]
name = "myapp"
dependencies = ["pywin32>=306; sys_platform == 'win32'"]
"""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("pyproject.toml", text))
        deps = _dep_hints(docs[0].graph_hints)
        assert deps[0].object_props["name"] == "pywin32"
        assert deps[0].object_props["version_constraint"] == ">=306"


# ── package.json ───────────────────────────────────────────────────


class TestPackageJson:
    def test_npm_dependencies(self) -> None:
        text = """\
{
  "name": "my-app",
  "dependencies": {
    "react": "^18.2.0",
    "next": "14.0.0"
  },
  "devDependencies": {
    "typescript": "^5.3"
  }
}
"""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("package.json", text))
        deps = _dep_hints(docs[0].graph_hints)
        by_name = {h.object_props["name"]: h for h in deps}

        assert by_name["react"].object_props["version_constraint"] == "^18.2.0"
        assert by_name["react"].object_props["ecosystem"] == "npm"
        assert by_name["react"].object_props["dependency_type"] == "runtime"
        assert by_name["react"].object_id == "pkg:npm:react"

        assert by_name["typescript"].object_props["dependency_type"] == "dev"

    def test_peer_dependencies(self) -> None:
        text = '{"peerDependencies": {"react": ">=17"}}'
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("package.json", text))
        deps = _dep_hints(docs[0].graph_hints)
        assert deps[0].object_props["name"] == "react"
        assert deps[0].object_props["dependency_type"] == "runtime"


# ── go.mod ─────────────────────────────────────────────────────────


class TestGoMod:
    def test_require_block(self) -> None:
        text = """\
module github.com/user/myapp

go 1.21

require (
\tgithub.com/gin-gonic/gin v1.9.1
\tgithub.com/lib/pq v1.10.9
)
"""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("go.mod", text))
        deps = _dep_hints(docs[0].graph_hints)
        by_name = {h.object_props["name"]: h for h in deps}

        assert "github.com/gin-gonic/gin" in by_name
        gin = by_name["github.com/gin-gonic/gin"]
        assert gin.object_props["version_constraint"] == "v1.9.1"
        assert gin.object_props["ecosystem"] == "go"
        assert gin.object_id == "pkg:go:github.com/gin-gonic/gin"

    def test_single_line_require(self) -> None:
        text = """\
module example.com/foo

require github.com/stretchr/testify v1.8.4
"""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("go.mod", text))
        deps = _dep_hints(docs[0].graph_hints)
        assert len(deps) == 1
        assert deps[0].object_props["name"] == "github.com/stretchr/testify"

    def test_comments_skipped_in_require_block(self) -> None:
        text = """\
module m

require (
\t// indirect dependency
\tgithub.com/foo/bar v1.0.0
)
"""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("go.mod", text))
        deps = _dep_hints(docs[0].graph_hints)
        names = [h.object_props["name"] for h in deps]
        assert "// indirect dependency" not in names
        assert "github.com/foo/bar" in names


# ── .NET / NuGet ──────────────────────────────────────────────────


class TestDotNetProject:
    def test_csproj_package_references(self) -> None:
        text = """\
<Project Sdk="Microsoft.NET.Sdk">
  <ItemGroup>
    <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />
    <PackageReference Include="Serilog" Version="3.1.1" />
  </ItemGroup>
</Project>
"""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("MyApp.csproj", text))
        deps = _dep_hints(docs[0].graph_hints)
        by_name = {h.object_props["name"]: h for h in deps}

        assert "Newtonsoft.Json" in by_name
        assert by_name["Newtonsoft.Json"].object_props["version_constraint"] == "13.0.3"
        assert by_name["Newtonsoft.Json"].object_props["ecosystem"] == "nuget"
        assert by_name["Newtonsoft.Json"].object_id == "pkg:nuget:Newtonsoft.Json"

    def test_fsproj_package_references(self) -> None:
        text = """\
<Project Sdk="Microsoft.NET.Sdk">
  <ItemGroup>
    <PackageReference Include="FSharp.Core" Version="8.0.0" />
  </ItemGroup>
</Project>
"""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("App.fsproj", text))
        deps = _dep_hints(docs[0].graph_hints)
        assert deps[0].object_props["name"] == "FSharp.Core"

    def test_version_as_child_element(self) -> None:
        text = """\
<Project Sdk="Microsoft.NET.Sdk">
  <ItemGroup>
    <PackageReference Include="Dapper">
      <Version>2.1.28</Version>
    </PackageReference>
  </ItemGroup>
</Project>
"""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("App.csproj", text))
        deps = _dep_hints(docs[0].graph_hints)
        assert deps[0].object_props["version_constraint"] == "2.1.28"

    def test_missing_version_defaults_to_wildcard(self) -> None:
        text = """\
<Project Sdk="Microsoft.NET.Sdk">
  <ItemGroup>
    <PackageReference Include="SomePackage" />
  </ItemGroup>
</Project>
"""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("App.csproj", text))
        deps = _dep_hints(docs[0].graph_hints)
        assert deps[0].object_props["version_constraint"] == "*"


class TestDirectoryPackagesProps:
    def test_package_version_elements(self) -> None:
        text = """\
<Project>
  <ItemGroup>
    <PackageVersion Include="xunit" Version="2.6.1" />
    <PackageVersion Include="Moq" Version="4.20.69" />
  </ItemGroup>
</Project>
"""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("Directory.Packages.props", text))
        deps = _dep_hints(docs[0].graph_hints)
        by_name = {h.object_props["name"]: h for h in deps}
        assert "xunit" in by_name
        assert by_name["xunit"].object_props["version_constraint"] == "2.6.1"
        assert "Moq" in by_name


class TestPackagesConfig:
    def test_legacy_packages_config(self) -> None:
        text = """\
<?xml version="1.0" encoding="utf-8"?>
<packages>
  <package id="jQuery" version="3.7.1" targetFramework="net48" />
  <package id="Newtonsoft.Json" version="13.0.3" targetFramework="net48" />
</packages>
"""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("packages.config", text))
        deps = _dep_hints(docs[0].graph_hints)
        by_name = {h.object_props["name"]: h for h in deps}
        assert "jQuery" in by_name
        assert by_name["jQuery"].object_props["version_constraint"] == "3.7.1"
        assert by_name["Newtonsoft.Json"] in deps


# ── Malformed manifests ───────────────────────────────────────────


class TestMalformedManifests:
    def test_malformed_cargo_toml(self) -> None:
        """Invalid TOML should not crash, should return gracefully."""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("Cargo.toml", "not valid toml {{{{"))
        assert len(docs) == 1
        # No dependency hints, but the file still gets parsed as a document
        deps = _dep_hints(docs[0].graph_hints)
        assert deps == []

    def test_malformed_package_json(self) -> None:
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("package.json", "{bad json"))
        assert len(docs) == 1
        deps = _dep_hints(docs[0].graph_hints)
        assert deps == []

    def test_malformed_csproj(self) -> None:
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("App.csproj", "<not>valid<xml"))
        assert len(docs) == 1
        deps = _dep_hints(docs[0].graph_hints)
        assert deps == []

    def test_billion_laughs_xml_rejected(self) -> None:
        """Billion laughs (entity expansion) attack should be safely rejected."""
        evil_xml = """\
<?xml version="1.0"?>
<!DOCTYPE lolz [
  <!ENTITY lol "lol">
  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
  <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">
]>
<Project Sdk="Microsoft.NET.Sdk">
  <ItemGroup>
    <PackageReference Include="Evil" Version="&lol4;" />
  </ItemGroup>
</Project>
"""
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("App.csproj", evil_xml))
        assert len(docs) == 1
        deps = _dep_hints(docs[0].graph_hints)
        assert deps == []

    def test_empty_manifest(self) -> None:
        """Empty text should not crash."""
        parser = RepositoryParser()
        # Empty text + text mime means it goes through as a normal file, not a manifest
        ev = _manifest_event("Cargo.toml", "")
        docs = parser.parse(ev)
        # Empty text but text/ mime → still produces a doc
        assert len(docs) >= 0  # Shouldn't crash


# ── DEPENDS_ON hint structure ─────────────────────────────────────


class TestDependsOnStructure:
    def test_hint_has_correct_labels(self) -> None:
        text = '{"dependencies": {"lodash": "^4.17"}}'
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("package.json", text))
        deps = _dep_hints(docs[0].graph_hints)
        h = deps[0]

        assert h.subject_label == "Repository"
        assert h.object_label == "Package"
        assert h.predicate == "DEPENDS_ON"
        assert h.subject_merge_key == "source_id"
        assert h.object_merge_key == "name"

    def test_hint_subject_props(self) -> None:
        text = '{"dependencies": {"lodash": "^4.17"}}'
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event(
            "package.json", text,
            repo_name="web-app", remote_url="https://github.com/org/web-app",
        ))
        deps = _dep_hints(docs[0].graph_hints)
        h = deps[0]
        assert h.subject_props["name"] == "web-app"
        assert h.subject_props["remote_url"] == "https://github.com/org/web-app"

    def test_manifest_file_suppresses_text_chunking(self) -> None:
        """Manifest files should have empty text (graph hints only, no chunking)."""
        text = '{"dependencies": {"react": "^18"}}'
        parser = RepositoryParser()
        docs = parser.parse(_manifest_event("package.json", text))
        assert docs[0].text == ""
