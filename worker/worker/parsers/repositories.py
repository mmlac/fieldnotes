"""Repository parser for file and commit events.

Transforms IngestEvents from the RepositorySource into ParsedDocuments with
GraphHints. File events produce Repository→File CONTAINS relationships.
Commit events produce Commit nodes with Person→Commit AUTHORED,
Commit→File MODIFIED, and Commit→Repository PART_OF relationships.

For manifest files (Cargo.toml, pyproject.toml, package.json, go.mod, *.csproj,
*.fsproj, *.vbproj, Directory.Packages.props, packages.config), dependency
sections are parsed and emitted as Repository → Package DEPENDS_ON GraphHints
instead of free-text chunks.
"""

from __future__ import annotations

import json
import logging
import re
import tomllib
import defusedxml
import defusedxml.ElementTree as ET
from typing import Any

from .base import BaseParser, GraphHint, ParsedDocument
from .registry import register

logger = logging.getLogger(__name__)

_MAX_MANIFEST_SIZE = 1 * 1024 * 1024  # 1 MiB — manifests should never be this large
_MAX_DEPS_PER_MANIFEST = (
    500  # Cap dependencies to prevent 100k GraphHints from a single manifest
)

# Manifest filenames that trigger dependency extraction
_MANIFEST_FILENAMES: set[str] = {
    "Cargo.toml",
    "pyproject.toml",
    "package.json",
    "go.mod",
    "Directory.Packages.props",
    "packages.config",
}

# Extensions that trigger dependency extraction (.NET project files)
_MANIFEST_EXTENSIONS: set[str] = {".csproj", ".fsproj", ".vbproj"}


@register
class RepositoryParser(BaseParser):
    """Parses repository file IngestEvents into ParsedDocuments with graph hints."""

    @property
    def source_type(self) -> str:
        return "repositories"

    def parse(self, event: dict[str, Any]) -> list[ParsedDocument]:
        source_id: str = event["source_id"]

        # Route commit events to dedicated handler
        if source_id.startswith("commit:"):
            return self._parse_commit(event)

        return self._parse_file(event)

    def _parse_commit(self, event: dict[str, Any]) -> list[ParsedDocument]:
        """Parse a commit IngestEvent into a ParsedDocument with graph hints."""
        source_id: str = event["source_id"]
        meta: dict[str, Any] = event.get("meta", {})

        sha: str = meta.get("sha", "")
        author_name: str = meta.get("author_name", "")
        author_email: str = meta.get("author_email", "")
        date: str = meta.get("date", "")
        repo_name: str = meta.get("repo_name", "")
        repo_path: str = meta.get("repo_path", "")
        remote_url: str | None = meta.get("remote_url")
        changed_files: list[str] = meta.get("changed_files", [])

        text: str = event.get("text", "")

        node_props: dict[str, Any] = {
            "sha": sha,
            "message": text,
            "date": date,
        }

        graph_hints: list[GraphHint] = []

        # Person → Commit via AUTHORED
        if author_email:
            norm_email = author_email.strip().lower()
            graph_hints.append(
                GraphHint(
                    subject_id=norm_email,
                    subject_label="Person",
                    predicate="AUTHORED",
                    object_id=source_id,
                    object_label="Commit",
                    subject_props={"name": author_name, "email": norm_email},
                    object_props={},
                    subject_merge_key="email",
                    object_merge_key="source_id",
                )
            )

        # Commit → Repository via PART_OF
        graph_hints.append(
            GraphHint(
                subject_id=source_id,
                subject_label="Commit",
                predicate="PART_OF",
                object_id=repo_path,
                object_label="Repository",
                subject_props={},
                object_props={
                    "name": repo_name,
                    "path": repo_path,
                    "remote_url": remote_url,
                },
                subject_merge_key="source_id",
                object_merge_key="source_id",
            )
        )

        # Commit → File via MODIFIED for each changed file
        for file_path in changed_files:
            file_source_id = f"repo:{repo_path}:{file_path}"
            graph_hints.append(
                GraphHint(
                    subject_id=source_id,
                    subject_label="Commit",
                    predicate="MODIFIED",
                    object_id=file_source_id,
                    object_label="File",
                    subject_props={},
                    object_props={},
                    subject_merge_key="source_id",
                    object_merge_key="source_id",
                )
            )

        return [
            ParsedDocument(
                source_type=self.source_type,
                source_id=source_id,
                operation="created",
                text=text,
                node_label="Commit",
                node_props=node_props,
                graph_hints=graph_hints,
                source_metadata={
                    "repo_name": repo_name,
                    "repo_path": repo_path,
                    "sha": sha,
                },
            )
        ]

    def _parse_file(self, event: dict[str, Any]) -> list[ParsedDocument]:
        """Parse a repository file IngestEvent into a ParsedDocument."""
        source_id: str = event["source_id"]
        operation: str = event.get("operation", "created")
        meta: dict[str, Any] = event.get("meta", {})

        if operation == "deleted":
            return [
                ParsedDocument(
                    source_type=self.source_type,
                    source_id=source_id,
                    operation="deleted",
                    text="",
                )
            ]

        repo_name: str = meta.get("repo_name", "")
        repo_path: str = meta.get("repo_path", "")
        remote_url: str | None = meta.get("remote_url")
        relative_path: str = meta.get("relative_path", "")

        # Skip binary files — text content will be absent for non-text mime types
        text: str = event.get("text", "")
        mime_type: str = event.get("mime_type", "text/plain")
        if not text and not mime_type.startswith("text/"):
            logger.warning(
                "Skipping binary file %s in repo %s (mime: %s)",
                relative_path,
                repo_name,
                mime_type,
            )
            return []

        # File node properties
        node_props: dict[str, Any] = {
            "path": relative_path,
            "name": relative_path.rsplit("/", 1)[-1] if relative_path else "",
            "ext": _file_ext(relative_path),
        }
        if meta.get("sha256"):
            node_props["sha256"] = meta["sha256"]
        if event.get("source_modified_at"):
            node_props["modified_at"] = event["source_modified_at"]

        # Repository → File CONTAINS relationship
        graph_hints: list[GraphHint] = [
            GraphHint(
                subject_id=repo_path,
                subject_label="Repository",
                predicate="CONTAINS",
                object_id=source_id,
                object_label="File",
                subject_props={
                    "name": repo_name,
                    "path": repo_path,
                    "remote_url": remote_url,
                },
                object_props={},
                subject_merge_key="source_id",
                object_merge_key="source_id",
            ),
        ]

        # Check if this is a manifest file → extract dependencies
        filename = relative_path.rsplit("/", 1)[-1] if relative_path else ""
        file_ext = _file_ext(relative_path)
        is_manifest = (filename in _MANIFEST_FILENAMES) or (
            file_ext in _MANIFEST_EXTENSIONS
        )
        if is_manifest and text:
            dep_hints = _extract_dependencies(
                filename, text, repo_path, repo_name, remote_url
            )
            graph_hints.extend(dep_hints)
            # Manifest files: emit graph hints only, no text chunking
            return [
                ParsedDocument(
                    source_type=self.source_type,
                    source_id=source_id,
                    operation=operation,
                    text="",
                    mime_type=mime_type,
                    node_label="File",
                    node_props=node_props,
                    graph_hints=graph_hints,
                    source_metadata={
                        "repo_name": repo_name,
                        "repo_path": repo_path,
                        "relative_path": relative_path,
                    },
                )
            ]

        return [
            ParsedDocument(
                source_type=self.source_type,
                source_id=source_id,
                operation=operation,
                text=text,
                mime_type=mime_type,
                node_label="File",
                node_props=node_props,
                graph_hints=graph_hints,
                source_metadata={
                    "repo_name": repo_name,
                    "repo_path": repo_path,
                    "relative_path": relative_path,
                },
            )
        ]


# ---------------------------------------------------------------------------
# Dependency extraction helpers
# ---------------------------------------------------------------------------


def _extract_dependencies(
    filename: str,
    text: str,
    repo_path: str,
    repo_name: str,
    remote_url: str | None,
) -> list[GraphHint]:
    """Dispatch to the right parser based on manifest filename."""
    if len(text) > _MAX_MANIFEST_SIZE:
        logger.warning(
            "Manifest %s in %s too large (%d bytes > %d), skipping dependency extraction",
            filename,
            repo_name,
            len(text),
            _MAX_MANIFEST_SIZE,
        )
        return []
    hints: list[GraphHint] = []
    try:
        if filename == "Cargo.toml":
            hints = _parse_cargo_toml(text, repo_path, repo_name, remote_url)
        elif filename == "pyproject.toml":
            hints = _parse_pyproject_toml(text, repo_path, repo_name, remote_url)
        elif filename == "package.json":
            hints = _parse_package_json(text, repo_path, repo_name, remote_url)
        elif filename == "go.mod":
            hints = _parse_go_mod(text, repo_path, repo_name, remote_url)
        elif _file_ext(filename) in _MANIFEST_EXTENSIONS:
            hints = _parse_dotnet_project(text, repo_path, repo_name, remote_url)
        elif filename == "Directory.Packages.props":
            hints = _parse_directory_packages_props(
                text, repo_path, repo_name, remote_url
            )
        elif filename == "packages.config":
            hints = _parse_packages_config(text, repo_path, repo_name, remote_url)
    except (
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
        ET.ParseError,
        KeyError,
        ValueError,
    ) as exc:
        logger.error(
            "Failed to parse dependencies from %s in %s: %s", filename, repo_name, exc
        )
        return []

    if len(hints) > _MAX_DEPS_PER_MANIFEST:
        logger.warning(
            "Manifest %s in %s has %d dependencies, truncating to %d",
            filename,
            repo_name,
            len(hints),
            _MAX_DEPS_PER_MANIFEST,
        )
        hints = hints[:_MAX_DEPS_PER_MANIFEST]
    return hints


def _dep_hint(
    repo_path: str,
    repo_name: str,
    remote_url: str | None,
    package_name: str,
    version_constraint: str,
    dep_type: str,
    ecosystem: str,
) -> GraphHint:
    """Build a Repository → Package DEPENDS_ON GraphHint."""
    return GraphHint(
        subject_id=repo_path,
        subject_label="Repository",
        predicate="DEPENDS_ON",
        object_id=f"pkg:{ecosystem}:{package_name}",
        object_label="Package",
        subject_props={
            "name": repo_name,
            "path": repo_path,
            "remote_url": remote_url,
        },
        object_props={
            "name": package_name,
            "version_constraint": version_constraint,
            "dependency_type": dep_type,
            "ecosystem": ecosystem,
        },
        subject_merge_key="source_id",
        object_merge_key="name",
    )


# -- Cargo.toml -------------------------------------------------------------


def _parse_cargo_toml(
    text: str, repo_path: str, repo_name: str, remote_url: str | None
) -> list[GraphHint]:
    data = tomllib.loads(text)
    hints: list[GraphHint] = []

    section_map = {
        "dependencies": "runtime",
        "dev-dependencies": "dev",
        "build-dependencies": "build",
    }
    for section, dep_type in section_map.items():
        deps = data.get(section, {})
        for pkg, spec in deps.items():
            version = _cargo_version(spec)
            hints.append(
                _dep_hint(
                    repo_path, repo_name, remote_url, pkg, version, dep_type, "cargo"
                )
            )
    return hints


def _cargo_version(spec: Any) -> str:
    """Extract version string from Cargo dependency spec (string or table)."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return spec.get("version", "*")
    return "*"


# -- pyproject.toml ----------------------------------------------------------


def _parse_pyproject_toml(
    text: str, repo_path: str, repo_name: str, remote_url: str | None
) -> list[GraphHint]:
    data = tomllib.loads(text)
    hints: list[GraphHint] = []

    # [project.dependencies]
    for req in data.get("project", {}).get("dependencies", []):
        name, constraint = _parse_pep508(req)
        hints.append(
            _dep_hint(
                repo_path, repo_name, remote_url, name, constraint, "runtime", "pypi"
            )
        )

    # [project.optional-dependencies.*]
    for group, reqs in data.get("project", {}).get("optional-dependencies", {}).items():
        for req in reqs:
            name, constraint = _parse_pep508(req)
            hints.append(
                _dep_hint(
                    repo_path, repo_name, remote_url, name, constraint, "dev", "pypi"
                )
            )

    # [tool.poetry.dependencies]
    poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
    for pkg, spec in poetry_deps.items():
        if pkg.lower() == "python":
            continue
        version = (
            spec
            if isinstance(spec, str)
            else spec.get("version", "*")
            if isinstance(spec, dict)
            else "*"
        )
        hints.append(
            _dep_hint(repo_path, repo_name, remote_url, pkg, version, "runtime", "pypi")
        )

    return hints


_PEP508_NAME_RE = re.compile(r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)")


def _parse_pep508(requirement: str) -> tuple[str, str]:
    """Extract package name and version constraint from a PEP 508 string."""
    requirement = requirement.strip()
    m = _PEP508_NAME_RE.match(requirement)
    if not m:
        return requirement, "*"
    name = m.group(1)
    rest = requirement[m.end() :].strip()
    # Strip extras like [extra1,extra2]
    if rest.startswith("["):
        bracket_end = rest.find("]")
        if bracket_end != -1:
            rest = rest[bracket_end + 1 :].strip()
    # Strip environment markers after ;
    if ";" in rest:
        rest = rest[: rest.index(";")].strip()
    return name, rest if rest else "*"


# -- package.json ------------------------------------------------------------


def _parse_package_json(
    text: str, repo_path: str, repo_name: str, remote_url: str | None
) -> list[GraphHint]:
    data = json.loads(text)
    hints: list[GraphHint] = []

    section_map = {
        "dependencies": "runtime",
        "devDependencies": "dev",
        "peerDependencies": "runtime",
    }
    for section, dep_type in section_map.items():
        deps = data.get(section, {})
        if not isinstance(deps, dict):
            continue
        for pkg, version in deps.items():
            hints.append(
                _dep_hint(
                    repo_path,
                    repo_name,
                    remote_url,
                    pkg,
                    version if isinstance(version, str) else "*",
                    dep_type,
                    "npm",
                )
            )
    return hints


# -- go.mod ------------------------------------------------------------------

_GO_REQUIRE_BLOCK_RE = re.compile(r"require\s*\((.*?)\)", re.DOTALL)
_GO_REQUIRE_SINGLE_RE = re.compile(r"^require\s+([^\s(]\S*)\s+(\S+)", re.MULTILINE)
_GO_REQUIRE_LINE_RE = re.compile(r"^\s*(\S+)\s+(\S+)", re.MULTILINE)


def _parse_go_mod(
    text: str, repo_path: str, repo_name: str, remote_url: str | None
) -> list[GraphHint]:
    hints: list[GraphHint] = []

    # Multi-line require blocks
    for block_match in _GO_REQUIRE_BLOCK_RE.finditer(text):
        block = block_match.group(1)
        for line_match in _GO_REQUIRE_LINE_RE.finditer(block):
            module = line_match.group(1)
            version = line_match.group(2)
            if module.startswith("//"):
                continue
            hints.append(
                _dep_hint(
                    repo_path, repo_name, remote_url, module, version, "runtime", "go"
                )
            )

    # Single-line requires (e.g. "require github.com/foo/bar v1.0.0")
    for m in _GO_REQUIRE_SINGLE_RE.finditer(text):
        module, version = m.group(1), m.group(2)
        hints.append(
            _dep_hint(
                repo_path, repo_name, remote_url, module, version, "runtime", "go"
            )
        )

    return hints


# -- .NET / NuGet ---------------------------------------------------------------


def _parse_dotnet_xml(text: str) -> ET.Element | None:
    """Parse XML text safely, returning root element or None on malformed input.

    Uses defusedxml to reject DTD definitions and entity expansion (billion laughs).
    """
    try:
        return ET.fromstring(text)
    except ET.ParseError:
        logger.warning("Malformed XML in .NET project file, skipping")
        return None
    except defusedxml.DefusedXmlException:
        # DTDForbidden / EntitiesForbidden / ExternalReferenceForbidden
        # for malicious XML payloads (billion laughs, XXE, etc.)
        logger.error(
            "Rejected unsafe XML in .NET project file (DTD/entity attack), skipping"
        )
        return None


def _parse_dotnet_project(
    text: str, repo_path: str, repo_name: str, remote_url: str | None
) -> list[GraphHint]:
    """Parse PackageReference elements from csproj/fsproj/vbproj files."""
    root = _parse_dotnet_xml(text)
    if root is None:
        return []
    hints: list[GraphHint] = []
    for ref in root.iter("PackageReference"):
        pkg = ref.get("Include") or ref.get("include") or ""
        if not pkg:
            continue
        # Version can be an attribute or a child element
        version = ref.get("Version") or ref.get("version") or ""
        if not version:
            version_el = ref.find("Version")
            if version_el is None:
                version_el = ref.find("version")
            if version_el is not None and version_el.text:
                version = version_el.text.strip()
        if not version:
            version = "*"
        hints.append(
            _dep_hint(
                repo_path, repo_name, remote_url, pkg, version, "runtime", "nuget"
            )
        )
    return hints


def _parse_directory_packages_props(
    text: str, repo_path: str, repo_name: str, remote_url: str | None
) -> list[GraphHint]:
    """Parse PackageVersion elements from Directory.Packages.props (Central Package Management)."""
    root = _parse_dotnet_xml(text)
    if root is None:
        return []
    hints: list[GraphHint] = []
    for ref in root.iter("PackageVersion"):
        pkg = ref.get("Include") or ref.get("include") or ""
        if not pkg:
            continue
        version = ref.get("Version") or ref.get("version") or "*"
        hints.append(
            _dep_hint(
                repo_path, repo_name, remote_url, pkg, version, "runtime", "nuget"
            )
        )
    return hints


def _parse_packages_config(
    text: str, repo_path: str, repo_name: str, remote_url: str | None
) -> list[GraphHint]:
    """Parse package elements from legacy packages.config (NuGet)."""
    root = _parse_dotnet_xml(text)
    if root is None:
        return []
    hints: list[GraphHint] = []
    for ref in root.iter("package"):
        pkg = ref.get("id") or ""
        if not pkg:
            continue
        version = ref.get("version") or "*"
        hints.append(
            _dep_hint(
                repo_path, repo_name, remote_url, pkg, version, "runtime", "nuget"
            )
        )
    return hints


def _file_ext(path: str) -> str:
    """Extract file extension from a path (e.g. '.md', '.toml')."""
    dot = path.rfind(".")
    if dot == -1 or dot == len(path) - 1:
        return ""
    return path[dot:]
