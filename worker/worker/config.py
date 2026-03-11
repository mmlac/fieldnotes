"""Config loader for the three-layer TOML configuration.

Parses ~/.fieldnotes/config.toml into typed dataclasses covering:
  - [modelproviders.*] — provider connection parameters
  - [models.*] — named model definitions bound to providers
  - [models.roles] — pipeline role → model alias mapping
  - [sources.*] — source adapter configuration
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = Path.home() / ".fieldnotes" / "config.toml"


def _check_type(section: str, key: str, value: Any, expected: type) -> None:
    """Raise TypeError if *value* is not an instance of *expected*."""
    if not isinstance(value, expected):
        raise TypeError(
            f"[{section}] {key}: expected {expected.__name__}, "
            f"got {type(value).__name__} ({value!r})"
        )


def _check_list_of(section: str, key: str, value: Any, item_type: type) -> None:
    """Raise TypeError if *value* is not a list of *item_type*."""
    _check_type(section, key, value, list)
    for i, item in enumerate(value):
        if not isinstance(item, item_type):
            raise TypeError(
                f"[{section}] {key}[{i}]: expected {item_type.__name__}, "
                f"got {type(item).__name__} ({item!r})"
            )


@dataclass
class CoreConfig:
    data_dir: str = "~/.fieldnotes/data"
    log_level: str = "info"


@dataclass
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    user: str = ""
    password: str = ""

    def __post_init__(self) -> None:
        if not self.user:
            self.user = os.environ.get("NEO4J_USER", "neo4j")
        if not self.password:
            self.password = os.environ.get("NEO4J_PASSWORD", "")

    def validate(self) -> None:
        """Raise if required fields are missing."""
        if not self.password:
            raise ValueError(
                "Neo4j password must be set via [neo4j] password in config.toml "
                "or the NEO4J_PASSWORD environment variable"
            )


@dataclass
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    collection: str = "fieldnotes"
    vector_size: int = 768


@dataclass
class ProviderConfig:
    """A single [modelproviders.<name>] section."""
    name: str
    type: str
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """A single [models.<alias>] section (not roles)."""
    alias: str
    provider: str
    model: str


@dataclass
class RolesConfig:
    """[models.roles] — maps pipeline roles to model aliases."""
    mapping: dict[str, str] = field(default_factory=dict)

    def get(self, role: str) -> str | None:
        return self.mapping.get(role)


@dataclass
class SourceConfig:
    """A single [sources.<name>] section."""
    name: str
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionConfig:
    enabled: bool = True
    concurrency: int = 2
    min_file_size_kb: int = 1
    max_file_size_mb: int = 20
    skip_patterns: list[str] = field(default_factory=lambda: [
        "icon", "avatar", "favicon", "logo", "badge", "emoji", "thumb",
    ])
    queue_size: int = 256


@dataclass
class ClusteringConfig:
    enabled: bool = True
    cron: str = "0 3 * * 0"
    min_corpus_size: int = 100
    min_interval_seconds: float = 60.0
    max_vectors: int = 500_000


@dataclass
class McpConfig:
    enabled: bool = True
    port: int = 3456


@dataclass
class Config:
    core: CoreConfig = field(default_factory=CoreConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    models: dict[str, ModelConfig] = field(default_factory=dict)
    roles: RolesConfig = field(default_factory=RolesConfig)
    sources: dict[str, SourceConfig] = field(default_factory=dict)
    vision: VisionConfig = field(default_factory=VisionConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    mcp: McpConfig = field(default_factory=McpConfig)


def load_config(path: Path | None = None) -> Config:
    """Load and parse config.toml into a Config object."""
    path = path or DEFAULT_CONFIG_PATH
    raw = tomllib.loads(path.read_text())
    cfg = _parse(raw)
    cfg.neo4j.validate()
    return cfg


def _validate_repositories_config(settings: dict[str, Any]) -> None:
    """Validate [sources.repositories] settings."""
    section = "sources.repositories"
    if "repo_roots" in settings:
        _check_list_of(section, "repo_roots", settings["repo_roots"], str)
    if "include_patterns" in settings:
        _check_list_of(section, "include_patterns", settings["include_patterns"], str)
    if "exclude_patterns" in settings:
        _check_list_of(section, "exclude_patterns", settings["exclude_patterns"], str)
    if "poll_interval_seconds" in settings:
        _check_type(section, "poll_interval_seconds", settings["poll_interval_seconds"], int)
    if "max_file_size" in settings:
        _check_type(section, "max_file_size", settings["max_file_size"], int)


def _parse(raw: dict[str, Any]) -> Config:
    cfg = Config()

    # [core]
    if "core" in raw:
        c = raw["core"]
        for k in ("data_dir", "log_level"):
            if k in c:
                _check_type("core", k, c[k], str)
        cfg.core = CoreConfig(
            data_dir=c.get("data_dir", cfg.core.data_dir),
            log_level=c.get("log_level", cfg.core.log_level),
        )

    # [neo4j]
    if "neo4j" in raw:
        n = raw["neo4j"]
        for k in ("uri", "user", "password"):
            if k in n:
                _check_type("neo4j", k, n[k], str)
        cfg.neo4j = Neo4jConfig(
            uri=n.get("uri", cfg.neo4j.uri),
            user=n.get("user", cfg.neo4j.user),
            password=n.get("password", cfg.neo4j.password),
        )

    # [qdrant]
    if "qdrant" in raw:
        q = raw["qdrant"]
        for k in ("host", "collection"):
            if k in q:
                _check_type("qdrant", k, q[k], str)
        for k in ("port", "vector_size"):
            if k in q:
                _check_type("qdrant", k, q[k], int)
        cfg.qdrant = QdrantConfig(
            host=q.get("host", cfg.qdrant.host),
            port=q.get("port", cfg.qdrant.port),
            collection=q.get("collection", cfg.qdrant.collection),
            vector_size=q.get("vector_size", cfg.qdrant.vector_size),
        )

    # [modelproviders.*]
    for name, pcfg in raw.get("modelproviders", {}).items():
        provider_type = pcfg["type"]
        settings = {k: v for k, v in pcfg.items() if k != "type"}
        cfg.providers[name] = ProviderConfig(
            name=name, type=provider_type, settings=settings,
        )

    # [models.*] and [models.roles]
    for alias, mcfg in raw.get("models", {}).items():
        if alias == "roles":
            cfg.roles = RolesConfig(mapping=dict(mcfg))
        else:
            cfg.models[alias] = ModelConfig(
                alias=alias,
                provider=mcfg["provider"],
                model=mcfg["model"],
            )

    # [sources.*]
    for name, scfg in raw.get("sources", {}).items():
        settings = dict(scfg)
        if name == "repositories":
            _validate_repositories_config(settings)
        cfg.sources[name] = SourceConfig(name=name, settings=settings)

    # [vision]
    if "vision" in raw:
        vi = raw["vision"]
        if "enabled" in vi:
            _check_type("vision", "enabled", vi["enabled"], bool)
        for k in ("concurrency", "min_file_size_kb", "max_file_size_mb", "queue_size"):
            if k in vi:
                _check_type("vision", k, vi[k], int)
        if "skip_patterns" in vi:
            _check_list_of("vision", "skip_patterns", vi["skip_patterns"], str)
        cfg.vision = VisionConfig(
            enabled=vi.get("enabled", cfg.vision.enabled),
            concurrency=vi.get("concurrency", cfg.vision.concurrency),
            min_file_size_kb=vi.get("min_file_size_kb", cfg.vision.min_file_size_kb),
            max_file_size_mb=vi.get("max_file_size_mb", cfg.vision.max_file_size_mb),
            skip_patterns=vi.get("skip_patterns", cfg.vision.skip_patterns),
            queue_size=vi.get("queue_size", cfg.vision.queue_size),
        )

    # [clustering]
    if "clustering" in raw:
        cl = raw["clustering"]
        if "enabled" in cl:
            _check_type("clustering", "enabled", cl["enabled"], bool)
        if "cron" in cl:
            _check_type("clustering", "cron", cl["cron"], str)
        if "min_corpus_size" in cl:
            _check_type("clustering", "min_corpus_size", cl["min_corpus_size"], int)
        if "min_interval_seconds" in cl:
            if not isinstance(cl["min_interval_seconds"], (int, float)):
                raise TypeError(
                    f"[clustering] min_interval_seconds: expected float, "
                    f"got {type(cl['min_interval_seconds']).__name__} "
                    f"({cl['min_interval_seconds']!r})"
                )
        if "max_vectors" in cl:
            _check_type("clustering", "max_vectors", cl["max_vectors"], int)
        cfg.clustering = ClusteringConfig(
            enabled=cl.get("enabled", cfg.clustering.enabled),
            cron=cl.get("cron", cfg.clustering.cron),
            min_corpus_size=cl.get("min_corpus_size", cfg.clustering.min_corpus_size),
            min_interval_seconds=cl.get("min_interval_seconds", cfg.clustering.min_interval_seconds),
            max_vectors=cl.get("max_vectors", cfg.clustering.max_vectors),
        )

    # [mcp]
    if "mcp" in raw:
        m = raw["mcp"]
        if "enabled" in m:
            _check_type("mcp", "enabled", m["enabled"], bool)
        if "port" in m:
            _check_type("mcp", "port", m["port"], int)
        cfg.mcp = McpConfig(
            enabled=m.get("enabled", cfg.mcp.enabled),
            port=m.get("port", cfg.mcp.port),
        )

    return cfg
