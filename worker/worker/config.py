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
import re
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
    skip_patterns: list[str] = field(
        default_factory=lambda: [
            "icon",
            "avatar",
            "favicon",
            "logo",
            "badge",
            "emoji",
            "thumb",
        ]
    )
    queue_size: int = 256


@dataclass
class ClusteringConfig:
    enabled: bool = True
    cron: str = "0 3 * * 0"
    min_corpus_size: int = 100
    min_interval_seconds: float = 60.0
    max_vectors: int = 500_000


@dataclass
class HealthConfig:
    enabled: bool = False
    port: int = 9100
    bind: str = "127.0.0.1"


@dataclass
class McpConfig:
    enabled: bool = True
    port: int = 3456
    auth_token: str | None = None

    def __post_init__(self) -> None:
        if not self.auth_token:
            self.auth_token = os.environ.get("FIELDNOTES_MCP_AUTH_TOKEN") or None


@dataclass
class MetricsConfig:
    pushgateway_url: str = "http://localhost:9091"
    push_interval: float = 15.0


@dataclass
class RateLimitConfig:
    """Parsed ``[rate_limits]`` section.

    All fields default to 0 (disabled / unlimited).
    """

    requests_per_minute: int = 0  # per-provider RPM; 0 = unlimited
    daily_token_budget: int = 0  # total tokens (input + output) per day; 0 = unlimited
    max_concurrency: int = 0  # max parallel LLM calls; 0 = unlimited


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
    health: HealthConfig = field(default_factory=HealthConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    rate_limits: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Expected embedding dimension used by the clustering pipeline.
    _EXPECTED_VECTOR_SIZE = 768

    def validate(self) -> list[str]:
        """Validate configuration thoroughly and return a list of warnings.

        Raises on hard errors (invalid cron, bad regex patterns).
        Returns warnings for soft issues (vector size mismatch).
        """
        from croniter import croniter

        warnings: list[str] = []

        # -- Hard errors --

        # (a) Validate cron expression
        cron_fields = self.clustering.cron.strip().split()
        if len(cron_fields) > 5:
            raise ValueError(
                f"[clustering] cron: {self.clustering.cron!r} has too many "
                f"fields (expected at most 5)"
            )
        if not croniter.is_valid(self.clustering.cron):
            raise ValueError(
                f"[clustering] cron: {self.clustering.cron!r} is not a valid "
                f"cron expression"
            )

        # (c) Validate vision.skip_patterns as valid regex
        for i, pattern in enumerate(self.vision.skip_patterns):
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"[vision] skip_patterns[{i}]: {pattern!r} is not a valid "
                    f"regex: {exc}"
                ) from exc

        # -- Warnings --

        # (b) Check role → model → provider references
        for role, model_alias in self.roles.mapping.items():
            if model_alias not in self.models:
                warnings.append(
                    f"[models.roles] role {role!r} references model "
                    f"{model_alias!r} which is not defined in [models.*]"
                )
            else:
                provider_name = self.models[model_alias].provider
                if provider_name not in self.providers:
                    warnings.append(
                        f"[models.{model_alias}] references provider "
                        f"{provider_name!r} which is not defined in "
                        f"[modelproviders.*]"
                    )

        # (c) Check vector_size against expected embedding dimensions
        if self.qdrant.vector_size != self._EXPECTED_VECTOR_SIZE:
            warnings.append(
                f"[qdrant] vector_size is {self.qdrant.vector_size}, but the "
                f"clustering pipeline expects {self._EXPECTED_VECTOR_SIZE}. "
                f"Ensure your embedding model produces "
                f"{self.qdrant.vector_size}-dimensional vectors."
            )

        # (d) Warn on clamped-looking min_interval_seconds boundaries
        if self.clustering.min_interval_seconds == 10.0:
            warnings.append(
                "[clustering] min_interval_seconds is at the minimum (10.0). "
                "This may cause excessive clustering runs."
            )
        if self.clustering.min_interval_seconds == 86_400.0:
            warnings.append(
                "[clustering] min_interval_seconds is at the maximum (86400). "
                "Clustering will run at most once per day."
            )

        for w in warnings:
            logger.warning(w)

        return warnings


def load_config(path: Path | None = None) -> Config:
    """Load and parse config.toml into a Config object."""
    path = path or DEFAULT_CONFIG_PATH
    raw = tomllib.loads(path.read_text())
    cfg = _parse(raw)
    cfg.neo4j.validate()
    cfg.validate()
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
        _check_type(
            section, "poll_interval_seconds", settings["poll_interval_seconds"], int
        )
    if "max_file_size" in settings:
        _check_type(section, "max_file_size", settings["max_file_size"], int)


def _validate_macos_apps_config(settings: dict[str, Any]) -> None:
    """Validate [sources.macos_apps] settings."""
    section = "sources.macos_apps"
    if "enabled" in settings:
        _check_type(section, "enabled", settings["enabled"], bool)
    if "scan_dirs" in settings:
        _check_list_of(section, "scan_dirs", settings["scan_dirs"], str)
    if "poll_interval_seconds" in settings:
        _check_type(
            section, "poll_interval_seconds", settings["poll_interval_seconds"], int
        )
    if "state_path" in settings:
        _check_type(section, "state_path", settings["state_path"], str)


def _validate_homebrew_config(settings: dict[str, Any]) -> None:
    """Validate [sources.homebrew] settings."""
    section = "sources.homebrew"
    if "enabled" in settings:
        _check_type(section, "enabled", settings["enabled"], bool)
    if "poll_interval_seconds" in settings:
        _check_type(
            section, "poll_interval_seconds", settings["poll_interval_seconds"], int
        )
    if "state_path" in settings:
        _check_type(section, "state_path", settings["state_path"], str)
    if "include_system" in settings:
        _check_type(section, "include_system", settings["include_system"], bool)


def _validate_google_calendar_config(settings: dict[str, Any]) -> None:
    """Validate [sources.google_calendar] settings."""
    section = "sources.google_calendar"
    if "poll_interval_seconds" in settings:
        _check_type(
            section, "poll_interval_seconds", settings["poll_interval_seconds"], int
        )
    if "max_initial_days" in settings:
        _check_type(section, "max_initial_days", settings["max_initial_days"], int)
    if "calendar_ids" in settings:
        _check_list_of(section, "calendar_ids", settings["calendar_ids"], str)
    if "client_secrets_path" in settings:
        _check_type(section, "client_secrets_path", settings["client_secrets_path"], str)


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
            name=name,
            type=provider_type,
            settings=settings,
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
        elif name == "macos_apps":
            _validate_macos_apps_config(settings)
        elif name == "homebrew":
            _validate_homebrew_config(settings)
        elif name == "google_calendar":
            _validate_google_calendar_config(settings)
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
            if cl["min_interval_seconds"] < 10.0:
                raise ValueError(
                    f"[clustering] min_interval_seconds must be >= 10.0, "
                    f"got {cl['min_interval_seconds']}"
                )
            if cl["min_interval_seconds"] > 86_400.0:
                raise ValueError(
                    f"[clustering] min_interval_seconds must be <= 86400 (24h), "
                    f"got {cl['min_interval_seconds']}"
                )
        if "max_vectors" in cl:
            _check_type("clustering", "max_vectors", cl["max_vectors"], int)
            if cl["max_vectors"] < 1:
                raise ValueError(
                    f"[clustering] max_vectors must be >= 1, got {cl['max_vectors']}"
                )
            if cl["max_vectors"] > 10_000_000:
                raise ValueError(
                    f"[clustering] max_vectors must be <= 10000000, "
                    f"got {cl['max_vectors']}"
                )
        cfg.clustering = ClusteringConfig(
            enabled=cl.get("enabled", cfg.clustering.enabled),
            cron=cl.get("cron", cfg.clustering.cron),
            min_corpus_size=cl.get("min_corpus_size", cfg.clustering.min_corpus_size),
            min_interval_seconds=cl.get(
                "min_interval_seconds", cfg.clustering.min_interval_seconds
            ),
            max_vectors=cl.get("max_vectors", cfg.clustering.max_vectors),
        )

    # [health]
    if "health" in raw:
        h = raw["health"]
        if "enabled" in h:
            _check_type("health", "enabled", h["enabled"], bool)
        if "port" in h:
            _check_type("health", "port", h["port"], int)
        if "bind" in h:
            _check_type("health", "bind", h["bind"], str)
        cfg.health = HealthConfig(
            enabled=h.get("enabled", cfg.health.enabled),
            port=h.get("port", cfg.health.port),
            bind=h.get("bind", cfg.health.bind),
        )

    # [mcp]
    if "mcp" in raw:
        m = raw["mcp"]
        if "enabled" in m:
            _check_type("mcp", "enabled", m["enabled"], bool)
        if "port" in m:
            _check_type("mcp", "port", m["port"], int)
        if "auth_token" in m:
            _check_type("mcp", "auth_token", m["auth_token"], str)
        cfg.mcp = McpConfig(
            enabled=m.get("enabled", cfg.mcp.enabled),
            port=m.get("port", cfg.mcp.port),
            auth_token=m.get("auth_token", cfg.mcp.auth_token),
        )

    # [metrics]
    if "metrics" in raw:
        mt = raw["metrics"]
        if "pushgateway_url" in mt:
            _check_type("metrics", "pushgateway_url", mt["pushgateway_url"], str)
        if "push_interval" in mt:
            _check_type("metrics", "push_interval", mt["push_interval"], (int, float))
        cfg.metrics = MetricsConfig(
            pushgateway_url=mt.get("pushgateway_url", cfg.metrics.pushgateway_url),
            push_interval=float(mt.get("push_interval", cfg.metrics.push_interval)),
        )

    # [rate_limits]
    if "rate_limits" in raw:
        rl = raw["rate_limits"]
        for k in ("requests_per_minute", "daily_token_budget", "max_concurrency"):
            if k in rl:
                _check_type("rate_limits", k, rl[k], int)
                if rl[k] < 0:
                    raise ValueError(f"[rate_limits] {k} must be >= 0, got {rl[k]}")
        cfg.rate_limits = RateLimitConfig(
            requests_per_minute=rl.get(
                "requests_per_minute",
                cfg.rate_limits.requests_per_minute,
            ),
            daily_token_budget=rl.get(
                "daily_token_budget",
                cfg.rate_limits.daily_token_budget,
            ),
            max_concurrency=rl.get(
                "max_concurrency",
                cfg.rate_limits.max_concurrency,
            ),
        )

    return cfg
