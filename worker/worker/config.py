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

from worker.parsers.attachments import DEFAULT_INDEXABLE_MIMETYPES

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = Path.home() / ".fieldnotes" / "config.toml"


# Account names for [sources.gmail.<name>] / [sources.google_calendar.<name>].
# Lowercase letter to start, then up to 30 of [a-z0-9_-].
_ACCOUNT_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,30}$")

# Reserved account names. Cursor keys use the form ``{source}:{account}``
# (e.g. ``gmail:personal``) and the legacy single-account migration uses
# the bare source word as the old key (``gmail``, ``calendar``).  Allowing
# an account literally named after a source produces ambiguous keys like
# ``gmail:gmail`` that collide with migration intermediate state.  Adjacent
# names (``gmail-work``, ``mygmail``) remain valid.
_RESERVED_ACCOUNT_NAMES = frozenset({"gmail", "calendar", "google_calendar", "slack"})


class MigrationRequiredError(Exception):
    """Raised when the legacy single-table source schema is detected.

    The new multi-account schema (``[sources.gmail.<account>]``) is required.
    Users must run the migration command to clean stored data, then update
    their config to the keyed form.
    """


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
class GmailAccountConfig:
    """A single ``[sources.gmail.<account>]`` section."""

    name: str
    enabled: bool = True
    client_secrets_path: str = "~/.fieldnotes/gmail_credentials.json"
    poll_interval_seconds: int = 300
    max_initial_threads: int = 500
    label_filter: str = "INBOX"
    download_attachments: bool = False
    attachment_indexable_mimetypes: list[str] = field(
        default_factory=lambda: list(DEFAULT_INDEXABLE_MIMETYPES)
    )
    attachment_max_size_mb: int = 25
    attachment_pdf_max_pages: int = 1000
    attachment_pdf_per_page_chars: int = 1_000_000
    attachment_pdf_timeout_seconds: int = 60


@dataclass
class CalendarAccountConfig:
    """A single ``[sources.google_calendar.<account>]`` section."""

    name: str
    enabled: bool = True
    client_secrets_path: str = "~/.fieldnotes/gmail_credentials.json"
    poll_interval_seconds: int = 300
    max_initial_days: int = 90
    calendar_ids: list[str] = field(default_factory=lambda: ["primary"])
    download_attachments: bool = False
    attachment_indexable_mimetypes: list[str] = field(
        default_factory=lambda: list(DEFAULT_INDEXABLE_MIMETYPES)
    )
    attachment_max_size_mb: int = 25
    attachment_pdf_max_pages: int = 1000
    attachment_pdf_per_page_chars: int = 1_000_000
    attachment_pdf_timeout_seconds: int = 60


@dataclass
class MeConfig:
    """Top-level ``[me]`` block: identifies the current user across sources."""

    emails: list[str] = field(default_factory=list)
    name: str | None = None


@dataclass
class SlackSourceConfig:
    """Parsed ``[sources.slack]`` section.

    Configures the Slack ingestion adapter: auth, polling, channel filters,
    and burst-window splitting parameters.
    """

    enabled: bool = False
    client_secrets_path: str = "~/.fieldnotes/slack_credentials.json"
    poll_interval_seconds: int = 300
    max_initial_days: int = 90
    include_channels: list[str] = field(default_factory=list)
    exclude_channels: list[str] = field(default_factory=list)
    include_dms: bool = True
    include_archived: bool = False
    window_max_tokens: int = 512
    window_gap_seconds: int = 1800
    window_overlap_messages: int = 3
    users_refresh_interval_seconds: int = 3600
    download_attachments: bool = False
    attachment_indexable_mimetypes: list[str] = field(
        default_factory=lambda: list(DEFAULT_INDEXABLE_MIMETYPES)
    )
    attachment_max_size_mb: int = 25
    attachment_pdf_max_pages: int = 1000
    attachment_pdf_per_page_chars: int = 1_000_000
    attachment_pdf_timeout_seconds: int = 60


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
class RerankerConfig:
    """Parsed ``[reranker]`` section.

    Controls the optional second-stage reranker that re-scores hybrid
    search candidates with a cross-encoder before they reach the LLM
    or the user.  The model itself is bound via ``[models.roles] rerank``.
    """

    enabled: bool = True
    top_k_pre: int = 50
    top_k_post: int = 10
    score_threshold: float = 0.0
    batch_size: int = 32


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
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    slack: SlackSourceConfig = field(default_factory=SlackSourceConfig)
    gmail: dict[str, GmailAccountConfig] = field(default_factory=dict)
    google_calendar: dict[str, CalendarAccountConfig] = field(default_factory=dict)
    me: MeConfig | None = None

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

        # (e) Warn when Slack include/exclude channels are both non-empty
        if self.slack.include_channels and self.slack.exclude_channels:
            warnings.append(
                "[sources.slack] include_channels and exclude_channels are both "
                "non-empty; exclude_channels will be ignored when include_channels "
                "is set"
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
    if "index_only_patterns" in settings:
        _check_list_of(
            section, "index_only_patterns", settings["index_only_patterns"], str
        )
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
    if "poll_interval_seconds" in settings:
        _check_type(
            section, "poll_interval_seconds", settings["poll_interval_seconds"], int
        )
    if "state_path" in settings:
        _check_type(section, "state_path", settings["state_path"], str)
    if "include_system" in settings:
        _check_type(section, "include_system", settings["include_system"], bool)


# Inclusive bounds enforced by _validate_attachment_settings on the
# attachment_max_size_mb knob.  1 MB is the smallest meaningful indexable
# attachment; 200 MB is well above any realistic email/calendar/Slack
# upload limit and stops accidental "index multi-GB tarballs" misconfigs.
_ATTACHMENT_MAX_SIZE_MIN_MB = 1
_ATTACHMENT_MAX_SIZE_MAX_MB = 200

# Inclusive bounds for the PDF-bomb defenses. Floors prevent an operator
# from accidentally disabling the guard with a zero/negative value;
# ceilings cap the worst case the parser will agree to attempt before
# giving up.
_ATTACHMENT_PDF_MAX_PAGES_MIN = 1
_ATTACHMENT_PDF_MAX_PAGES_MAX = 100_000
_ATTACHMENT_PDF_PER_PAGE_CHARS_MIN = 1_000
_ATTACHMENT_PDF_PER_PAGE_CHARS_MAX = 100_000_000
_ATTACHMENT_PDF_TIMEOUT_SECONDS_MIN = 1
_ATTACHMENT_PDF_TIMEOUT_SECONDS_MAX = 3600


def _validate_attachment_settings(section: str, settings: dict[str, Any]) -> None:
    """Validate the shared attachment knobs on any source section.

    Checks ``attachment_indexable_mimetypes`` (list[str]),
    ``attachment_max_size_mb`` (int in [1, 200]) and the three PDF-bomb
    defense knobs. Type / range errors raise; missing keys are fine
    (defaults will fill in).
    """
    if "attachment_indexable_mimetypes" in settings:
        _check_list_of(
            section,
            "attachment_indexable_mimetypes",
            settings["attachment_indexable_mimetypes"],
            str,
        )
    if "attachment_max_size_mb" in settings:
        _check_type(
            section,
            "attachment_max_size_mb",
            settings["attachment_max_size_mb"],
            int,
        )
        v = settings["attachment_max_size_mb"]
        if not _ATTACHMENT_MAX_SIZE_MIN_MB <= v <= _ATTACHMENT_MAX_SIZE_MAX_MB:
            raise ValueError(
                f"[{section}] attachment_max_size_mb must be in "
                f"[{_ATTACHMENT_MAX_SIZE_MIN_MB}, {_ATTACHMENT_MAX_SIZE_MAX_MB}], "
                f"got {v}"
            )
    for key, lo, hi in (
        (
            "attachment_pdf_max_pages",
            _ATTACHMENT_PDF_MAX_PAGES_MIN,
            _ATTACHMENT_PDF_MAX_PAGES_MAX,
        ),
        (
            "attachment_pdf_per_page_chars",
            _ATTACHMENT_PDF_PER_PAGE_CHARS_MIN,
            _ATTACHMENT_PDF_PER_PAGE_CHARS_MAX,
        ),
        (
            "attachment_pdf_timeout_seconds",
            _ATTACHMENT_PDF_TIMEOUT_SECONDS_MIN,
            _ATTACHMENT_PDF_TIMEOUT_SECONDS_MAX,
        ),
    ):
        if key in settings:
            _check_type(section, key, settings[key], int)
            v = settings[key]
            if not lo <= v <= hi:
                raise ValueError(f"[{section}] {key} must be in [{lo}, {hi}], got {v}")


def _parse_slack_config(settings: dict[str, Any]) -> SlackSourceConfig:
    """Validate and parse [sources.slack] settings into SlackSourceConfig."""
    section = "sources.slack"
    defaults = SlackSourceConfig()

    if "enabled" in settings:
        _check_type(section, "enabled", settings["enabled"], bool)
    if "client_secrets_path" in settings:
        _check_type(
            section, "client_secrets_path", settings["client_secrets_path"], str
        )
    if "poll_interval_seconds" in settings:
        _check_type(
            section, "poll_interval_seconds", settings["poll_interval_seconds"], int
        )
    if "max_initial_days" in settings:
        _check_type(section, "max_initial_days", settings["max_initial_days"], int)
    if "include_channels" in settings:
        _check_list_of(section, "include_channels", settings["include_channels"], str)
    if "exclude_channels" in settings:
        _check_list_of(section, "exclude_channels", settings["exclude_channels"], str)
    if "include_dms" in settings:
        _check_type(section, "include_dms", settings["include_dms"], bool)
    if "include_archived" in settings:
        _check_type(section, "include_archived", settings["include_archived"], bool)
    if "window_max_tokens" in settings:
        _check_type(section, "window_max_tokens", settings["window_max_tokens"], int)
        v = settings["window_max_tokens"]
        if not 128 <= v <= 4096:
            raise ValueError(
                f"[{section}] window_max_tokens must be in [128, 4096], got {v}"
            )
    if "window_gap_seconds" in settings:
        _check_type(section, "window_gap_seconds", settings["window_gap_seconds"], int)
        v = settings["window_gap_seconds"]
        if not 60 <= v <= 86_400:
            raise ValueError(
                f"[{section}] window_gap_seconds must be in [60, 86400], got {v}"
            )
    if "window_overlap_messages" in settings:
        _check_type(
            section,
            "window_overlap_messages",
            settings["window_overlap_messages"],
            int,
        )
        v = settings["window_overlap_messages"]
        if not 0 <= v <= 10:
            raise ValueError(
                f"[{section}] window_overlap_messages must be in [0, 10], got {v}"
            )
    if "users_refresh_interval_seconds" in settings:
        _check_type(
            section,
            "users_refresh_interval_seconds",
            settings["users_refresh_interval_seconds"],
            int,
        )
        v = settings["users_refresh_interval_seconds"]
        if not 60 <= v <= 86_400:
            raise ValueError(
                f"[{section}] users_refresh_interval_seconds must be in "
                f"[60, 86400], got {v}"
            )
    if "download_files" in settings:
        _check_type(section, "download_files", settings["download_files"], bool)
    if "download_attachments" in settings:
        _check_type(
            section, "download_attachments", settings["download_attachments"], bool
        )
    _validate_attachment_settings(section, settings)

    # Legacy alias: [sources.slack].download_files predates the unified
    # attachment knobs.  When both keys are present, download_attachments
    # wins and we warn; when only the legacy key is present, we silently
    # promote it.
    if "download_attachments" in settings and "download_files" in settings:
        logger.warning(
            "[%s] both 'download_attachments' and legacy 'download_files' set; "
            "download_attachments wins",
            section,
        )
        download_attachments = settings["download_attachments"]
    elif "download_attachments" in settings:
        download_attachments = settings["download_attachments"]
    elif "download_files" in settings:
        download_attachments = settings["download_files"]
    else:
        download_attachments = defaults.download_attachments

    cfg = SlackSourceConfig(
        enabled=settings.get("enabled", defaults.enabled),
        client_secrets_path=settings.get(
            "client_secrets_path", defaults.client_secrets_path
        ),
        poll_interval_seconds=settings.get(
            "poll_interval_seconds", defaults.poll_interval_seconds
        ),
        max_initial_days=settings.get("max_initial_days", defaults.max_initial_days),
        include_channels=list(
            settings.get("include_channels", defaults.include_channels)
        ),
        exclude_channels=list(
            settings.get("exclude_channels", defaults.exclude_channels)
        ),
        include_dms=settings.get("include_dms", defaults.include_dms),
        include_archived=settings.get("include_archived", defaults.include_archived),
        window_max_tokens=settings.get("window_max_tokens", defaults.window_max_tokens),
        window_gap_seconds=settings.get(
            "window_gap_seconds", defaults.window_gap_seconds
        ),
        users_refresh_interval_seconds=settings.get(
            "users_refresh_interval_seconds",
            defaults.users_refresh_interval_seconds,
        ),
        window_overlap_messages=settings.get(
            "window_overlap_messages", defaults.window_overlap_messages
        ),
        download_attachments=download_attachments,
        attachment_indexable_mimetypes=list(
            settings.get(
                "attachment_indexable_mimetypes",
                defaults.attachment_indexable_mimetypes,
            )
        ),
        attachment_max_size_mb=settings.get(
            "attachment_max_size_mb", defaults.attachment_max_size_mb
        ),
        attachment_pdf_max_pages=settings.get(
            "attachment_pdf_max_pages", defaults.attachment_pdf_max_pages
        ),
        attachment_pdf_per_page_chars=settings.get(
            "attachment_pdf_per_page_chars",
            defaults.attachment_pdf_per_page_chars,
        ),
        attachment_pdf_timeout_seconds=settings.get(
            "attachment_pdf_timeout_seconds",
            defaults.attachment_pdf_timeout_seconds,
        ),
    )

    # If enabled, the client_secrets file must exist on disk.
    if cfg.enabled:
        expanded = Path(cfg.client_secrets_path).expanduser()
        if not expanded.is_file():
            raise ValueError(
                f"[{section}] enabled=true but client_secrets_path does not exist: "
                f"{expanded} (expected JSON file with Slack client_id + client_secret)"
            )

    return cfg


# Leaf config keys recognised under the legacy single-account shape.
# Used by ``_detect_old_multiaccount_shape`` to distinguish "user wrote
# the old shape" from "user wrote a sub-table whose name we don't know".
_GMAIL_LEAF_KEYS: frozenset[str] = frozenset(
    {
        "enabled",
        "client_secrets_path",
        "poll_interval_seconds",
        "max_initial_threads",
        "label_filter",
        "download_attachments",
        "attachment_indexable_mimetypes",
        "attachment_max_size_mb",
        "attachment_pdf_max_pages",
        "attachment_pdf_per_page_chars",
        "attachment_pdf_timeout_seconds",
    }
)
_CALENDAR_LEAF_KEYS: frozenset[str] = frozenset(
    {
        "enabled",
        "client_secrets_path",
        "poll_interval_seconds",
        "max_initial_days",
        "calendar_ids",
        "download_attachments",
        "attachment_indexable_mimetypes",
        "attachment_max_size_mb",
        "attachment_pdf_max_pages",
        "attachment_pdf_per_page_chars",
        "attachment_pdf_timeout_seconds",
    }
)
_SOURCE_LEAF_KEYS: dict[str, frozenset[str]] = {
    "gmail": _GMAIL_LEAF_KEYS,
    "google_calendar": _CALENDAR_LEAF_KEYS,
}


def _detect_old_multiaccount_shape(source: str, settings: dict[str, Any]) -> None:
    """Reject the legacy single-table ``[sources.<source>]`` shape.

    The new multi-account schema places every account under a sub-table
    (``[sources.<source>.<account>]``). The legacy shape is only flagged
    when a *known leaf config key* (e.g. ``client_secrets_path``) appears
    as a direct child with a non-dict value. Empty tables are tolerated
    (with a warning) and account sub-tables are ignored.
    """
    leaf_keys = _SOURCE_LEAF_KEYS.get(source, frozenset())
    offending = sorted(
        k for k, v in settings.items() if k in leaf_keys and not isinstance(v, dict)
    )
    if not offending:
        if not settings:
            logger.warning(
                "[sources.%s] is empty; treating as if absent. Add accounts "
                "as [sources.%s.<account>] sub-tables or remove the empty "
                "section header.",
                source,
                source,
            )
        return

    has_account_subtables = any(isinstance(v, dict) for v in settings.values())
    if has_account_subtables:
        raise MigrationRequiredError(
            f"[sources.{source}] is ambiguous: the section mixes account "
            f"sub-tables with legacy top-level keys "
            f"({', '.join(offending)}). Move those keys into a "
            f"[sources.{source}.<account>] sub-table (or remove them) "
            f"before running."
        )
    raise MigrationRequiredError(
        f"[sources.{source}] uses the legacy single-account shape "
        f"(found leaf keys: {', '.join(offending)}). "
        f"Multi-account schema is required. Run "
        f"`fieldnotes migrate gmail-multiaccount` to clean old data, "
        f"then update your config to [sources.{source}.<account>] "
        f"form. See README."
    )


def _validate_account_name(source: str, account: str) -> None:
    """Account names must match ``^[a-z][a-z0-9_-]{0,30}$`` and not be reserved."""
    if not _ACCOUNT_NAME_RE.match(account):
        raise ValueError(
            f"[sources.{source}] account name {account!r} is invalid: must "
            f"match ^[a-z][a-z0-9_-]{{0,30}}$"
        )
    if account in _RESERVED_ACCOUNT_NAMES:
        raise ValueError(
            f"[sources.{source}] account name {account!r} is reserved: "
            f"names matching a source word ({', '.join(sorted(_RESERVED_ACCOUNT_NAMES))}) "
            f"would produce cursor keys that collide with the legacy "
            f"single-account migration state. Use a distinct name "
            f"(e.g. {account!r} → {account!r}-work or my{account})."
        )


def _parse_gmail_account(account: str, settings: dict[str, Any]) -> GmailAccountConfig:
    """Validate and parse a single ``[sources.gmail.<account>]`` section."""
    section = f"sources.gmail.{account}"
    defaults = GmailAccountConfig(name=account)

    if "enabled" in settings:
        _check_type(section, "enabled", settings["enabled"], bool)
    if "client_secrets_path" in settings:
        _check_type(
            section, "client_secrets_path", settings["client_secrets_path"], str
        )
    if "poll_interval_seconds" in settings:
        _check_type(
            section, "poll_interval_seconds", settings["poll_interval_seconds"], int
        )
    if "max_initial_threads" in settings:
        _check_type(
            section, "max_initial_threads", settings["max_initial_threads"], int
        )
    if "label_filter" in settings:
        _check_type(section, "label_filter", settings["label_filter"], str)
    if "download_attachments" in settings:
        _check_type(
            section, "download_attachments", settings["download_attachments"], bool
        )
    _validate_attachment_settings(section, settings)

    enabled = settings.get("enabled", True)
    # An account section must define a credentials path when enabled.
    # Bare ``[sources.gmail.foo]`` with no fields is invalid.
    if enabled and "client_secrets_path" not in settings:
        raise ValueError(
            f"[{section}] must define client_secrets_path "
            f"(account section cannot be empty when enabled)"
        )

    return GmailAccountConfig(
        name=account,
        enabled=enabled,
        client_secrets_path=settings.get(
            "client_secrets_path", defaults.client_secrets_path
        ),
        poll_interval_seconds=settings.get(
            "poll_interval_seconds", defaults.poll_interval_seconds
        ),
        max_initial_threads=settings.get(
            "max_initial_threads", defaults.max_initial_threads
        ),
        label_filter=settings.get("label_filter", defaults.label_filter),
        download_attachments=settings.get(
            "download_attachments", defaults.download_attachments
        ),
        attachment_indexable_mimetypes=list(
            settings.get(
                "attachment_indexable_mimetypes",
                defaults.attachment_indexable_mimetypes,
            )
        ),
        attachment_max_size_mb=settings.get(
            "attachment_max_size_mb", defaults.attachment_max_size_mb
        ),
        attachment_pdf_max_pages=settings.get(
            "attachment_pdf_max_pages", defaults.attachment_pdf_max_pages
        ),
        attachment_pdf_per_page_chars=settings.get(
            "attachment_pdf_per_page_chars",
            defaults.attachment_pdf_per_page_chars,
        ),
        attachment_pdf_timeout_seconds=settings.get(
            "attachment_pdf_timeout_seconds",
            defaults.attachment_pdf_timeout_seconds,
        ),
    )


def _parse_calendar_account(
    account: str, settings: dict[str, Any]
) -> CalendarAccountConfig:
    """Validate and parse a single ``[sources.google_calendar.<account>]``."""
    section = f"sources.google_calendar.{account}"
    defaults = CalendarAccountConfig(name=account)

    if "enabled" in settings:
        _check_type(section, "enabled", settings["enabled"], bool)
    if "client_secrets_path" in settings:
        _check_type(
            section, "client_secrets_path", settings["client_secrets_path"], str
        )
    if "poll_interval_seconds" in settings:
        _check_type(
            section, "poll_interval_seconds", settings["poll_interval_seconds"], int
        )
    if "max_initial_days" in settings:
        _check_type(section, "max_initial_days", settings["max_initial_days"], int)
    if "calendar_ids" in settings:
        _check_list_of(section, "calendar_ids", settings["calendar_ids"], str)
    if "download_attachments" in settings:
        _check_type(
            section, "download_attachments", settings["download_attachments"], bool
        )
    _validate_attachment_settings(section, settings)

    enabled = settings.get("enabled", True)
    if enabled and "client_secrets_path" not in settings:
        raise ValueError(
            f"[{section}] must define client_secrets_path "
            f"(account section cannot be empty when enabled)"
        )

    download_attachments = settings.get(
        "download_attachments", defaults.download_attachments
    )
    return CalendarAccountConfig(
        name=account,
        enabled=enabled,
        client_secrets_path=settings.get(
            "client_secrets_path", defaults.client_secrets_path
        ),
        poll_interval_seconds=settings.get(
            "poll_interval_seconds", defaults.poll_interval_seconds
        ),
        max_initial_days=settings.get("max_initial_days", defaults.max_initial_days),
        calendar_ids=list(settings.get("calendar_ids", defaults.calendar_ids)),
        download_attachments=download_attachments,
        attachment_indexable_mimetypes=list(
            settings.get(
                "attachment_indexable_mimetypes",
                defaults.attachment_indexable_mimetypes,
            )
        ),
        attachment_max_size_mb=settings.get(
            "attachment_max_size_mb", defaults.attachment_max_size_mb
        ),
        attachment_pdf_max_pages=settings.get(
            "attachment_pdf_max_pages", defaults.attachment_pdf_max_pages
        ),
        attachment_pdf_per_page_chars=settings.get(
            "attachment_pdf_per_page_chars",
            defaults.attachment_pdf_per_page_chars,
        ),
        attachment_pdf_timeout_seconds=settings.get(
            "attachment_pdf_timeout_seconds",
            defaults.attachment_pdf_timeout_seconds,
        ),
    )


def _parse_me_config(raw_me: dict[str, Any]) -> MeConfig:
    """Validate and parse the top-level ``[me]`` block."""
    from worker.parsers.base import canonicalize_email

    section = "me"
    if "emails" not in raw_me:
        raise ValueError(
            f"[{section}] emails: required (must be a non-empty list of strings)"
        )
    _check_list_of(section, "emails", raw_me["emails"], str)
    if not raw_me["emails"]:
        raise ValueError(f"[{section}] emails: must be a non-empty list of strings")
    if "name" in raw_me and raw_me["name"] is not None:
        _check_type(section, "name", raw_me["name"], str)

    return MeConfig(
        emails=[canonicalize_email(e) for e in raw_me["emails"]],
        name=raw_me.get("name"),
    )


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
        elif name == "slack":
            cfg.slack = _parse_slack_config(settings)
            continue  # parsed into cfg.slack; skip generic sources dict
        elif name == "gmail":
            _detect_old_multiaccount_shape("gmail", settings)
            for account, account_settings in settings.items():
                _validate_account_name("gmail", account)
                cfg.gmail[account] = _parse_gmail_account(
                    account, dict(account_settings)
                )
            continue  # parsed into cfg.gmail; skip generic sources dict
        elif name == "google_calendar":
            _detect_old_multiaccount_shape("google_calendar", settings)
            for account, account_settings in settings.items():
                _validate_account_name("google_calendar", account)
                cfg.google_calendar[account] = _parse_calendar_account(
                    account, dict(account_settings)
                )
            continue  # parsed into cfg.google_calendar; skip generic sources dict
        cfg.sources[name] = SourceConfig(name=name, settings=settings)

    # [me]
    if "me" in raw:
        cfg.me = _parse_me_config(dict(raw["me"]))

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

    # [reranker]
    if "reranker" in raw:
        rr = raw["reranker"]
        if "enabled" in rr:
            _check_type("reranker", "enabled", rr["enabled"], bool)
        for k in ("top_k_pre", "top_k_post", "batch_size"):
            if k in rr:
                _check_type("reranker", k, rr[k], int)
                if rr[k] < 1:
                    raise ValueError(f"[reranker] {k} must be >= 1, got {rr[k]}")
        if "score_threshold" in rr:
            if not isinstance(rr["score_threshold"], (int, float)):
                raise TypeError(
                    f"[reranker] score_threshold: expected float, "
                    f"got {type(rr['score_threshold']).__name__}"
                )
        if (
            "top_k_pre" in rr
            and "top_k_post" in rr
            and rr["top_k_post"] > rr["top_k_pre"]
        ):
            raise ValueError(
                f"[reranker] top_k_post ({rr['top_k_post']}) cannot exceed "
                f"top_k_pre ({rr['top_k_pre']})"
            )
        cfg.reranker = RerankerConfig(
            enabled=rr.get("enabled", cfg.reranker.enabled),
            top_k_pre=rr.get("top_k_pre", cfg.reranker.top_k_pre),
            top_k_post=rr.get("top_k_post", cfg.reranker.top_k_post),
            score_threshold=float(
                rr.get("score_threshold", cfg.reranker.score_threshold)
            ),
            batch_size=rr.get("batch_size", cfg.reranker.batch_size),
        )

    return cfg
