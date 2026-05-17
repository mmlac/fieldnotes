![Fieldnotes](./fieldnotes-banner.png)


# Fieldnotes

Fieldnotes is a personal knowledge graph that continuously indexes your digital life — local files, Obsidian vaults, email threads, git repositories, OmniFocus tasks, and installed applications — and exposes that knowledge as structured context for LLM agents. It combines a property graph (Neo4j) for relationship traversal with a vector store (Qdrant) for semantic retrieval, connected by a hybrid query layer and served over the Model Context Protocol (MCP) so any compatible AI assistant can query everything you know.

## Table of Contents

- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [Attachments](#attachments)
- [Migrating from Single-Account](#migrating-from-single-account)
- [Data Sources](#data-sources)
  - [Cross-Source Tag Unification](#cross-source-tag-unification-omnifocus--obsidian)
  - [Cross-Source Person Linking](#cross-source-person-linking-gmail--google-calendar--obsidian--slack)
  - [Person Extraction from Text and Tasks](#person-extraction-from-text-and-tasks)
  - [Entity Resolution Pipeline](#entity-resolution-pipeline)
- [Interacting With Your Data](#interacting-with-your-data)
- [CLI Reference](#cli-reference)
- [Backup & Restore](#backup--restore)
- [MCP Server](#mcp-server)
- [Pipeline Architecture](#pipeline-architecture)
- [Observability](#observability)
- [Local Development](#local-development)
- [Project Structure](#project-structure)

## How It Works

Fieldnotes watches your configured sources for changes in real time. When a file is saved, an email arrives, or a commit is pushed, the pipeline picks it up and runs it through a sequence of stages:

```
Source Event (file / email / commit / task / app)
        │
        ▼
   ┌─────────┐
   │  Parser  │  → extracts text, metadata, and graph hints
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │  Chunker  │  → splits text into ~512-token windows (64-token overlap)
   └────┬──────┘
        │
   ┌────┴─────────────────────┐
   │                          │
   ▼                          ▼
┌──────────┐           ┌───────────┐
│ Embedder │           │ Extractor │  → LLM-based entity and triple extraction
└────┬─────┘           └─────┬─────┘
     │                       │
     │                  ┌────▼─────┐
     │                  │ Resolver │  → deduplicates entities (fuzzy matching)
     │                  └────┬─────┘
     │                       │
     └───────┬───────────────┘
             ▼
        ┌─────────┐
        │  Writer  │  → persists to Neo4j (graph) + Qdrant (vectors)
        └─────────┘
```

Images follow a parallel path through a vision model that extracts descriptions, OCR text, and entities before rejoining the main pipeline at the embedding stage.

Files that can't be parsed for content (e.g. `.3mf`, `.psd`, `.mp4`) are still indexed as metadata-only records — the filename, extension, path, and a human-readable description are embedded so they're discoverable via semantic search. You can also force this behavior for specific files or directories using `index_only_patterns` in any source config — matching files are indexed by filename only, without reading their content.

Topic discovery runs on a schedule (default: weekly) or on demand via `fieldnotes cluster`, using UMAP dimensionality reduction and HDBSCAN clustering over the full vector corpus, with an LLM naming each discovered cluster.

## Requirements

| Dependency | Version | Purpose | Notes |
|------------|---------|---------|-------|
| **Python** | 3.11+ | Worker runtime (pipeline, CLI, MCP server) | macOS ships 3.x; also via Homebrew, pyenv, or system package manager |
| **Docker & Docker Compose** | — | Infrastructure services | Runs Neo4j, Qdrant, Prometheus, Grafana, Pushgateway |
| **Git** | — | Repository source scanning | Needed if `[sources.repositories]` is enabled (default) |
| **Ollama** | — | Default local LLM provider | Not required if using OpenAI or Anthropic API keys instead. Install from [ollama.ai](https://ollama.ai) |
| **libheif** | 1.17+ | HEIF/HEIC image support | Required by `pillow-heif` for processing Apple photos. See install instructions below |

### Installing libheif

`pillow-heif` builds a native C extension against `libheif`. You must install the library **before** `pip install`:

**macOS (Homebrew):**
```bash
brew install libheif
```

> **Note:** `pillow-heif` versions 0.18–1.0 fail to build against system libheif
> 1.20+ due to a header reorganization (`heif_camera_intrinsic_matrix` moved to
> `heif_properties.h`). Fieldnotes requires `pillow-heif>=1.1` which includes the fix.

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install libheif-dev
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install libheif-devel
```

### Services managed by Docker Compose

These are started automatically by `fieldnotes init --with-docker` or `docker compose up -d`:

| Service | Image | Default Port | Purpose |
|---------|-------|--------------|---------|
| Neo4j | `neo4j:2026.02.3-community` | 7687 | Property graph (entities, relationships, triples) |
| Qdrant | `qdrant/qdrant:v1.17.0` | 6333 | Vector database (semantic search) |
| Prometheus | `prom/prometheus:v3.10.0` | 9090 | Metrics collection |
| Pushgateway | `prom/pushgateway:v1.11.2` | 9091 | Metrics aggregation for batch jobs |
| Grafana | `grafana/grafana-oss:12.4.1` | 3000 | Observability dashboards |

## Installation

```bash

# With pipx
pipx install fieldnotes

# Also available but not recommended:

# With pip
pip install fieldnotes

# With uv
uv tool install fieldnotes



# From source
git clone https://github.com/mmlac/fieldnotes.git
cd fieldnotes
pip install -e ".[dev]"
```

## Quick Start

```bash
# 1. Bootstrap configuration and start infrastructure
#    Option A — fully automatic (recommended):
fieldnotes init --with-docker
# Runs the interactive wizard, extracts docker-compose.yml +
# Grafana/Prometheus configs to ~/.fieldnotes/infrastructure/,
# generates .env with passwords, and starts Docker containers.

#    Option A — with a custom compose file:
fieldnotes init --compose-file /path/to/docker-compose.yml

#    Option B — manual Docker setup:
fieldnotes init                     # interactive wizard only
export NEO4J_PASSWORD=changeme      # must match config.toml
export GRAFANA_PASSWORD=changeme
docker compose -f ~/.fieldnotes/infrastructure/docker-compose.yml up -d

# 2. Wait for services to be healthy
docker compose -f ~/.fieldnotes/infrastructure/docker-compose.yml ps

# 3. Pull the default models (if using Ollama)
ollama pull nomic-embed-text   # embedding model
ollama pull llama3.2           # chat model (extraction, queries, completions)

# 4. Verify everything is ready
fieldnotes doctor

# 5. Start the daemon (pipeline + MCP server)
fieldnotes serve --daemon

# 6. Search your knowledge graph
fieldnotes search "what do I know about kubernetes"

# 7. Ask a question (RAG + LLM synthesis)
fieldnotes ask "summarize my recent project decisions"
```

## Configuration

Fieldnotes reads `~/.fieldnotes/config.toml` (override with `fieldnotes -c /path/to/config.toml`). Run `fieldnotes init` to generate a default config pre-configured with an Ollama provider, embedding + chat model bindings, and all role assignments. If `NEO4J_PASSWORD` is set in your environment when you run `init`, it will be injected into the config automatically.

### Core

```toml
[core]
data_dir = "~/.fieldnotes/data"   # persistent storage for Docker volumes
log_level = "info"                # debug | info | warning | error
```

### Databases

```toml
[neo4j]
uri = "bolt://localhost:7687"
user = "neo4j"                    # or NEO4J_USER env var
password = ""                     # or NEO4J_PASSWORD env var (required)

[qdrant]
host = "localhost"
port = 6333
collection = "fieldnotes"
vector_size = 768                 # must match your embedding model
```

> **Note:** If `vector_size` doesn't match your embedding model's output dimensions, fieldnotes will log a warning at startup. The default `768` matches `nomic-embed-text`.

### Model Providers

Fieldnotes uses a three-layer model config: **providers** register API connections, **models** name specific model+provider pairs, and **roles** bind pipeline stages to models.

```toml
# Layer 1: Provider connections
[modelproviders.ollama]
type = "ollama"
base_url = "http://localhost:11434"

[modelproviders.openai]
type = "openai"
api_key = ""                      # or OPENAI_API_KEY env var

[modelproviders.anthropic]
type = "anthropic"
api_key = ""                      # or ANTHROPIC_API_KEY env var

# Layer 2: Named models
[models.local_embed]
provider = "ollama"
model = "nomic-embed-text"

[models.local_chat]
provider = "ollama"
model = "llama3.2"

# Layer 3: Role bindings (which model does what)
[models.roles]
embed = "local_embed"             # vector embeddings
extract = "local_chat"            # entity/triple extraction
extract_fallback = "local_chat"   # retry on malformed JSON
query = "local_chat"              # NL → Cypher translation
vision = "local_chat"             # image analysis
clustering = "local_chat"         # topic naming
completion = "local_chat"         # ask tool synthesis
rerank = "bge_reranker"           # second-stage cross-encoder (see [reranker])
```

#### Reranker

Fieldnotes supports an optional second-stage **cross-encoder reranker** that re-scores hybrid search candidates before they reach the LLM or the user. This is a significant precision improvement for `search` and `ask` — the cross-encoder can distinguish fine-grained relevance that bi-encoder embeddings miss.

The default model (`BAAI/bge-reranker-v2-m3`) runs locally via the `sentence_transformers` provider. It lazy-loads on first use (~1.5 GB RAM, no daemon-startup cost) and falls back gracefully to the original ranking if the model fails.

```toml
# Provider for local in-process models
[modelproviders.local]
type = "sentence_transformers"
device = "auto"                   # auto | cpu | cuda | mps
cache_dir = "~/.fieldnotes/data/models"

# Cross-encoder model
[models.bge_reranker]
provider = "local"
model = "BAAI/bge-reranker-v2-m3"

# Reranker tuning
[reranker]
enabled = true                    # set false to skip reranking entirely
top_k_pre = 50                    # candidates pulled from vector search
top_k_post = 10                   # results kept after reranking
score_threshold = 0.0             # drop candidates below this score
batch_size = 32                   # cross-encoder batch size
```

Only vector search results are reranked — graph results pass through as the precision lane. The reranker is wired into CLI search (`--no-rerank`, `--rerank-top-k`), interactive ask, and the MCP `search`/`ask` tools (`rerank` parameter).

### Sources

```toml
[sources.files]
watch_paths = ["~/Documents"]
include_extensions = [                   # optional filter — omit to allow all
  ".md", ".txt", ".pdf",                # documents
  ".json", ".yaml", ".yml", ".toml",    # structured data
  ".csv", ".html",                       # tabular / web
  ".pages", ".key",                      # Apple iWork (macOS only, needs app installed)
  ".png", ".jpg", ".jpeg", ".gif",       # images (vision pipeline)
  ".webp", ".bmp", ".tiff", ".heic",
]
exclude_patterns = ["node_modules/"]
index_only_patterns = ["*.iso", "*.dmg"] # index filename only, skip content
recursive = true
max_file_size = 104857600               # 100 MB

# Pattern matching: `exclude_patterns` and `index_only_patterns` are glob
# patterns. A pattern containing `/` (e.g. `Library/Uni`) matches contiguous
# path segments anywhere in the path — so `Library/Uni` matches
# `~/Documents/Library/Uni/syllabus.pdf` but not `~/Library/foo/Uni`.
# Single-segment patterns (e.g. `*.iso`, `node_modules/`) match the full path,
# basename, or any single path segment. Both patterns and paths are
# NFC-normalized before comparison, so accented characters (`Bücher`) match
# regardless of whether the on-disk filename is stored as NFC (Linux) or NFD
# (macOS HFS+/APFS).

[sources.obsidian]
vault_paths = ["~/obsidian-vault"]
# index_only_patterns = ["attachments/"] # index filename only, skip content

# Gmail and Google Calendar are configured per account. Each
# `[sources.gmail.<account>]` / `[sources.google_calendar.<account>]`
# section adds one mailbox or calendar. Account labels must match
# `^[a-z][a-z0-9_-]{0,30}$` (e.g. `personal`, `work`, `clientx`). All
# accounts can share a single `client_secrets_path` (one OAuth client
# in your Google Cloud project) — each account gets its own token file
# at `~/.fieldnotes/data/{gmail,calendar}_token-<account>.json`.

[sources.gmail.personal]
client_secrets_path = "~/.fieldnotes/credentials.json"
poll_interval_seconds = 300
max_initial_threads = 500
label_filter = "INBOX"
download_attachments = false           # opt-in: fetch + parse attachment bodies
# attachment_indexable_mimetypes = [...] # override the default allowlist (see Attachments)
# attachment_max_size_mb = 25           # inclusive upper bound; range [1..200]

[sources.gmail.work]
client_secrets_path = "~/.fieldnotes/credentials.json"
poll_interval_seconds = 300
max_initial_threads = 500
label_filter = "INBOX"

[sources.google_calendar.personal]
client_secrets_path = "~/.fieldnotes/credentials.json"
poll_interval_seconds = 300
max_initial_days = 90
calendar_ids = ["primary"]
download_attachments = false           # opt-in: also requires drive.readonly scope
# attachment_indexable_mimetypes = [...] # override the default allowlist (see Attachments)
# attachment_max_size_mb = 25           # inclusive upper bound; range [1..200]

[sources.google_calendar.work]
client_secrets_path = "~/.fieldnotes/credentials.json"
poll_interval_seconds = 300
max_initial_days = 90
calendar_ids = ["primary"]

[sources.slack]
enabled = false
# JSON file with Slack app client_id + client_secret. Required when enabled.
client_secrets_path = "~/.fieldnotes/slack_credentials.json"
poll_interval_seconds = 300
max_initial_days = 90                  # backfill window per conversation
include_channels = []                  # channel names or IDs; [] = all joined channels
exclude_channels = []                  # mutually exclusive with include_channels (warn)
include_dms = true                     # include 1:1 and multi-party DMs
include_archived = false
window_max_tokens = 512                # burst-window upper bound (token count) [128..4096]
window_gap_seconds = 1800              # quiet-gap that closes a window (30 min) [60..86400]
window_overlap_messages = 3            # whole-message overlap between windows [0..10]
download_attachments = false           # opt-in: fetch + parse uploaded files
# download_files = false               # legacy alias for download_attachments (deprecated)
# attachment_indexable_mimetypes = [...] # override the default allowlist (see Attachments)
# attachment_max_size_mb = 25           # inclusive upper bound; range [1..200]
```

> **Drive scope requirement (Calendar):** `download_attachments = true` on a `[sources.google_calendar.<account>]` section requires the `https://www.googleapis.com/auth/drive.readonly` OAuth scope. See [Google Calendar OAuth Setup](#google-calendar-oauth-setup) for the walkthrough — flipping the knob on for an account whose token was issued without Drive raises a `ReauthRequiredError` until you delete `calendar_token-<account>.json` and re-run the consent flow.

> **Slack key migration:** `download_files` predates the unified attachment knobs and is aliased to `download_attachments`. Existing configs continue to work; new configs should use `download_attachments`. If both keys are set, `download_attachments` wins and a warning is logged.

#### Gmail OAuth Setup

Gmail indexing requires a Google Cloud OAuth2 credential. This is a one-time setup:

1. **Create a Google Cloud project** at [console.cloud.google.com](https://console.cloud.google.com) (e.g., name it "FieldNotes").
2. **Enable the Gmail API**: Navigate to *APIs & Services → Library*, search for "Gmail API", and click *Enable*.
3. **Configure the OAuth consent screen**: Go to *APIs & Services → OAuth consent screen*. Choose **External** as user type. Fill in the required app name and email fields. On the *Scopes* page, add `gmail.readonly`. On the *Test users* page, **add your own Gmail address**. Leave the app in **Testing** mode — this is sufficient for personal use and avoids Google's app verification process.
4. **Create OAuth credentials**: Go to *APIs & Services → Credentials → Create Credentials → OAuth client ID*. Select **Desktop application** as the application type.
5. **Download the credentials**: Click *Download JSON* and save it as `~/.fieldnotes/credentials.json` (or the path you set in `client_secrets_path`).
6. **First run**: When the daemon starts with Gmail enabled, it will open your browser for a one-time consent screen — once per configured account. Approve read-only access (`gmail.readonly` scope). The resulting token is saved to `~/.fieldnotes/data/gmail_token-<account>.json` (mode `0600`) and refreshed automatically from then on.

> **Note:** If you're running fieldnotes as a system service (headless), complete the first OAuth flow manually with `fieldnotes serve --daemon` in a terminal before installing the service. Each configured account triggers its own consent flow on first start; the saved tokens will be reused.

> **Multi-account tip:** The same `client_secrets_path` (one Google Cloud OAuth client) works for any number of accounts — when the consent screen opens for each `[sources.gmail.<account>]` section, sign in with that account's Google identity. Per-account tokens land at `gmail_token-<account>.json` so they don't collide.

> **Troubleshooting:** If the consent screen shows _"Access Blocked: \<AppName\> has not completed the Google verification process"_, the `credentials.json` file belongs to a different Google Cloud project that was published but never verified. Create your own project following the steps above — step 3 (consent screen) is the most commonly missed part.

#### Google Calendar OAuth Setup

Google Calendar uses the same OAuth credentials file and Google Cloud project as Gmail. If you already configured Gmail, you only need to enable the Calendar API:

1. **Enable the Google Calendar API**: In the same Google Cloud project, go to *APIs & Services → Library*, search for "Google Calendar API", and click *Enable*.
2. **Add the Calendar scope**: Go to *APIs & Services → OAuth consent screen → Edit App → Scopes* and add `calendar.events.readonly`. (If you set up Gmail and Calendar at the same time, add both scopes in one go.)
3. **First run**: When the daemon starts with Google Calendar enabled, it will open your browser for a one-time consent screen per configured account. Approve read-only access (`calendar.events.readonly` scope). The token is saved to `~/.fieldnotes/data/calendar_token-<account>.json` (mode `0600`) and refreshed automatically.
4. **Multiple calendars per account**: Set `calendar_ids` to a list of calendar IDs. Use `"primary"` for the account's main calendar. Other calendars can be found in Google Calendar settings under *Settings for other calendars → Integrate calendar → Calendar ID*.
5. **Multiple accounts**: Add additional `[sources.google_calendar.<account>]` sections — one per Google identity you want to index. Account labels must match `^[a-z][a-z0-9_-]{0,30}$`. See also the [`[me]`](#me-self-identity) block below to declare your own emails so the graph treats them as a single self-Person.

> **Tip:** The same `credentials.json` file works for both Gmail and Google Calendar across every account — they share one OAuth client. Each `(source, account)` pair has its own token file and consent flow.

##### Drive scope (Calendar attachments)

Calendar events frequently link to attachments stored on Google Drive (a `.pdf`
brief, a `.png` agenda, etc.). Fetching their bytes requires the `drive.readonly`
scope, which is **not** granted by default — the calendar adapter only requests
it when `download_attachments = true` on at least one calendar account. Adding
the scope to an existing install is a one-time setup:

1. **Enable the Google Drive API**: In the same Google Cloud project, go to
   *APIs & Services → Library*, search for "Google Drive API", and click *Enable*.
2. **Add the Drive scope to consent**: Go to *APIs & Services → OAuth consent
   screen → Edit App → Scopes* and add `https://www.googleapis.com/auth/drive.readonly`.
3. **Flip the per-account knob**: Set `download_attachments = true` under each
   `[sources.google_calendar.<account>]` section that should fetch attachments.
4. **Re-run the install for affected accounts**: Delete
   `~/.fieldnotes/data/calendar_token-<account>.json` for each calendar account
   so the next daemon start re-runs the OAuth consent flow with the expanded
   scope set. Tokens issued before the scope was added are rejected with a
   `ReauthRequiredError` to fail loud rather than silently fall back to
   metadata-only.

> **Read-only:** The `drive.readonly` scope only lets fieldnotes *read* attachments
> linked from calendar events — it does not list, modify, or upload anything in your
> Drive. Bytes are streamed in-memory, parsed, and discarded (see
> [Attachments](#attachments)).

#### `[me]` Self-Identity

Declare your own email addresses in a top-level `[me]` block so the graph treats them as a single self-Person:

```toml
[me]
emails = ["me@personal.com", "me@work.com"]
name = "Your Name"  # optional — falls back to the longest existing display name
```

**Why this matters:** Email is the canonical merge key for Person nodes (see [Cross-Source Person Linking](#cross-source-person-linking-gmail--google-calendar--obsidian--slack)), so two of *your* mailboxes appear as two different Persons by default. The `[me]` block tells the [`reconcile_self_person()`](docs/similarity.md) reconciliation step to merge them: it `MERGE`s a Person on each canonicalized email, sets `is_self = true`, and creates `SAME_AS` edges across all of them (`match_type = 'self_identity'`, `confidence = 1.0`).

**Usage notes:**
- `emails` is required and must be a non-empty list. Addresses are canonicalized (lowercased, trimmed, `@googlemail.com` → `@gmail.com`) so capitalization and Gmail aliases don't matter.
- `name` is optional. When set it overrides the survivor display name; otherwise the longest existing `name` among matched Persons wins.
- The OAuth client (your Google Cloud project) and `client_secrets_path` can be reused across all `[sources.gmail.<account>]` and `[sources.google_calendar.<account>]` sections — no additional setup is required to plug `[me]` in.
- A single email is a no-op for `SAME_AS` edges; the lone Person is still flagged `is_self`.

After it runs, `MATCH (p:Person {is_self: true})` resolves directly to your self-Persons, and `SAME_AS` traversals from any of them reach the rest of your alias cluster.

#### Slack OAuth Setup

Slack indexing requires a Slack app you create in your own workspace. This is a one-time setup:

1. **Create a Slack app**: Go to [api.slack.com/apps](https://api.slack.com/apps) and click *Create New App*.
2. **Choose "From scratch"**: Pick a name (e.g., "Fieldnotes") and select your workspace.
3. **Add OAuth scopes**: On *OAuth & Permissions*, under *Bot Token Scopes*, add:
   - `channels:history`
   - `channels:read`
   - `groups:history`
   - `groups:read`
   - `im:history`
   - `im:read`
   - `mpim:history`
   - `mpim:read`
   - `users:read`
   - `users:read.email`
4. **Set the redirect URL**: On *OAuth & Permissions → Redirect URLs*, add `http://localhost:3000/oauth/callback` — the listener fieldnotes runs locally during the install flow.
5. **Install to workspace**: Still on *OAuth & Permissions*, click *Install to Workspace* and approve. Copy the *Client ID* and *Client Secret* from *Basic Information → App Credentials*.
6. **Save credentials**: Write a JSON file at `~/.fieldnotes/slack_credentials.json` containing the client id and secret:

   ```json
   {
     "client_id": "1234567890.1234567890",
     "client_secret": "abcdef0123456789abcdef0123456789"
   }
   ```

7. **First daemon run**: When the daemon starts with Slack enabled, it opens your browser to complete the OAuth flow. Approve the requested scopes. The resulting bot token is saved to `~/.fieldnotes/data/slack_token.json` (mode `0600`) and validated on subsequent runs.

> **Troubleshooting:** A `missing_scope` error from any Slack API call means the app needs additional scopes. Add them to the app's OAuth & Permissions page, then **re-install the app to the workspace** — newly added scopes are not granted to existing tokens. Delete `~/.fieldnotes/data/slack_token.json` and the daemon will re-run the OAuth flow on next start.

```toml
[sources.repositories]
repo_roots = ["~/projects"]
include_patterns = ["README*", "CHANGELOG*", "CONTRIBUTING*", "docs/**/*.md", "*.toml", "ADR/**/*.md"]
exclude_patterns = ["node_modules/", ".git/", "vendor/", "target/", "__pycache__/"]
index_only_patterns = ["*.lock"]         # index filename only, skip content
poll_interval_seconds = 300
max_file_size = 104857600
```

OmniFocus support is auto-detected on macOS and indexes tasks, projects, tags, and parent-child relationships via JXA (JavaScript for Automation):

```toml
[sources.omnifocus]
# enabled = true             # auto-detected on macOS
# poll_interval_seconds = 300
# state_path = "~/.fieldnotes/state/omnifocus.json"
```

macOS apps and Homebrew sources auto-discover installed software and rescan on a configurable interval:

```toml
[sources.macos_apps]
poll_interval_seconds = 21600   # default: 6 hours

[sources.homebrew]
poll_interval_seconds = 21600   # default: 6 hours
```

### Attachments

Gmail messages, Slack messages, and Google Calendar events can carry file
attachments — PDFs, screenshots, plain-text logs, JSON dumps, the occasional
`.docx` brief. Fieldnotes treats every attachment as a first-class document
that's reachable from the parent message or event, but it indexes the *body*
of each attachment only when it knows how to parse the format.

#### Two-tier indexing

Two independent decisions per attachment, made at fetch time by
[`classify_attachment`](worker/worker/parsers/attachments.py):

1. **Indexable (`download_and_index`)** — MIME type is on the per-source
   `attachment_indexable_mimetypes` allowlist *and* size is at or below
   `attachment_max_size_mb`. The bytes are downloaded in-memory, handed to the
   matching parser (PDF → `pymupdf`, image → vision pipeline, text/JSON/YAML/CSV
   → text loader), and the extracted content is chunked and embedded alongside
   the parent.
2. **Metadata-only** — anything else (unsupported MIME, oversize file, or fetch
   disabled). The Attachment Document records filename, MIME, size, and a
   parent link, but the body is never downloaded.

The default allowlist mirrors the parsers we ship today:

| Category   | MIME types |
|------------|------------|
| Documents  | `application/pdf`, `text/plain`, `text/markdown`, `text/csv` |
| Structured | `application/json`, `application/yaml`, `application/x-yaml` |
| Images     | `image/png`, `image/jpeg`, `image/gif`, `image/webp`, `image/heic`, `image/heif`, `image/tiff`, `image/bmp` |

Override per source by setting `attachment_indexable_mimetypes` on the section.
Adding a MIME type the worker can't parse just wastes bandwidth — keep the
allowlist narrow.

#### Stream-and-forget

Attachments are fetched in-memory, parsed, then discarded. **No on-disk cache.**
The downloader (see [`stream_and_parse`](worker/worker/parsers/attachments.py))
takes a closure that returns the raw bytes, hands them to the parser, and
releases them as soon as the parsed result is in hand. The parent message or
event link is stored on the Attachment Document so search results can navigate
back to the source: `mailto:`-style Gmail thread URL, Slack message permalink,
or the Calendar event's `htmlLink`.

> **`parent_url` scope:** Gmail links open the **thread** that contains the
> message (not a deep-link to the specific message — Gmail's per-message
> permalinks are unstable as threads grow); Slack links open the **message
> permalink**; Calendar links open the **event detail page** (`htmlLink`).

#### Filename in body

Before chunking, the parser weaves attachment filenames into the parent
message or event text under an `Attachments:` section. That means semantic
queries like *"the AWS architecture diagram alice sent"* hit the parent email
even when the attached `.pptx` is metadata-only and never gets its own content
chunks. The filename is part of the parent's embedding, not a separate index.

#### Three-layer retrieval

Each attached message/event ends up represented at three layers in the index:

1. **Parent Document** — message/event chunks with the `Attachments:` section
   stitched into the body, so filename matches surface the parent.
2. **Attachment Document** — one per file, carrying filename + MIME + parent
   URL. Always written, regardless of indexable status.
3. **Attachment-content chunks** — only for `download_and_index` files. The
   parsed text (PDF text, OCR + vision description for images, raw text for
   text/JSON/YAML/CSV) is chunked, embedded, and linked back to the
   Attachment Document.

This three-layer structure means *"what did the spec say about quotas?"*
matches the PDF body when it's indexable, *"the AWS diagram alice sent"* still
matches the parent email even when the attachment is metadata-only, and
*"that quarterly_review.docx Bob shared"* finds the parent message via the
filename — even though the Word body is currently unparseable.

#### Counting attachments: intent vs. outcome

Each parent Document (Email, SlackMessage, CalendarEvent) carries three
counters that distinguish how many attachments were *seen* from how many
ended up with extractable text:

| Property                          | Meaning |
|-----------------------------------|---------|
| `attachments_count_intended`      | Total attachments discovered on the parent (post-dedupe). |
| `attachments_count_indexed`       | Successfully fetched **and** parsed — the Attachment Document carries body text and content chunks. |
| `attachments_count_metadata_only` | Skipped (non-indexable MIME / oversize / fetch disabled) **or** fell back after a fetch/parse error — the Attachment Document is metadata-only. |

`intended = indexed + metadata_only` always holds. Use
`attachments_count_indexed` when ranking by "messages with parseable
attachment text"; use `attachments_count_intended` for diagnostic queries
("how many emails arrived with attachments at all?").

> **Deprecated alias.** A legacy `has_attachments` property is still
> emitted as a copy of `attachments_count_intended` for one release so
> existing Cypher queries keep working. New code should use the explicit
> counters; `has_attachments` will be removed in a future cleanup.

#### Office formats — deferred

Microsoft Office formats (`.docx`, `.xlsx`, `.pptx`) are intentionally **not**
in the default allowlist: we don't ship parsers for them yet. They land as
metadata-only Attachment Documents (filename + parent link only) so they
remain discoverable via the filename-in-body path. A follow-up epic will add
real parsers; once those land, you'll be able to opt in by appending the MIME
types (`application/vnd.openxmlformats-officedocument.wordprocessingml.document`
and friends) to `attachment_indexable_mimetypes` on the relevant source.

> **Doctor:** `fieldnotes doctor` reports per-source attachment status (ON/OFF,
> Drive scope present for Calendar) and a 24h failure counter, so you can see
> at a glance whether attachment fetches are working.

### Features

```toml
[clustering]
enabled = true
cron = "0 3 * * 0"               # Sunday 3 AM
min_corpus_size = 100            # skip if fewer vectors
min_interval_seconds = 60.0      # minimum gap between runs (10–86400)
max_vectors = 500000             # max vectors per run (1–10000000)

[vision]
enabled = true
concurrency = 2                  # parallel vision processing tasks
min_file_size_kb = 1             # skip files smaller than this
max_file_size_mb = 20            # skip files larger than this
queue_size = 256                 # vision processing queue depth
skip_patterns = ["icon", "avatar", "favicon", "logo", "badge", "emoji", "thumb"]
                                 # regex patterns — matching files are skipped

[mcp]
enabled = true
port = 3456
auth_token = ""                  # or FIELDNOTES_MCP_AUTH_TOKEN env var (optional)

[health]
enabled = false                  # enable HTTP health check endpoint
port = 9100
bind = "127.0.0.1"

[rate_limits]
requests_per_minute = 0          # per-provider rate limit (0 = unlimited)
daily_token_budget = 0           # total tokens/day across all LLM calls (0 = unlimited)
max_concurrency = 0              # max parallel LLM calls (0 = unlimited)

[reranker]
enabled = true                   # second-stage cross-encoder (see Reranker section above)
top_k_pre = 50                   # candidates from vector search
top_k_post = 10                  # results kept after reranking
score_threshold = 0.0            # drop candidates below this score
batch_size = 32                  # cross-encoder batch size
```

### Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | *required* |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `FIELDNOTES_DATA` | Docker volume root | `~/.fieldnotes/data` |
| `FIELDNOTES_MCP_AUTH_TOKEN` | MCP server auth token | — |
| `GRAFANA_PASSWORD` | Grafana admin password | *required* |

## Migrating from Single-Account

If you upgraded from a release that used the flat `[sources.gmail]` / `[sources.google_calendar]` schema, the daemon will refuse to start until you migrate. The `fieldnotes migrate gmail-multiaccount` command is a one-shot retag that does everything in place.

**1. Stop the daemon:**

```bash
fieldnotes service stop
```

**2. Run the migrate command:**

```bash
# Interactive (prompts for an account label and confirmation):
fieldnotes migrate gmail-multiaccount

# Non-interactive (explicit label, no confirmation):
fieldnotes migrate gmail-multiaccount --account personal --yes

# See what would change without writing anything:
fieldnotes migrate gmail-multiaccount --dry-run
```

The default label is `default` when you pass `--yes` without `--account`. Account labels must match `^[a-z][a-z0-9_-]{0,30}$`.

**3. Restart the daemon:**

```bash
fieldnotes service start
```

That's it. The migrate command rewrites everything you'd otherwise have to update by hand:

| What | Action |
|---|---|
| `~/.fieldnotes/data/queue.db` | `queue.source_id` and embedded payload `source_id` retagged to `gmail://<account>/...` and `google-calendar://<account>/...`; `cursors.key` rewritten to `gmail:<account>` / `calendar:<account>` |
| Neo4j | `Document.source_id`, derivative `Chunk.id`, and fallback `Person.source_id` updated in place |
| Qdrant | Chunk `payload.source_id` rewritten (vectors preserved in place) |
| Token files | `gmail_token.json` → `gmail_token-<account>.json`, same for `calendar_token.json` |
| Legacy cursor JSON | `gmail_cursor.json`, `calendar_cursor.json` deleted (cursors live in `queue.db` now) |
| `~/.fieldnotes/config.toml` | Legacy `[sources.gmail]` / `[sources.google_calendar]` sections rewritten under `[sources.gmail.<account>]` / `[sources.google_calendar.<account>]` |

**Auto-backup of `config.toml`:** The original config is copied to `~/.fieldnotes/config.toml.backup-<UTC-timestamp>` *before* the rewrite. To revert, replace `config.toml` with the backup and re-run `migrate` if you want a different account label.

**Cursors are preserved automatically.** Polling position is stored in `queue.db` (`cursors` table) keyed by `gmail:<account>` / `calendar:<account>`, so the daemon resumes exactly where the single-account version left off — no re-scan, no re-OAuth.

**Daemon-running guard:** Migrate refuses to run while the daemon is alive (`queue.db` is shared and concurrent writes race even under WAL). Stop the service first. The escape hatch is `--force-running`, but be warned: any in-flight queue items the daemon picks up *during* the migrate window keep their old-shape `source_id` and need a manual second `migrate` run to clean up.

**Interleaved-migration safety:** If new-shape data already exists for the chosen account label (e.g. you ran the migration once, then added some fresh data, then re-ran with the same label), migrate refuses to proceed and asks for manual review.

**After migrating:** Optionally add a [`[me]`](#me-self-identity) block to declare your own email aliases, and add additional `[sources.gmail.<account>]` / `[sources.google_calendar.<account>]` sections to bring further accounts online. Each new account triggers its own one-time OAuth consent flow on the next daemon start.

## Data Sources

| Source | Sync Mode | What It Indexes |
|---|---|---|
| **Files** | Initial scan + real-time (watchdog) | Markdown, text, and other configured file types |
| **Obsidian** | Initial scan + real-time (watchdog) | Notes with frontmatter, wikilinks, and #tags |
| **Gmail** | Backfill + polling (configurable interval) | Email threads — subjects, bodies, metadata |
| **Git Repositories** | Initial scan + polling (configurable interval) | READMEs, changelogs, docs, commit messages |
| **OmniFocus** | Polling (default: every 5 minutes) | Tasks, projects, tags, due dates, subtask hierarchy, and person mentions (macOS only) |
| **macOS Apps** | Polling (default: every 6 hours) | Installed application bundles (Info.plist) |
| **Homebrew** | Polling (default: every 6 hours) | Installed formulae and casks with descriptions |
| **Google Calendar** | Backfill + polling (configurable interval) | Events, attendees, organizers, locations |
| **Slack** | Backfill + polling (configurable interval) | Channel threads, DM/MPIM bursts, with attendees and mentions |

Each source emits `IngestEvent` dicts into the pipeline queue. Modified files trigger a delete-before-rewrite cycle that cleans stale graph data (edges, chunks, orphan entities) in a single Neo4j transaction before writing the updated version.

### Git Repositories

The Git Repositories source indexes documentation and commit history from local Git repos. Point `repo_roots` at one or more parent directories (e.g. `~/Code`) and the source discovers repositories one level deep — so `~/Code/projectX/.git` is found automatically, but `~/Code/org/projectX/.git` is not.

Each discovered repo is scanned for files matching `include_patterns` (default: READMEs, changelogs, docs, TOML configs, ADRs). Commit messages from the last `max_commits` (default: 200) are also ingested. The source re-polls on a configurable interval (`poll_interval_seconds`, default: 300).

### Cross-Source Tag Unification (OmniFocus + Obsidian)

Fieldnotes automatically merges tags across OmniFocus and Obsidian so that a single Tag node in the knowledge graph connects both your tasks and your notes. This means a query like "what do I know about ProjectX/Deploy" returns OmniFocus tasks *and* Obsidian notes in one result — no manual linking required.

**How it works:** OmniFocus uses hierarchical tags (e.g., `Work/Deploy`). When an Obsidian note has a `categories` frontmatter field (a list) and a filename, the Obsidian parser synthesizes a matching hierarchical tag for each category that targets the same graph node OmniFocus created. Both sources MERGE onto one `Tag` node keyed by `omnifocus-tag:{category}/{name}`.

**Setup:**

1. **In OmniFocus**, organize your tags hierarchically — child tags nested under a parent produce a `Parent/Child` path:

   ```
   Tags
   ├── Work
   │   ├── Deploy               ← resulting tag path: Work/Deploy
   │   └── CodeReview           ← resulting tag path: Work/CodeReview
   └── Personal
       └── Taxes                ← resulting tag path: Personal/Taxes
   ```

2. **In Obsidian**, add a `categories` field to your note's frontmatter. The note's filename stem becomes the second part of the hierarchy:

   ```yaml
   ---
   categories:
     - Work
   ---
   Notes about the deployment process...
   ```

   A file named `Deploy.md` with this frontmatter will be linked to the same `Work/Deploy` tag as your OmniFocus tasks. Multiple categories generate multiple tag links — a note in both `Work` and `Personal` merges onto both parent tags.

**What gets unified:**

| OmniFocus tag | Obsidian frontmatter | Merged Tag node |
|---|---|---|
| `Work/Deploy` | `categories: [Work]`, filename `Deploy.md` | `omnifocus-tag:Work/Deploy` |
| `Research/LLMs` | `categories: [Research]`, filename `LLMs.md` | `omnifocus-tag:Research/LLMs` |
| `Personal/Taxes` | `categories: [Personal]`, filename `Taxes.md` | `omnifocus-tag:Personal/Taxes` |

**Notes:**
- Regular Obsidian `#tags` and frontmatter `tags: [...]` still create their own Tag nodes (prefixed `tag:`) — this unification only applies to the `categories` frontmatter field.
- If an Obsidian note has `categories` but no filename that can serve as a name, no synthesized tag is emitted.
- Each `categories` value must match the OmniFocus parent tag name exactly (case-sensitive).
- The frontmatter key name is configurable via `categories_key` in `[sources.obsidian]` (default: `categories`).

### Cross-Source Person Linking (Gmail + Google Calendar + Obsidian + Slack)

When multiple people-aware sources are enabled, Fieldnotes automatically merges Person nodes across sources into a single graph identity based on email address. A user mentioned in Slack, an attendee at a meeting, the sender of an email, and a person mentioned in an Obsidian note all link to the same Person node — no manual linking required.

> **View the consolidated identity:** run [`fieldnotes person <email>`](#person-profile--what-do-i-know-about-this-person) to see every source rolled up against a single Person, including the `SAME_AS` cluster the reconcile chain built.

> **See it for a single day:** run [`fieldnotes itinerary`](#itinerary--whats-on-my-plate-today) to view today's meetings with each attendee's open tasks, semantically related notes, and the most recent email or Slack thread covering the room — all resolved through the same `SAME_AS` clusters.

**How it works:** The Gmail parser, Google Calendar parser, Obsidian parser, and Slack parser all emit `GraphHint` records with `object_merge_key="email"` and an `object_id` of the form `person:{email}`. The pipeline Writer `MERGE`s Person nodes by email, so a Slack message author, an attendee in a calendar event, a correspondent in a Gmail thread, and a contact in an Obsidian note are guaranteed to resolve to the same graph node. Periodic `reconcile_persons()` runs create `SAME_AS` edges when the same email appears across sources, keeping the longest display name.

**Slack-specific identity:** When a Slack user's profile email isn't visible (the workspace doesn't grant `users:read.email`, or the user has hidden it), the Slack parser falls back to keying the Person on `slack-user:{team_id}/{user_id}`. As soon as the email becomes available on a later message, a separate email-keyed Person is emitted that also carries `(slack_user_id, team_id)`. A dedicated `reconcile_persons_by_slack_user()` step links the two via `SAME_AS`. See [docs/similarity.md](docs/similarity.md) for the full reconcile chain.

**Email canonicalization:** All parsers normalise emails through a shared `canonicalize_email()` function that lower-cases, trims whitespace, and rewrites `@googlemail.com` → `@gmail.com` (Google treats these as the same mailbox). This prevents duplicate Person nodes for the same real-world identity.

**Linking your own multiple email addresses:** Email is the canonical merge key, so two of *your* mailboxes (e.g. `me@personal.com` and `me@work.com`) appear as two different Persons by default. Declare them in a top-level [`[me]` block](#me-self-identity) and a `reconcile_self_person()` step links them as a single self-Person with `is_self = true`. See the [`[me]`](#me-self-identity) section for details and `MATCH (p:Person {is_self: true})` queryability.

**Linking people from Obsidian notes:**

Add an `emails` field to your note's frontmatter — either as a comma-separated string or a YAML list:

```yaml
---
title: Alice Smith
emails: alice@example.com, alice@googlemail.com
---
Meeting notes about Alice...
```

or equivalently:

```yaml
---
title: Alice Smith
emails:
  - alice@example.com
  - alice@googlemail.com
---
```

Each email creates a `MENTIONS` edge from the File node to a Person node. Because `@googlemail.com` is canonicalised to `@gmail.com`, the Person node will automatically merge with any Gmail or Calendar activity from that address.

**Graph relationships created by the Calendar parser:**

| Predicate | Direction | Description |
|---|---|---|
| `ORGANIZED_BY` | CalendarEvent → Person | The event organizer |
| `ATTENDED_BY` | CalendarEvent → Person | Each attendee (excludes the organizer to avoid duplication) |
| `CREATED_BY` | CalendarEvent → Person | Only emitted when the creator is different from the organizer |

**What this enables:**
- "Who was in the meeting about X?" — traverses `ATTENDED_BY` edges from the CalendarEvent
- "Show me all interactions with alice@example.com" — returns emails, meetings, *and* Obsidian notes via a single Person node
- "What was discussed before and after the design review?" — links meeting context to email follow-ups and your notes through shared Person nodes
- "What do I know about Bob?" — finds the Person node by email and traverses all `SENT`, `TO`, `ATTENDED_BY`, and `MENTIONS` edges across every source

### Person Extraction from Text and Tasks

Beyond structured sources like Gmail and Google Calendar, Fieldnotes discovers person mentions in unstructured text and task metadata:

**Email extraction from document text:** During pipeline processing, all document text is scanned for email addresses using a regex before chunking. Each discovered email creates a `MENTIONS` edge from the document to a `Person` node keyed by that (canonicalized) email. This means a Markdown file or PDF that mentions `alice@example.com` in its body automatically links to the same Person node created by Gmail or Calendar — no frontmatter required.

**OmniFocus person extraction:** The OmniFocus parser extracts person mentions from tasks using three strategies:

| Strategy | Example | Confidence |
|---|---|---|
| Email addresses in task name or note | `alice@example.com` | 1.0 |
| @Mentions (uppercase-first names) | `@Alice`, `@Bob Smith` | 0.9 |
| Tags under `People/` hierarchy | `People/Alice` | 0.95 |

Lowercase @mentions like `@home` or `@work` are ignored (treated as context tags, not people). Extracted Person nodes merge with existing persons by email or name.

### Entity Resolution Pipeline

After each ingest batch, a multi-step reconciliation chain runs to unify person identities across all sources:

| Step | What It Does |
|---|---|
| **1. Email-based reconciliation** | Creates `SAME_AS` edges between Person nodes that share an email address across different sources. Keeps the longest display name. |
| **2. Slack-identity merge** | Groups Person nodes by `(team_id, slack_user_id)` and creates `SAME_AS` edges (`match_type='slack_user_id'`) — closes the no-email-fallback gap when a Slack profile email isn't visible. |
| **3. Fuzzy name matching** | Uses RapidFuzz `token_sort_ratio` to find Person nodes with near-identical names (threshold ≥ 95). Creates `SAME_AS` edges with `match_type='fuzzy_name'`. |
| **4. Entity→Person bridging** | Links `Entity` nodes of type Person (extracted by the LLM) to structured `Person` nodes (from Gmail, Calendar, etc.) using fuzzy matching (threshold ≥ 93). Creates `SAME_AS` edges with `match_type='entity_person_bridge'`. |
| **5. Cross-source entity resolution** | Deduplicates entities with the same label appearing across different sources using a 3-strategy cascade: exact match → fuzzy string match → embedding cosine similarity. |
| **6. Transitive SAME_AS closure** | If A↔B and B↔C have `SAME_AS` edges but A↔C does not, creates the missing A↔C edge (up to 4 hops). Ensures the full identity cluster is fully connected. |
| **7. Self-identity** *(only when [`[me]`](#me-self-identity) is configured)* | `MERGE`s a Person on each canonicalized email in `[me].emails`, sets `is_self=true`, and creates `SAME_AS` edges across all of them (`match_type='self_identity'`, `confidence=1.0`). Runs last so it can link your own aliases that the email-merge step left split. |

Each step is fault-tolerant — a failure in one step logs a warning and proceeds to the next. All `SAME_AS` edges carry metadata: `confidence`, `match_type`, and `cross_source` flag.

> **Details:** See [docs/similarity.md](docs/similarity.md) for the full entity resolution analysis including thresholds, match strategies, and architecture notes.

### Cross-Source Document Linking (REFERENCES edges)

When one document explicitly references another — an Obsidian note that embeds a `gmail://` link, a calendar event description that links to an Obsidian file, an email that pastes a Slack permalink — Fieldnotes creates a `REFERENCES` edge in the knowledge graph to capture that explicit connection.

**What it connects:** `REFERENCES` edges link any of the four main text-bearing node types (`CalendarEvent`, `Email`, `SlackMessage`, `File`) to any other indexed node. The subject is always the document containing the link; the object is the document being linked to.

**Predicate:** `REFERENCES` — carried on a directed edge `(subject)-[:REFERENCES]->(object)`.

**Merge keys:** Both subject and object nodes MERGE on `source_id`, so edges automatically bind to existing nodes as long as the target has already been indexed. Objects not yet in the graph are created as stub nodes and gain full content on the next ingest cycle.

**Confidence:** `1.0` — extracted from structured source-id URLs and Slack permalinks, not LLM inference.

**Sources of REFERENCES edges:**

| URL pattern in text | Object label | Example |
|---|---|---|
| `gmail://<account>/message/<id>` | `Email` | `gmail://work@gmail.com/message/abc123` |
| `google-calendar://<account>/event/<id>` | `CalendarEvent` | `google-calendar://work@gmail.com/event/xyz` |
| `omnifocus://task/<id>` (normalized to `omnifocus://<id>`) | `Task` | `omnifocus://task/A1B2C3` |
| `slack://<team>/<channel>/<ts>` | `SlackMessage` | `slack://T01/C01/1234567890.000100` |
| `obsidian://open?vault=<name>&file=<rel_path>` | `File` | `obsidian://open?vault=Personal&file=Meetings%2FKris.md` |
| `https://<workspace>.slack.com/archives/<channel>/p<ts>` | `SlackMessage` | Slack permalink from browser |

**Re-indexing existing data:** REFERENCES edges are only produced during fresh ingest. To backfill the existing corpus, run:

```bash
fieldnotes reindex-references [--dry-run] [--label LABEL]
```

See [`reindex-references`](#reindex-references-dry-run---label-label) in the CLI reference for details.

### Initial Scan and Cursor Persistence

On first startup, file and obsidian sources walk all configured directories and index every matching file. A SHA256-based cursor is saved to disk after the scan completes, recording the hash and mtime of every indexed file.

On subsequent startups, the cursor is loaded and diffed against the current filesystem state — only new, modified, or deleted files generate events. Unchanged files are skipped entirely.

| Source | Cursor location | What It Tracks |
|---|---|---|
| **Files** | `~/.fieldnotes/data/files_cursor.json` | Per-file SHA256 + mtime |
| **Obsidian** | `~/.fieldnotes/data/obsidian_cursor.json` | Per-file SHA256 + mtime |
| **Gmail** | `queue.db` (key `gmail:<account>`) | Gmail History API ID per account |
| **Git Repos** | `~/.fieldnotes/data/repo_cursor.json` | Per-repo HEAD commit SHA |
| **OmniFocus** | `~/.fieldnotes/state/omnifocus.json` | Per-task content hash |
| **Google Calendar** | `queue.db` (key `calendar:<account>`) | Per-calendar syncToken per account |
| **Slack** | `~/.fieldnotes/data/slack_cursor.json` | Per-conversation latest_ts |

Gmail and Google Calendar cursors live in the SQLite persistent queue (`~/.fieldnotes/data/queue.db`, `cursors` table). The legacy per-source JSON cursor files (`gmail_cursor.json`, `calendar_cursor.json`, and any per-account variants) are deprecated transitional state — the daemon does not read or write them under the multi-account schema, and `fieldnotes migrate gmail-multiaccount` deletes them. Other sources (Files, Obsidian, Git Repos, OmniFocus, Slack) still use their own JSON state files since they predate the SQLite persistent queue.

Cursors are checkpointed every 5 minutes during operation and saved on graceful shutdown, so a restart only re-processes files that changed since the last checkpoint.

To prevent duplicate events between the initial scan and the watchdog starting, a short dedup window (default 5 seconds) filters out watchdog events for files already processed during the scan. The watchdog starts only after the scan completes.

## Interacting With Your Data

Beyond search and Q&A, fieldnotes provides three tools for staying on top of your indexed knowledge.

### Timeline — "What was I working on?"

View a chronological feed of activity across all your sources:

```bash
# What happened in the last 24 hours (default)
fieldnotes timeline

# What was I doing last week?
fieldnotes timeline --since 7d

# Just OmniFocus tasks from Monday to Wednesday
fieldnotes timeline --since 2026-03-16 --until 2026-03-18 --source omnifocus
```

Output groups entries by day and shows the source type, title, and a text snippet:

```
Timeline: 2026-03-18 → 2026-03-19

  2026-03-19 14:32  [File]    Deploy.md (modified)
                              Notes about the deployment process...
  2026-03-19 13:15  [Task]    Review auth middleware (completed)
                              Check the token refresh logic against...
  2026-03-19 10:04  [Email]   Re: Q1 Planning (created)
                              Hey team, here's the updated timeline...
```

### Connection Suggestions — "What's related that I haven't linked?"

Discover documents that are semantically similar but not explicitly connected in the graph. This surfaces latent relationships you might not notice manually:

```bash
# Find unlinked similar documents across all sources
fieldnotes connections

# Only show cross-source connections (notes ↔ tasks, emails ↔ commits, etc.)
fieldnotes connections --cross-source

# What's related to a specific document?
fieldnotes connections --source-id "obsidian://vault/Deploy.md"

# Tighten the similarity threshold
fieldnotes connections --threshold 0.90 --limit 10
```

The `--cross-source` flag is the high-value mode — it finds relationships *between* different tools in your workflow (e.g., an Obsidian note that covers the same topic as an OmniFocus task or a Gmail thread).

### Person Profile — "What do I know about this person?"

Pull a single person's full footprint from the graph: recent interactions across every source, the topics you discuss with them, the people you both share, their open tasks, files that mention them, and the SAME_AS identity cluster the resolver built for them.

```bash
# Fastest path — look up by email
fieldnotes person alice@example.com

# Slack-only identity (no email visible)
fieldnotes person slack-user:T-TEAM/U-ALICE

# Fuzzy name lookup (forces name match even for inputs that look like emails)
fieldnotes person --search "Alice"

# Yourself (requires the [me] block — the canonical Person {is_self: true})
fieldnotes person --self

# Widen the recency window (default: 30d)
fieldnotes person alice@example.com --since 90d --limit 20

# Machine-readable output (stable schema — see below)
fieldnotes person alice@example.com --json
```

**Identifier resolution order:** The CLI accepts the same identifier shapes as the [persons curation surface](#cli-reference) — emails are matched exactly, `slack-user:<team>/<user>` matches the Slack-keyed Person fallback, and anything else falls through to fuzzy name matching. An ambiguous fuzzy name returns a non-zero exit and a candidate table on stderr (or a JSON `error: "ambiguous"` payload with `--json`) so you can re-run with a more specific identifier.

**Meeting prep with `--summary`:** Add `--summary` to generate a short LLM-written brief that answers *"what do I need to discuss with this person next?"*. The brief is grounded in the same data the profile shows — open tasks, recent emails and meetings, last-touch files, and shared topics — so it cites only what's in your graph. A plain `fieldnotes person …` call makes no LLM request; the brief only runs when `--summary` is set.

```bash
# Brief grounded in the last 30 days (default horizon)
fieldnotes person alice@example.com --summary

# Pull a specific calendar event into the brief context (agenda, attendees, attachments)
fieldnotes person alice@example.com --summary --meeting cal://q2-planning-2026-04-29

# Stretch the lookback window for the brief
fieldnotes person alice@example.com --summary --horizon 90d
```

**JSON shape (`--json`):** the payload is the same regardless of whether you arrive through the CLI or the [`person` MCP tool](#mcp-server) — that's the parity guarantee the integration test pins.

```json
{
  "identifier": "alice@example.com",
  "resolved": {"name": "Alice Example", "email": "alice@example.com", "is_self": false},
  "sources_present": ["calendar", "file", "gmail", "omnifocus", "slack"],
  "last_seen": "2026-04-26T12:00:00Z",
  "total_interactions": 27,
  "recent_interactions": [
    {"timestamp": "...", "source_type": "gmail", "title": "...", "snippet": "...", "edge_kind": "SENT"}
  ],
  "top_topics":      [{"topic_name": "Q2 planning", "doc_count": 3}],
  "related_people":  [{"name": "Bob", "email": "bob@example.com", "shared_count": 4}],
  "open_tasks":      [{"title": "...", "project": "...", "tags": ["..."], "due": null, "defer": null, "flagged": true}],
  "files_mentioning":[{"path": "/notes/alice.md", "mtime": "...", "source": "obsidian"}],
  "identity_cluster":[{"member": "alice.alt@example.com", "match_type": "fuzzy_name", "confidence": 0.97, "cross_source": true}],
  "next_brief": "…"
}
```

`next_brief` is only present when `--summary` is set; every other key is always present (empty arrays for sections with no edges). The reference fixture lives at [`worker/tests/integration/person_profile_schema.json`](worker/tests/integration/person_profile_schema.json) and is asserted by the e2e integration test on every CI run.

### Itinerary — "What's on my plate today?"

Roll up the calendar for a single day, then enrich every event with the open OmniFocus tasks for its attendees, the most semantically similar Obsidian / file / Slack notes, and the most recent email or Slack thread that covers all attendees. Built for **morning prep before standup**: one terminal command (or one MCP call) to walk into your first meeting knowing what's already on the table.

```bash
# Today, Rich-rendered, with a per-meeting LLM brief
fieldnotes itinerary

# Tomorrow's agenda
fieldnotes itinerary --day tomorrow

# Specific date (ISO)
fieldnotes itinerary --day 2026-04-29

# Filter to one configured Google Calendar account
fieldnotes itinerary --account work

# Skip the per-event LLM brief (no `completion`-role calls)
fieldnotes itinerary --brief

# Machine-readable JSON (stable schema — see below)
fieldnotes itinerary --json
```

**Day formats:** `--day` accepts `today` (default), `tomorrow`, or an explicit ISO date `YYYY-MM-DD`. Anything else exits non-zero with a parse error on stderr.

**Per-meeting brief (`next_brief`):** by default — i.e. without `--brief` — Fieldnotes makes one [`completion`-role](#configuration) call per event, grounded in the same linked tasks/notes/threads the profile shows, so each brief cites only what's already in your graph. The brief lands in `next_brief` (and on the Rich row, prefixed `▸`). Pass `--brief` to skip the LLM entirely; `next_brief` is then `null` on every event and the [`completion`](#configuration) role is never resolved. This is the right flag for offline runs, throwaway commands, or shells where you don't want LLM cost per invocation.

**JSON shape (`--json`):** the payload is the same regardless of whether you arrive through the CLI or the [`itinerary` MCP tool](#mcp-server) — that's the parity guarantee the integration test pins.

```json
{
  "day": "2026-04-29",
  "timezone": "America/Los_Angeles",
  "events": [
    {
      "event_id": "12345",
      "source_id": "google_calendar.work:abc",
      "title": "Q2 sync",
      "start": "2026-04-29T16:00:00Z",
      "end":   "2026-04-29T17:00:00Z",
      "account": "work",
      "calendar_id": "alice@example.com",
      "organizer": {"name": "Alice Example", "email": "alice@example.com"},
      "attendees": [{"name": "Bob Builder", "email": "bob@example.com"}],
      "location": "Zoom",
      "html_link": "https://calendar.google.com/...",
      "linked": {
        "tasks": [{"title": "Email Bob about Q2", "project": "Work", "tags": [], "due": "2026-04-28", "defer": null, "flagged": true, "source_id": "of://open-1"}],
        "notes": [{"source_id": "/notes/q2-plan.md", "title": "Q2 plan", "snippet": "...", "mtime": "2026-04-26T...", "attendee_overlap": true, "score": 0.81}],
        "thread": {"kind": "email", "source_id": "gmail://thread/q2", "title": "Q2 planning", "last_ts": "2026-04-27T...", "last_from": "bob@example.com"}
      },
      "next_brief": "..."
    }
  ]
}
```

`next_brief` is `null` whenever `--brief` is set (or when the LLM call is skipped); every other key is always present (`linked.thread` is `null` if no email/Slack window covered all attendees inside `--horizon`, and the linked arrays may be empty). The reference fixture lives at [`worker/tests/integration/itinerary_schema.json`](worker/tests/integration/itinerary_schema.json) and is asserted by the e2e integration test on every CI run.

### Daily Digest — "What changed recently?"

Get an aggregate summary of activity across all sources:

```bash
# What changed today?
fieldnotes digest

# Weekly summary with LLM-generated narrative
fieldnotes digest --since 7d --summarize

# Machine-readable output
fieldnotes digest --json
```

The digest shows per-source counts, top highlights, newly discovered cross-source connections, and new topics:

```
Digest: last 24 hours (2026-03-18 → 2026-03-19)

  Obsidian       3 modified
                 Deploy.md, auth-notes.md, meeting-2026-03-19.md

  OmniFocus      1 completed, 2 modified
                 Review auth middleware, Update CI pipeline, Fix deploy script

  Gmail          4 new emails
                 Re: Q1 Planning, Auth migration thread, ...

  Cross-source   5 new connections discovered
  Topics         1 new topic: "Deployment Infrastructure"
```

With `--summarize`, an LLM reads the digest and generates a narrative summary of your day or week.

## CLI Reference

```
fieldnotes [-c CONFIG] [-v] <command>
```

### Commands

**`init`** — Bootstrap `~/.fieldnotes/config.toml` and data directories.
  - When run in a terminal, launches an interactive wizard that prompts for your Neo4j password, model provider (Ollama/OpenAI/Anthropic), document paths, and Obsidian vault location.
  - `--non-interactive` — skip prompts and use defaults (useful in scripts).
  - `--with-docker` — extract the bundled `docker-compose.yml`, Prometheus, and Grafana configs to `~/.fieldnotes/infrastructure/`, generate a `.env` file with `NEO4J_PASSWORD` and `GRAFANA_PASSWORD`, create data directories for bind mounts, then run `docker compose up -d` automatically.
  - `--compose-file PATH` — use a custom `docker-compose.yml` instead of the bundled one (implies `--with-docker`). The compose file's parent directory is used as the Docker Compose project root; `.env` is written there.

**`doctor`** — Pre-flight checks for a healthy setup. Verifies:
  - Config file exists and parses correctly.
  - Model role → model → provider chain is complete.
  - Ollama is reachable and configured models are pulled.
  - OpenAI / Anthropic API keys are set.
  - Neo4j and Qdrant are reachable.
  - Source watch paths exist on disk.
  - Per-account Gmail and Google Calendar auth status (one row per `[sources.gmail.<account>]` / `[sources.google_calendar.<account>]` section).
  - `[me]` self-identity block parses, has at least one canonicalized email, and `name` is a string when set.
  - Required tools (`ollama`, `docker`) are on PATH.

**`up [--compose-file PATH]`** — Start Docker infrastructure (`docker compose up -d`). Uses `~/.fieldnotes/infrastructure/docker-compose.yml` by default.

**`stop [--compose-file PATH]`** — Stop Docker containers without removing them (`docker compose stop`). Data volumes are preserved.

**`down [--compose-file PATH]`** — Tear down Docker infrastructure (`docker compose down`). Containers and networks are removed; data volumes under `~/.fieldnotes/data/` are preserved.

**`migrate gmail-multiaccount [--account LABEL] [--yes] [--dry-run] [--force-running]`** — One-shot retag of legacy single-account Gmail and Google Calendar artifacts under a chosen multi-account label. Rewrites `queue.db` rows + cursor keys, Neo4j `Document.source_id` / `Chunk.id` / fallback `Person.source_id`, Qdrant chunk `payload.source_id`, renames token files (`gmail_token.json` → `gmail_token-<account>.json`), deletes deprecated cursor JSON files, and rewrites `~/.fieldnotes/config.toml` (with backup at `config.toml.backup-<UTC-timestamp>`). See [Migrating from Single-Account](#migrating-from-single-account) for the full walkthrough.
  - `--account LABEL` — Account label to retag under (default: `default` when `--yes` is set; otherwise interactive prompt). Must match `^[a-z][a-z0-9_-]{0,30}$`.
  - `--yes` — Skip the confirmation prompt; assume `default` when `--account` is omitted.
  - `--dry-run` — Print counts of what would change without mutating any state. Shows queue rows, cursor rows, Neo4j Documents/Chunks, fallback Persons, and Qdrant points.
  - `--force-running` — Run even when the daemon is alive. **Use with care:** any in-flight queue items the daemon picks up during the migrate window keep their old-shape `source_id` and need a manual second `migrate` run to clean up.

  Example dry-run:

  ```bash
  fieldnotes migrate gmail-multiaccount --dry-run
  Migration target: account='default'
    queue.db rows:       42
    queue.db cursors:    2
    Neo4j Documents:     128
    Neo4j Chunks:        417
    Fallback Persons:    9
    Qdrant points:       417
  Dry run — no changes made.
  ```

**`reindex-references [--dry-run] [--label LABEL]`** — Backfill `REFERENCES` edges for the existing corpus. Walks `CalendarEvent`, `Email`, `SlackMessage`, and `File` (Obsidian) nodes in Neo4j, re-runs `extract_source_link_hints` against their stored chunk text, and upserts the resulting edges. Idempotent — safe to re-run; MERGE ensures duplicate edges are not created.
  - `--dry-run` — Print how many edges would be created without writing to Neo4j.
  - `--label LABEL` — Scope to one node type: `CalendarEvent`, `Email`, `SlackMessage`, or `ObsidianNote`. Default: all four.

  Example:

  ```bash
  fieldnotes reindex-references --dry-run
  Dry run — would create 147 REFERENCES edge(s) across 312 node(s).

  fieldnotes reindex-references --label ObsidianNote
  Created 89 REFERENCES edge(s) across 203 node(s).
  ```

**`search <query> [-k N] [--no-rerank] [--rerank-top-k N]`** — Hybrid search combining graph traversal and vector similarity. Returns ranked results with source metadata. Results are reranked by a cross-encoder by default (see [Reranker](#reranker)); use `--no-rerank` to disable or `--rerank-top-k` to control how many candidates survive reranking.

**`ask [question]`** — Interactive Q&A against the knowledge graph. Retrieves context via hybrid search and synthesizes an answer with an LLM.
  - With no argument, starts a REPL with conversation history, streaming output, and question reformulation for follow-ups.
  - `--resume [id]` — resume a previous conversation (omit id for most recent).
  - `--history` — list past conversations.
  - `--no-stream` — disable streaming output.
  - `--json` — structured JSON output.

**`cluster [--min-cluster-size N] [--force]`** — Run the clustering pipeline manually. Connects to Neo4j and Qdrant, fetches all vectors, runs UMAP + HDBSCAN, labels clusters via LLM, and writes topic nodes. Use `--force` to run even if the corpus is below `min_corpus_size`. Prints progress and a summary of discovered topics.

**`topics list`** — List all discovered topics with document counts.

**`topics show <name>`** — Topic details: description, linked documents, related entities.

**`topics gaps`** — Topics discovered by clustering that aren't in your manual taxonomy.

**`timeline [--since SINCE] [--until UNTIL] [--source SOURCE] [--limit N] [--json]`** — Show a chronological timeline of activity across all indexed sources. Answers "what was I working on?" by listing file modifications, task completions, emails, and commits ordered by time.
  - `--since` — Start of the time range. Accepts relative values (`24h`, `7d`, `2w`) or ISO 8601 timestamps. Default: `24h`.
  - `--until` — End of the time range. Default: now.
  - `--source` — Filter to a single source type (`obsidian`, `omnifocus`, `gmail`, `file`, `repositories`, `slack`).
  - `--limit` — Maximum entries. Default: 50.
  - `--json` — Structured JSON output.

**`connections [--source-id ID] [--source SOURCE] [--threshold F] [--limit N] [--cross-source] [--json]`** — Surface documents that are semantically similar but not explicitly linked in the knowledge graph. Useful for discovering latent relationships — for example, Obsidian notes related to OmniFocus tasks, or emails related to code commits.
  - `--source-id` — Focus on connections for a specific document.
  - `--source` — Focus on a specific source type as seeds.
  - `--threshold` — Minimum cosine similarity score (0–1). Default: 0.82.
  - `--cross-source` — Only show connections between different source types (e.g., notes ↔ tasks). This is the high-value mode.
  - `--limit` — Maximum suggestions. Default: 20.
  - `--json` — Structured JSON output.

**`persons {inspect|split|confirm|merge}`** — Curate the Person identity graph. The reconcile chain (email → slack → fuzzy name → entity-bridge → cross-source → transitive closure → self-identity) makes mistakes over time; these subcommands let you fix and pin merges. Identifiers are emails (`alice@example.com`), slack ids (`slack:T123/U456`), or exact display names. Pass `--json` for machine-readable output. Every mutation appends to `<data_dir>/curation_audit.jsonl` so reversals are traceable.
  - `persons inspect <id>` — Show every `SAME_AS` and `NEVER_SAME_AS` edge incident on the matched Person, with `match_type`, `confidence`, and direction.
  - `persons split <id> <member>` — Break the `SAME_AS` edge between the cluster and `<member>` and install a `NEVER_SAME_AS` block so the next reconcile pass does not recreate the merge.
  - `persons confirm <a> <b>` — Lock a good merge as user-confirmed. Writes `SAME_AS` with `match_type='user_confirmed'`, `confidence=1.0`. Reconcile steps treat user-confirmed edges as ground truth and never overwrite them.
  - `persons merge <a> <b>` — Manual merge for cases the automated chain missed (different names, no shared email/slack id). Equivalent to `confirm` when no edge exists yet.

**`person <identifier> [--since SINCE] [--limit N] [--self] [--search NAME] [--summary] [--meeting CAL_ID] [--horizon SINCE] [--json]`** — Profile a single person. Renders recent interactions, top topics, related people, open OmniFocus tasks, files mentioning them, and the `SAME_AS` identity cluster. See [Person Profile](#person-profile--what-do-i-know-about-this-person) for the full overview.
  - `<identifier>` — Email (`alice@example.com`), `slack-user:<team>/<user>`, or a fuzzy name fragment. Resolution order: email → slack-user → fuzzy name. Ambiguous fuzzy matches return non-zero with a candidate table.
  - `--self` — Resolve the canonical `Person {is_self: true}`. Requires a `[me]` block in config.
  - `--search NAME` — Force fuzzy-name lookup even for inputs that look like email addresses.
  - `--since` — Recency window for the recent-interactions section. Default: `30d`.
  - `--limit N` — Cap rows in each list section (recent interactions, topics, related people, files). Default: 10.
  - `--summary` — Add an LLM-generated `next_brief` answering "what do I need to discuss with this person next?". Uses the configured `completion` role.
  - `--meeting CAL_ID` — Pull a specific calendar event's summary, attendees, and attachments into the brief context (only meaningful with `--summary`).
  - `--horizon SINCE` — Lookback window for the `--summary` brief inputs. Default: `30d`.
  - `--json` — Stable machine-readable schema (matches the `person` MCP tool payload).

**`itinerary [--day DAY] [--account ACCOUNT] [--brief] [--horizon SINCE] [--json]`** — Render a single day's calendar agenda with each event's linked open OmniFocus tasks, vector-similar notes (File / Obsidian / Slack), and the most recent email or Slack thread covering all attendees. By default makes one [`completion`-role](#configuration) call per event to populate `next_brief`. See [Itinerary](#itinerary--whats-on-my-plate-today) for the full overview.
  - `--day` — `today` (default), `tomorrow`, or an explicit `YYYY-MM-DD`. Other formats exit non-zero.
  - `--account` — Restrict to one configured `[sources.google_calendar.<account>]`. Unknown account names exit non-zero with the configured set on stderr.
  - `--brief` — Skip the per-event LLM brief; `next_brief` stays `null` on every event and the `completion` role is never resolved.
  - `--horizon` — Lookback window for linked tasks/notes/threads. Relative form (`30d`, `24h`, `2w`, `3m`). Default: `30d`.
  - `--json` — Stable machine-readable schema (matches the `itinerary` MCP tool payload).

**`digest [--since SINCE] [--summarize] [--json]`** — Summarize recent activity across all indexed sources. Returns aggregate counts per source type with top highlights, cross-source connections discovered, and new topics.
  - `--since` — Time range start. Default: `24h`. Same relative format as `timeline`.
  - `--summarize` — Generate an LLM-powered summary paragraph of the activity.
  - `--json` — Structured JSON output.

**`serve --daemon [--progress | --no-progress]`** — Run the ingest pipeline as a long-running background service. Prints a startup summary showing configured sources, model providers, and role assignments. When connected to a TTY, shows live progress bars for queue depth and per-file pipeline stage (auto-detected; override with `--progress` / `--no-progress`).

**`serve --mcp`** — Run only the MCP server (stdio transport, for Claude Desktop).

**`service install|uninstall|status|start|stop`** — Manage fieldnotes as a system service (launchd on macOS, systemd on Linux).

  - **macOS**: Installs a launchd plist at `~/Library/LaunchAgents/com.fieldnotes.daemon.plist` with auto-restart.
  - **Linux**: Installs a systemd user unit at `~/.config/systemd/user/fieldnotes.service`.
  - **Logs**: `~/.fieldnotes/logs/daemon.log` (created on `service install`).

**`backup [create] [--keep N]`** — Create a compressed backup of all fieldnotes data and databases. Stops Docker containers for a consistent snapshot, archives configuration (`config.toml`, `credentials.json`, `.env`) and all database data (`data/neo4j`, `data/qdrant`, `data/prometheus`, `data/grafana`, `state/`) to `~/.fieldnotes/backups/fieldnotes-<timestamp>.tar.gz`, then restarts containers.
  - `--keep N` — After creating the backup, delete all but the **N** most recent backups.

**`backup list`** — List existing backups with filenames, sizes, and creation timestamps.

**`backup schedule [--remove] [--keep N]`** — Install or remove a daily scheduled backup (02:00 local time).
  - On macOS, installs a launchd plist at `~/Library/LaunchAgents/com.fieldnotes.backup.plist`.
  - On Linux, installs a systemd user timer at `~/.config/systemd/user/fieldnotes-backup.timer`.
  - `--remove` — Uninstall the scheduled backup.
  - `--keep N` — Pass `--keep N` to each scheduled backup run so old backups are pruned automatically.

**`restore <backup>`** — Restore fieldnotes data from a backup archive. Accepts a filename from `backup list` or a full path. Stops Docker containers, extracts the archive over `~/.fieldnotes/`, then restarts containers. Validates archive contents and blocks path-traversal attempts.

**`setup-claude`** — Configure Claude Desktop to use the fieldnotes MCP server.

## Backup & Restore

Fieldnotes can back up and restore your entire setup — configuration, credentials, and all database data (Neo4j, Qdrant, Prometheus, Grafana) — in a single compressed archive.

### Creating a Backup

```bash
# Create a backup (stops containers, archives, restarts)
fieldnotes backup

# Same thing, explicit subcommand
fieldnotes backup create

# Create a backup and keep only the 5 most recent
fieldnotes backup create --keep 5
```

Backups are saved to `~/.fieldnotes/backups/fieldnotes-<YYYYMMDD-HHMMSS>.tar.gz`. Docker containers are stopped during the snapshot and restarted afterward.

### Listing Backups

```bash
fieldnotes backup list
```

```
Backup                                       Size  Created
----------------------------------------------------------------------
fieldnotes-20260318-020000.tar.gz          12.3 MB  2026-03-18 02:00 UTC
fieldnotes-20260319-020000.tar.gz          12.5 MB  2026-03-19 02:00 UTC
fieldnotes-20260320-143022.tar.gz          12.8 MB  2026-03-20 14:30 UTC
```

### Restoring From a Backup

```bash
# Restore by name (from backup list output)
fieldnotes restore fieldnotes-20260318-020000.tar.gz

# Or by full path
fieldnotes restore /path/to/fieldnotes-20260318-020000.tar.gz
```

Restore stops running containers, extracts the archive over `~/.fieldnotes/`, and restarts containers. The archive is validated for path-traversal safety before extraction.

### Scheduled Backups

```bash
# Install a daily backup at 02:00 local time
fieldnotes backup schedule

# With automatic pruning — keep only the last 7 backups
fieldnotes backup schedule --keep 7

# Remove the scheduled backup
fieldnotes backup schedule --remove
```

On macOS this creates a launchd plist; on Linux a systemd user timer. Logs go to `~/.fieldnotes/logs/backup.log`.

### What's Included in a Backup

| Item | Contents |
|---|---|
| `config.toml` | All configuration settings |
| `credentials.json` | OAuth / API credentials |
| `data/` | Neo4j, Qdrant, Prometheus, and Grafana databases |
| `state/` | Source cursors (OmniFocus, sync state) |
| `infrastructure/.env` | Docker passwords and environment variables |

## MCP Server

Fieldnotes exposes tools over the [Model Context Protocol](https://modelcontextprotocol.io) via stdio transport, making it available to Claude Desktop, Claude Code, and other MCP-compatible clients.

### Tools

| Tool | Description |
|---|---|
| `search(query, top_k?, source_type?, rerank?)` | Hybrid graph + vector search with optional source filtering and cross-encoder reranking |
| `ask(question, source_type?, rerank?)` | RAG + LLM synthesis — retrieves context and generates an answer |
| `timeline(since?, until?, source_type?, limit?)` | Chronological activity feed across all sources within a time range |
| `suggest_connections(source_id?, source_type?, threshold?, limit?, cross_source?)` | Find semantically similar but unlinked documents across the knowledge graph |
| `digest(since?, summarize?)` | Aggregate activity summary with per-source counts, highlights, and new connections |
| `itinerary(day?, account?, brief?, horizon?)` | Aggregated daily agenda: calendar events with linked open OmniFocus tasks, vector-similar notes, and the most recent email/Slack thread covering all attendees. `day` accepts `today` (default), `tomorrow`, or `YYYY-MM-DD`. `brief=true` skips the LLM per-event summary (`next_brief` stays null). Returns the same payload shape as `fieldnotes itinerary --json` — see [Itinerary](#itinerary--whats-on-my-plate-today). |
| `list_topics(source?)` | List topics (`all`, `cluster`, or `user`) with document counts |
| `show_topic(name)` | Topic details: description, documents, related entities and topics |
| `topic_gaps()` | Cluster-discovered topics missing from your manual taxonomy |
| `person(identifier, since?, limit?, summary?, meeting_id?, horizon?)` | Profile a person: recent interactions, topics, related people, open tasks, files, identity cluster (resolution: email → slack-user → fuzzy name). Set `summary=true` for an LLM-written `next_brief` (optionally seeded with `meeting_id`). Returns the same payload shape as `fieldnotes person --json` — see [Person Profile](#person-profile--what-do-i-know-about-this-person). |
| `ingest_status()` | Index health: source counts, last sync times, circuit breaker states |

The `source_type` parameter on `search`, `ask`, `timeline`, and `digest` accepts any configured source key — including `slack`.

#### Searching attachments

Because [Attachments](#attachments) are indexed across three layers (parent message/event, Attachment Document, and — for indexable formats — attachment-content chunks), the same `search` and `ask` tools cover both filename and content queries with no special syntax. Examples:

```text
# Filename-style queries hit the parent (filename is woven into its body)
search("the AWS architecture diagram alice sent")
search("quarterly_review.docx Bob shared in #leadership")

# Content-style queries hit the attachment-content chunks for indexable MIMEs
search("PDF page that mentions the Q3 quota carve-out")
ask("what did the design brief say about retention?")
```

Filename queries work even when the attachment itself is metadata-only (e.g. an unparsed `.docx` or `.pptx`); content queries require the attachment to fall on the `attachment_indexable_mimetypes` allowlist.

### Claude Desktop Integration

```bash
fieldnotes setup-claude
```

This registers fieldnotes as an MCP server in Claude Desktop's configuration. After restarting Claude Desktop, it can query your knowledge graph directly during conversations.

## Pipeline Architecture

### Databases

**Neo4j** (property graph) stores the knowledge graph:
- **Nodes**: Document, Entity (Person, Technology, Project, Organization, Concept), Topic
- **Edges**: MENTIONS, APPEARS_IN, BELONGS_TO, DEPICTS, RELATED_TO_TOPIC, and extracted relationship triples
- **Queries**: Natural language translated to Cypher via LangChain (read-only execution)

**Qdrant** (vector store) enables semantic search:
- 768-dimensional embeddings (default: `nomic-embed-text` via Ollama)
- Payload includes text, source type, source ID, date, and chunk index
- Top-k cosine similarity with optional source filtering

### Pipeline Stages

1. **Parser** — Source-specific adapter extracts text, metadata, and graph hints (pre-known entities/edges) from raw content. Files with unsupported MIME types — or those matching `index_only_patterns` — get a metadata-only record (filename, extension, path, size) with a human-readable description so they remain discoverable via search. **Attachments** on Gmail / Slack / Calendar items are routed through the same per-MIME parsers (PDF / vision / text loader) via the shared [`stream_and_parse`](worker/worker/parsers/attachments.py) helper — see [Attachments](#attachments) for the indexable/metadata-only split and parent-link semantics.
2. **Chunker** — Sentence-aware splitter produces ~512-token chunks with 64-token overlap. Short chunks are merged to avoid fragmentation. Slack documents opt into a *whole-message overlap* mode (`chunk_strategy={"mode": "message_overlap"}`) so chunk boundaries fall on message gaps and never split a message mid-text.
3. **Embedder** — Generates 768-dim vectors via the `embed` role model. Batches 64 texts per call.
4. **Extractor** — LLM extracts named entities (typed: Person, Technology, etc.) and relationship triples from each chunk. Falls back to `extract_fallback` role on JSON parse errors.
5. **Resolver** — Deduplicates entities across chunks using a 3-strategy cascade: exact string match → fuzzy match (rapidfuzz `token_sort_ratio` with length-aware thresholds) → embedding cosine similarity. Resolves references to canonical names.
6. **Writer** — Persists everything in a single Neo4j transaction per document: upsert source node, write entities, write chunks, write graph hint edges, clean orphans. Upserts chunk vectors to Qdrant. After each batch, runs the [entity resolution pipeline](#entity-resolution-pipeline) to unify person identities across sources.

### Vision Pipeline

Images are processed asynchronously through a vision model that extracts:
- A natural-language description
- OCR text
- Named entities

The output becomes a synthetic text chunk that flows through the standard embedding and writing stages. Images are linked to extracted entities via `DEPICTS` edges.

### Topic Clustering

Runs on a configurable schedule (default: Sunday 3 AM) or on demand via `fieldnotes cluster`:
1. UMAP reduces the full vector corpus to 2D
2. HDBSCAN discovers density-based clusters
3. An LLM names each cluster based on representative documents
4. Topic nodes and BELONGS_TO edges are written to Neo4j
5. Installed applications are linked to relevant topics via RELATED_TO_TOPIC edges

## Observability

Fieldnotes pushes metrics to a Prometheus Pushgateway running in Docker. Prometheus scrapes the gateway, and Grafana provides pre-built dashboards.

### Docker Compose Services

The full stack runs five containers. When installed via `pipx install fieldnotes`, all Docker infrastructure files (compose file, Prometheus config, Grafana dashboards and provisioning) are bundled with the package and extracted to `~/.fieldnotes/infrastructure/` on first `fieldnotes init --with-docker`.

If you cloned the repository, the same files are also available in the repo root.

```bash
# Option A: automatic (extracts bundled infra + generates .env + starts containers)
fieldnotes init --with-docker

# Option B: manual
export NEO4J_PASSWORD=changeme
export GRAFANA_PASSWORD=changeme
docker compose -f ~/.fieldnotes/infrastructure/docker-compose.yml up -d
```

| Service | Image | Port | Memory Limit | Purpose |
|---|---|---|---|---|
| neo4j | `neo4j:2026.02.3-community` | 7687 | 1 GB | Knowledge graph storage |
| qdrant | `qdrant/qdrant:v1.17.0` | 6333 | 512 MB | Vector similarity search |
| pushgateway | `prom/pushgateway:v1.11.2` | 9091 | 64 MB | Metrics collection endpoint |
| prometheus | `prom/prometheus:v3.10.0` | 9090 | 256 MB | Metrics storage and querying |
| grafana | `grafana/grafana-oss:12.4.1` | 3000 | 128 MB | Dashboards and visualization |

All services bind to `127.0.0.1` only. Data is persisted under `$FIELDNOTES_DATA` (default `~/.fieldnotes/data`). Total infrastructure memory footprint is approximately 2 GB.

### Key Metrics

- `worker_documents_processed` / `worker_documents_failed` — ingest throughput
- `worker_pipeline_duration_seconds` — per-document processing time by stage
- `worker_llm_request_duration_seconds` — LLM API latency by provider and role
- `worker_llm_tokens` — token usage (input/output)
- `worker_entities_extracted` / `worker_entities_resolved` — extraction yield
- `worker_chunks_embedded` — embedding throughput
- `worker_circuit_breaker_rejections` — fault tolerance activations
- `worker_queue_depth` — pending ingest events

Access Grafana at `http://localhost:3000` (user: `admin`, password: your `GRAFANA_PASSWORD`). A fieldnotes dashboard is auto-provisioned with panels for ingest throughput, LLM latency, entity counts, and pipeline health.

## Local Development

All development commands use a virtualenv managed by `make`. The `.venv` is created automatically on first run — no manual activation needed.

```bash
# Clone and set up
git clone https://github.com/mmlac/fieldnotes.git
cd fieldnotes

# Install in editable mode with dev dependencies (creates worker/.venv)
make install-dev

# Start infrastructure (Neo4j, Qdrant, Prometheus, Pushgateway, Grafana)
export NEO4J_PASSWORD=changeme
export GRAFANA_PASSWORD=changeme
make docker-up
```

### Make Targets

| Target | Description |
|---|---|
| `make install` | Install fieldnotes worker into `worker/.venv` |
| `make install-dev` | Editable install with dev deps (pytest, ruff). **Re-run whenever `worker/pyproject.toml` changes** (new dependency added, version bumped) — the venv is not auto-refreshed. A stale venv shows up as `ModuleNotFoundError` during test collection; `tests/test_cli_imports.py` is the fast-failing canary for this. |
| `make lint` | Run ruff linter on worker |
| `make fmt` | Auto-format worker code with ruff |
| `make test` | Run worker test suite |
| `make test-ci` | Lint + tests (CI mode) |
| `make build` | Build worker sdist and wheel into `worker/dist/` |
| `make publish-test` | Upload worker to TestPyPI |
| `make publish` | Upload worker to PyPI |
| `make docker-up` | Start Docker services |
| `make docker-down` | Stop Docker services |
| `make version` | Print current version |

### Running Commands Directly

If you need to run `fieldnotes` or other commands inside the venv:

```bash
# Option 1: activate the venv
source worker/.venv/bin/activate
fieldnotes search "test query"

# Option 2: run directly via venv path
worker/.venv/bin/fieldnotes search "test query"
```

### Publishing to PyPI

```bash
# Dry run on TestPyPI first
TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-test-... make publish-test

# Install from TestPyPI to verify
pip install -i https://test.pypi.org/simple/ fieldnotes

# Publish for real
TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-... make publish
```

## Project Structure

```
worker/
├── cli/                    # CLI entry point and interactive Q&A
│   ├── __init__.py         # Argument parsing and command dispatch
│   ├── ask.py              # Interactive REPL with streaming
│   ├── cluster.py          # Manual clustering run
│   ├── reformulator.py     # Follow-up question reformulation
│   └── history.py          # Conversation persistence
├── sources/                # Data source adapters
│   ├── files.py            # Filesystem watcher (watchdog)
│   ├── obsidian.py         # Obsidian vault watcher
│   ├── gmail.py            # Gmail polling with cursor sync
│   ├── gmail_auth.py       # Gmail OAuth2 flow
│   ├── calendar.py         # Google Calendar polling with syncToken sync
│   ├── calendar_auth.py    # Google Calendar OAuth2 flow
│   ├── repositories.py     # Git repository scanner
│   ├── omnifocus.py        # OmniFocus task polling via JXA (macOS)
│   ├── macos_apps.py       # macOS app discovery (polling)
│   └── homebrew.py         # Homebrew package listing (polling)
├── parsers/                # Document type parsers
│   ├── base.py             # Shared utilities (canonicalize_email, email extraction, GraphHint)
│   ├── files.py            # Text, PDF, image, iWork, and metadata-only
│   ├── iwork.py            # Apple Pages and Keynote (osascript)
│   ├── obsidian.py         # Obsidian notes (wikilinks, frontmatter, emails)
│   ├── gmail.py            # Email messages
│   ├── calendar.py         # Calendar events (attendees, organizer, location)
│   ├── repositories.py     # Git commits and READMEs
│   ├── omnifocus.py        # OmniFocus tasks (projects, tags, subtasks, person mentions)
│   └── apps.py             # Application metadata
├── pipeline/               # Ingest pipeline stages
│   ├── __init__.py         # Pipeline orchestration and reconciliation chain
│   ├── chunker.py          # Sentence-aware text splitter
│   ├── embedder.py         # Vector embedding
│   ├── extractor.py        # LLM entity/triple extraction
│   ├── resolver.py         # Entity deduplication
│   ├── writer.py           # Neo4j + Qdrant persistence + entity reconciliation
│   ├── progress.py         # Live Rich progress bars for daemon ingest
│   ├── vision.py           # Image analysis
│   └── app_describer.py    # LLM app descriptions
├── clustering/             # Topic discovery
│   ├── cluster.py          # UMAP + HDBSCAN
│   ├── labeler.py          # LLM topic naming
│   ├── writer.py           # Topic persistence
│   └── app_linker.py       # App-to-topic linking
├── query/                  # Search and retrieval
│   ├── graph.py            # NL → Cypher (LangChain)
│   ├── vector.py           # Qdrant similarity search
│   ├── hybrid.py           # Result merging
│   ├── reranker.py         # Cross-encoder reranking (sentence-transformers)
│   └── topics.py           # Topic browsing
├── models/                 # LLM provider abstraction
│   ├── providers/          # Ollama, OpenAI, Anthropic, sentence-transformers
│   └── resolver.py         # Role-based model resolution
├── service/                # System service management
│   ├── launchd.py          # macOS
│   └── systemd.py          # Linux
├── infrastructure/         # Bundled Docker Compose + Grafana/Prometheus configs
│   ├── docker-compose.yml  # (extracted to ~/.fieldnotes/infrastructure/)
│   ├── prometheus.yml
│   └── grafana/            # Provisioning and dashboard JSON
├── templates/              # Bundled launchd/systemd templates for scheduled backups
├── config.py               # TOML config loader (with role→model→provider validation)
├── init.py                 # Interactive init wizard (--with-docker, --non-interactive)
├── infra.py                # Docker Compose lifecycle (up, stop, down)
├── backup.py               # Backup, restore, and scheduled backup management
├── doctor.py               # Pre-flight diagnostic checks
├── mcp_server.py           # MCP server (stdio transport)
├── serve_daemon.py         # Daemon mode with startup summary
├── metrics.py              # Prometheus metrics
├── circuit_breaker.py      # Fault tolerance
└── config.toml.example     # Default configuration template

docs/
└── similarity.md           # Entity resolution analysis and improvement roadmap
```

## License

MIT
