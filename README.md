![Fieldnotes](./fieldnotes-banner.png)


# Fieldnotes

Fieldnotes is a personal knowledge graph that continuously indexes your digital life — local files, Obsidian vaults, email threads, git repositories, OmniFocus tasks, and installed applications — and exposes that knowledge as structured context for LLM agents. It combines a property graph (Neo4j) for relationship traversal with a vector store (Qdrant) for semantic retrieval, connected by a hybrid query layer and served over the Model Context Protocol (MCP) so any compatible AI assistant can query everything you know.

## Table of Contents

- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Data Sources](#data-sources)
  - [Cross-Source Tag Unification](#cross-source-tag-unification-omnifocus--obsidian)
  - [Cross-Source Person Linking](#cross-source-person-linking-gmail--google-calendar--obsidian)
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

Files that can't be parsed for content (e.g. `.3mf`, `.psd`, `.mp4`) are still indexed as metadata-only records — the filename, extension, path, and a human-readable description are embedded so they're discoverable via semantic search.

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

> **Note:** `pillow-heif>=0.22` requires `libheif>=1.22`. If you see compilation
> errors about `heif_camera_intrinsic_matrix`, your libheif is older than what
> `pillow-heif` expects. Fieldnotes pins `pillow-heif<0.22` for compatibility
> with libheif 1.17–1.21.

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
```

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
recursive = true
max_file_size = 104857600               # 100 MB

[sources.obsidian]
vault_paths = ["~/obsidian-vault"]

[sources.gmail]
poll_interval_seconds = 300
max_initial_threads = 500
label_filter = "INBOX"
client_secrets_path = "~/.fieldnotes/credentials.json"

[sources.google_calendar]
poll_interval_seconds = 300
max_initial_days = 90
calendar_ids = ["primary"]
client_secrets_path = "~/.fieldnotes/credentials.json"
```

#### Gmail OAuth Setup

Gmail indexing requires a Google Cloud OAuth2 credential. This is a one-time setup:

1. **Create a Google Cloud project** at [console.cloud.google.com](https://console.cloud.google.com) (e.g., name it "FieldNotes").
2. **Enable the Gmail API**: Navigate to *APIs & Services → Library*, search for "Gmail API", and click *Enable*.
3. **Configure the OAuth consent screen**: Go to *APIs & Services → OAuth consent screen*. Choose **External** as user type. Fill in the required app name and email fields. On the *Scopes* page, add `gmail.readonly`. On the *Test users* page, **add your own Gmail address**. Leave the app in **Testing** mode — this is sufficient for personal use and avoids Google's app verification process.
4. **Create OAuth credentials**: Go to *APIs & Services → Credentials → Create Credentials → OAuth client ID*. Select **Desktop application** as the application type.
5. **Download the credentials**: Click *Download JSON* and save it as `~/.fieldnotes/credentials.json` (or the path you set in `client_secrets_path`).
6. **First run**: When the daemon starts with Gmail enabled, it will open your browser for a one-time consent screen. Approve read-only access (`gmail.readonly` scope). The resulting token is saved to `~/.fieldnotes/data/gmail_token.json` (mode `0600`) and refreshed automatically from then on.

> **Note:** If you're running fieldnotes as a system service (headless), complete the first OAuth flow manually with `fieldnotes serve --daemon` in a terminal before installing the service. The saved token will be reused.

> **Troubleshooting:** If the consent screen shows _"Access Blocked: \<AppName\> has not completed the Google verification process"_, the `credentials.json` file belongs to a different Google Cloud project that was published but never verified. Create your own project following the steps above — step 3 (consent screen) is the most commonly missed part.

#### Google Calendar OAuth Setup

Google Calendar uses the same OAuth credentials file and Google Cloud project as Gmail. If you already configured Gmail, you only need to enable the Calendar API:

1. **Enable the Google Calendar API**: In the same Google Cloud project, go to *APIs & Services → Library*, search for "Google Calendar API", and click *Enable*.
2. **Add the Calendar scope**: Go to *APIs & Services → OAuth consent screen → Edit App → Scopes* and add `calendar.events.readonly`. (If you set up Gmail and Calendar at the same time, add both scopes in one go.)
3. **First run**: When the daemon starts with Google Calendar enabled, it will open your browser for a one-time consent screen. Approve read-only access (`calendar.events.readonly` scope). The token is saved to `~/.fieldnotes/data/calendar_token.json` (mode `0600`) and refreshed automatically.
3. **Multiple calendars**: Set `calendar_ids` to a list of calendar IDs. Use `"primary"` for your main calendar. Other calendars can be found in Google Calendar settings under *Settings for other calendars → Integrate calendar → Calendar ID*.

> **Tip:** The same `credentials.json` file works for both Gmail and Google Calendar — they share the OAuth client, but each source has its own token file and consent flow.

```toml
repo_roots = ["~/projects"]
include_patterns = ["README*", "CHANGELOG*", "CONTRIBUTING*", "docs/**/*.md", "*.toml", "ADR/**/*.md"]
exclude_patterns = ["node_modules/", ".git/", "vendor/", "target/", "__pycache__/"]
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

### Cross-Source Person Linking (Gmail + Google Calendar + Obsidian)

When multiple people-aware sources are enabled, Fieldnotes automatically merges Person nodes across sources into a single graph identity based on email address. An attendee at a meeting, the sender of an email, and a person mentioned in an Obsidian note all link to the same Person node — no manual linking required.

**How it works:** The Gmail parser, Google Calendar parser, and Obsidian parser all emit `GraphHint` records with `object_merge_key="email"` and an `object_id` of the form `person:{email}`. The pipeline Writer `MERGE`s Person nodes by email, so an attendee in a calendar event, a correspondent in a Gmail thread, and a contact in an Obsidian note are guaranteed to resolve to the same graph node. Periodic `reconcile_persons()` runs create `SAME_AS` edges when the same email appears across sources, keeping the longest display name.

**Email canonicalization:** All parsers normalise emails through a shared `canonicalize_email()` function that lower-cases, trims whitespace, and rewrites `@googlemail.com` → `@gmail.com` (Google treats these as the same mailbox). This prevents duplicate Person nodes for the same real-world identity.

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

After each ingest batch, a five-step reconciliation chain runs to unify person identities across all sources:

| Step | What It Does |
|---|---|
| **1. Email-based reconciliation** | Creates `SAME_AS` edges between Person nodes that share an email address across different sources. Keeps the longest display name. |
| **2. Fuzzy name matching** | Uses RapidFuzz `token_sort_ratio` to find Person nodes with near-identical names (threshold ≥ 95). Creates `SAME_AS` edges with `match_type='fuzzy_name'`. |
| **3. Entity→Person bridging** | Links `Entity` nodes of type Person (extracted by the LLM) to structured `Person` nodes (from Gmail, Calendar, etc.) using fuzzy matching (threshold ≥ 93). Creates `SAME_AS` edges with `match_type='entity_person_bridge'`. |
| **4. Cross-source entity resolution** | Deduplicates entities with the same label appearing across different sources using a 3-strategy cascade: exact match → fuzzy string match → embedding cosine similarity. |
| **5. Transitive SAME_AS closure** | If A↔B and B↔C have `SAME_AS` edges but A↔C does not, creates the missing A↔C edge (up to 4 hops). Ensures the full identity cluster is fully connected. |

Each step is fault-tolerant — a failure in one step logs a warning and proceeds to the next. All `SAME_AS` edges carry metadata: `confidence`, `match_type`, and `cross_source` flag.

> **Details:** See [docs/similarity.md](docs/similarity.md) for the full entity resolution analysis including thresholds, match strategies, and architecture notes.

### Initial Scan and Cursor Persistence

On first startup, file and obsidian sources walk all configured directories and index every matching file. A SHA256-based cursor is saved to disk after the scan completes, recording the hash and mtime of every indexed file.

On subsequent startups, the cursor is loaded and diffed against the current filesystem state — only new, modified, or deleted files generate events. Unchanged files are skipped entirely.

| Source | Cursor File | What It Tracks |
|---|---|---|
| **Files** | `~/.fieldnotes/data/files_cursor.json` | Per-file SHA256 + mtime |
| **Obsidian** | `~/.fieldnotes/data/obsidian_cursor.json` | Per-file SHA256 + mtime |
| **Gmail** | `~/.fieldnotes/data/gmail_cursor.json` | Gmail History API ID |
| **Git Repos** | `~/.fieldnotes/data/repo_cursor.json` | Per-repo HEAD commit SHA |
| **OmniFocus** | `~/.fieldnotes/state/omnifocus.json` | Per-task content hash |
| **Google Calendar** | `~/.fieldnotes/data/calendar_cursor.json` | Per-calendar syncToken |

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
  - Required tools (`ollama`, `docker`) are on PATH.

**`up [--compose-file PATH]`** — Start Docker infrastructure (`docker compose up -d`). Uses `~/.fieldnotes/infrastructure/docker-compose.yml` by default.

**`stop [--compose-file PATH]`** — Stop Docker containers without removing them (`docker compose stop`). Data volumes are preserved.

**`down [--compose-file PATH]`** — Tear down Docker infrastructure (`docker compose down`). Containers and networks are removed; data volumes under `~/.fieldnotes/data/` are preserved.

**`search <query> [-k N]`** — Hybrid search combining graph traversal and vector similarity. Returns ranked results with source metadata.

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
  - `--source` — Filter to a single source type (`obsidian`, `omnifocus`, `gmail`, `file`, `repositories`).
  - `--limit` — Maximum entries. Default: 50.
  - `--json` — Structured JSON output.

**`connections [--source-id ID] [--source SOURCE] [--threshold F] [--limit N] [--cross-source] [--json]`** — Surface documents that are semantically similar but not explicitly linked in the knowledge graph. Useful for discovering latent relationships — for example, Obsidian notes related to OmniFocus tasks, or emails related to code commits.
  - `--source-id` — Focus on connections for a specific document.
  - `--source` — Focus on a specific source type as seeds.
  - `--threshold` — Minimum cosine similarity score (0–1). Default: 0.82.
  - `--cross-source` — Only show connections between different source types (e.g., notes ↔ tasks). This is the high-value mode.
  - `--limit` — Maximum suggestions. Default: 20.
  - `--json` — Structured JSON output.

**`digest [--since SINCE] [--summarize] [--json]`** — Summarize recent activity across all indexed sources. Returns aggregate counts per source type with top highlights, cross-source connections discovered, and new topics.
  - `--since` — Time range start. Default: `24h`. Same relative format as `timeline`.
  - `--summarize` — Generate an LLM-powered summary paragraph of the activity.
  - `--json` — Structured JSON output.

**`serve --daemon`** — Run the ingest pipeline as a long-running background service. Prints a startup summary showing configured sources, model providers, and role assignments.

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
| `search(query, top_k?, source_type?)` | Hybrid graph + vector search with optional source filtering |
| `ask(question, source_type?)` | RAG + LLM synthesis — retrieves context and generates an answer |
| `timeline(since?, until?, source_type?, limit?)` | Chronological activity feed across all sources within a time range |
| `suggest_connections(source_id?, source_type?, threshold?, limit?, cross_source?)` | Find semantically similar but unlinked documents across the knowledge graph |
| `digest(since?, summarize?)` | Aggregate activity summary with per-source counts, highlights, and new connections |
| `list_topics(source?)` | List topics (`all`, `cluster`, or `user`) with document counts |
| `show_topic(name)` | Topic details: description, documents, related entities and topics |
| `topic_gaps()` | Cluster-discovered topics missing from your manual taxonomy |
| `ingest_status()` | Index health: source counts, last sync times, circuit breaker states |

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

1. **Parser** — Source-specific adapter extracts text, metadata, and graph hints (pre-known entities/edges) from raw content. Files with unsupported MIME types get a metadata-only record (filename, extension, path, size) with a human-readable description so they remain discoverable via search.
2. **Chunker** — Sentence-aware splitter produces ~512-token chunks with 64-token overlap. Short chunks are merged to avoid fragmentation.
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
| `make install-dev` | Editable install with dev deps (pytest, ruff) |
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
│   └── topics.py           # Topic browsing
├── models/                 # LLM provider abstraction
│   ├── providers/          # Ollama, OpenAI, Anthropic
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
