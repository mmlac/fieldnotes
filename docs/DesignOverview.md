# Fieldnotes

**A Personal Knowledge Graph for LLM Agents**

> *"The best knowledge system is the one that knows what you already know."*

---

## Overview

Fieldnotes is an open-source personal knowledge graph that continuously indexes your digital life — local files, Obsidian vaults, email threads, and code repositories — and exposes that knowledge as structured context for LLM agents. It combines a property graph (Neo4j) for relationship traversal with a vector store (Qdrant) for semantic retrieval, and surfaces both through a unified query interface designed for agent consumption.

The goal is not to replace your existing tools. Fieldnotes runs alongside them, watching for changes, extracting meaning, and connecting the dots across sources that have never talked to each other before.

---

## Motivation

Modern knowledge workers accumulate information across a fragmented landscape: notes in markdown files, decisions in email threads, architecture rationale in READMEs, and project history in commit logs. LLMs can reason across all of this — but only if they can access it in a coherent, queryable form.

Existing RAG solutions treat knowledge as a flat document corpus. They answer "what chunk is most similar to this query?" but cannot answer "who have I emailed about this topic, and which of my local files relate to those conversations?" That second question requires a graph.

Fieldnotes is built on the belief that **relationships are the signal**. Entities matter less than the edges between them.

---

## Design Principles

**1. Continuous over batch.** Fieldnotes watches your sources in real time. New files, incoming emails, and fresh commits are indexed as they arrive — not on a schedule you have to remember to run.

**2. Relationships first.** Every ingested artifact is modeled as a node with edges to entities, people, topics, and other artifacts. Vector similarity is a fallback, not the primary retrieval strategy.

**3. Local by default.** All inference runs locally via Ollama. No API calls, no data leaving your machine, no ongoing cost. Cloud LLM providers can be swapped in for extraction quality improvements, but the system is designed to run entirely air-gapped.

**4. Agent-native.** The query interface is designed for LLM consumption first, human consumption second. Fieldnotes exposes an MCP server so agents (including GasTown Polecats) can query it as a tool during task execution.

**5. Right tool for the job.** The daemon runtime is Go — a single binary, low idle memory, goroutine-native concurrency for I/O-bound watching and polling. The ML pipeline is Python — the only viable language for HDBSCAN, UMAP, LangChain, and mlx-lm. Adding a new source type requires only a new adapter (one Go `Source`, one Python `BaseParser`), with zero changes to the core pipeline. The adapter contract is the community contribution surface.

**6. Honest about limitations.** Fieldnotes does not attempt to solve hallucination. It is a grounding layer — it provides factual context for LLMs to reason over. The LLM is still responsible for synthesis.

---

## Implementation Language

Fieldnotes uses a **split architecture**: a Go daemon for the always-on runtime, and a Python worker for the ML pipeline. This is a deliberate choice, not a compromise.

### Go Daemon (`fieldnotes-daemon`)

The daemon handles all I/O-bound, always-on work:

- File watching and event dispatch
- Gmail polling
- Git repository scanning
- Obsidian vault parsing (wikilinks, frontmatter, tags)
- Ingestion queue management
- MCP server
- HTTP API (feeds the Python worker)

Go is the right fit here for the same reasons it suits any long-running systems process. Goroutines are a natural model for the concurrent vision worker queue, parallel embedding calls, and async Gmail polling. Idle memory footprint is dramatically lower than Python — relevant when this process runs 24/7 alongside Neo4j and Qdrant. Distribution is a single binary with no dependency management, no virtualenv, no `--break-system-packages`. It is installed with `brew install fieldnotes` or a one-line curl, which matters for open-source adoption and for the blog narrative.

### Python Worker (`fieldnotes-worker`)

The worker handles all ML-heavy, ecosystem-dependent work:

- Text chunking
- Embedding via Ollama (`nomic-embed-text`)
- Entity and triple extraction (LLM)
- Vision processing
- Entity resolution (`rapidfuzz`)
- Neo4j + Qdrant writes
- UMAP + HDBSCAN clustering (weekly cron)
- LLM cluster labeling

Python is non-negotiable here. `hdbscan` and `umap-learn` are Python-only with no serious Go equivalents. LangChain's NL→Cypher chain, the Neo4j Python driver, and `qdrant-client` are all Python-first. `mlx-lm` for fine-tuning is Python-only. The Go daemon dispatches ingest events to the worker over a local HTTP queue; the worker processes them and writes results to Neo4j and Qdrant directly.

### Communication

For POC: the daemon POSTs ingest events to a local HTTP endpoint on the worker (`http://localhost:4242/ingest`). For production: a lightweight Redis queue between the two, allowing the worker to scale independently and batch events during bulk indexing without blocking the watcher.

### POC Path

**For the initial POC, the worker runs standalone with Python source shims.** `worker/worker/sources/` contains lightweight Python implementations of the same watching and polling logic that will eventually live in the Go daemon — `watchdog` for files, direct Gmail API calls for email. They emit `IngestEvent` dicts to the worker's internal queue rather than over HTTP. Every other component (`parsers/`, `pipeline/`, `clustering/`, `query/`) is identical between Phase 1 and the target architecture. The shims are the only thing replaced when the Go daemon is built in Phase 4.

The Go `daemon/` skeleton — `Source` interface, `IngestEvent` struct, registry — is committed from day one even before any Go adapters are written. This keeps the contract stable and means Phase 4 is a drop-in replacement rather than a design-time decision made under pressure.

```
Phase 1 (shims):  Python sources + Python worker — fast to build, validates the full pipeline
Phase 4 (target): Go daemon + Python worker — right tool for each job, single binary distribution
```

---

## Model Provider Architecture

Fieldnotes uses multiple models for different pipeline roles, each with different latency, quality, and cost requirements. The model system is designed with the same plugin philosophy as the source adapters: providers are registered implementations of a common interface, models are named references into those providers, and pipeline roles are mapped to model names in config. Swapping a role from a local Ollama model to a Claude API call — or mixing providers per role — is a config change, not a code change.

### The Three-Layer Config

The model configuration is split into three explicit layers, each with a distinct responsibility:

```toml
# Layer 1: Provider instances
# Define the connection parameters for each backend.
# Multiple instances of the same provider type are supported
# (e.g. two Ollama endpoints on different machines).

[modelproviders.local-ollama]
type = "ollama"
url  = "http://localhost:11434"

[modelproviders.anthropic]
type    = "anthropic"
api_key = ""          # or ANTHROPIC_API_KEY env var

[modelproviders.openai]
type    = "openai"
api_key = ""          # or OPENAI_API_KEY env var

[modelproviders.remote-ollama]
type = "ollama"
url  = "http://192.168.1.50:11434"   # second machine on local network


# Layer 2: Named model definitions
# Each entry binds a human-readable alias to a provider instance + model string.
# These are the names referenced in [models.roles].

[models.qwen-27b]
provider  = "local-ollama"
model     = "qwen3.5:27b"

[models.qwen-9b]
provider  = "local-ollama"
model     = "qwen3.5:9b"

[models.qwen-72b]
provider  = "local-ollama"
model     = "qwen3.5:72b"

[models.qwen-moe]
provider  = "local-ollama"
model     = "qwen3.5:122b-a10b"

[models.nomic-embed]
provider  = "local-ollama"
model     = "nomic-embed-text"

[models.haiku]
provider  = "anthropic"
model     = "claude-haiku-4-5"

[models.sonnet]
provider  = "anthropic"
model     = "claude-sonnet-4-6"

[models.opus]
provider  = "anthropic"
model     = "claude-opus-4-6"

[models.gpt-mini]
provider  = "openai"
model     = "gpt-5-mini"

[models.gpt-flagship]
provider  = "openai"
model     = "gpt-5.4"


# Layer 3: Role assignments
# Map each pipeline role to a named model (and optional fallback).
# This is the only thing that changes when switching between
# a local-first setup and a cloud-first setup.

[models.roles]
embed          = "nomic-embed"    # always-on embedding — local only recommended
extract        = "qwen-27b"       # hot path: every ingested chunk
extract_fallback = "qwen-9b"      # bulk initial indexing, or if primary is slow
vision         = "qwen-9b"        # async image queue — multimodal required
cluster_label  = "qwen-72b"       # weekly background cron
query          = "qwen-moe"       # interactive queries and agent tool calls
```

Switching to cloud models for everything is a roles-only change:

```toml
[models.roles]
embed          = "nomic-embed"    # embedding stays local — no cloud embed model needed
extract        = "haiku"
extract_fallback = "sonnet"
vision         = "haiku"
cluster_label  = "opus"
query          = "opus"
```

Per-role provider mixing is a first-class pattern — not a workaround:

```toml
[models.roles]
embed          = "nomic-embed"    # local
extract        = "haiku"          # Anthropic — cost-optimised
vision         = "gpt-mini"       # OpenAI — cost-optimised
cluster_label  = "opus"           # Anthropic — quality
query          = "opus"           # Anthropic — 1M context
```

### The Provider Interface

Each provider backend implements a common `ModelProvider` protocol. The pipeline calls `complete()` or `embed()` on a resolved `Model` instance — it has no knowledge of whether the underlying transport is Ollama's local HTTP API, Anthropic's Messages API, or OpenAI's chat completions endpoint.

```python
# worker/worker/models/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class CompletionRequest:
    """Normalised input for any text completion call."""
    system:      str
    messages:    list[dict[str, Any]]   # [{"role": "user"|"assistant", "content": "..."}]
    tools:       list[dict] | None = None   # for function calling / tool use
    max_tokens:  int = 1024
    temperature: float = 0.0            # 0.0 = deterministic; extraction always uses 0.0

@dataclass
class CompletionResponse:
    """Normalised output from any text completion call."""
    text:          str
    tool_calls:    list[dict] | None = None
    input_tokens:  int = 0
    output_tokens: int = 0
    cached_tokens: int = 0              # prompt cache hits (Anthropic / OpenAI)

@dataclass
class EmbedRequest:
    texts: list[str]

@dataclass
class EmbedResponse:
    vectors:      list[list[float]]
    model:        str
    input_tokens: int = 0


class ModelProvider(ABC):
    """Base class for all model provider backends."""

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Stable identifier — matches [modelproviders.<name>] type field."""
        ...

    @abstractmethod
    def configure(self, cfg: dict[str, Any]) -> None:
        """Receive the provider's config section on startup."""
        ...

    @abstractmethod
    def complete(self, model: str, req: CompletionRequest) -> CompletionResponse:
        """Run a chat completion. model is the raw model string (e.g. 'qwen3.5:27b')."""
        ...

    def embed(self, model: str, req: EmbedRequest) -> EmbedResponse:
        """Run a batch embedding call. Not all providers support this."""
        raise NotImplementedError(f"{self.provider_type} does not support embedding")
```

### The Provider Registry

The registry mirrors the source adapter pattern exactly — providers self-register via a decorator, and the config loader resolves them by type string.

```python
# worker/worker/models/registry.py

from typing import Type
from .base import ModelProvider

_registry: dict[str, Type[ModelProvider]] = {}

def register(cls: Type[ModelProvider]) -> Type[ModelProvider]:
    instance = cls()
    _registry[instance.provider_type] = cls
    return cls

def get_provider(provider_type: str) -> Type[ModelProvider]:
    cls = _registry.get(provider_type)
    if cls is None:
        raise ValueError(
            f"No model provider registered for type={provider_type!r}. "
            f"Known types: {sorted(_registry)}"
        )
    return cls
```

### The Model Registry

At startup, the config loader builds two maps: one from provider instance name to configured `ModelProvider`, and one from model alias to a `ResolvedModel` that holds the provider instance and raw model string. The pipeline only ever interacts with `ResolvedModel` — it does not know which provider is underneath.

```python
# worker/worker/models/resolver.py

from dataclasses import dataclass
from .base import ModelProvider, CompletionRequest, CompletionResponse, EmbedRequest, EmbedResponse

@dataclass
class ResolvedModel:
    """A model alias fully resolved to a provider instance + raw model string."""
    alias:    str
    model:    str           # raw string passed to the provider API
    provider: ModelProvider

    def complete(self, req: CompletionRequest) -> CompletionResponse:
        return self.provider.complete(self.model, req)

    def embed(self, req: EmbedRequest) -> EmbedResponse:
        return self.provider.embed(self.model, req)


class ModelRegistry:
    """Resolves model aliases to ResolvedModel instances."""

    def __init__(self, cfg: dict):
        from .registry import get_provider

        # Build provider instances from [modelproviders.*]
        self._providers: dict[str, ModelProvider] = {}
        for name, pcfg in cfg.get("modelproviders", {}).items():
            provider_type = pcfg["type"]
            cls = get_provider(provider_type)
            instance = cls()
            instance.configure(pcfg)
            self._providers[name] = instance

        # Build model alias → ResolvedModel map from [models.*]
        self._models: dict[str, ResolvedModel] = {}
        for alias, mcfg in cfg.get("models", {}).items():
            if alias == "roles":
                continue  # skip the roles table
            provider_name = mcfg["provider"]
            self._models[alias] = ResolvedModel(
                alias=alias,
                model=mcfg["model"],
                provider=self._providers[provider_name],
            )

        # Build role → ResolvedModel map from [models.roles]
        self._roles: dict[str, ResolvedModel] = {}
        for role, alias in cfg.get("models", {}).get("roles", {}).items():
            self._roles[role] = self._models[alias]

    def for_role(self, role: str) -> ResolvedModel:
        m = self._roles.get(role)
        if m is None:
            raise ValueError(f"No model assigned to role {role!r}")
        return m

    def by_alias(self, alias: str) -> ResolvedModel:
        m = self._models.get(alias)
        if m is None:
            raise ValueError(f"Unknown model alias {alias!r}")
        return m
```

The pipeline accesses models exclusively through `ModelRegistry.for_role()`:

```python
# worker/worker/pipeline/extractor.py

class Extractor:
    def __init__(self, registry: ModelRegistry):
        self._model = registry.for_role("extract")
        self._fallback = registry.for_role("extract_fallback")

    def extract(self, chunk: str) -> ExtractionResult:
        req = CompletionRequest(
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": chunk}],
            tools=[EXTRACTION_TOOL_SCHEMA],
            temperature=0.0,
        )
        try:
            resp = self._model.complete(req)
            return self._parse(resp)
        except (JSONDecodeError, ValidationError):
            # automatic fallback on malformed output
            resp = self._fallback.complete(req)
            return self._parse(resp)
```

### Built-in Provider Implementations

#### Ollama

Calls Ollama's local HTTP API. Supports both `complete()` and `embed()`. Uses the `/api/chat` endpoint for completions and `/api/embeddings` for vectors. Tool use is supported via Ollama's tool call format (available for models that support it).

```python
# worker/worker/models/providers/ollama.py

from ..registry import register
from ..base import ModelProvider, CompletionRequest, CompletionResponse, EmbedRequest, EmbedResponse

@register
class OllamaProvider(ModelProvider):
    provider_type = "ollama"

    def configure(self, cfg: dict) -> None:
        self._base_url = cfg.get("url", "http://localhost:11434").rstrip("/")

    def complete(self, model: str, req: CompletionRequest) -> CompletionResponse:
        payload = {
            "model":    model,
            "messages": [{"role": "system", "content": req.system}] + req.messages,
            "stream":   False,
            "options":  {"temperature": req.temperature},
        }
        if req.tools:
            payload["tools"] = req.tools

        r = httpx.post(f"{self._base_url}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        msg = data["message"]
        return CompletionResponse(
            text=msg.get("content", ""),
            tool_calls=msg.get("tool_calls"),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )

    def embed(self, model: str, req: EmbedRequest) -> EmbedResponse:
        r = httpx.post(
            f"{self._base_url}/api/embeddings",
            json={"model": model, "prompt": req.texts[0]},  # Ollama: one at a time
            timeout=30,
        )
        r.raise_for_status()
        return EmbedResponse(vectors=[r.json()["embedding"]], model=model)
```

#### Anthropic

Calls the Anthropic Messages API. Supports `complete()` only — embedding is not available from Anthropic. Handles tool use natively via the `tools` parameter. Extracts prompt cache token counts from the response for cost tracking.

```python
# worker/worker/models/providers/anthropic.py

@register
class AnthropicProvider(ModelProvider):
    provider_type = "anthropic"

    def configure(self, cfg: dict) -> None:
        import anthropic
        api_key = cfg.get("api_key") or os.environ["ANTHROPIC_API_KEY"]
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, model: str, req: CompletionRequest) -> CompletionResponse:
        kwargs = dict(
            model=model,
            max_tokens=req.max_tokens,
            system=req.system,
            messages=req.messages,
            temperature=req.temperature,
        )
        if req.tools:
            kwargs["tools"] = req.tools

        msg = self._client.messages.create(**kwargs)

        text = next((b.text for b in msg.content if b.type == "text"), "")
        tool_calls = [
            {"name": b.name, "input": b.input}
            for b in msg.content if b.type == "tool_use"
        ]
        usage = msg.usage
        return CompletionResponse(
            text=text,
            tool_calls=tool_calls or None,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=getattr(usage, "cache_read_input_tokens", 0),
        )
```

#### OpenAI

Calls the OpenAI Chat Completions API. Supports `complete()` and `embed()` (via the embeddings endpoint). Tool use via the `tools` parameter.

```python
# worker/worker/models/providers/openai.py

@register
class OpenAIProvider(ModelProvider):
    provider_type = "openai"

    def configure(self, cfg: dict) -> None:
        import openai
        api_key = cfg.get("api_key") or os.environ["OPENAI_API_KEY"]
        self._client = openai.OpenAI(api_key=api_key)

    def complete(self, model: str, req: CompletionRequest) -> CompletionResponse:
        messages = [{"role": "system", "content": req.system}] + req.messages
        kwargs = dict(model=model, messages=messages, temperature=req.temperature)
        if req.tools:
            kwargs["tools"] = req.tools

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        tool_calls = None
        if choice.tool_calls:
            tool_calls = [
                {"name": tc.function.name, "input": json.loads(tc.function.arguments)}
                for tc in choice.tool_calls
            ]
        return CompletionResponse(
            text=choice.content or "",
            tool_calls=tool_calls,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
        )

    def embed(self, model: str, req: EmbedRequest) -> EmbedResponse:
        resp = self._client.embeddings.create(model=model, input=req.texts)
        return EmbedResponse(
            vectors=[d.embedding for d in resp.data],
            model=model,
            input_tokens=resp.usage.prompt_tokens,
        )
```

### Repository Layout for Models

```
worker/
└── worker/
    └── models/
        ├── base.py                     # ModelProvider, CompletionRequest/Response, etc.
        ├── registry.py                 # @register decorator + get_provider()
        ├── resolver.py                 # ResolvedModel, ModelRegistry
        ├── __init__.py                 # imports all providers to trigger registration
        └── providers/
            ├── ollama.py               # OllamaProvider
            ├── anthropic.py            # AnthropicProvider
            └── openai.py               # OpenAIProvider
```

### Hardware & Model Recommendations

The plugin architecture is provider-agnostic, but the following configurations are tested and recommended. Which one to use depends entirely on hardware availability and cost tolerance.

**Local-first (M5 Max, 128GB unified memory)**

Unified memory on Apple Silicon is shared between CPU and GPU, making the full 128GB available for model weights. At Q4 quantization, these models never need to be loaded simultaneously — peak usage stays well under the hardware ceiling.

| Role alias | Model | Provider | Size at Q4 | Est. t/s |
|---|---|---|---|---|
| `embed` | `nomic-embed-text` | ollama | ~0.3GB | — |
| `extract` | `qwen3.5:27b` | ollama | ~16GB | 40–60 |
| `extract_fallback` | `qwen3.5:9b` | ollama | ~6GB | 80–120 |
| `vision` | `qwen3.5:9b` | ollama | ~6GB | 2–5s/img |
| `cluster_label` | `qwen3.5:72b` | ollama | ~45GB | 20–35 |
| `query` | `qwen3.5:122b-a10b` | ollama | ~65GB | 15–25 |

The `122b-a10b` MoE model is well-suited to this hardware: large total parameter count for broad knowledge, sparse activation (10B active params per pass) for practical inference speed. It fits in 128GB with ~60GB headroom for the OS, Ollama overhead, and Docker containers.

**Fallback (32–64GB):** `extract` → `qwen3.5:9b`, `cluster_label` → `qwen3.5:27b`, `query` → `qwen3.5:27b`.

**Cloud-first (Anthropic)**

| Role alias | Model alias | Raw model | Notes |
|---|---|---|---|
| `embed` | `nomic-embed` | `nomic-embed-text` via Ollama | Stay local — no cloud embed needed |
| `extract` | `haiku` | `claude-haiku-4-5` | $1/$5 per MTok; 3x cheaper than Sonnet |
| `extract_fallback` | `sonnet` | `claude-sonnet-4-6` | Step up if JSON compliance degrades |
| `vision` | `haiku` | `claude-haiku-4-5` | Natively multimodal, cost-efficient |
| `cluster_label` | `opus` | `claude-opus-4-6` | Weekly cron — quality over cost |
| `query` | `opus` | `claude-opus-4-6` | 1M context window; best multi-hop reasoning |

At moderate personal use (500 chunks/day, weekly clustering, 20 queries/day), this configuration costs approximately $14–27/month. Prompt caching on the shared extraction system prompt drops effective extraction costs to under $0.50/month. Switching `extract` from Haiku to Sonnet roughly triples that line item.

**Mixed (local embed + extraction, cloud for quality tasks)**

```toml
[models.roles]
embed          = "nomic-embed"   # local
extract        = "qwen-27b"      # local — saves ~$9/mo vs Haiku at this volume
extract_fallback = "qwen-9b"     # local
vision         = "haiku"         # cloud — Haiku handles vision well at low cost
cluster_label  = "opus"          # cloud — quality matters, runs once a week
query          = "opus"          # cloud — best reasoning for interactive use
```

This is the recommended configuration for M5 Max users who want the best of both: fast local extraction during the day, frontier-quality responses for interactive queries and weekly labeling.





```
┌───────────────────────────────────────────────────────────────────────┐
│                              Sources                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │
│  │ Local Files │ │  Obsidian   │ │    Gmail    │ │     Git     │    │
│  │ (watchdog)  │ │   Vaults    │ │  (MCP/API)  │ │    Repos    │    │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘    │
└─────────┼───────────────┼───────────────┼───────────────┼────────────┘
          │               │               │               │
          └───────────────▼───────────────▼───────────────┘
                      Ingestion Pipeline
           ┌──────────────────────────────────────┐
           │  1. Parse & chunk                    │
           │  2. Embed  (nomic-embed-text/Ollama)  │
           │  3. Extract entities & triples (LLM)  │
           │  4. Resolve entities (fuzzy dedup)    │
           │  5. Write to stores                  │
           └────────────┬─────────────┬───────────┘
                        │             │
               ┌────────▼────┐  ┌─────▼──────┐
               │   Neo4j     │  │   Qdrant   │
               │  (graph)    │  │  (vectors) │
               └────────┬────┘  └─────┬──────┘
                        │             │
                        └──────┬──────┘
                         Query Layer
                   ┌────────────────────┐
                   │  - NL → Cypher     │
                   │  - Vector search   │
                   │  - Hybrid merge    │
                   │  - MCP server      │
                   └────────────────────┘
```

---

## Data Model

### Node Types

| Label | Key Properties | Description |
|---|---|---|
| `File` | `path`, `name`, `ext`, `modified_at`, `sha256` | A local file on disk |
| `Image` | `path`, `name`, `ext`, `modified_at`, `sha256`, `vision_processed` | An image file — carries a vision-extracted text description |
| `Vault` | `path`, `name` | An Obsidian vault root |
| `Email` | `message_id`, `subject`, `date`, `snippet` | A single email message |
| `Thread` | `thread_id`, `subject`, `last_date` | A Gmail thread |
| `Repository` | `name`, `path`, `remote_url` | A git repository |
| `Commit` | `sha`, `message`, `date` | A git commit |
| `Person` | `email`, `name` | A person inferred from email headers |
| `Entity` | `name`, `type`, `confidence` | A named entity (project, concept, technology, org) |
| `Topic` | `name`, `description`, `source` | A topic label — `source` is either `"cluster"` (HDBSCAN-derived) or `"user"` (Obsidian #tag) |
| `Chunk` | `id`, `text`, `source_id` | A text chunk with a Qdrant vector reference |

### Edge Types

| Relationship | From → To | Description |
|---|---|---|
| `SENT` | Person → Email | Email authorship |
| `TO` | Email → Person | Email recipient |
| `PART_OF` | Email → Thread | Thread membership |
| `TAGGED` | Thread/File → Topic | Cluster-assigned topic (`source: "cluster"`) |
| `TAGGED_BY_USER` | File → Topic | User-declared Obsidian #tag (`source: "user"`) |
| `DEPICTS` | Image → Entity | Vision-extracted entity visible in an image |
| `ATTACHED_TO` | Image → File | Image embedded in or co-located with a note |
| `IN_VAULT` | File → Vault | Vault membership |
| `MENTIONS` | File/Email/Commit → Entity | Named entity mention |
| `CONTAINS` | Repository → File | Repo membership |
| `AUTHORED` | Commit → File | Files changed in commit |
| `DEPENDS_ON` | Repository → Entity | Package dependency |
| `RELATED_TO` | Entity → Entity | Co-occurrence relationship |
| `HAS_CHUNK` | File/Email → Chunk | Chunked text reference |
| `SAME_AS` | Entity → Entity | Entity deduplication link |

### Example Subgraph

```
(Markus:Person)-[:SENT]->(:Email {subject: "Daytona Tier 3 access request"})
    -[:MENTIONS]->(:Entity {name: "Daytona", type: "Technology"})
    <-[:MENTIONS]-(:File {name: "daytona-integration.md"})
    -[:LINKS_TO]->(:File {name: "sandbox-architecture.md"})   ← Obsidian wikilink
    -[:TAGGED_BY_USER]->(:Topic {name: "platform-engineering", source: "user"})

(:File {name: "daytona-integration.md"})
    -[:MENTIONS]->(gastown:Repository {name: "GasTown"})
    -[:DEPENDS_ON]->(:Entity {name: "daytona-sdk"})
```

This subgraph connects an outbound email to a local design doc, through an Obsidian wikilink to a related architecture note, to a repository dependency — across four source types, through shared entity nodes. The `LINKS_TO` edge is derived directly from the user's own `[[wikilink]]`, requiring no LLM inference.

---

## Source Adapters

### Plugin Architecture

Every source in Fieldnotes — files, Obsidian vaults, Gmail, Git repositories, and any future adapter — plugs into a single well-defined interface. The rule is simple: adding a new source type should require writing exactly one new adapter, touching zero existing code, and dropping one new entry in `config.toml`. The core pipeline, the entity extractor, the Neo4j writer, and the MCP query layer are completely unaware of where content came from.

This is not over-engineering for a personal tool. It is the only design that makes the project worth building in the open. The adapters are where the community contribution surface lives — someone writes a Slack adapter, a Google Docs adapter, a Raindrop.io adapter, and it just works, because the contract is narrow and the pipeline is source-agnostic.

#### The Contract

The adapter system is split across both halves of the architecture. The Go daemon owns the **source interface** — the polling/watching side that detects changes and emits events. The Python worker owns the **parser interface** — the content-extraction side that transforms raw bytes into structured `ParsedDocument` objects. Both sides implement a simple interface; both sides are pluggable independently.

```
┌──────────────────────────────────────────────────────────────────┐
│  Go Daemon                                                       │
│                                                                  │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐  │
│  │  FileSource    │    │ ObsidianSource │    │  GmailSource   │  │
│  │  (fsnotify)    │    │  (fsnotify +   │    │  (OAuth poll)  │  │
│  │                │    │  vault detect) │    │                │  │
│  └───────┬────────┘    └───────┬────────┘    └───────┬────────┘  │
│          └──────────────┬──────┘─────────────────────┘           │
│                         │  Source interface                       │
│                         ▼                                         │
│                   IngestEvent{}          ─────────────────────►  │
│                   (JSON over HTTP)          to Python worker      │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  Python Worker                                                   │
│                                                                  │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐  │
│  │  FileParser    │    │ObsidianParser  │    │  EmailParser   │  │
│  │  (.md,.pdf,..) │    │  (frontmatter, │    │ (HTML strip,   │  │
│  │                │    │  wikilinks,    │    │  Person nodes) │  │
│  │                │    │  tags)         │    │                │  │
│  └───────┬────────┘    └───────┬────────┘    └───────┬────────┘  │
│          └──────────────┬──────┘─────────────────────┘           │
│                         │  Parser interface                       │
│                         ▼                                         │
│                  ParsedDocument{}                                 │
│                  → chunker → embedder → extractor → writer        │
└──────────────────────────────────────────────────────────────────┘
```

#### Go: The Source Interface

Each adapter in the daemon implements the `Source` interface. The interface is intentionally minimal — a source knows how to start watching, how to stop, and how to describe itself. Everything else (event routing, queue management, retry logic) is handled by the daemon's `Dispatcher`.

```go
// daemon/internal/sources/source.go

// Source is the interface every adapter must implement.
// It is responsible for detecting changes in an external system
// and emitting IngestEvents to the dispatcher.
type Source interface {
    // Name returns the stable identifier for this source type.
    // Used as the "source_type" field in IngestEvent and in config.toml section names.
    Name() string

    // Configure initialises the source from its config section.
    // Called once at startup before Start().
    Configure(cfg map[string]any) error

    // Start begins watching/polling and emits events to the provided channel.
    // Must be safe to call in a goroutine. Blocks until ctx is cancelled.
    Start(ctx context.Context, events chan<- IngestEvent) error

    // Healthcheck returns a non-nil error if the source is unhealthy
    // (e.g. lost OAuth token, missing directory). Called by the status API.
    Healthcheck() error
}
```

The `IngestEvent` is the single envelope passed from every source to the dispatcher, and from the dispatcher to the Python worker. Its shape is fixed; adapters fill the fields that are relevant to them and leave the rest empty.

```go
// daemon/internal/sources/event.go

type IngestEvent struct {
    // Identity
    ID         string    `json:"id"`          // UUIDv4, set by dispatcher
    SourceType string    `json:"source_type"` // matches Source.Name()
    SourceID   string    `json:"source_id"`   // stable external identifier (path, message_id, etc.)
    Operation  Operation `json:"operation"`   // Created | Modified | Deleted

    // Content
    // For text-based sources: Text is populated, RawBytes is nil.
    // For binary sources (images, PDFs): RawBytes is populated, Text is empty.
    // The Python parser decides how to handle each content type.
    Text      string `json:"text,omitempty"`
    RawBytes  []byte `json:"raw_bytes,omitempty"`  // base64 in JSON
    MimeType  string `json:"mime_type,omitempty"`

    // Source-specific structured metadata.
    // Adapters may include anything here — the Python parser for this source type
    // knows how to interpret it. The core pipeline ignores unknown fields.
    Meta map[string]any `json:"meta,omitempty"`

    // Timestamps
    SourceModifiedAt time.Time `json:"source_modified_at"`
    EnqueuedAt       time.Time `json:"enqueued_at"`
}

type Operation string

const (
    OperationCreated  Operation = "created"
    OperationModified Operation = "modified"
    OperationDeleted  Operation = "deleted"
)
```

#### Go: The Registry

Sources register themselves at startup. There is no config-file-driven adapter discovery — that would require dynamic loading. Instead, adapters are compiled in, and the registry maps config section names to constructor functions. Adding a new adapter means writing its `Source` implementation and adding one line to the registry.

```go
// daemon/internal/sources/registry.go

// SourceFactory is a constructor function for a Source implementation.
type SourceFactory func() Source

var registry = map[string]SourceFactory{}

// Register adds a source factory to the registry.
// Call from an init() function in the adapter's package.
func Register(name string, factory SourceFactory) {
    if _, exists := registry[name]; exists {
        panic(fmt.Sprintf("source already registered: %s", name))
    }
    registry[name] = factory
}

// Build constructs all sources enabled in the config.
func Build(cfg *Config) ([]Source, error) {
    var sources []Source
    for name, sectionCfg := range cfg.Sources {
        factory, ok := registry[name]
        if !ok {
            return nil, fmt.Errorf("unknown source type %q in config — is the adapter compiled in?", name)
        }
        s := factory()
        if err := s.Configure(sectionCfg); err != nil {
            return nil, fmt.Errorf("configuring source %q: %w", name, err)
        }
        sources = append(sources, s)
    }
    return sources, nil
}
```

Each adapter registers itself in its `init()` function:

```go
// daemon/internal/sources/files/adapter.go

func init() {
    sources.Register("files", func() sources.Source { return &FileSource{} })
}
```

The `main.go` entrypoint imports each adapter package for its side effect:

```go
// daemon/cmd/fieldnotes/main.go

import (
    _ "github.com/fieldnotes/daemon/internal/sources/files"
    _ "github.com/fieldnotes/daemon/internal/sources/obsidian"
    _ "github.com/fieldnotes/daemon/internal/sources/gmail"
    _ "github.com/fieldnotes/daemon/internal/sources/repositories"
)
```

A community adapter ships as a separate Go module. The user forks `fieldnotes-daemon`, adds the import, and rebuilds. In the future, a plugin host (e.g. hashicorp/go-plugin) could enable out-of-process adapters loaded from binaries — but that is a Phase 5 consideration.

#### Python: The Parser Interface

On the worker side, each source type has a corresponding `Parser` class. The parser receives an `IngestEvent` (deserialised from the HTTP POST) and returns a `ParsedDocument`. The rest of the pipeline — chunker, embedder, extractor, writer — operates exclusively on `ParsedDocument` objects and is completely source-agnostic.

```python
# worker/worker/parsers/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass
class GraphHint:
    """A pre-extracted graph fact that bypasses the LLM extractor.

    Used when the adapter has high-confidence structured data (Obsidian wikilinks,
    email headers, git commit metadata) that does not need LLM inference.
    Hints are written to Neo4j directly after the write step, before entity resolution.
    """
    subject_id: str           # source_id of the subject node
    subject_label: str        # Neo4j label: "File", "Email", "Person", etc.
    predicate: str            # relationship type: "LINKS_TO", "SENT", etc.
    object_id: str            # source_id of the object node
    object_label: str
    object_props: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0   # 1.0 = certain (derived from structure, not LLM)

@dataclass
class ParsedDocument:
    """The normalised output of any adapter parser.

    Every parser must produce one of these. The pipeline beyond this point
    is entirely source-agnostic — it only sees ParsedDocuments.
    """
    # Identity
    source_type: str           # matches Source.Name() in the Go daemon
    source_id: str             # stable external identifier (path, message_id, etc.)
    operation: str             # "created" | "modified" | "deleted"

    # Content for the text pipeline (chunker → embedder → LLM extractor)
    text: str                  # plain text body; empty for image-only documents
    mime_type: str = "text/plain"

    # Node properties written directly to Neo4j (no LLM involved)
    # Keys become node properties; the writer merges these on upsert.
    node_label: str = "File"
    node_props: dict[str, Any] = field(default_factory=dict)

    # Pre-extracted graph facts (high-confidence, bypass LLM extractor)
    graph_hints: list[GraphHint] = field(default_factory=list)

    # Binary content for the vision pipeline (images only)
    image_bytes: bytes | None = None

    # Source metadata passed through to Qdrant payload
    source_metadata: dict[str, Any] = field(default_factory=dict)


class BaseParser(ABC):
    """Base class for all adapter parsers."""

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Must match the Go Source.Name() for this adapter."""
        ...

    @abstractmethod
    def parse(self, event: dict[str, Any]) -> list[ParsedDocument]:
        """Transform a raw IngestEvent dict into one or more ParsedDocuments.

        Returns a list because some events produce multiple documents —
        an Obsidian note with embedded images yields one text document
        and N image documents. An email thread yields one document per message.
        """
        ...

    def configure(self, cfg: dict[str, Any]) -> None:
        """Optional: receive the source's config section on startup."""
        pass
```

#### Python: The Registry

The Python registry mirrors the Go registry. Parsers register themselves, the worker resolves the correct parser for each incoming event by `source_type`, and unknown source types are rejected with a clear error rather than silently dropped.

```python
# worker/worker/parsers/registry.py

from typing import Type
from .base import BaseParser

_registry: dict[str, Type[BaseParser]] = {}

def register(cls: Type[BaseParser]) -> Type[BaseParser]:
    """Class decorator that registers a parser. Use on every BaseParser subclass."""
    instance = cls()
    _registry[instance.source_type] = cls
    return cls

def get(source_type: str) -> BaseParser:
    cls = _registry.get(source_type)
    if cls is None:
        raise ValueError(
            f"No parser registered for source_type={source_type!r}. "
            f"Known types: {sorted(_registry)}"
        )
    return cls()
```

Parsers register themselves with the decorator:

```python
# worker/worker/parsers/files.py

from .registry import register
from .base import BaseParser, ParsedDocument

@register
class FileParser(BaseParser):
    source_type = "files"

    def parse(self, event: dict) -> list[ParsedDocument]:
        ...
```

The worker's startup routine imports all parser modules to trigger registration:

```python
# worker/worker/parsers/__init__.py

from . import files      # noqa: F401
from . import obsidian   # noqa: F401
from . import gmail      # noqa: F401
from . import images     # noqa: F401
from . import repositories  # noqa: F401
```

#### Configuration Convention

Every adapter follows the same `config.toml` pattern. The section name is the adapter's stable `name` / `source_type` string. The Go daemon uses the section name to look up the registered factory; the Python worker uses the `source_type` field on each event to look up the registered parser.

```toml
# Built-in adapters use [sources.<n>]
[sources.files]
watch_paths = ["~/notes"]

[sources.obsidian]
vault_paths = ["~/notes/work"]

[sources.gmail]
poll_interval_seconds = 300

# A community adapter works identically — no special registration needed
[sources.google_docs]
credentials_file = "~/.fieldnotes/google_credentials.json"
shared_drives = ["Engineering", "Product"]
poll_interval_seconds = 600
```

#### What an Adapter Must and Must Not Do

**Must do:**
- Implement `Source` (Go) and `BaseParser` (Python)
- Emit one `IngestEvent` per logical document (one file, one email, one doc)
- Populate `source_id` with a stable, globally unique identifier that does not change when content changes (a file path, a message ID, a document ID — not a hash)
- Set `operation` correctly so the pipeline can handle deletions without re-processing

**Must not do:**
- Call the LLM extractor directly — that is the pipeline's job
- Write to Neo4j or Qdrant — that is the writer's job
- Parse content that belongs to another adapter (the file adapter does not special-case `.md` files in an Obsidian vault — the Obsidian adapter handles those)
- Fail silently — log the error and emit nothing rather than emitting a partial event

**May do:**
- Populate `graph_hints` with high-confidence structural facts (wikilinks, email headers, commit metadata) that should bypass the LLM extractor entirely
- Return multiple `ParsedDocument` objects from a single event (email threads, notes with embedded images)
- Implement its own deduplication (e.g. skipping re-indexing if SHA256 hasn't changed)
- Add arbitrary fields to `node_props` — the writer uses `MERGE` on `source_id` and sets all provided properties

---

### Built-in Adapters

#### File Adapter

Watches one or more configured directory trees using `fsnotify` (Go). On `created` or `modified` events, the file is read and emitted as an `IngestEvent`. On `deleted`, a deletion event is emitted and the writer removes the corresponding Neo4j node and Qdrant vectors.

The Go side handles watching and emitting. The Python `FileParser` handles content extraction — text files are read directly, PDFs are processed with `pymupdf`, HTML is stripped with `beautifulsoup4`.

```go
// daemon/internal/sources/files/adapter.go

type FileSource struct {
    watchPaths        []string
    includeExtensions map[string]bool
    excludePatterns   []glob.Glob
}

func (s *FileSource) Name() string { return "files" }

func (s *FileSource) Start(ctx context.Context, events chan<- IngestEvent) error {
    watcher, _ := fsnotify.NewWatcher()
    for _, path := range s.watchPaths {
        filepath.WalkDir(path, func(p string, d fs.DirEntry, _ error) error {
            if d.IsDir() { watcher.Add(p) }
            return nil
        })
    }
    for {
        select {
        case event := <-watcher.Events:
            if s.shouldIndex(event.Name) {
                events <- s.buildEvent(event)
            }
        case <-ctx.Done():
            return nil
        }
    }
}
```

```python
# worker/worker/parsers/files.py

@register
class FileParser(BaseParser):
    source_type = "files"

    def parse(self, event: dict) -> list[ParsedDocument]:
        path = event["source_id"]
        mime = event.get("mime_type", "text/plain")

        if mime == "application/pdf":
            text = self._extract_pdf(event["raw_bytes"])
        elif mime.startswith("text/"):
            text = event["text"]
        else:
            return []  # unsupported format — skip silently

        return [ParsedDocument(
            source_type="files",
            source_id=path,
            operation=event["operation"],
            text=text,
            node_label="File",
            node_props={
                "path": path,
                "name": os.path.basename(path),
                "ext": os.path.splitext(path)[1],
                "modified_at": event["source_modified_at"],
                "sha256": event["meta"].get("sha256"),
            },
            source_metadata={"source_type": "file", "path": path},
        )]
```

**Allowlist configuration:**

```toml
[sources.files]
watch_paths = ["~/notes", "~/Documents"]
include_extensions = [".md", ".txt", ".pdf", ".html"]
exclude_patterns = ["node_modules/", ".git/", "*.log"]
```

#### Obsidian Adapter

Obsidian vaults are a first-class source type, distinct from generic file watching. The Go adapter detects vaults by the presence of a `.obsidian/` directory. The Python parser is vault-aware: it extracts three layers of structure that generic file parsing ignores — YAML frontmatter, `[[wikilinks]]`, and `#tags` — all of which bypass the LLM extractor entirely via `graph_hints`.

The vault detection runs once at startup and on new directory events — if a new `.obsidian/` directory appears under a watched path, the vault is registered automatically.

```go
// daemon/internal/sources/obsidian/adapter.go

type ObsidianSource struct {
    vaultPaths     []string
    detectedVaults map[string]VaultMeta
}

func (s *ObsidianSource) Name() string { return "obsidian" }

func (s *ObsidianSource) buildEvent(filePath string, op Operation) IngestEvent {
    vault := s.vaultForPath(filePath)
    return IngestEvent{
        SourceType: "obsidian",
        SourceID:   filePath,
        Operation:  op,
        Meta: map[string]any{
            "vault_path":    vault.RootPath,
            "vault_name":    vault.Name,
            "relative_path": strings.TrimPrefix(filePath, vault.RootPath+"/"),
        },
    }
}
```

```python
# worker/worker/parsers/obsidian.py

WIKILINK_RE = re.compile(r'\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]')
TAG_RE      = re.compile(r'(?<!\w)#([a-zA-Z][a-zA-Z0-9_/-]*)')
EMBED_RE    = re.compile(r'!\[\[([^\]]+\.(png|jpg|jpeg|webp|gif))\]\]', re.IGNORECASE)

@register
class ObsidianParser(BaseParser):
    source_type = "obsidian"

    def parse(self, event: dict) -> list[ParsedDocument]:
        path = event["source_id"]
        meta = event.get("meta", {})
        post = frontmatter.loads(event["text"])
        fm, body = post.metadata, post.content

        hints = []

        # Wikilinks → LINKS_TO edges (confidence 0.95, no LLM needed)
        for target in WIKILINK_RE.findall(body):
            target_path = self._resolve_wikilink(target, meta["vault_path"])
            hints.append(GraphHint(
                subject_id=path,        subject_label="File",
                predicate="LINKS_TO",
                object_id=target_path,  object_label="File",
                object_props={"path": target_path, "status": "unresolved"},
                confidence=0.95,
            ))

        # Tags → TAGGED_BY_USER edges (source: "user", never overwritten by clustering)
        for tag in TAG_RE.findall(body):
            hints.append(GraphHint(
                subject_id=path,              subject_label="File",
                predicate="TAGGED_BY_USER",
                object_id=f"topic:user:{tag}", object_label="Topic",
                object_props={"name": tag, "source": "user"},
                confidence=1.0,
            ))

        # Embedded images → separate ParsedDocuments for the vision pipeline
        image_docs = self._extract_image_docs(body, path, meta["vault_path"])

        node_props = {
            "path":       path,
            "name":       os.path.basename(path),
            "vault_path": meta["vault_path"],
            **self._extract_frontmatter_props(fm),
        }

        text_doc = ParsedDocument(
            source_type="obsidian",
            source_id=path,
            operation=event["operation"],
            text=body,
            node_label="File",
            node_props=node_props,
            graph_hints=hints,
            source_metadata={"source_type": "obsidian", "vault": meta["vault_name"]},
        )
        return [text_doc] + image_docs
```

The `graph_hints` for wikilinks and tags are written to Neo4j before entity resolution runs, so the user's own explicit graph structure is in place before LLM inference adds its lower-confidence inferences. Wikilink-sourced entity nodes act as anchors during resolution.

**Frontmatter as web clip detection:** The frontmatter parser detects pages captured via Obsidian Web Clipper by the presence of a `url` or `source_url` field, and sets `clipped: true` + `source_url` on the node. This enables Cypher queries against web-clipped content without any separate browser integration.

**Configuration:**

```toml
[sources.obsidian]
vault_paths = ["~/notes/personal", "~/notes/work"]
index_attachments = false
daily_notes_folder = "Daily"
```

#### Image Adapter

Images are not a source adapter in the traditional sense — they are a content type handled within other adapters (Obsidian embeds, files watched by the file adapter). The `ImageParser` is registered for `source_type = "image"` events, which are emitted by the Obsidian and File adapters when they encounter binary image files.

This separation means the vision pipeline is not coupled to any specific source. A Google Drive adapter can emit image events for `.png` files stored in Drive; the `ImageParser` handles them identically regardless of origin.

```python
# worker/worker/parsers/images.py

@register
class ImageParser(BaseParser):
    source_type = "image"

    def parse(self, event: dict) -> list[ParsedDocument]:
        return [ParsedDocument(
            source_type="image",
            source_id=event["source_id"],
            operation=event["operation"],
            text="",           # populated after vision extraction
            node_label="Image",
            node_props={
                "path":             event["source_id"],
                "sha256":           event["meta"].get("sha256"),
                "vision_processed": False,
                "parent_source_id": event["meta"].get("parent_source_id"),
            },
            image_bytes=base64.b64decode(event["raw_bytes"]),
            source_metadata={"source_type": "image"},
        )]
```

The pipeline detects `image_bytes is not None` and routes the document through the vision worker before the text chunker. After vision extraction, `text` is populated with the description and visible text, and the document proceeds through the standard pipeline.

#### Gmail Adapter

The Gmail adapter is the reference example of a **polling adapter** — it has no filesystem events to watch, so the Go source runs a ticker goroutine and compares against a persisted cursor to detect new content.

```go
// daemon/internal/sources/gmail/adapter.go

type GmailSource struct {
    pollInterval time.Duration
    maxInitial   int
    labelFilter  []string
    cursor       *Cursor  // persisted to ~/.fieldnotes/data/gmail_cursor.json
    client       *gmail.Service
}

func (s *GmailSource) Name() string { return "gmail" }

func (s *GmailSource) Start(ctx context.Context, events chan<- IngestEvent) error {
    s.backfill(ctx, events)   // initial run
    ticker := time.NewTicker(s.pollInterval)
    for {
        select {
        case <-ticker.C:
            s.poll(ctx, events)
        case <-ctx.Done():
            return nil
        }
    }
}
```

The Python `GmailParser` strips HTML, extracts sender/recipient relationships as `GraphHints` (confidence 1.0 — email headers are ground truth), and returns one `ParsedDocument` per message.

```python
# worker/worker/parsers/gmail.py

@register
class GmailParser(BaseParser):
    source_type = "gmail"

    def parse(self, event: dict) -> list[ParsedDocument]:
        meta = event["meta"]
        body = self._strip_html(event["text"])

        hints = [
            GraphHint(
                subject_id=meta["sender_email"], subject_label="Person",
                predicate="SENT",
                object_id=meta["message_id"],    object_label="Email",
                object_props={"message_id": meta["message_id"]},
                confidence=1.0,
            ),
            GraphHint(
                subject_id=meta["message_id"],   subject_label="Email",
                predicate="PART_OF",
                object_id=meta["thread_id"],     object_label="Thread",
                object_props={"thread_id": meta["thread_id"], "subject": meta["subject"]},
                confidence=1.0,
            ),
        ] + [
            GraphHint(
                subject_id=meta["message_id"], subject_label="Email",
                predicate="TO",
                object_id=recipient,           object_label="Person",
                object_props={"email": recipient},
                confidence=1.0,
            )
            for recipient in meta.get("recipients", [])
        ]

        return [ParsedDocument(
            source_type="gmail",
            source_id=meta["message_id"],
            operation=event["operation"],
            text=body,
            node_label="Email",
            node_props={
                "message_id": meta["message_id"],
                "subject":    meta["subject"],
                "date":       meta["date"],
            },
            graph_hints=hints,
            source_metadata={"source_type": "email", "thread_id": meta["thread_id"]},
        )]
```

```toml
[sources.gmail]
poll_interval_seconds = 300
max_initial_threads = 500
label_filter = []
```

#### Repository Adapter

Scans configured repository roots. Indexes only high-signal files; source code is excluded by default.

```toml
[sources.repositories]
repo_roots = ["~/code"]
include_patterns = [
    "README*", "CHANGELOG*", "CONTRIBUTING*",
    "docs/**/*.md", "*.toml", "ADR/**/*.md",
]
```

Git commit history is extracted via `gitpython`: the last N commits per repo are indexed as `Commit` nodes with their messages and changed file lists. For `.toml` files, only dependency sections are parsed — the values are written as `DEPENDS_ON` edges rather than free text chunks.

---

### Writing a New Adapter: Google Docs Example

The following is a complete walkthrough of what it takes to add a Google Docs adapter. It is the reference implementation a community contributor would follow.

**What it needs to do:** Poll the Google Drive API for documents in configured shared drives or folders. On new or modified documents, fetch content as plain text via the Docs export API. Emit one `IngestEvent` per document. Persist a cursor (last modified timestamp per document) to avoid re-indexing unchanged content.

**Step 1: Go source** — `daemon/internal/sources/googledocs/adapter.go`

```go
package googledocs

import (
    "context"
    "time"

    sources "github.com/fieldnotes/daemon/internal/sources"
    "google.golang.org/api/drive/v3"
)

func init() {
    sources.Register("google_docs", func() sources.Source { return &GoogleDocsSource{} })
}

type GoogleDocsSource struct {
    credentialsFile string
    sharedDrives    []string
    pollInterval    time.Duration
    cursor          map[string]time.Time  // doc_id → last indexed modified time
    driveClient     *drive.Service
}

func (s *GoogleDocsSource) Name() string { return "google_docs" }

func (s *GoogleDocsSource) Configure(cfg map[string]any) error {
    s.credentialsFile, _ = cfg["credentials_file"].(string)
    s.pollInterval = time.Duration(cfg["poll_interval_seconds"].(float64)) * time.Second
    // parse shared_drives, initialise OAuth clients...
    return nil
}

func (s *GoogleDocsSource) Start(ctx context.Context, events chan<- sources.IngestEvent) error {
    s.poll(ctx, events)  // initial backfill
    ticker := time.NewTicker(s.pollInterval)
    for {
        select {
        case <-ticker.C:
            s.poll(ctx, events)
        case <-ctx.Done():
            return nil
        }
    }
}

func (s *GoogleDocsSource) poll(ctx context.Context, events chan<- sources.IngestEvent) {
    for _, driveID := range s.sharedDrives {
        files, _ := s.driveClient.Files.List().
            Corpora("drive").DriveId(driveID).
            SupportsAllDrives(true).IncludeItemsFromAllDrives(true).
            Q("mimeType='application/vnd.google-apps.document'").
            Fields("files(id, name, modifiedTime, webViewLink)").
            Do()

        for _, f := range files.Files {
            modifiedAt, _ := time.Parse(time.RFC3339, f.ModifiedTime)
            if last, seen := s.cursor[f.Id]; seen && !modifiedAt.After(last) {
                continue  // unchanged since last poll
            }

            resp, _ := s.driveClient.Files.Export(f.Id, "text/plain").Download()
            text, _ := io.ReadAll(resp.Body)

            events <- sources.IngestEvent{
                SourceType:       "google_docs",
                SourceID:         f.Id,
                Operation:        sources.OperationModified,
                Text:             string(text),
                MimeType:         "text/plain",
                SourceModifiedAt: modifiedAt,
                Meta: map[string]any{
                    "title":         f.Name,
                    "web_view_link": f.WebViewLink,
                    "drive_id":      driveID,
                },
            }
            s.cursor[f.Id] = modifiedAt
        }
    }
}

func (s *GoogleDocsSource) Healthcheck() error {
    _, err := s.driveClient.About.Get().Fields("user").Do()
    return err
}
```

**Step 2: Python parser** — `worker/worker/parsers/googledocs.py`

```python
from .registry import register
from .base import BaseParser, ParsedDocument

@register
class GoogleDocsParser(BaseParser):
    source_type = "google_docs"

    def parse(self, event: dict) -> list[ParsedDocument]:
        meta = event.get("meta", {})
        return [ParsedDocument(
            source_type="google_docs",
            source_id=event["source_id"],
            operation=event["operation"],
            text=event["text"],
            node_label="File",        # reuses File node type — no schema migration needed
            node_props={
                "source_id":   event["source_id"],
                "name":        meta.get("title", "Untitled"),
                "source_url":  meta.get("web_view_link"),
                "modified_at": event["source_modified_at"],
                "drive_id":    meta.get("drive_id"),
                "clipped":     False,
            },
            source_metadata={
                "source_type": "google_docs",
                "drive_id":    meta.get("drive_id"),
            },
        )]
```

**Step 3: Register the imports**

In `daemon/cmd/fieldnotes/main.go`:
```go
_ "github.com/fieldnotes/daemon/internal/sources/googledocs"
```

In `worker/worker/parsers/__init__.py`:
```python
from . import googledocs  # noqa: F401
```

**Step 4: Config**

```toml
[sources.google_docs]
credentials_file = "~/.fieldnotes/google_credentials.json"
shared_drives = ["Engineering", "Product"]
poll_interval_seconds = 600
```

**That is everything.** The Google Docs adapter reuses `File` nodes in the graph with no schema migration. Its documents flow through the exact same chunker, embedder, and entity extractor as local files. They are immediately queryable via the same Cypher patterns and MCP tools as any other source. The only code that knows Google Docs exists is the two files above.

---

## Ingestion Pipeline

Each source adapter emits an `IngestEvent` containing raw content and source metadata. The pipeline processes events sequentially (with optional async batching for initial bulk loads).

### Step 1: Parse & Chunk

The pipeline first resolves the correct `BaseParser` for the event's `source_type` and calls `parse()`, receiving one or more `ParsedDocument` objects. For documents with structured pre-extracted `graph_hints` (Obsidian notes, emails), those hints are queued for direct Neo4j writes. The remaining `text` field then enters the generic chunker.

For all source types, text is split into overlapping chunks using a sentence-aware splitter:

- Chunk size: 512 tokens
- Overlap: 64 tokens
- Minimum chunk size: 100 tokens (shorter chunks are merged)

### Step 1b: Vision Extraction (Images Only)

Documents with `image_bytes is not None` are routed to the vision worker before the text chunker. This runs as a parallel async queue — text documents and image documents are processed concurrently.

**Supported formats:** `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif` (first frame only)

**Model:** `qwen3.5:9b` via Ollama. Qwen3.5-9B is natively multimodal and already in use for extraction fallback — no additional model pull required. It processes images at approximately 2–5 seconds each on the M5 Max, which is acceptable for the async worker queue.

**What the vision model extracts:**

The image is passed to the model with a structured prompt asking for three things:

```
Given this image, provide:
1. A 2-3 sentence description of what is shown
2. Any text visible in the image (UI labels, code, diagrams, captions)
3. Named entities visible or implied (people, logos, products, tools, concepts)

Return ONLY valid JSON:
{
  "description": "...",
  "visible_text": "...",
  "entities": [{"name": "...", "type": "..."}]
}
```

The `description` and `visible_text` fields are concatenated into a synthetic text chunk, embedded, and stored in Qdrant exactly like any other chunk — making images semantically searchable. The `entities` list flows into the standard entity resolution step (Step 4) with a confidence score of `0.80`.

**Why this matters for screenshots and diagrams:**

A screenshot of a terminal session, an architecture diagram, or a clipped UI is currently a black box to the pipeline. Vision extraction makes these first-class indexed artifacts:

- A screenshot of a Daytona dashboard clipped into Obsidian gets `DEPICTS` edges to the `Daytona` entity
- An architecture diagram pasted into a note has its visible labels extracted and linked to existing entities
- A photo of a whiteboard session from a planning meeting becomes searchable by its content

**Attachment detection in Obsidian:**

Obsidian embeds images with `![[filename.png]]` syntax. The Obsidian adapter parses these embed references alongside wikilinks and queues the referenced image files for vision processing. An `ATTACHED_TO` edge is written from the `Image` node to the parent `File` node, preserving the context that the image appeared in a specific note.

```python
EMBED_RE = re.compile(r'!\[\[([^\]]+\.(png|jpg|jpeg|webp|gif))\]\]', re.IGNORECASE)

def extract_image_embeds(content: str) -> list[str]:
    """Extract embedded image filenames from Obsidian note content."""
    return [m.group(1) for m in EMBED_RE.finditer(content)]
```

**Performance considerations:**

Vision inference is significantly slower than text embedding — approximately 2-5 seconds per image on an M-series Mac with Ollama. To avoid blocking the ingestion pipeline, image processing runs in a separate async worker queue with configurable concurrency. Images are deduplicated by SHA256 hash so re-indexing a vault does not reprocess unchanged images.

```toml
[vision]
enabled = true
model = "qwen3.5:9b"       # uses the same model as entity extraction
concurrency = 2             # parallel vision workers
min_file_size_kb = 10       # skip tiny images (icons, favicons)
max_file_size_mb = 20       # skip very large images
skip_patterns = ["avatar*", "icon*", ".obsidian/**"]
```

Each chunk is embedded using `nomic-embed-text` via Ollama. The resulting vector is stored in Qdrant alongside the chunk text and source metadata as payload.

```python
collection: "fieldnotes"
vector_size: 768
distance: Cosine
payload: {
    "source_type": "file|obsidian|email|repo|image",
    "source_id": "<neo4j node id>",
    "chunk_index": 0,
    "text": "...",          # for images: vision-extracted description + visible text
    "date": "2026-03-10T00:00:00Z"
}
```

### Step 3: Entity & Triple Extraction

A local LLM (Qwen3.5-27B via Ollama; falls back to Qwen3.5-9B or any OpenAI-compatible endpoint) is prompted to extract named entities and relationships from each chunk. The 27B is preferred over smaller models because extraction quality directly determines graph fidelity — it produces fewer malformed JSON responses and more accurate relationship triples, and at 40–60 t/s on the M5 Max it keeps pace with continuous ingestion without queue buildup.

**Extraction prompt (system):**

```
You are a knowledge graph extraction engine. Given a text chunk, extract:
1. Named entities: people, technologies, projects, organizations, concepts
2. Relationships between entities as subject-predicate-object triples

Return ONLY valid JSON in this format:
{
  "entities": [{"name": "...", "type": "Person|Technology|Project|Organization|Concept"}],
  "triples": [{"subject": "...", "predicate": "...", "object": "..."}]
}
```

The extraction model is intentionally separate from the query model. During bulk initial indexing of large corpora, swap to Qwen3.5-9B for speed, then switch back to Qwen3.5-27B for incremental updates where quality matters more than throughput.

### Step 4: Entity Resolution

Before writing to Neo4j, extracted entity names are resolved against existing entities using:

1. **Exact match** on lowercased name
2. **Fuzzy match** using `rapidfuzz` (threshold: 88 similarity)
3. **Embedding similarity** as a fallback (cosine > 0.92 triggers a `SAME_AS` edge rather than a merge, preserving both forms)

Entities sourced from Obsidian wikilinks carry a higher confidence score (`0.95`) than LLM-extracted entities (`0.75`). During resolution, wikilink-sourced entities act as anchors — ambiguous LLM mentions are preferentially merged into them rather than spawning new nodes. This means your own deliberate note links shape the entity namespace, and LLM extraction fills in the gaps.

### Step 5: Write to Stores

Neo4j and Qdrant writes happen in a single transaction boundary (best-effort — Neo4j ACID, Qdrant eventual). The source node (File/Email/Thread/Commit) is upserted first, then entity nodes, then edges, then chunk nodes with their Qdrant vector IDs.

---

## Topic Clustering

Topic clustering runs as a background job, not in the hot ingestion path. It operates over the full embedding space to discover emergent themes.

### Algorithm

1. **Pull all chunk embeddings** from Qdrant (filtered by source type if needed)
2. **Dimensionality reduction** with UMAP (768 → 32 dims) for clustering stability
3. **Cluster** with HDBSCAN (`min_cluster_size=10`, `metric='euclidean'`)
4. **Label** each cluster: take the 20 most central chunks (by distance to cluster centroid), send their text to Qwen3.5-72B, ask for a 2-4 word topic label and a one-sentence description. The 72B is used here specifically because the weekly background schedule has no latency constraint and label quality matters — it produces specific, insightful names rather than generic ones.
5. **Write** `Topic` nodes to Neo4j with `source: "cluster"`, link cluster members via `TAGGED` edges

User-defined Obsidian `#tags` are written as `Topic` nodes with `source: "user"` at ingest time and are never touched by the clustering job. The two topic types coexist in the graph and are queryable independently or together:

```cypher
-- All topics (both user-defined and cluster-derived)
MATCH (t:Topic) RETURN t.name, t.source ORDER BY t.source

-- Only user-curated topics from Obsidian
MATCH (f:File)-[:TAGGED_BY_USER]->(t:Topic {source: "user"}) RETURN f.name, t.name

-- Cluster-discovered topics not yet in your Obsidian taxonomy (gaps in your thinking)
MATCH (t:Topic {source: "cluster"})
WHERE NOT EXISTS { MATCH (:Topic {name: t.name, source: "user"}) }
RETURN t.name, t.description
```

The last query is particularly useful: it surfaces themes the clustering algorithm found in your corpus that you haven't explicitly named in your notes yet.

### Schedule

```toml
[clustering]
enabled = true
cron = "0 3 * * 0"    # weekly, Sunday 3am
min_corpus_size = 100  # don't cluster until there's enough data
```

Clustering is idempotent — re-running replaces existing topic labels rather than accumulating duplicates.

---

## Query Layer

The query layer is the primary interface for LLM agents. It exposes two retrieval strategies that are always combined before returning context.

### Graph Query (NL → Cypher)

Natural language queries are translated to Cypher using LangChain's `GraphCypherQAChain` backed by Neo4j. The chain is given the full schema as context.

Example:

```
Query:  "What have I written and emailed about Daytona?"

Cypher: MATCH (e:Entity {name: "Daytona"})
        OPTIONAL MATCH (f:File)-[:MENTIONS]->(e)
        OPTIONAL MATCH (em:Email)-[:MENTIONS]->(e)
        RETURN f.path, f.name, em.subject, em.date
        ORDER BY em.date DESC
        LIMIT 20
```

### Vector Search

Parallel semantic search over Qdrant using the query embedding. Returns top-k chunks with source metadata.

### Hybrid Merge

Results from both strategies are merged and deduplicated before being returned as context. Graph results are ranked first (higher precision), vector results fill in the gaps (higher recall). The merged context is formatted as a structured prompt fragment:

```
[Graph context]
File: ~/notes/daytona-integration.md (modified 2026-02-14)
Email thread: "Daytona Tier 3 access request" (2026-01-22, 3 messages)
Repository: GasTown — depends on daytona-sdk

[Semantic context]
Chunk from daytona-integration.md: "The sandbox orchestration model requires..."
Chunk from email (2026-01-22): "We'd like to request complimentary access for..."
```

### MCP Server

Fieldnotes exposes its query layer as an MCP server, making it directly consumable by Claude, GasTown Polecats, or any MCP-compatible agent.

**Tools exposed:**

| Tool | Description |
|---|---|
| `fieldnotes_search` | Hybrid graph + vector search. Returns structured context. |
| `fieldnotes_entity` | Look up all artifacts connected to a named entity. |
| `fieldnotes_timeline` | Return artifacts related to a topic, ordered by date. |
| `fieldnotes_topics` | List all discovered topic clusters with summaries. |
| `fieldnotes_graph` | Execute a raw Cypher query (read-only). |

---

## Configuration

All configuration lives in `~/.fieldnotes/config.toml`.

```toml
[core]
data_dir  = "~/.fieldnotes/data"
log_level = "info"

[neo4j]
uri      = "bolt://localhost:7687"
user     = "neo4j"
password = "fieldnotes"

[qdrant]
host       = "localhost"
port       = 6333
collection = "fieldnotes"

# ── Model provider instances ──────────────────────────────────────────────────

[modelproviders.local-ollama]
type = "ollama"
url  = "http://localhost:11434"

# Uncomment to add cloud providers:
# [modelproviders.anthropic]
# type    = "anthropic"
# api_key = ""          # or ANTHROPIC_API_KEY env var
#
# [modelproviders.openai]
# type    = "openai"
# api_key = ""          # or OPENAI_API_KEY env var

# ── Named model definitions ───────────────────────────────────────────────────

[models.nomic-embed]
provider = "local-ollama"
model    = "nomic-embed-text"

[models.qwen-27b]
provider = "local-ollama"
model    = "qwen3.5:27b"

[models.qwen-9b]
provider = "local-ollama"
model    = "qwen3.5:9b"

[models.qwen-72b]
provider = "local-ollama"
model    = "qwen3.5:72b"

[models.qwen-moe]
provider = "local-ollama"
model    = "qwen3.5:122b-a10b"

# ── Role assignments ──────────────────────────────────────────────────────────

[models.roles]
embed            = "nomic-embed"   # always-on embedding
extract          = "qwen-27b"      # hot path — every ingested chunk
extract_fallback = "qwen-9b"       # bulk initial indexing
vision           = "qwen-9b"       # async image queue
cluster_label    = "qwen-72b"      # weekly background cron
query            = "qwen-moe"      # interactive queries and agent tool calls

# ── Sources ───────────────────────────────────────────────────────────────────

[sources.files]
watch_paths        = []
include_extensions = [".md", ".txt", ".pdf"]

[sources.obsidian]
vault_paths        = []
index_attachments  = false
daily_notes_folder = "Daily"

[sources.gmail]
enabled              = false
poll_interval_seconds = 300
max_initial_threads  = 500

[sources.repositories]
repo_roots = []

# ── Other ─────────────────────────────────────────────────────────────────────

[clustering]
enabled = true
cron    = "0 3 * * 0"

[mcp]
enabled = true
port    = 3456
```

---

## Infrastructure

For local development and personal use, all dependencies run in Docker:

```yaml
# docker-compose.yml
services:
  neo4j:
    image: neo4j:5-community
    ports: ["7474:7474", "7687:7687"]
    environment:
      NEO4J_AUTH: neo4j/fieldnotes
    volumes:
      - neo4j_data:/data

  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  neo4j_data:
  qdrant_data:
```

Ollama runs natively on macOS (Metal-accelerated). In the target architecture (Phase 4+), the Go daemon is managed by `launchd` on macOS as a persistent background service, and the Python worker is launched by the daemon on startup. In Phase 1, the worker runs standalone — `worker/worker/main.py` starts both the Python source shims and the pipeline in a single process, with no Go daemon required.

---

## Repository Structure

The repository is a monorepo containing both the Go daemon and the Python worker. Each is independently buildable and deployable.

```
fieldnotes/
│
├── daemon/                             # Go — always-on runtime (Phase 2)
│   ├── cmd/
│   │   └── fieldnotes/
│   │       └── main.go                 # entrypoint — imports all source adapters
│   ├── internal/
│   │   ├── sources/
│   │   │   ├── source.go               # Source interface + IngestEvent types
│   │   │   ├── registry.go             # Register() + Build()
│   │   │   ├── files/
│   │   │   │   └── adapter.go          # fsnotify file watcher
│   │   │   ├── obsidian/
│   │   │   │   └── adapter.go          # vault detection, embed parsing
│   │   │   ├── gmail/
│   │   │   │   └── adapter.go          # OAuth polling + cursor
│   │   │   └── repositories/
│   │   │       └── adapter.go          # git log + file allowlist scanner
│   │   ├── dispatcher/
│   │   │   ├── dispatcher.go           # fan-in from all sources → worker queue
│   │   │   └── redis.go                # optional Redis queue (Phase 4)
│   │   ├── mcp/
│   │   │   └── server.go               # MCP server + tool definitions
│   │   └── api/
│   │       └── handler.go              # HTTP API (query passthrough, status)
│   ├── go.mod
│   └── go.sum
│
└── worker/                             # Python — ML pipeline (Phase 1+)
    ├── worker/
    │   ├── parsers/
    │   │   ├── base.py                 # BaseParser, ParsedDocument, GraphHint
    │   │   ├── registry.py             # @register decorator + get()
    │   │   ├── __init__.py             # imports all parsers to trigger registration
    │   │   ├── files.py                # FileParser
    │   │   ├── obsidian.py             # ObsidianParser (frontmatter, wikilinks, tags)
    │   │   ├── gmail.py                # GmailParser
    │   │   ├── images.py               # ImageParser (vision pipeline entry)
    │   │   └── repositories.py         # RepositoryParser
    │   ├── models/
    │   │   ├── base.py                 # ModelProvider, CompletionRequest/Response, EmbedRequest/Response
    │   │   ├── registry.py             # @register decorator + get_provider()
    │   │   ├── resolver.py             # ResolvedModel, ModelRegistry
    │   │   ├── __init__.py             # imports all providers to trigger registration
    │   │   └── providers/
    │   │       ├── ollama.py           # OllamaProvider
    │   │       ├── anthropic.py        # AnthropicProvider
    │   │       └── openai.py           # OpenAIProvider
    │   ├── sources/                    # Phase 1 only — Python source shims
    │   │   ├── base.py                 # PythonSource interface (mirrors Go Source)
    │   │   ├── files.py                # watchdog-based file watcher
    │   │   ├── obsidian.py             # vault detection shim
    │   │   ├── gmail.py                # Gmail polling shim
    │   │   └── repositories.py         # git scanner shim
    │   ├── pipeline/
    │   │   ├── chunker.py              # sentence-aware text splitter
    │   │   ├── embedder.py             # nomic-embed-text via Ollama
    │   │   ├── vision.py               # vision worker queue (async)
    │   │   ├── extractor.py            # entity/triple extraction (LLM)
    │   │   ├── resolver.py             # entity dedup (rapidfuzz + embeddings)
    │   │   └── writer.py               # Neo4j + Qdrant write layer
    │   ├── clustering/
    │   │   ├── cluster.py              # UMAP + HDBSCAN
    │   │   └── labeler.py              # LLM topic labeling
    │   ├── query/
    │   │   ├── graph.py                # NL → Cypher via LangChain
    │   │   ├── vector.py               # Qdrant semantic search
    │   │   └── hybrid.py               # result merging
    │   ├── api.py                      # HTTP server — receives IngestEvents
    │   ├── main.py                     # entrypoint — starts sources + HTTP server
    │   └── config.py                   # config.toml loader
    └── pyproject.toml
```

In Phase 1, `worker/worker/sources/` contains lightweight Python shims that implement the same watching and polling logic that will eventually live in the Go daemon. They emit `IngestEvent` dicts directly to the worker's internal queue rather than over HTTP. The `parsers/` package, the full pipeline, and all downstream code are identical between Phase 1 and the target architecture — the shims are the only thing that gets replaced when the Go daemon is built in Phase 4.

The Go `daemon/` tree is committed from day one as a skeleton with the `Source` interface, `IngestEvent` types, and registry defined — even before the adapters are implemented. This keeps the contract stable and makes Phase 4 a drop-in replacement rather than a refactor.

---

## Phased Roadmap

### Phase 1 — Core Pipeline (POC, pure Python)
- [ ] `models/` package: `ModelProvider` interface, `OllamaProvider`, `ModelRegistry` with three-layer config loading
- [ ] Python source shims in `worker/worker/sources/` (watchdog file watcher, Obsidian vault detection)
- [ ] File and Obsidian parsers (`parsers/files.py`, `parsers/obsidian.py`) with `GraphHint` support
- [ ] Full pipeline: chunker → embedder → entity extractor → resolver → Neo4j + Qdrant writer
- [ ] Go `daemon/` skeleton committed: `Source` interface, `IngestEvent` types, registry defined (no adapters yet)
- [ ] Basic hybrid query (NL→Cypher + vector)
- [ ] CLI: `fieldnotes search "<query>"`

### Phase 2 — Email + Clustering + Vision
- [ ] Gmail adapter
- [ ] Person nodes and email relationship graph
- [ ] HDBSCAN topic clustering + LLM labeling
- [ ] Vision extraction worker queue (Qwen3.5-9B via Ollama)
- [ ] `Image` nodes with `DEPICTS` and `ATTACHED_TO` edges
- [ ] `fieldnotes topics` CLI command

### Phase 3 — Repo Integration
- [ ] Repository adapter (README/docs/commits)
- [ ] Dependency graph from Cargo.toml / pyproject.toml
- [ ] Cross-source entity resolution improvements

### Phase 4 — Go Daemon Refactor
- [ ] `daemon/` — Go rewrite of all adapters (file watcher, Obsidian parser, Gmail poller, git scanner)
- [ ] `worker/` — Python ML pipeline promoted from `poc/`, unchanged
- [ ] HTTP queue interface between daemon and worker
- [ ] Go daemon distributed as single binary (`brew install fieldnotes`)
- [ ] `launchd` plist for macOS background service management
- [ ] Replace `watchdog` Python dependency with `fsnotify` in Go

### Phase 5 — Agent Interface
- [ ] MCP server in Go daemon (replaces Python MCP server from POC)
- [ ] GasTown Polecat integration
- [ ] `fieldnotes_search` tool in Claude Desktop via MCP

### Phase 6 — Fine-tuning (Experimental)
- [ ] Generate fine-tuning dataset from extraction errors
- [ ] LoRA fine-tune Qwen3.5-9B on personal corpus via mlx-lm
- [ ] A/B eval: base model vs fine-tuned on extraction quality

---

## Non-Goals

- **Not a search engine.** Fieldnotes is a context provider for agents, not a replacement for Spotlight or grep.
- **Not a note-taking app.** It indexes existing artifacts; it does not create or edit them.
- **Not a cloud service.** There are no plans for a hosted version. Local-first is a design constraint, not a feature.
- **Not a general RAG framework.** Fieldnotes is opinionated about its data model and query interface. It is not designed to be a generic library.

---

## Prior Art & Inspiration

- **Vannevar Bush, "As We May Think" (1945)** — the original vision of a personal knowledge machine (the Memex)
- **Microsoft GraphRAG** — community detection and hierarchical summarization over document corpora
- **Obsidian** — local-first, graph-native note-taking; Fieldnotes ingests Obsidian vaults as a first-class source and extends their graph into the broader knowledge network
- **Rewind.ai** — personal data indexing, but cloud-dependent and retrieval-only
- **LlamaIndex KnowledgeGraphIndex** — inspiration for the extraction pipeline design

---

*Fieldnotes is built by Markus. Honest tools for practitioners.*