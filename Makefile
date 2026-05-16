.PHONY: help venv install install-dev lint fmt test test-ci build clean publish publish-test docker-up docker-down version

# `uv sync` reads worker/.python-version, downloads CPython 3.14 if absent,
# materializes worker/.venv, and installs the locked dependency set.

VENV    := worker/.venv
PYTHON  := $(VENV)/bin/python
VERSION  = $(shell uv version --short --project worker)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Environment ──────────────────────────────────────────────────────

venv: ## Create virtual environment (idempotent)
	uv sync --project worker

# ── Worker (Python) ──────────────────────────────────────────────────

install: ## Install fieldnotes worker (locked versions from uv.lock)
	uv sync --project worker

install-dev: ## Install worker with dev dependencies (pytest, ruff)
	uv sync --project worker --all-extras

lint: ## Run ruff linter on worker
	uv run --directory worker ruff check worker/ tests/

fmt: ## Auto-format worker code with ruff
	uv run --directory worker ruff check --fix worker/ tests/
	uv run --directory worker ruff format worker/ tests/

test: ## Run worker tests
	uv run --directory worker pytest tests/ -v

test-ci: lint ## Run linter + tests (CI mode)
	uv run --directory worker pytest tests/ -v --tb=short

# ── Build & Release ──────────────────────────────────────────────────

build: clean ## Build worker sdist and wheel
	uv build --project worker

clean: ## Remove build artifacts
	rm -rf worker/dist/ worker/build/ worker/*.egg-info worker/worker/*.egg-info

publish-test: build ## Upload worker to TestPyPI
	uv publish --directory worker --publish-url https://test.pypi.org/legacy/

publish: build ## Upload worker to PyPI (production)
	uv publish --directory worker

# ── Infrastructure ───────────────────────────────────────────────────

docker-up: ## Start Neo4j, Qdrant, and observability stack
	docker compose up -d

docker-down: ## Stop all Docker services
	docker compose down

# ── Info ─────────────────────────────────────────────────────────────

version: ## Print current version
	@echo $(VERSION)
