.PHONY: help install install-dev lint fmt test test-ci build clean publish publish-test docker-up docker-down

PYTHON   ?= python3
VERSION  := $(shell $(PYTHON) -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Development ──────────────────────────────────────────────────────

install: ## Install fieldnotes in the current environment
	$(PYTHON) -m pip install .

install-dev: ## Install with dev dependencies (pytest, ruff)
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## Run ruff linter
	$(PYTHON) -m ruff check worker/ tests/

fmt: ## Auto-format with ruff
	$(PYTHON) -m ruff check --fix worker/ tests/
	$(PYTHON) -m ruff format worker/ tests/

test: ## Run tests
	$(PYTHON) -m pytest tests/ -v

test-ci: lint ## Run linter + tests (CI mode)
	$(PYTHON) -m pytest tests/ -v --tb=short

# ── Build & Release ──────────────────────────────────────────────────

build: clean ## Build sdist and wheel
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

clean: ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info worker/*.egg-info

publish-test: build ## Upload to TestPyPI
	$(PYTHON) -m pip install --upgrade twine
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish: build ## Upload to PyPI (production)
	$(PYTHON) -m pip install --upgrade twine
	$(PYTHON) -m twine upload dist/*

# ── Infrastructure ───────────────────────────────────────────────────

docker-up: ## Start Neo4j, Qdrant, and observability stack
	docker compose -f ../docker-compose.yml up -d

docker-down: ## Stop all Docker services
	docker compose -f ../docker-compose.yml down

# ── Info ─────────────────────────────────────────────────────────────

version: ## Print current version
	@echo $(VERSION)
