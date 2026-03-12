.PHONY: help venv install install-dev lint fmt test test-ci build clean publish publish-test docker-up docker-down

VENV     := .venv
PYTHON   := $(VENV)/bin/python
PIP      := $(VENV)/bin/pip
VERSION   = $(shell $(PYTHON) -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Environment ──────────────────────────────────────────────────────

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

venv: $(VENV)/bin/activate ## Create virtual environment

# ── Development ──────────────────────────────────────────────────────

install: venv ## Install fieldnotes into venv
	$(PIP) install .

install-dev: venv ## Install with dev dependencies (pytest, ruff)
	$(PIP) install -e ".[dev]"

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

build: venv clean ## Build sdist and wheel
	$(PIP) install --upgrade build
	$(PYTHON) -m build

clean: ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info worker/*.egg-info

publish-test: build ## Upload to TestPyPI
	$(PIP) install --upgrade twine
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish: build ## Upload to PyPI (production)
	$(PIP) install --upgrade twine
	$(PYTHON) -m twine upload dist/*

# ── Infrastructure ───────────────────────────────────────────────────

docker-up: ## Start Neo4j, Qdrant, and observability stack
	docker compose -f ../docker-compose.yml up -d

docker-down: ## Stop all Docker services
	docker compose -f ../docker-compose.yml down

# ── Info ─────────────────────────────────────────────────────────────

version: ## Print current version
	@echo $(VERSION)
