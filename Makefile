.PHONY: help venv install install-dev lint fmt test test-ci build clean publish publish-test docker-up docker-down

VENV     := worker/.venv
PYTHON   := $(VENV)/bin/python
PIP      := $(VENV)/bin/pip
VERSION   = $(shell $(PYTHON) -c "import tomllib; print(tomllib.load(open('worker/pyproject.toml','rb'))['project']['version'])")

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Environment ──────────────────────────────────────────────────────

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

venv: $(VENV)/bin/activate ## Create virtual environment

# ── Worker (Python) ──────────────────────────────────────────────────

install: venv ## Install fieldnotes worker into venv
	$(PIP) install ./worker

install-dev: venv ## Install worker with dev dependencies (pytest, ruff)
	$(PIP) install -e "./worker[dev]"

lint: ## Run ruff linter on worker
	cd worker && ../$(PYTHON) -m ruff check worker/ tests/

fmt: ## Auto-format worker code with ruff
	cd worker && ../$(PYTHON) -m ruff check --fix worker/ tests/
	cd worker && ../$(PYTHON) -m ruff format worker/ tests/

test: ## Run worker tests
	cd worker && ../$(PYTHON) -m pytest tests/ -v

test-ci: lint ## Run linter + tests (CI mode)
	cd worker && ../$(PYTHON) -m pytest tests/ -v --tb=short

# ── Build & Release ──────────────────────────────────────────────────

build: venv clean ## Build worker sdist and wheel
	$(PIP) install --upgrade build
	cd worker && ../$(PYTHON) -m build

clean: ## Remove build artifacts
	rm -rf worker/dist/ worker/build/ worker/*.egg-info worker/worker/*.egg-info

publish-test: build ## Upload worker to TestPyPI
	$(PIP) install --upgrade twine
	$(PYTHON) -m twine upload --repository testpypi worker/dist/*

publish: build ## Upload worker to PyPI (production)
	$(PIP) install --upgrade twine
	$(PYTHON) -m twine upload worker/dist/*

# ── Infrastructure ───────────────────────────────────────────────────

docker-up: ## Start Neo4j, Qdrant, and observability stack
	docker compose up -d

docker-down: ## Stop all Docker services
	docker compose down

# ── Info ─────────────────────────────────────────────────────────────

version: ## Print current version
	@echo $(VERSION)
