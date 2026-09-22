.DEFAULT_GOAL := help
PY ?= python3
VENV := .venv
BIN := $(VENV)/bin

help: ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PY) -m venv $(VENV)

install: $(BIN)/python ## Create the venv and install the project with dev extras
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[postgres,server,dev]"

bootstrap: ## Schema, corpus ingest, detector training, benchmark sweep
	$(BIN)/cleartusk bootstrap

migrate: ## Apply database migrations
	$(BIN)/alembic upgrade head

revision: ## Autogenerate a migration (make revision m="add table")
	$(BIN)/alembic revision --autogenerate -m "$(m)"

serve: ## Run the development server
	$(BIN)/cleartusk serve --debug

benchmark: ## Re-clean the corpus and record the scorecard
	$(BIN)/cleartusk benchmark --csv

train: ## Train the call detector
	$(BIN)/cleartusk train-detector

stats: ## Print the analytics summary
	$(BIN)/cleartusk stats

test: ## Run the test suite
	$(BIN)/python -m pytest

coverage: ## Run the tests with a coverage report
	$(BIN)/python -m pytest --cov=cleartusk --cov-report=term-missing

lint: ## Lint with ruff
	$(BIN)/python -m ruff check .

format: ## Autoformat with ruff
	$(BIN)/python -m ruff format .
	$(BIN)/python -m ruff check --fix .

docker-up: ## Start Postgres + the web app in Docker
	docker compose up --build

clean: ## Remove caches and generated runtime artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov **/__pycache__
	rm -rf data/runtime/media data/runtime/reports

.PHONY: help install bootstrap migrate revision serve benchmark train stats test coverage lint format docker-up clean
