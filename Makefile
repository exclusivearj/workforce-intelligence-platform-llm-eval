.PHONY: help install setup embed eval test test-unit test-integration lint clean teardown

# Source env files into a recipe's shell so DB/eval targets work without manually
# exporting. Repo-root ../.env first (shared Postgres creds + ANTHROPIC_API_KEY), then a
# module-local ./.env if present (overrides root — handy for putting ANTHROPIC_API_KEY
# next to this module). Applied only to the targets below — NOT the test targets, which
# stay hermetic. Harmless when both files are absent.
LOAD_ENV := set -a; [ -f ../.env ] && . ../.env; [ -f .env ] && . .env; set +a

help:
	@echo ""
	@echo "workforce-intelligence-platform :: llm-eval (module 2)"
	@echo "──────────────────────────────────────────────────────"
	@echo "  Prereqs (run from the repo root first):"
	@echo "    make infra-up && make ingestion-setup && make ingestion-dbt"
	@echo "    → brings up pgvector Postgres and builds analytics.dim_employees,"
	@echo "      which the masking view reads from."
	@echo ""
	@echo "  make install           Install package + dev dependencies"
	@echo "  make setup             Install, apply masking views, embed safe context"
	@echo "  make embed             Re-embed safe_employee_context into pgvector"
	@echo "  make eval              Run RAGAS eval (Claude judge; needs ANTHROPIC_API_KEY)"
	@echo ""
	@echo "  make test              Unit + integration tests"
	@echo "  make test-unit         Unit tests + coverage (no infra required)"
	@echo "  make test-integration  Integration tests (requires Docker / testcontainers)"
	@echo "  make lint              Ruff lint over src/ + tests/"
	@echo "  make clean             Remove caches + coverage artifacts"
	@echo "  make teardown          Drop llm objects + shut down the shared stack"
	@echo ""

install:
	pip install -e ".[dev]"

setup: install
	@echo "Applying masking views + embedding safe employee context..."
	@echo "(requires Postgres up and analytics.dim_employees populated)"
	@$(LOAD_ENV); python -m src.pipeline.setup

embed:
	@$(LOAD_ENV); python -c "from src.embeddings.pipeline import run_embedding_pipeline; print(run_embedding_pipeline())"

eval:
	@echo "Running RAGAS eval — metric scoring uses a Claude LLM judge."
	@echo "Requires ANTHROPIC_API_KEY; judge model = RAGAS_LLM_MODEL (default claude-opus-4-8)."
	@$(LOAD_ENV); [ -n "$$ANTHROPIC_API_KEY" ] || { echo "ERROR: ANTHROPIC_API_KEY is not set (checked env + ../.env). Set it to run the Claude judge; make setup/embed need no key."; exit 1; }; \
		python -m src.pipeline.eval

test-unit:
	pytest tests/unit/ -v --cov=src --cov-report=term-missing --cov-fail-under=80

test-integration:
	pytest tests/integration/ -v -m integration

test: test-unit test-integration

lint:
	ruff check src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .coverage coverage.xml htmlcov .pytest_cache

# Graceful teardown — the inverse of `make setup`. Drops the PII masking views and
# truncates the llm tables (keeping the tables so `make setup` rebuilds cleanly),
# clears local caches, then shuts down the shared Docker stack (repo-root
# `infra-down`). DB cleanup runs first while Postgres is up and is skipped if it is
# already down. Volumes are preserved (sibling-module data + Airflow metadata
# survive) — use repo-root `make infra-reset` to also wipe them.
teardown:
	@echo "Tearing down llm-eval (graceful)..."
	@echo "  - dropping masking views + truncating llm tables (skipped if Postgres is down)..."
	@$(LOAD_ENV); python -c "from src.utils.db import get_connection, teardown_data; print('\n'.join('    ' + a for a in teardown_data(get_connection())))" || echo "    (Postgres not reachable — skipped DB cleanup)"
	@$(MAKE) clean
	@echo "  - shutting down the shared Docker stack (postgres/trino/airflow)..."
	@$(MAKE) -C .. infra-down
	@echo "llm-eval teardown complete. Volumes preserved — run repo-root 'make infra-reset' to also wipe them."
