.PHONY: help install setup embed eval test test-unit test-integration lint clean

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
	@echo "  make eval              Run RAGAS eval (needs OPENAI_API_KEY — see README)"
	@echo ""
	@echo "  make test              Unit + integration tests"
	@echo "  make test-unit         Unit tests + coverage (no infra required)"
	@echo "  make test-integration  Integration tests (requires Docker / testcontainers)"
	@echo "  make lint              Ruff lint over src/ + tests/"
	@echo "  make clean             Remove caches + coverage artifacts"
	@echo ""

install:
	pip install -e ".[dev]"

setup: install
	@echo "Applying masking views + embedding safe employee context..."
	@echo "(requires Postgres up and analytics.dim_employees populated)"
	python -m src.pipeline.setup

embed:
	python -c "from src.embeddings.pipeline import run_embedding_pipeline; print(run_embedding_pipeline())"

eval:
	@echo "Running RAGAS eval — metric scoring uses an LLM judge (OpenAI by default)."
	@echo "Set OPENAI_API_KEY, or inject a custom evaluator into run_eval(). See README."
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
