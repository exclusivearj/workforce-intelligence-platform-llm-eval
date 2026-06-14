.PHONY: setup embed eval test test-unit test-integration lint clean

setup:
	pip install -e ".[dev]"
	@echo "Applying masking views + embedding safe employee context..."
	python -m src.pipeline.setup

embed:
	python -c "from src.embeddings.pipeline import run_embedding_pipeline; print(run_embedding_pipeline())"

eval:
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
	rm -rf .coverage htmlcov
