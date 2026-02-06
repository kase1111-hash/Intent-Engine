.PHONY: install dev test lint typecheck format check clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest --cov=intent_engine --cov-report=term-missing

lint:
	ruff check intent_engine/ tests/

typecheck:
	mypy intent_engine/

format:
	ruff format intent_engine/ tests/
	ruff check --fix intent_engine/ tests/

check: lint typecheck test

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/ coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
