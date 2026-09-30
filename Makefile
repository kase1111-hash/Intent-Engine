.PHONY: install dev check-prosody-protocol test lint typecheck format check clean

# prosody-protocol is not on PyPI, so it is installed from GitHub before this
# package. The default is the commit CI pins (PROSODY_PROTOCOL in
# .github/workflows/ci.yml, read from there so the two cannot drift apart).
# Override it to try another one, for example a local checkout:
#   make dev PROSODY_PROTOCOL="prosody-protocol[audio] @ file:///path/to/Prosody-Protocol"
PROSODY_PROTOCOL ?= $(shell sed -n 's/^  PROSODY_PROTOCOL: "\(.*\)"$$/\1/p' .github/workflows/ci.yml)

check-prosody-protocol:
	@test -n '$(PROSODY_PROTOCOL)' || { \
		echo "PROSODY_PROTOCOL is empty: set it to a pip requirement for prosody-protocol[audio]" >&2; \
		exit 1; }

install: check-prosody-protocol
	pip install "$(PROSODY_PROTOCOL)"
	pip install -e .

dev: check-prosody-protocol
	pip install "$(PROSODY_PROTOCOL)"
	pip install -e ".[dev]"

# Same command as the CI test job, including its 80% coverage gate. Plain
# `pytest` runs the same tests without the gate; both collect tests/ and
# examples/ (see testpaths in pyproject.toml).
test:
	pytest --cov=intent_engine --cov-report=term-missing --cov-fail-under=80

lint:
	ruff check intent_engine/ tests/ examples/

typecheck:
	mypy intent_engine/

# The tree is not `ruff format` clean (a bare run would rewrite more than half
# of the Python files) and CI does not check formatting, so this takes the
# files to format instead of the whole tree:
#   make format FILES="intent_engine/engine.py tests/test_engine.py"
format:
ifndef FILES
	$(error set FILES to the files you changed, for example: make format FILES="intent_engine/engine.py". A whole-tree format is deliberately not offered, see CONTRIBUTING.md)
endif
	ruff format $(FILES)
	ruff check --fix $(FILES)

check: lint typecheck test

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/ coverage.xml .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
