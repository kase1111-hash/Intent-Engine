# Contributing to Intent Engine

Thank you for your interest in contributing to Intent Engine!

## Getting Started

1. Fork the repository
2. Clone your fork and create a feature branch
3. Install development dependencies:

```bash
pip install -e ".[dev]"
pre-commit install
```

## Development Workflow

### Running Tests

```bash
make test
```

Tests require 80% coverage to pass. Run `make check` to run linting, type checking, and tests together.

### Code Style

- We use **ruff** for linting and formatting (`make lint`, `make format`)
- We use **mypy** in strict mode for type checking (`make typecheck`)
- Target Python 3.10+
- Line length limit: 100 characters

### Key Conventions

- **Never reimplement Prosody Protocol components.** Use `prosody_protocol` for all IML parsing, validation, prosody analysis, and emotion classification.
- **Provider adapters must be provider-agnostic.** All STT, LLM, and TTS providers implement the abstract base class from their respective `base.py`.
- **Lazy imports for optional dependencies.** Provider SDKs are imported inside methods, not at module level, to keep the core package lightweight.
- **All IML output must validate.** Use `prosody_protocol.IMLValidator` before returning IML to callers.

### Submitting Changes

1. Ensure all tests pass: `make check`
2. Write tests for new functionality
3. Keep commits focused and descriptive
4. Open a pull request against `main`

## Reporting Issues

Please open an issue on GitHub with:
- A clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
