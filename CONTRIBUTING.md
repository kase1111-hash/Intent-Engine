# Contributing to Intent Engine

Thank you for your interest in contributing to Intent Engine!

## Getting Started

1. Fork the repository
2. Clone your fork and create a feature branch
3. Create a virtual environment (Python 3.10 or newer) and install the development dependencies. Intent Engine is built on the [Prosody Protocol](https://github.com/kase1111-hash/Prosody-Protocol) SDK, which is not published to PyPI, so install it from GitHub first (with its `audio` extra), then this package:

```bash
pip install "prosody-protocol[audio] @ git+https://github.com/kase1111-hash/Prosody-Protocol.git@4d4f0bb930b33f5d66015f8565a87e16c02e5fe2"
pip install -e ".[dev]"
```

`make dev` runs both commands. CI installs the same commit (prosody-protocol 0.1.0a3, pinned as `PROSODY_PROTOCOL` in `.github/workflows/ci.yml`, which is the authority if this page falls behind), and `make dev` reads it from there. Skip the first command and the second one fails with `No matching distribution found for prosody-protocol`.

## Development Workflow

### Running Tests

```bash
make test
```

This runs `pytest` with coverage and fails below 80%, the same gate CI enforces. Plain `pytest` runs the same tests without the gate. Both collect `tests/` and `examples/` (see `testpaths` in `pyproject.toml`). Run `make check` to run linting, type checking, and tests together.

The optional provider SDKs are not part of the `dev` extra, so the tests that need one (and the tests of the example integrations, which need their web frameworks) skip themselves when it is missing. Mocked tests cannot notice an SDK release breaking an adapter; the SDK-backed ones can. To run them, install what CI's `sdk-contracts` job installs:

```bash
pip install "prosody-protocol[audio,api] @ git+https://github.com/kase1111-hash/Prosody-Protocol.git@4d4f0bb930b33f5d66015f8565a87e16c02e5fe2"
pip install -e ".[dev,claude,openai,deepgram,assemblyai,elevenlabs,examples]"
pytest
```

They use fake servers on `127.0.0.1` and stubs, so no network access or API keys are needed. `tests/tts/test_espeak_real.py` additionally needs the `espeak` extra and the system eSpeak NG library, and skips without them.

### Code Style

- We use **ruff** for linting (`make lint`, which checks `intent_engine/`, `tests/` and `examples/` as CI does)
- We use **mypy** in strict mode for type checking (`make typecheck`, which checks `intent_engine/` only, as CI does)
- Target Python 3.10+
- Line length limit: 100 characters

**Formatting.** The code base is not `ruff format` clean, and CI does not check formatting. Running it over the whole tree would rewrite more than half of the Python files, so `make format` does not do that: it takes the files you changed and formats them, then applies `ruff check --fix`:

```bash
make format FILES="intent_engine/engine.py tests/test_engine.py"
```

A file that is not format clean changes throughout when you do this, so format only the files you are changing anyway and leave the others alone.

**Pre-commit hooks.** The repository has a `.pre-commit-config.yaml`, but its hooks do not work yet, so do not run `pre-commit install` (`make check` is the check to run before you push). Its mypy hook builds its own environment and asks PyPI for `prosody-protocol>=0.1.0a1`, which is not there, so the hook cannot even be set up. Its ruff hook is pinned to ruff v0.4.4, which reports lint (rule `UP038`) that the ruff installed by the `dev` extra does not, and its ruff-format hook reformats dozens of files.

### Key Conventions

- **Never reimplement Prosody Protocol components.** Use `prosody_protocol` for all IML parsing, validation, prosody analysis, and emotion classification.
- **Provider adapters must be provider-agnostic.** All STT, LLM, and TTS providers implement the abstract base class from their respective `base.py`.
- **Lazy imports for optional dependencies.** Provider SDKs are imported inside methods, not at module level, to keep the core package lightweight.
- **Do not block the event loop.** Adapter methods are coroutines. An SDK call that blocks (a synchronous client, a local model) runs in a worker thread, for example with `asyncio.to_thread`.
- **All IML output must validate.** Use `prosody_protocol.IMLValidator` before returning IML to callers.
- **Keep the LLM system prompt valid.** The IML examples in `intent_engine/llm/prompts.py` are checked against `prosody_protocol.IMLValidator` by `tests/llm/test_prompt_iml_conformance.py`. Bump `PROMPT_VERSION` when you change the prompt.

### Submitting Changes

1. Ensure all tests pass: `make check`
2. Write tests for new functionality
3. Keep commits focused and descriptive
4. Open a pull request against `main`

CI runs the lint job (ruff, mypy), the tests with coverage on Python 3.10, 3.11 and 3.12, the SDK-backed tests, and `pip-audit`.

## Reporting Issues

Please open an issue on GitHub with:
- A clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
