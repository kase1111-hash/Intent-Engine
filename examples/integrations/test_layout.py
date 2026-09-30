"""Checks on how the examples are laid out and documented.

The examples were once moved without updating their imports (nothing could
be collected), and three of them were named after the SDKs they wrap, so
copying the folder next to a bot script replaced ``import discord``.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

INTEGRATIONS = Path(__file__).resolve().parent
MODULES = sorted(
    p.stem
    for p in INTEGRATIONS.glob("*.py")
    if not p.stem.startswith("test_") and p.stem not in {"__init__", "conftest"}
)
# Import names of the packages the examples (or code copied next to them) use.
SDK_NAMES = [
    "discord",
    "twilio",
    "slack",
    "slack_sdk",
    "slack_bolt",
    "fastapi",
    "starlette",
    "uvicorn",
    "httpx",
    "pydantic",
    "anthropic",
    "openai",
    "prosody_protocol",
    "intent_engine",
]


def test_modules_are_found() -> None:
    assert {"twilio_voice", "slack_bot", "discord_bot", "rest_server"} <= set(MODULES)


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_from_the_repository_root(module: str) -> None:
    # The path the docstrings and the README tell people to use, without any
    # optional SDK installed (they are all imported lazily).
    importlib.import_module(f"examples.integrations.{module}")


def test_no_module_is_named_like_an_sdk() -> None:
    clashes = sorted(set(MODULES) & set(SDK_NAMES))
    assert clashes == [], f"{clashes} would shadow the real packages if this folder is on sys.path"


@pytest.mark.parametrize(
    "name", ["discord", "twilio", "slack_sdk", "slack_bolt", "httpx", "fastapi"]
)
def test_sdk_is_not_shadowed_when_run_from_the_examples_directory(name: str) -> None:
    pytest.importorskip(name)
    out = subprocess.run(
        [sys.executable, "-c", f"import {name}; print({name}.__file__)"],
        cwd=INTEGRATIONS,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert Path(out).resolve().parent != INTEGRATIONS


def test_docs_do_not_mention_the_removed_package_path() -> None:
    for path in [*INTEGRATIONS.glob("*.py"), INTEGRATIONS / "README.md"]:
        if path.name == Path(__file__).name:
            continue
        assert "intent_engine.integrations" not in path.read_text(), path.name


@pytest.mark.parametrize("module", [m for m in MODULES if not m.startswith("_")])
def test_readme_lists_every_example(module: str) -> None:
    assert f"{module}.py" in (INTEGRATIONS / "README.md").read_text()
