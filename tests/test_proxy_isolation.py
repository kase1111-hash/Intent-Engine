"""The suite must not depend on the developer's proxy settings.

Many tests talk to a fake server on ``127.0.0.1`` through a real HTTP client
(the vendor SDKs, ``httpx``).  Those clients honour ``HTTP_PROXY``,
``ALL_PROXY`` and friends, so on a machine behind a proxy that does not list
the loopback address in ``NO_PROXY`` every such request went to the proxy and
failed with "connection refused".  ``tests/conftest.py`` therefore gives every
test an environment in which loopback traffic never meets a proxy.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")


class _Hello(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        body = b"hello"
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture()
def loopback_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Hello)
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    ).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()


class TestLoopbackIsolation:
    """Assertions about the environment a test sees; the subprocess test below
    runs them under a hostile proxy configuration."""

    def test_no_proxy_variable_reaches_the_test(self) -> None:
        for name in _PROXY_VARIABLES:
            assert name not in os.environ, name
            assert name.lower() not in os.environ, name.lower()

    def test_loopback_is_excluded_from_proxies(self) -> None:
        for name in ("NO_PROXY", "no_proxy"):
            hosts = os.environ[name].split(",")
            assert {"127.0.0.1", "localhost", "::1"} <= set(hosts), name

    def test_urllib_reaches_a_loopback_server(self, loopback_url: str) -> None:
        with urllib.request.urlopen(loopback_url, timeout=10) as reply:
            assert reply.read() == b"hello"

    def test_httpx_reaches_a_loopback_server(self, loopback_url: str) -> None:
        httpx = pytest.importorskip("httpx")

        assert httpx.get(loopback_url, timeout=10).text == "hello"


def test_the_fixture_holds_under_a_hostile_proxy_environment() -> None:
    """Run the checks above in a fresh pytest whose environment routes every
    request to a dead proxy and does not exempt loopback."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.lower().endswith("_proxy") and not key.startswith("COV_CORE_")
    }
    for name in _PROXY_VARIABLES:
        env[name] = env[name.lower()] = "http://127.0.0.1:9"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_proxy_isolation.py::TestLoopbackIsolation",
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
