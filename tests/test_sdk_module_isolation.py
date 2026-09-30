"""A test that fakes a provider SDK in ``sys.modules`` must not remove the real one.

Older adapter tests install a fake with ``sys.modules[name] = fake`` and end
with ``sys.modules.pop(name)``.  When the real SDK was already imported, that
pop leaves the next test to import a second copy of it, and a patch made on
the first copy (``monkeypatch.setattr(elevenlabs, ...)``) misses the module
the adapter uses.  ``tests/conftest.py`` puts the loaded SDK modules back
after every test; this runs the pattern in a fresh pytest to prove it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_PROBE = textwrap.dedent(
    '''
    import sys
    import types

    # "already imported", as an installed SDK is once a test has used it
    REAL = types.ModuleType("openai")
    sys.modules["openai"] = REAL


    def test_a_fakes_the_sdk_and_pops_it():
        sys.modules["openai"] = types.ModuleType("openai")
        try:
            assert sys.modules["openai"] is not REAL
        finally:
            sys.modules.pop("openai", None)


    def test_b_still_sees_the_real_module():
        assert sys.modules["openai"] is REAL
    '''
)


def test_a_popped_fake_does_not_take_the_real_sdk_with_it(tmp_path: Path) -> None:
    probe = tmp_path / "test_probe.py"
    probe.write_text(_PROBE)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "tests.conftest",
            "--rootdir",
            str(tmp_path),
            str(probe),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
