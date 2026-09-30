"""A hostile rules file must not hang or exhaust the process while it is rejected.

YAML aliases share objects, so a few hundred bytes of nested aliases describe
a value whose ``repr`` is exponentially long.  Error messages that quote the
offending value must therefore be bounded.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from intent_engine.constitutional.rules import parse_rules_yaml


def _list_bomb(levels: int, fan: int = 9) -> str:
    """Anchored flow lists, each level holding ``fan`` references to the one below."""
    text = "&a0 [x]"
    for level in range(1, levels + 1):
        text = f"&a{level} [{text}" + ",".join([""] + [f"*a{level - 1}"] * (fan - 1)) + "]"
    return text


def _dict_bomb(levels: int, fan: int = 9) -> str:
    text = "&m0 {k: x}"
    for level in range(1, levels + 1):
        first = f"k0: {text}"
        rest = [f"k{i}: *m{level - 1}" for i in range(1, fan)]
        text = f"&m{level} {{{', '.join([first, *rest])}}}"
    return text


def _rules_file(tmp_path: Path, body: str, *, triggers: str = "[delete]") -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(
        f"rules:\n  bomb:\n    triggers: {triggers}\n    {body}\n", encoding="utf-8"
    )
    return path


def _variants(levels: int) -> dict[str, tuple[str, str]]:
    """name -> (rule body, triggers) placing a bomb in one field."""
    lst, dct = _list_bomb(levels), _dict_bomb(levels)
    ok = "[delete]"
    required = "required_prosody: {emotion: [calm]}\n    "
    return {
        "retries": (f"{required}verification: {{retries: {lst}}}", ok),
        "method": (f"{required}verification: {{method: {lst}}}", ok),
        "verification_as_list": (f"{required}verification: {lst}", ok),
        "required_as_list": (f"required_prosody: {lst}", ok),
        "speaking_rate": (f"required_prosody: {{speaking_rate: {lst}}}", ok),
        "pitch_variance": (f"required_prosody: {{pitch_variance: {lst}}}", ok),
        "emotion_list": (f"forbidden_prosody: {{emotion: {lst}}}", ok),
        "emotion_mapping": (f"forbidden_prosody: {{emotion: {dct}}}", ok),
        "triggers": ("forbidden_prosody: {emotion: [angry]}", lst),
    }


BOMB_FIELDS = sorted(_variants(1))


class TestAliasBombs:
    @pytest.mark.parametrize("field", BOMB_FIELDS)
    def test_messages_stay_short_for_a_small_bomb(self, tmp_path: Path, field: str) -> None:
        # 6 levels: about 4 MB of text per unbounded repr, quick enough to run anywhere
        body, triggers = _variants(6)[field]

        with pytest.raises(ValueError) as excinfo:
            parse_rules_yaml(_rules_file(tmp_path, body, triggers=triggers))

        assert len(str(excinfo.value)) < 2_000
        assert "bomb" in str(excinfo.value)  # still names the rule

    @pytest.mark.parametrize("field", BOMB_FIELDS)
    def test_a_tiny_file_cannot_hang_the_loader(self, tmp_path: Path, field: str) -> None:
        body, triggers = _variants(10)[field]
        path = _rules_file(tmp_path, body, triggers=triggers)
        assert path.stat().st_size < 2_000
        repo_root = str(Path(__file__).resolve().parents[2])
        script = textwrap.dedent(
            """
            import resource, sys
            try:  # a runaway repr should fail fast instead of eating the machine
                resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
            except (ImportError, ValueError, OSError):
                pass
            from intent_engine.constitutional.rules import parse_rules_yaml
            try:
                parse_rules_yaml(sys.argv[1])
            except ValueError as exc:
                print(len(str(exc)))
            """
        )
        pythonpath = os.pathsep.join([repo_root, os.environ.get("PYTHONPATH", "")])
        env = {**os.environ, "PYTHONPATH": pythonpath}

        result = subprocess.run(
            [sys.executable, "-c", script, str(path)],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            check=False,
        )

        assert result.returncode == 0, result.stderr[-500:]
        assert int(result.stdout.strip()) < 2_000

    def test_ordinary_error_messages_keep_the_offending_value(self, tmp_path: Path) -> None:
        body = "required_prosody: {emotion: [calm]}\n    verification: {retries: -3}"
        path = _rules_file(tmp_path, body)

        with pytest.raises(ValueError, match="-3"):
            parse_rules_yaml(path)
