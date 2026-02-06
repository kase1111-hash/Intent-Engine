"""YAML rule schema and parser for constitutional rules.

Defines the data structures for constitutional rules and a parser
that loads them from YAML files. Rules define trigger conditions,
required/forbidden prosody characteristics, and verification
requirements for sensitive actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProsodyCondition:
    """Prosody conditions for a constitutional rule.

    Attributes
    ----------
    emotion:
        List of acceptable (or forbidden) emotion labels from the
        Prosody Protocol core vocabulary.
    pitch_variance:
        Expected pitch variance level: ``"low"``, ``"normal"``, or
        ``"high"``.  ``None`` means no constraint.
    speaking_rate:
        Acceptable speaking rate range as ``[min, max]`` in
        syllables per second.  ``None`` means no constraint.
    """

    emotion: list[str] = field(default_factory=list)
    pitch_variance: str | None = None
    speaking_rate: tuple[float, float] | None = None


@dataclass(frozen=True)
class Verification:
    """Verification requirements when prosody checks fail.

    Attributes
    ----------
    method:
        Verification method (``"explicit_confirmation"`` or
        ``"two_factor"``).
    retries:
        Number of retry attempts allowed.
    """

    method: str = "explicit_confirmation"
    retries: int = 1


@dataclass(frozen=True)
class ConstitutionalRule:
    """A single constitutional rule definition.

    Attributes
    ----------
    name:
        Rule identifier.
    triggers:
        List of intent keywords or phrases that activate this rule.
    required_prosody:
        Prosody conditions that MUST be met for the action to be
        allowed without verification.
    forbidden_prosody:
        Prosody conditions that BLOCK the action entirely.
    verification:
        Verification requirements when prosody checks fail.
        ``None`` means the action is denied outright.
    """

    name: str
    triggers: list[str] = field(default_factory=list)
    required_prosody: ProsodyCondition | None = None
    forbidden_prosody: ProsodyCondition | None = None
    verification: Verification | None = None


def _parse_prosody_condition(data: dict[str, Any] | None) -> ProsodyCondition | None:
    """Parse a prosody condition from a YAML dict."""
    if not data:
        return None

    speaking_rate = None
    if "speaking_rate" in data:
        sr = data["speaking_rate"]
        if isinstance(sr, (list, tuple)) and len(sr) == 2:
            speaking_rate = (float(sr[0]), float(sr[1]))

    return ProsodyCondition(
        emotion=data.get("emotion", []),
        pitch_variance=data.get("pitch_variance"),
        speaking_rate=speaking_rate,
    )


def _parse_verification(data: dict[str, Any] | None) -> Verification | None:
    """Parse a verification block from a YAML dict."""
    if not data:
        return None

    return Verification(
        method=data.get("method", "explicit_confirmation"),
        retries=int(data.get("retries", 1)),
    )


def parse_rules_yaml(path: str | Path) -> list[ConstitutionalRule]:
    """Parse constitutional rules from a YAML file.

    Parameters
    ----------
    path:
        Path to the YAML file containing rule definitions.

    Returns
    -------
    list[ConstitutionalRule]
        Parsed rules.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If the YAML structure is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "rules" not in data:
        raise ValueError(
            f"Invalid rules file: expected a top-level 'rules' key in {path}"
        )

    rules_data = data["rules"]
    if not isinstance(rules_data, dict):
        raise ValueError(
            f"Invalid rules file: 'rules' must be a mapping in {path}"
        )

    rules: list[ConstitutionalRule] = []
    for rule_name, rule_data in rules_data.items():
        if not isinstance(rule_data, dict):
            raise ValueError(
                f"Invalid rule '{rule_name}': expected a mapping"
            )

        rules.append(
            ConstitutionalRule(
                name=str(rule_name),
                triggers=rule_data.get("triggers", []),
                required_prosody=_parse_prosody_condition(
                    rule_data.get("required_prosody")
                ),
                forbidden_prosody=_parse_prosody_condition(
                    rule_data.get("forbidden_prosody")
                ),
                verification=_parse_verification(
                    rule_data.get("verification")
                ),
            )
        )

    return rules
