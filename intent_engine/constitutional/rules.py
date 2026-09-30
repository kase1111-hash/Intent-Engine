"""YAML rule schema and parser for constitutional rules.

Defines the data structures for constitutional rules and a parser
that loads them from YAML files. Rules define trigger conditions,
required/forbidden prosody characteristics, and verification
requirements for sensitive actions.

Rule file schema (see ``spec.md``)::

    rules:
      <rule_name>:
        triggers:                       # required, at least one phrase
          - "<phrase>"
        required_prosody:               # optional: every listed condition
          emotion: [<label>, ...]       #   must hold to pass without
          pitch_variance: <low|normal|high>  # verification
          speaking_rate: [<min>, <max>] # syllables per second
        forbidden_prosody:              # optional: blocks the action
          emotion: [<label>, ...]       #   (only ``emotion`` is supported)
        verification:                   # optional: what to do when
          method: <explicit_confirmation|two_factor>  # required_prosody
          retries: <int >= 0>           # fails; omit to deny outright
                                        # (needs required_prosody)

``speaking_rate`` is in syllables per second of speaking time, the unit of
``SpanFeatures.speech_rate`` (conversational speech is roughly 3-6), not
words per minute or a ratio to a normal pace.  ``pitch_variance`` is
measured in semitones (see ``PITCH_VARIANCE_THRESHOLDS`` in the evaluator).
Triggers are matched against the intent as whole word sequences, ignoring
case and ``_``/``-`` separators (``"delete all"`` matches
``"delete_all_files"``).

The schema is strict: an unknown key, a misspelled section or a
condition of the wrong shape raises ``ValueError`` when the rules are
loaded instead of being ignored, so a safety condition can never be
dropped silently.  That includes settings that could never take effect:
a ``verification`` block without ``required_prosody`` (it only applies
when that fails) and a condition that lists nothing.  A rule with neither
``required_prosody`` nor ``forbidden_prosody`` is valid (it always allows)
but logs a warning when loaded.  In particular the voice-quality, jitter, shimmer and
intensity measurements that ``prosody_protocol`` extracts are not
available as rule conditions.
"""

from __future__ import annotations

import logging
import math
import re
import reprlib
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Prosody Protocol core emotion vocabulary.
CORE_EMOTIONS = frozenset({
    "neutral", "sincere", "sarcastic", "frustrated", "joyful",
    "uncertain", "angry", "sad", "fearful", "surprised",
    "disgusted", "calm", "empathetic",
})

# Labels the built-in ``prosody_protocol.RuleBasedEmotionClassifier`` can emit
# (a3).  The other core labels only reach the filter from another source.
CLASSIFIER_EMOTIONS = frozenset({"neutral", "calm", "sad", "angry", "joyful", "fearful"})

# Accepted ``pitch_variance`` levels.
PITCH_VARIANCE_LEVELS = ("low", "normal", "high")

# Accepted verification methods, from least to most demanding: when several
# rules ask for verification, the last one listed here wins.
VERIFICATION_METHODS = ("explicit_confirmation", "two_factor")

# Error messages quote the offending YAML value.  Aliases share objects, so a
# few hundred bytes of nested anchors describe a value whose full repr is
# exponentially long (an "alias bomb"); quote a bounded excerpt instead.
_excerpt = reprlib.Repr()
_excerpt.maxlevel = 2
_excerpt.maxlist = _excerpt.maxtuple = _excerpt.maxset = _excerpt.maxdict = 4
_excerpt.maxstring = _excerpt.maxother = 60


def _show(value: object) -> str:
    """A short ``repr`` of *value* that is safe for arbitrarily nested data."""
    return _excerpt.repr(value)


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# ... and the end of a run of capitals before a capitalised word (HTTPServer)
_CAMEL_ACRONYM_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SEPARATORS = re.compile(r"[\W_]+")
# Format characters (zero-width joiner, soft hyphen, ...), combining marks and
# control characters: invisible, and whether one was meant as a word separator
# or as nothing cannot be told.
_IGNORABLE_CATEGORIES = frozenset({"Cf", "Mn", "Me", "Cc"})


def normalize_phrase(text: str) -> str:
    """Return *text* as lower-case words separated by single spaces.

    Case, ``_``/``-``/punctuation/whitespace separators and camelCase
    boundaries are ignored (and Unicode is NFKC-folded), so
    ``"delete_all_files"``, ``"Delete all files"`` and
    ``"deleteAllFiles"`` all normalise to ``"delete all files"``.
    """
    text = _CAMEL_BOUNDARY.sub(" ", unicodedata.normalize("NFKC", text))
    return _SEPARATORS.sub(" ", text.casefold()).strip()


def phrase_readings(text: str) -> tuple[str, ...]:
    """Return the distinct ways *text* can be read as lower-case words.

    The first is :func:`normalize_phrase`.  The others cover what that cannot
    decide: an invisible character inside a word (soft hyphen, zero-width
    joiner, NUL, a combining mark that does not compose) may have been meant
    as a separator or as nothing, and a run of capitals may or may not end a
    word (``deleteALLFiles`` is ``delete all files``, ``getIDs`` is
    ``get ids``).  Matching under any reading errs on the side of guarding an
    action; only NFKC folding is applied, so look-alike letters from another
    script are different letters.
    """
    folded = unicodedata.normalize("NFKC", text)
    bare = "".join(c for c in folded if unicodedata.category(c) not in _IGNORABLE_CATEGORIES)
    readings: dict[str, None] = {}
    for candidate in (folded, bare):
        for boundary in (_CAMEL_BOUNDARY, _CAMEL_ACRONYM_BOUNDARY):
            reading = _SEPARATORS.sub(" ", boundary.sub(" ", candidate).casefold()).strip()
            readings[reading] = None
    return tuple(readings)


def normalize_emotion(label: str) -> str:
    """Return an emotion label in the form rules and comparisons use."""
    return label.strip().lower()


def _str_tuple(value: Iterable[str], what: str) -> tuple[str, ...]:
    """Return *value* as a tuple of strings.

    A bare ``str`` is rejected: it would be iterated character by character.
    """
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise TypeError(f"{what} must be a list of strings, got {_show(value)}")
    items = tuple(value)
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"{what} must contain only strings, got {_show(item)}")
    return items


def _rate_range(value: Iterable[float]) -> tuple[float, float]:
    """Return *value* as a validated ``(min, max)`` pair of rates."""
    def bad() -> ValueError:
        return ValueError(
            f"speaking_rate must be [min, max] in syllables/second with "
            f"0 <= min <= max, got {_show(value)}"
        )

    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise bad()
    pair = tuple(value)
    if len(pair) != 2:
        raise bad()
    for bound in pair:
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            raise bad()
        if not math.isfinite(bound) or bound < 0:
            raise bad()
    low, high = float(pair[0]), float(pair[1])
    if low > high:
        raise bad()
    return (low, high)


@dataclass(frozen=True)
class ProsodyCondition:
    """Prosody conditions for a constitutional rule.

    Lists passed to the constructor are stored as tuples and validated, so
    a rule cannot be changed (or broken) after it has been built.

    Attributes
    ----------
    emotion:
        Acceptable (or forbidden) emotion labels, compared
        case-insensitively; normally from the Prosody Protocol core
        vocabulary.
    pitch_variance:
        Expected pitch movement within words: ``"low"``, ``"normal"``, or
        ``"high"``.  It is measured in semitones from the voiced F0
        contour, so it does not depend on the speaker's register (see
        ``PITCH_VARIANCE_THRESHOLDS`` in the evaluator for the bounds).
        ``None`` means no constraint.
    speaking_rate:
        Acceptable speaking rate range as ``[min, max]`` in
        syllables per second of speaking time, the unit of
        ``SpanFeatures.speech_rate`` (conversational speech is roughly
        3-6).  ``None`` means no constraint.
    """

    emotion: Sequence[str] = ()
    pitch_variance: str | None = None
    speaking_rate: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        labels = tuple(normalize_emotion(e) for e in _str_tuple(self.emotion, "emotion"))
        if not all(labels):
            raise ValueError("emotion labels must not be empty")
        object.__setattr__(self, "emotion", labels)

        level = self.pitch_variance
        if level is not None:
            if not isinstance(level, str) or level.strip().lower() not in PITCH_VARIANCE_LEVELS:
                raise ValueError(
                    f"pitch_variance must be one of {list(PITCH_VARIANCE_LEVELS)}, "
                    f"got {_show(level)}"
                )
            object.__setattr__(self, "pitch_variance", level.strip().lower())

        if self.speaking_rate is not None:
            object.__setattr__(self, "speaking_rate", _rate_range(self.speaking_rate))

        if not labels and self.pitch_variance is None and self.speaking_rate is None:
            raise ValueError(
                "a prosody condition needs at least one of emotion, pitch_variance "
                "and speaking_rate; an empty one checks nothing"
            )


@dataclass(frozen=True)
class Verification:
    """Verification requirements when prosody checks fail.

    Attributes
    ----------
    method:
        Verification method (``"explicit_confirmation"`` or
        ``"two_factor"``).
    retries:
        Number of retry attempts allowed.  Informational for now: the
        filter neither enforces it nor reports it in its ``Decision``.
    """

    method: str = "explicit_confirmation"
    retries: int = 1

    def __post_init__(self) -> None:
        method = self.method
        if not isinstance(method, str) or method.strip().lower() not in VERIFICATION_METHODS:
            raise ValueError(
                f"verification method must be one of {list(VERIFICATION_METHODS)}, "
                f"got {_show(method)}"
            )
        object.__setattr__(self, "method", method.strip().lower())

        retries = self.retries
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError(f"verification retries must be an integer >= 0, got {_show(retries)}")


@dataclass(frozen=True)
class ConstitutionalRule:
    """A single constitutional rule definition.

    Attributes
    ----------
    name:
        Rule identifier.
    triggers:
        Intent keywords or phrases that activate this rule (matched
        as whole word sequences, see ``match_triggers``).
    required_prosody:
        Prosody conditions that MUST be met for the action to be
        allowed without verification.  A measured condition
        (``pitch_variance``, ``speaking_rate``) that could not be measured
        is not met, and neither is an ``emotion`` condition when the
        emotion is unknown.
    forbidden_prosody:
        Emotions that BLOCK the action entirely (only ``emotion`` can be
        forbidden).  Nothing is blocked when the emotion is unknown, so
        pair it with a required emotion list to fail closed.
    verification:
        Verification requirements when ``required_prosody`` fails (so it
        needs ``required_prosody``).  ``None`` means the action is denied
        outright.
    """

    name: str
    triggers: Sequence[str] = ()
    required_prosody: ProsodyCondition | None = None
    forbidden_prosody: ProsodyCondition | None = None
    verification: Verification | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"rule name must be a non-empty string, got {_show(self.name)}")

        triggers = _str_tuple(self.triggers, "triggers")
        for trigger in triggers:
            if not normalize_phrase(trigger):
                raise ValueError(f"triggers must contain words, got {_show(trigger)}")
        object.__setattr__(self, "triggers", triggers)

        for section in ("required_prosody", "forbidden_prosody"):
            value = getattr(self, section)
            if value is not None and not isinstance(value, ProsodyCondition):
                raise TypeError(f"{section} must be a ProsodyCondition, got {_show(value)}")
        if self.verification is not None and not isinstance(self.verification, Verification):
            raise TypeError(f"verification must be a Verification, got {_show(self.verification)}")

        forbidden = self.forbidden_prosody
        if forbidden is not None and (
            forbidden.pitch_variance is not None or forbidden.speaking_rate is not None
        ):
            raise ValueError(
                "forbidden_prosody supports only 'emotion' "
                "(pitch_variance and speaking_rate can only be required)"
            )

        if self.verification is not None and self.required_prosody is None:
            # Verification is what a failed required_prosody asks for; without
            # it the block is never consulted and would read like a safeguard.
            raise ValueError(
                "verification only applies when required_prosody fails; add "
                "required_prosody or remove verification"
            )


_RULE_KEYS = ("triggers", "required_prosody", "forbidden_prosody", "verification")
_CONDITION_KEYS = ("emotion", "pitch_variance", "speaking_rate")
_VERIFICATION_KEYS = ("method", "retries")


class _UniqueKeyLoader(yaml.SafeLoader):
    """``SafeLoader`` that rejects duplicate mapping keys.

    PyYAML silently keeps the last of two equal keys, which would drop an
    earlier (possibly stricter) rule that has the same name.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=True)
            try:
                if key in seen:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"found duplicate key {_show(key)}; the earlier definition "
                        "would be silently discarded",
                        key_node.start_mark,
                    )
                seen.add(key)
            except TypeError:
                break  # Unhashable key: the parent class reports it.
        return super().construct_mapping(node, deep=deep)


def _check_keys(section: str, data: dict[Any, Any], allowed: tuple[str, ...]) -> None:
    """Reject keys of *data* that are not in *allowed*."""
    unknown = sorted(str(key) for key in data if key not in allowed)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in {section}; supported keys: {list(allowed)}")


def _require_mapping(section: str, data: Any) -> dict[Any, Any]:
    """Return *data* if it is a non-empty mapping, else raise."""
    if not isinstance(data, dict) or not data:
        raise ValueError(
            f"{section} must be a mapping with at least one entry (omit it to set none), "
            f"got {_show(data)}"
        )
    return data


def _parse_prosody_condition(section: str, data: Any) -> ProsodyCondition:
    """Parse a prosody condition from a YAML dict."""
    data = _require_mapping(section, data)
    _check_keys(section, data, _CONDITION_KEYS)
    for key in ("pitch_variance", "speaking_rate"):
        if key in data and data[key] is None:
            raise ValueError(f"{section}: {key} must not be empty")
    if "emotion" in data and data["emotion"] == []:
        raise ValueError(f"{section}: emotion must list at least one label")

    try:
        return ProsodyCondition(**data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{section}: {exc}") from exc


def _parse_verification(data: Any) -> Verification:
    """Parse a verification block from a YAML dict."""
    data = _require_mapping("verification", data)
    _check_keys("verification", data, _VERIFICATION_KEYS)
    try:
        return Verification(**data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"verification: {exc}") from exc


def _parse_rule(name: str, data: Any) -> ConstitutionalRule:
    """Parse one rule from a YAML dict."""
    if not isinstance(data, dict):
        raise ValueError("expected a mapping")
    _check_keys("the rule", data, _RULE_KEYS)

    triggers = data.get("triggers")
    if not isinstance(triggers, list) or not triggers:
        raise ValueError("triggers must be a non-empty list of phrases")

    sections: dict[str, Any] = {}
    for key in ("required_prosody", "forbidden_prosody"):
        if key in data:
            sections[key] = _parse_prosody_condition(key, data[key])
    if "verification" in data:
        sections["verification"] = _parse_verification(data["verification"])

    try:
        rule = ConstitutionalRule(name=name, triggers=triggers, **sections)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc

    if rule.required_prosody is None and rule.forbidden_prosody is None:
        logger.warning(
            "Constitutional rule '%s' has neither required_prosody nor forbidden_prosody: "
            "it matches its triggers but never restricts anything.",
            name,
        )

    labels = sorted({
        emotion
        for condition in (rule.required_prosody, rule.forbidden_prosody)
        if condition
        for emotion in condition.emotion
    })
    for emotion in labels:
        if emotion not in CORE_EMOTIONS:
            logger.warning(
                "Emotion %r in constitutional rule '%s' is not in the Prosody Protocol "
                "core vocabulary; it may never match.",
                emotion,
                name,
            )
    external = [e for e in labels if e in CORE_EMOTIONS and e not in CLASSIFIER_EMOTIONS]
    if external:
        logger.warning(
            "Constitutional rule '%s' uses emotion label(s) %s that the built-in "
            "RuleBasedEmotionClassifier never emits; they only match when the caller "
            "supplies emotions from another source.",
            name,
            ", ".join(external),
        )
    return rule


def parse_rules_yaml(path: str | Path) -> list[ConstitutionalRule]:
    """Parse constitutional rules from a YAML file.

    The file is read as UTF-8.  See the module docstring for the schema.

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
        If the file cannot be read or is not valid YAML, or the rules
        do not follow the schema.  The message names the file (and the
        rule, where one is at fault).
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Rules file not found: {path}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot read rules file {path}: {exc}") from exc

    try:
        # _UniqueKeyLoader is a SafeLoader: no arbitrary Python objects.
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc

    if not isinstance(data, dict) or "rules" not in data:
        raise ValueError(f"{path}: invalid rules file, expected a top-level 'rules' key")
    unknown = sorted(str(key) for key in data if key != "rules")
    if unknown:
        raise ValueError(f"{path}: unknown top-level key(s) {unknown}; only 'rules' is supported")

    rules_data = data["rules"]
    if not isinstance(rules_data, dict):
        raise ValueError(f"{path}: invalid rules file, 'rules' must be a mapping")
    if not rules_data:
        raise ValueError(f"{path}: 'rules' is empty, which would allow every action")

    rules: list[ConstitutionalRule] = []
    for rule_name, rule_data in rules_data.items():
        if not isinstance(rule_name, str):
            raise ValueError(f"{path}: rule name must be a string, got {_show(rule_name)}")
        try:
            rules.append(_parse_rule(rule_name, rule_data))
        except ValueError as exc:
            raise ValueError(f"{path}: rule '{rule_name}': {exc}") from exc

    return rules
