"""Decision dataclass -- output of ConstitutionalFilter.evaluate()."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    """Output of ``ConstitutionalFilter.evaluate()``.

    Describes whether an action is permitted, whether additional
    verification is needed, and why a request was denied.
    """

    allow: bool
    """Whether the action is permitted."""

    requires_verification: bool = False
    """Whether additional confirmation is needed before proceeding."""

    verification_method: str | None = None
    """Verification method to use (e.g., ``"explicit_confirmation"``, ``"two_factor"``)."""

    denial_reason: str | None = None
    """Human-readable reason for denial, or ``None`` if allowed."""
