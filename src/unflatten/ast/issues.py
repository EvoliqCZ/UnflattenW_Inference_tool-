"""Structured validator issue model (RFC001 §50)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["info", "warning", "error", "fatal"]


@dataclass(frozen=True)
class Issue:
    """A single validation finding.

    Validators never raise on invalid documents — they return a list of
    Issue instances instead. `path` is a JSON Pointer (RFC 6901), e.g.
    "/sections/0/children/3".
    """

    code: str
    severity: Severity
    node_id: str | None
    path: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
