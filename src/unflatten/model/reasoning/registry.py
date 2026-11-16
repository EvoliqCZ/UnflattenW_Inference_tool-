"""Specialization registry -- RFC004-ERRATA-005 E1 rules 7-8.

Rule 7: "Doménová neutralita rámca. ... Špecializácia sa registruje ako
`reasoning.<family>.<version>` a inštancuje sa iba tá, ktorú vybral
vlastník." Rule 8: `continuation.*` (multi-page paragraph/table
continuation) belongs to the deterministic assembler under DEC-033 and is
not a selectable reasoning family until a separate decision reopens it.

This registry enforces both at registration time (`register`) and again at
instantiation time (`instantiate`), and never falls back to a default
specialization -- the caller must always name the id it wants.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from unflatten.model.reasoning.module import ReasoningModule

#: `reasoning.<family>.<version>`; family is a lowercase snake_case token,
#: version is any lowercase alnum/dot token (e.g. "v0" or "0.1.0").
_SPEC_ID_PATTERN = re.compile(
    r"^reasoning\.(?P<family>[a-z][a-z0-9_]*)\.(?P<version>[a-z0-9][a-z0-9_.]*)$"
)

#: E1 rule 8: DEC-033 owns continuation across pages; not selectable here.
_FORBIDDEN_FAMILY = "continuation"


class ReasoningSpecializationRejected(ValueError):
    """`spec_id` fails the naming rule or names a forbidden family (E1 rules 7-8)."""


class ReasoningSpecializationNotRegistered(KeyError):
    """`instantiate()` was asked for an id nothing has registered.

    There is no default specialization to fall back to (rule 7: "iba tá,
    ktorú vybral vlastník").
    """


_REGISTRY: dict[str, Callable[[], ReasoningModule]] = {}


def _validate_spec_id(spec_id: str) -> str:
    match = _SPEC_ID_PATTERN.match(spec_id)
    if not match:
        raise ReasoningSpecializationRejected(
            f"spec_id {spec_id!r} does not match 'reasoning.<family>.<version>' (E1 rule 7)"
        )
    family = match.group("family")
    if family == _FORBIDDEN_FAMILY:
        raise ReasoningSpecializationRejected(
            f"spec_id {spec_id!r} names the forbidden family "
            f"{_FORBIDDEN_FAMILY!r}: multi-page continuation belongs to the "
            "deterministic assembler under DEC-033 and is not selectable "
            "here until a separate decision reopens it (E1 rule 8)"
        )
    return spec_id


def register(spec_id: str, factory: Callable[[], ReasoningModule]) -> None:
    """Register a specialization factory under `spec_id`.

    Raises `ReasoningSpecializationRejected` if `spec_id` is malformed or
    names the forbidden `continuation` family. Re-registering the same
    `spec_id` overwrites the previous factory (last write wins), matching
    how other registries in this package behave (e.g. `events_vocab`).
    """
    _validate_spec_id(spec_id)
    _REGISTRY[spec_id] = factory


def instantiate(spec_id: str) -> ReasoningModule:
    """Build the specialization explicitly selected by `spec_id`.

    Never falls back to a default and never guesses: an unregistered id is a
    hard `ReasoningSpecializationNotRegistered`, not a silently returned
    no-op module.
    """
    _validate_spec_id(spec_id)
    try:
        factory = _REGISTRY[spec_id]
    except KeyError:
        raise ReasoningSpecializationNotRegistered(
            f"no reasoning specialization is registered under {spec_id!r}"
        ) from None
    return factory()


def registered_spec_ids() -> frozenset[str]:
    """The set of currently registered ids -- for tests and diagnostics only."""
    return frozenset(_REGISTRY)


def _unregister_all_for_tests() -> None:
    """Test-only reset so one test's registrations cannot leak into another."""
    _REGISTRY.clear()
