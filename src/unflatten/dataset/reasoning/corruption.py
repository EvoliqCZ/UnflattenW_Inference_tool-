"""D7 — controlled-corruption generator (RFC003-ERRATA-003 E3 rule 6,
`candidate_origin == "controlled_corruption"`).

Applies a register-defined corruption to a gold Canonical AST to produce a
corrective example whose inverse (the fix the model must recover) is known
by construction, for `task_kind == "reasoning_corrective_v0"`. This module
is the FRAMEWORK the errata ratifies:

- the seed is a pure function of example identity only — no worker id,
  batch index or wall-clock time (E3 rule 6: "free RNG is forbidden");
- `reasoning_corrective_v0` may only be materialized in the `train` split
  (E3 rule 3) — enforced at this call boundary, the split is supplied by
  the caller and never guessed;
- privacy is inherited from the authority, never declared by the
  corruption itself (E3 rule 7) — this module does not carry a
  `privacy_class` field at all, by construction.

The corruption register itself (which distortions a family applies) is
supplied by the caller's policy plug-in; none is chosen here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.dataset.identity import domain_hash

#: Identity domain for a corruption policy's own hash (bound into the
#: Input Ancestry `model_candidate` binding's `corruption_policy_hash`).
CORRUPTION_POLICY_HASH_DOMAIN = "rfc003.controlled-corruption-policy/v1"

#: Identity domain for the deterministic per-example seed (E3 rule 6).
CORRUPTION_SEED_DOMAIN = "rfc003.controlled-corruption-seed/v1"

#: The only split `reasoning_corrective_v0` may materialize in (E3 rule 3).
TRAIN_SPLIT = "train"


class CorruptionSplitRefusedError(ValueError):
    """Raised when the caller asks for anything but the `train` split."""


class CorruptionPolicy(Protocol):
    """A versioned plug-in supplying the corruption register."""

    policy_id: str
    policy_version: str

    def corrupt(
        self, canonical_ast: dict[str, Any], *, seed: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return `(corrupted_candidate_ast, inverse_operation)`.

        `seed` is the deterministic per-example seed from
        :func:`corruption_seed` — the policy MUST use it (or a value
        derived from it) for every random choice and no other entropy
        source, or the determinism guarantee this framework exists to
        provide does not hold.
        """
        ...


def corruption_policy_hash(policy: CorruptionPolicy) -> str:
    """`H(CORRUPTION_POLICY_HASH_DOMAIN, {policy_id, policy_version})`."""
    return domain_hash(
        CORRUPTION_POLICY_HASH_DOMAIN,
        {"policy_id": policy.policy_id, "policy_version": policy.policy_version},
    )


def corruption_seed(*, example_identity_hash: str) -> int:
    """Deterministic unsigned 64-bit seed from example identity alone.

    Mirrors `identity.operation_seed`'s projection: the first 8 bytes of
    the domain digest, big-endian. No worker id, batch index or wall-clock
    time ever enters this computation (E3 rule 6).
    """
    digest = hashlib.sha256(
        to_canonical_json(
            [CORRUPTION_SEED_DOMAIN, {"example_identity_hash": example_identity_hash}]
        )
    ).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class CorrectiveExample:
    """One generated `reasoning_corrective_v0` corruption result."""

    corrupted_candidate: dict[str, Any]
    inverse_operation: dict[str, Any]
    seed: int
    corruption_policy_hash: str


def generate_corrective_example(
    policy: CorruptionPolicy,
    *,
    canonical_ast: dict[str, Any],
    example_identity_hash: str,
    split: str,
) -> CorrectiveExample:
    """Run `policy` deterministically, refusing any split but `train`.

    Same `(policy, canonical_ast, example_identity_hash)` always yields a
    byte-identical `CorrectiveExample` — the seed is a pure function of
    `example_identity_hash` and nothing else reaches the policy.
    """
    if split != TRAIN_SPLIT:
        raise CorruptionSplitRefusedError(
            f"reasoning_corrective_v0 controlled corruption may only run for "
            f"split={TRAIN_SPLIT!r} (RFC003-ERRATA-003 E3 rule 3), got {split!r}"
        )
    seed = corruption_seed(example_identity_hash=example_identity_hash)
    corrupted, inverse = policy.corrupt(canonical_ast, seed=seed)
    return CorrectiveExample(
        corrupted_candidate=corrupted,
        inverse_operation=inverse,
        seed=seed,
        corruption_policy_hash=corruption_policy_hash(policy),
    )
