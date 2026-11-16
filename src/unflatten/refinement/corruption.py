"""RFC006 §38 Corruption Registry / offline corruption episode factory (R0).

V0 auto-patchable corruption operators only (§38.3): this factory
perturbs exactly one currently-present scalar typed property via
`set_typed_property`, which is already `target_catalog`'s whole
V0-patchable surface (font/run style, spacing/alignment/indent, table
width, cell padding/border/shading, row height, image extent —
`_STRUCTURE_LOCAL_KEYS`-excluded span/grid and section page geometry are
never touched, matching §38.4's diagnostic-only list). No RL/GRPO/PPO in the refiner (RFC006-ERRATA-001 §42.8(a)):
`_perturb` is a pure deterministic function of the seed, never a learned
policy.

Seed (§38.2): derived from parent lineage + corruption policy hash +
operator id/version + variant index — never worker/batch/time.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from unflatten.ast.hashing import ast_hash
from unflatten.model.typed_property_pointer_map import enum_domain
from unflatten.refinement.patch_applier import apply_patch_set
from unflatten.refinement.patch_language import ASTPatchSet, build_patch_set, invert_patch_set
from unflatten.refinement.target_catalog import TargetCatalog, TargetEntry, build_target_catalog

OPERATOR_ID = "set_typed_property.v0"
OPERATOR_VERSION = "0.1.0"


class NoPatchableTargetError(ValueError):
    """`build_target_catalog(gold_ast)` produced no entries to corrupt."""


class CrossSplitError(ValueError):
    """An episode's recorded `split` does not match the split its
    caller declared for the batch (RFC006 §38 R0: split is supplied by
    the caller, never inferred, and never mixed)."""


@dataclass(frozen=True)
class CorruptionEpisode:
    """`(corrupted, inverse_patch)` training episode (RFC006 §38.1,
    simplified to the fields this package actually produces)."""

    example_id: str
    split: str
    operator_id: str
    operator_version: str
    seed: int
    target_slot_id: str
    source_ast_hash: str
    corrupt_ast: dict[str, Any]
    corrupt_ast_hash: str
    inverse_patch: ASTPatchSet


@dataclass(frozen=True)
class CorruptionEpisodeSet:
    """A batch of episodes, all bound to one declared `split`."""

    split: str
    episodes: tuple[CorruptionEpisode, ...]


def derive_seed(
    *, example_id: str, config_hash: str, operator_id: str = OPERATOR_ID,
    operator_version: str = OPERATOR_VERSION, variant_index: int = 0,
) -> int:
    """RFC006 §38.2: parent lineage + corruption policy hash + operator
    id/version + variant index. Deliberately excludes worker/batch/time."""
    payload = "\0".join(
        (
            "rfc006.corruption-seed.v0",
            example_id,
            config_hash,
            operator_id,
            operator_version,
            str(variant_index),
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _perturb(entry: TargetEntry, rng: random.Random) -> Any:
    """A deterministic, different-from-current replacement value for one
    target entry, drawn from `rng` alone."""
    if entry.property_family == "boolean":
        return not entry.expected_value
    if entry.property_family == "enum":
        pointer_key = (entry.node_type, entry.node_property_pointer)
        domain = [v for v in enum_domain(*pointer_key) if v != entry.expected_value]
        if not domain:
            raise NoPatchableTargetError(
                f"{pointer_key!r} enum domain has no alternate value"
            )
        return domain[rng.randrange(len(domain))]
    # integer: a nonzero deterministic offset, never producing the same value.
    offset = rng.randrange(1, 1000)
    if rng.random() < 0.5:
        offset = -offset
    return entry.expected_value + offset


def make_episode(
    gold_ast: dict[str, Any],
    *,
    example_id: str,
    split: str,
    config_hash: str,
    variant_index: int = 0,
) -> CorruptionEpisode:
    """Build one deterministic corruption episode from a gold canonical
    AST (RFC006 §38, R0 — discovery only, no gradient)."""
    catalog: TargetCatalog = build_target_catalog(gold_ast)
    if not catalog.entries:
        raise NoPatchableTargetError(f"no patchable target in example {example_id!r}")

    seed = derive_seed(example_id=example_id, config_hash=config_hash, variant_index=variant_index)
    rng = random.Random(seed)
    entry = catalog.entries[rng.randrange(len(catalog.entries))]
    replacement = _perturb(entry, rng)

    forward = build_patch_set(catalog, [(entry.target_slot_id, replacement)])
    corrupt_ast, _ = apply_patch_set(gold_ast, forward)
    corrupt_hash = ast_hash(corrupt_ast)
    inverse = invert_patch_set(forward, corrupt_hash)

    return CorruptionEpisode(
        example_id=example_id,
        split=split,
        operator_id=OPERATOR_ID,
        operator_version=OPERATOR_VERSION,
        seed=seed,
        target_slot_id=entry.target_slot_id,
        source_ast_hash=catalog.base_ast_hash,
        corrupt_ast=corrupt_ast,
        corrupt_ast_hash=corrupt_hash,
        inverse_patch=inverse,
    )


def build_episode_set(split: str, episodes: list[CorruptionEpisode]) -> CorruptionEpisodeSet:
    """Seal a batch of episodes into one `split`. Refuses any episode
    whose own `split` disagrees — mixing is never silently allowed."""
    for episode in episodes:
        if episode.split != split:
            raise CrossSplitError(
                f"episode {episode.example_id!r} has split {episode.split!r}, expected {split!r}"
            )
    return CorruptionEpisodeSet(split=split, episodes=tuple(episodes))
