"""Producer of the S2 pyramid tier map (DEC-013, `spec-s2-exit-gate-metrics-design` F1).

The tier map `{document_bundle_id: pyramid_level}` is the input the S2
exit gate's criterion 6 consumes (`s2_gate` `--pyramid-tier-map`). It is
produced from the synthetic factory's own records for the cohort's
`source_dir`:

- `run_manifest.json` `documents[]` -- which `recipe_id` generated each bundle;
- `index.jsonl` -- the factory's per-bundle `pyramid_level` record.

The level is taken from the recipe (`recipes.PYRAMID_LEVELS`, the DEC-013
ratified assignment) and cross-checked against the index record: two
independent records of the same fact, so a contradiction is a factory
defect to be refused, not a value to arbitrate.

The map is scoped to the VERIFIED cohort, not the factory's full output:
the factory renders more documents than the cohort admitted (the P3
documents DEC-013 excluded), and criterion 6 counts `tier_counts` over
every value it is given -- an unadmitted document in the map would
silently distort the tier distribution the gate is supposed to measure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from unflatten.dataset.synthetic import recipes
from unflatten.model.s2_cohort import S2Cohort

_LEVEL_RE = re.compile(r"^P[1-5]$")


class TierMapRefused(ValueError):
    """The factory records cannot produce a trustworthy tier map for the
    cohort: a generation mismatch, an incomplete factory output, an
    unratified recipe, or two factory records that contradict each other.
    A refusal, not a repair -- the producer never picks a side."""


@dataclass(frozen=True)
class PyramidTierMap:
    """A verified, cohort-scoped tier map.

    `tiers` is read-only and contains exactly the cohort's bundles;
    `cohort_hash` and `factory_dir` pin what it was produced from.
    """

    tiers: MappingProxyType[str, str]
    factory_dir: str
    cohort_hash: str

    def as_dict(self) -> dict[str, str]:
        return dict(self.tiers)


def _refuse(message: str) -> TierMapRefused:
    return TierMapRefused(f"pyramid tier map refused: {message}")


def _read_manifest(factory: Path) -> dict[str, str]:
    manifest = json.loads((factory / "run_manifest.json").read_text(encoding="utf-8"))
    documents = manifest.get("documents") if isinstance(manifest, dict) else None
    if not isinstance(documents, list) or not documents:
        raise _refuse("run_manifest.json must carry a non-empty 'documents' list")
    recipe_by_bundle: dict[str, str] = {}
    for i, entry in enumerate(documents):
        if not isinstance(entry, dict):
            raise _refuse(f"run_manifest.json documents[{i}] must be an object")
        bundle_id = entry.get("document_bundle_id")
        recipe_id = entry.get("recipe_id")
        if (
            not isinstance(bundle_id, str)
            or not bundle_id
            or not isinstance(recipe_id, str)
            or not recipe_id
        ):
            raise _refuse(
                f"run_manifest.json documents[{i}] must carry non-empty 'document_bundle_id' "
                f"and 'recipe_id' strings"
            )
        if bundle_id in recipe_by_bundle:
            raise _refuse(f"duplicate document_bundle_id {bundle_id!r} in run_manifest.json")
        recipe_by_bundle[bundle_id] = recipe_id
    return recipe_by_bundle


def _read_index(factory: Path) -> dict[str, str]:
    level_by_bundle: dict[str, str] = {}
    for lineno, line in enumerate(
        (factory / "index.jsonl").read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        envelope = row.get("envelope") if isinstance(row, dict) else None
        bundle_id = envelope.get("document_bundle_id") if isinstance(envelope, dict) else None
        level = row.get("pyramid_level") if isinstance(row, dict) else None
        if not isinstance(bundle_id, str) or not bundle_id:
            raise _refuse(f"index.jsonl line {lineno} has no envelope.document_bundle_id")
        if not isinstance(level, str) or not _LEVEL_RE.match(level):
            raise _refuse(f"index.jsonl line {lineno} has no valid pyramid_level: {level!r}")
        if bundle_id in level_by_bundle and level_by_bundle[bundle_id] != level:
            raise _refuse(
                f"index.jsonl records bundle {bundle_id!r} at both "
                f"{level_by_bundle[bundle_id]} and {level} -- conflicting factory records"
            )
        level_by_bundle[bundle_id] = level
    return level_by_bundle


def build_pyramid_tier_map(factory_dir: Path | str, cohort: S2Cohort) -> PyramidTierMap:
    """Build the cohort-scoped tier map from the factory output at
    `factory_dir`.

    `cohort` must already be a VERIFIED `S2Cohort` (the loader's hash
    check is the precondition: a tier map over an unverified cohort would
    inherit its defects). Raises `TierMapRefused` on any generation
    mismatch, incomplete output, unratified recipe, or record
    contradiction; `FileNotFoundError` on a missing factory file.
    """
    factory = Path(factory_dir)
    declared = Path(cohort.source_dir).resolve()
    if factory.resolve() != declared:
        raise _refuse(
            f"factory directory {factory} does not match the cohort's pinned "
            f"source_dir {declared} -- a generation mismatch"
        )

    recipe_by_bundle = _read_manifest(factory)
    index_level_by_bundle = _read_index(factory)

    tiers: dict[str, str] = {}
    for bundle in cohort.bundles:
        bundle_id = bundle.document_bundle_id
        if bundle_id not in recipe_by_bundle:
            raise _refuse(
                f"cohort bundle {bundle_id!r} is not in the factory run_manifest.json"
            )
        if bundle_id not in index_level_by_bundle:
            raise _refuse(
                f"cohort bundle {bundle_id!r} is not in the factory index.jsonl -- "
                "incomplete factory output"
            )
        recipe_id = recipe_by_bundle[bundle_id]
        try:
            level = recipes.pyramid_level(recipe_id)
        except recipes.UnknownRecipeError as exc:
            raise _refuse(
                f"cohort bundle {bundle_id!r} was generated by recipe {recipe_id!r}, "
                "which has no DEC-013 pyramid level"
            ) from exc
        if index_level_by_bundle[bundle_id] != level:
            raise _refuse(
                f"index.jsonl records bundle {bundle_id!r} at "
                f"{index_level_by_bundle[bundle_id]} but its recipe {recipe_id!r} is "
                f"{level} -- the factory records contradict"
            )
        tiers[bundle_id] = level

    return PyramidTierMap(
        tiers=MappingProxyType(tiers),
        factory_dir=str(factory),
        cohort_hash=cohort.cohort_hash,
    )


def write_pyramid_tier_map(tier_map: PyramidTierMap, path: Path | str) -> Path:
    """Write the map as the plain `{bundle_id: level}` JSON the
    `--pyramid-tier-map` flag reads."""
    path = Path(path)
    path.write_text(
        json.dumps(tier_map.as_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return path
