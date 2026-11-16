"""Model-free assembly of train-cohort sealed artefacts for the S4.5 failure taxonomy (DP3).

``assemble_train_sealed_artefacts`` turns the results of an *injected*
``decode_fn`` into ``SealedCohortArtefact`` objects, one per distinct page index
``k`` (``build_report`` keys gold by bundle id and one ``page_index`` per cohort,
so a bundle's page 0 and page 3 cannot share an artefact). It never opens a
checkpoint, never calls a model and has no optimiser: the real decoder is bound
by a later task under its own ratification. Torch-free.

Guards, each a named refusal:

* the role passes the taxonomy's ``check_role`` (no test / locked / p6) AND the
  selection's train-cohort assertion (token ``train``, no val / dev / p5 ...). The
  page index is never part of the role string (a ``p5`` token would be refused);
* ``cache_identity["split"]`` must be ``"train"``. This is the opposite-sign
  guard of ``eval.s2_gate.decode.refuse_unless_eval_cache_identity_ok`` (which
  refuses ``train``): the taxonomy-for-selection reads the train cache only;
* a decoded page must carry the bundle id it was asked for, and a page index in
  ``pages`` must be one of ``page_indices`` (no silently dropped page).

The artefact has no field for ``k``: the i-th returned artefact belongs to
``sorted(set(page_indices))[i]``, and the caller passes ``inputs={"page_index":
k}`` to ``build_report``.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any, Final

from unflatten.eval.failure_taxonomy import report as ft
from unflatten.eval.s2_gate import sealed
from unflatten.model.hashing import domain_hash
from unflatten.model.rl import selection as sel

TRAIN_TAXONOMY_ROLE: Final = "s4_5_train_failure_mining"
COHORT_DOMAIN: Final = "unflatten.rl.train-taxonomy-cohort/v0"
TRAIN_SPLIT: Final = "train"

PageKey = tuple[str, int]


class TaxonomyAssemblyRefused(ValueError):
    """The inputs cannot back a train-only sealed artefact."""


def _page_key(page: Any) -> PageKey:
    key = getattr(page, "key", page)
    if (
        not isinstance(key, tuple)
        or len(key) != 2
        or not isinstance(key[0], str)
        or isinstance(key[1], bool)
        or not isinstance(key[1], int)
    ):
        raise TaxonomyAssemblyRefused(f"page {page!r} has no (bundle, page_index) key")
    return key[0], key[1]


def _check_role(role: str) -> None:
    try:
        ft.check_role(role)
        sel._assert_train_cohort(role)
    except (ft.ArtefactRoleRefused, sel.SelectionRefused) as exc:
        raise TaxonomyAssemblyRefused(f"role {role!r} refused: {exc}") from exc


def _train_identity(cache_identity: Mapping[str, Any], checkpoint_id: str) -> dict[str, Any]:
    split = cache_identity.get("split")
    if split != TRAIN_SPLIT:
        raise TaxonomyAssemblyRefused(
            f"cache identity split is {split!r}, not {TRAIN_SPLIT!r}: the failure taxonomy "
            "feeding the frozen hard-example selection reads the train cache only"
        )
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise TaxonomyAssemblyRefused("checkpoint_id must be a non-empty string")
    own = cache_identity.get("checkpoint_id")
    if own is not None and own != checkpoint_id:
        raise TaxonomyAssemblyRefused(
            f"cache identity checkpoint_id {own!r} != checkpoint_id {checkpoint_id!r}"
        )
    return {**dict(cache_identity), "checkpoint_id": checkpoint_id}


def _group_by_index(pages: Iterable[Any], page_indices: Collection[int]) -> dict[int, list[Any]]:
    allowed = set(page_indices)
    if not allowed:
        raise TaxonomyAssemblyRefused("page_indices is empty")
    grouped: dict[int, list[Any]] = {k: [] for k in sorted(allowed)}
    seen: set[PageKey] = set()
    for page in pages:
        key = _page_key(page)
        if key in seen:
            raise TaxonomyAssemblyRefused(f"page {key} appears twice")
        seen.add(key)
        if key[1] not in allowed:
            raise TaxonomyAssemblyRefused(
                f"page {key} has page_index {key[1]} outside page_indices {sorted(allowed)}"
            )
        grouped[key[1]].append(page)
    empty = [k for k, v in grouped.items() if not v]
    if empty:
        raise TaxonomyAssemblyRefused(f"page_indices {empty} have no page: no empty artefact")
    return grouped


def _decode_one(
    decode_fn: Callable[[Any], sealed.PageDecodeResult], page: Any
) -> sealed.PageDecodeResult:
    bundle = _page_key(page)[0]
    result = decode_fn(page)
    if not isinstance(result, sealed.PageDecodeResult):
        raise TaxonomyAssemblyRefused(
            f"decode_fn returned {type(result).__name__} for {bundle!r}, "
            "expected a sealed PageDecodeResult"
        )
    if result.document_bundle_id != bundle:
        raise TaxonomyAssemblyRefused(
            f"decode_fn answered bundle {result.document_bundle_id!r} for page of {bundle!r}"
        )
    return result


def assemble_train_sealed_artefacts(
    decode_fn: Callable[[Any], sealed.PageDecodeResult],
    pages: Iterable[Any],
    *,
    checkpoint_id: str,
    cache_identity: Mapping[str, Any],
    page_indices: Sequence[int],
    role: str = TRAIN_TAXONOMY_ROLE,
    arm: str = sealed.DecodeArm.TRAINED,
) -> list[sealed.SealedCohortArtefact]:
    """One sealed artefact per distinct page index (ascending), from ``decode_fn``.

    ``pages`` are ``(bundle, k)`` keys or objects with a ``.key`` of that shape;
    ``decode_fn`` receives the page object unchanged."""
    _check_role(role)
    identity = _train_identity(cache_identity, checkpoint_id)
    if arm not in sealed.VALID_ARMS:
        raise TaxonomyAssemblyRefused(f"arm {arm!r} is not one of {sorted(sealed.VALID_ARMS)}")
    grouped = _group_by_index(pages, page_indices)
    out: list[sealed.SealedCohortArtefact] = []
    for k, group in grouped.items():
        results = tuple(_decode_one(decode_fn, p) for p in group)
        n = len(results)
        cohort = domain_hash(
            COHORT_DOMAIN,
            {
                "role": role,
                "checkpoint_id": checkpoint_id,
                "page_index": k,
                "bundles": sorted(r.document_bundle_id for r in results),
            },
        )
        out.append(
            sealed.SealedCohortArtefact(
                role=role,
                arm=arm,
                eval_cache_identity=dict(identity),
                cohort_manifest_sha256=cohort,
                accounting=sealed.CohortAccounting(
                    eligible=n,
                    withdrawn_text_coverage_floor=0,
                    attempted=n,
                    decoded=n,
                    capability_excluded=0,
                    grammar_refused=0,
                    text_encoding_failed=0,
                    over_length=0,
                ),
                pages=results,
            )
        )
    return out


def bundle_lineage_from_split_records(records: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """``{bundle: lineage_root_id}`` over the train-split records of a verified release.

    ``records`` is ``adapters.release.read_split_records_verified`` output (bundle ==
    ``authority_id``, as in ``rl.data``). Records of any other split are left out, so
    a bundle that is not a verified train record stays unresolved (counted as
    ``unmapped_bundles`` by ``build_selection``, never guessed). A bundle seen twice
    with different lineages is refused."""
    mapping: dict[str, str] = {}
    for rec in records:
        if rec.get("split_name") != TRAIN_SPLIT:
            continue
        bundle, lineage = rec.get("authority_id"), rec.get("lineage_root_id")
        if not isinstance(bundle, str) or not isinstance(lineage, str):
            raise TaxonomyAssemblyRefused(f"split record lacks authority_id/lineage_root_id: {rec}")
        if mapping.setdefault(bundle, lineage) != lineage:
            raise TaxonomyAssemblyRefused(f"bundle {bundle!r} maps to two lineages")
    return mapping
