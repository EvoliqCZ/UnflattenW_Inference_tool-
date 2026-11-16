"""Build-time typed property targets for ONE page — RFC004 §24/§26
(DEC-021), RFC003-ERRATA-001 E3/E4, DEC-020.

Reads the typed surface of `dataset.targets.masks.apply_masks` output —
the single production withdrawal path — and turns it into the three flat
record types a typed head trains against:

``TypedPropertyTarget``      one per assigned `(node_id, pointer)` leaf
``TypedRepeatedCardinality`` one per repeated container instance
``TypedUnionArm``            one per union container instance

Three rules here are load-bearing, and each of them is a decision this
project has already paid for once:

**Raw values only.** A stored value is exactly the label's value —
never binned, quantized or re-encoded. Binning is *load* policy
(DEC-020): baking a codec into the cache would make an ablation of that
codec a rebuild.

**All 183 assigned pointers, including the 28 with no head family
today** (24 `string`, 4 `object`). Dropping them would cost a second
full rebuild when the RFC004 §25 colour codec lands. An `object`
pointer reaches this module only when its value is `null` —
`paragraph:/style/borders = null` is a real, known, supervisable label
("no borders"), not an absence.

**Both mask booleans, never one.** The `masks.py` placement pass forces
`observable` *and* `supervision` false, while the projection pass copies
a `NO_MODEL_CHANNEL` entry's scopes verbatim — observable ink with no
channel to emit it. A consumer storing only `supervision_mask` cannot
tell those two apart, and RFC005's masked-loss rule needs the
distinction.

Torch-free (it reads `sequence.loss_units`, which are plain dataclasses)
and mints no hash domain. Never binds `masks.apply_masks`'s recomputed
`target_bundle_hash`: identity stays with the canonical bundle
(RFC003-ERRATA-001 E4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from unflatten.dataset.targets.properties import normalize_pointer
from unflatten.model.capability_profile import STANDALONE_PAGE_PROFILE_IDS
from unflatten.model.loss_unit import EventRangeLocator
from unflatten.model.page_pieces import PieceKind, piece_kinds_for_page
from unflatten.model.piece_withdrawal import (  # noqa: F401 -- pre-move name
    LIST_UNOBSERVABLE_ON_CONTINUATION as _LIST_UNOBSERVABLE_ON_CONTINUATION,
)
from unflatten.model.piece_withdrawal import (  # noqa: F401 -- pre-move name
    PARAGRAPH_UNOBSERVABLE_ON_CONTINUATION as _PARAGRAPH_UNOBSERVABLE_ON_CONTINUATION,
)
from unflatten.model.piece_withdrawal import (  # noqa: F401 -- pre-move name
    PARAGRAPH_UNOBSERVABLE_SPACING_AFTER as _PARAGRAPH_UNOBSERVABLE_SPACING_AFTER,
)
from unflatten.model.piece_withdrawal import (  # noqa: F401 -- pre-move name
    ROW_UNOBSERVABLE_ON_CONTINUATION_TABLE as _ROW_UNOBSERVABLE_ON_CONTINUATION_TABLE,
)
from unflatten.model.piece_withdrawal import (
    withdrawn_pointers_for_piece as _withdrawn_pointers_for_piece,
)
from unflatten.model.property_head_registry import (
    EXCLUDED_PROPERTY_POINTERS,
    head_for_pointer,
)
from unflatten.model.typed_property_pointer_map import (
    DEFERRED,
    FAMILY,
    repeated_containers,
    union_containers,
)

#: The discriminant leaf of every union container in the surface
#: (RFC004 §26 / RFC004-ERRATA-001 §26.9): `line/mode`, `width/mode`,
#: `anchor/{horizontal,vertical,wrap}/mode`. A union with no such leaf is
#: refused rather than given a default arm.
_UNION_DISCRIMINANT = "mode"

#: DEC-033 §10 reason codes for `withdraw_page_piece_typed` -- disjoint
#: from the m2b `mask_projection` enum (`mask_projection.schema.json:73`),
#: which is document-scoped and stays untouched by this page-scoped pass.
WITHDRAWAL_REASON_CODES: tuple[str, ...] = (
    "CONTINUATION_PIECE_UNOBSERVABLE",
    "PAGE_PROJECTION_DROPPED",
)


class TypedTargetError(ValueError):
    """A typed target could not be derived fail-closed: a broken §29 mask
    conjunction, a pointer outside the registry, a repeated container with
    a gap in its indices, or a union whose arm cannot be named."""


@dataclass(frozen=True, slots=True)
class TypedPropertyTarget:
    """One assigned `(node_id, pointer)` leaf on this page.

    `node_property_pointer` keeps its indices — the index IS data, and a
    normalized pointer cannot say *which* tab stop it labels. `head_id` is
    re-derived from the registry rather than carried, so writer/reader
    registry drift is a refusal instead of a silent mismatch.
    """

    node_id: str
    node_type: str
    node_property_pointer: str
    value_type: str
    value: int | str | bool | None
    head_id: str
    known_mask: bool
    observable_mask: bool
    supervision_mask: bool
    anchor_token_index: int
    origin: str
    verification: str


@dataclass(frozen=True, slots=True)
class TypedRepeatedCardinality:
    """The element count of one repeated container instance.

    The true count, never capped: the full release carries maxima of
    8 / 64 / 118 across the three containers against 3 / 26 / 37 on a
    200-document sample — a sample-derived cap would have been 2.5-3.2x
    low in every one of them, deleting tab stops and table columns and
    teaching the model the truncated list was correct.
    """

    node_id: str
    container_pointer: str
    count: int
    known_mask: bool
    observable_mask: bool
    supervision_mask: bool
    anchor_token_index: int


@dataclass(frozen=True, slots=True)
class TypedUnionArm:
    """The active arm of one union container instance.

    Recorded even though `arm` is just the `mode` leaf's value: the
    *absence* of `offset_mpt` under `mode == "align"` is a schema-mandated
    `not_applicable` (RFC003 §29.1), and a consumer iterating leaf records
    alone cannot tell it from a withdrawal. On the anchor head that
    distinction covers 20 pointers.
    """

    node_id: str
    container_pointer: str
    arm: str
    arm_domain: tuple[str, ...]
    known_mask: bool
    observable_mask: bool
    supervision_mask: bool
    anchor_token_index: int


def effective_typed_masks(field: dict[str, Any]) -> tuple[bool, bool, bool]:
    """`(known, observable, supervision)` for one typed field.

    The scope keys are OPTIONAL: a field on a resolved node whose pointer
    the projection never assessed carries neither, so `field["…_mask"]`
    would be a `KeyError`. The defaults are not invented — they reproduce
    what `masks.py::_stamp_supervision_scopes` does on its
    assessed-and-unmasked branch, where `supervision` follows from
    `known AND observable`. Defaulting `supervision` to `True` instead
    would promote an unstated value to a supervised label.

    Raises `TypedTargetError` when the RFC003 §29 conjunction
    `supervision ⇒ known ∧ observable` is broken, or when an unknown field
    still carries a value.

    The converse of the second check — `value is None ⇒ not known` — is
    deliberately NOT asserted: `known ∧ value is None` is the legitimate
    "explicitly null" state, and asserting it would refuse every node
    without borders.
    """
    known = bool(field["known_mask"])
    observable = bool(field.get("observable_mask", True))
    supervision = bool(field.get("supervision_mask", known and observable))
    pointer = field.get("node_property_pointer")
    if supervision and not (known and observable):
        raise TypedTargetError(
            f"{pointer!r} has supervision_mask=True with known_mask={known} and "
            f"observable_mask={observable} — RFC003 §29.3 requires "
            "supervision_mask ⇒ known_mask ∧ observable_mask"
        )
    if not known and field.get("value") is not None:
        raise TypedTargetError(
            f"{pointer!r} has known_mask=False but carries the value "
            f"{field.get('value')!r} — an unknown label must be null (RFC003 §29.1)"
        )
    return known, observable, supervision


def build_typed_property_targets(
    *,
    masked_typed: dict[str, Any],
    fragment: dict[str, Any] | None,
    sequence: Any,
    structural: dict[str, Any],
    capability_profile_id: str | None = None,
) -> tuple[
    tuple[TypedPropertyTarget, ...],
    tuple[TypedRepeatedCardinality, ...],
    tuple[TypedUnionArm, ...],
    int,
]:
    """Derive this page's typed records from `masked_typed`.

    `masked_typed` is `masks.apply_masks(...)[0]["typed"]` — the whole
    DOCUMENT's typed surface. `fragment` selects the part of it this page
    owns; without that selection every node of every other page would
    count as "not opened here" and the fourth return value would report
    the document's size instead of this page's resumptions.

    Returns `(targets, cardinalities, union_arms,
    typed_targets_node_not_opened_here)`. The fourth element counts the
    typed-target-owning nodes that have NO opening structural event in
    this page's token stream — a resumed node. Those nodes get no typed
    target at all here: an anchor may never be a resume token, a nearest
    neighbour, or index 0.

    `capability_profile_id` selects node ownership (DEC-033 §10 /
    RFC003-ERRATA-002 E2): under `STANDALONE_V1_PROFILE_ID` a node
    contributes iff this page's fragment carries a present+supervised
    slice for it (`_present_supervised_node_ids`); every other id,
    including the default `None`, keeps the pre-existing
    `typed_target_owner` reading (`_owned_node_ids`) byte-identically.
    """
    anchors = _anchor_token_indices(sequence, structural)
    targets: list[TypedPropertyTarget] = []
    cardinalities: list[TypedRepeatedCardinality] = []
    union_arms: list[TypedUnionArm] = []
    not_opened_here = 0

    node_ids = (
        _present_supervised_node_ids(masked_typed, fragment)
        if capability_profile_id in STANDALONE_PAGE_PROFILE_IDS
        else _owned_node_ids(masked_typed, fragment)
    )
    for node_id in node_ids:
        entry = masked_typed[node_id]
        anchor = anchors.get(node_id)
        if anchor is None:
            not_opened_here += 1
            continue
        node_type = entry["node_type"]
        node_targets = [
            target
            for field in entry["fields"]
            for target in _target_for_field(node_id, node_type, field, anchor)
        ]
        targets.extend(node_targets)
        cardinalities.extend(
            _cardinalities(node_id, node_type, entry["fields"], anchor)
        )
        union_arms.extend(_union_arms(node_id, node_type, entry["fields"], anchor))

    return tuple(targets), tuple(cardinalities), tuple(union_arms), not_opened_here


def typed_target_payload(
    target: TypedPropertyTarget | TypedRepeatedCardinality | TypedUnionArm,
) -> dict[str, Any]:
    """The plain-dict payload for one record — the shape `loss_units`
    already uses on an `ExactTextCacheRow`."""
    if isinstance(target, TypedPropertyTarget):
        return {
            "node_id": target.node_id,
            "node_type": target.node_type,
            "node_property_pointer": target.node_property_pointer,
            "value_type": target.value_type,
            "value": target.value,
            "head_id": target.head_id,
            "known_mask": target.known_mask,
            "observable_mask": target.observable_mask,
            "supervision_mask": target.supervision_mask,
            "anchor_token_index": target.anchor_token_index,
            "origin": target.origin,
            "verification": target.verification,
        }
    if isinstance(target, TypedRepeatedCardinality):
        return {
            "node_id": target.node_id,
            "container_pointer": target.container_pointer,
            "count": target.count,
            "known_mask": target.known_mask,
            "observable_mask": target.observable_mask,
            "supervision_mask": target.supervision_mask,
            "anchor_token_index": target.anchor_token_index,
        }
    if isinstance(target, TypedUnionArm):
        return {
            "node_id": target.node_id,
            "container_pointer": target.container_pointer,
            "arm": target.arm,
            "arm_domain": list(target.arm_domain),
            "known_mask": target.known_mask,
            "observable_mask": target.observable_mask,
            "supervision_mask": target.supervision_mask,
            "anchor_token_index": target.anchor_token_index,
        }
    raise TypedTargetError(f"{type(target).__name__} is not a typed target record")


def typed_target_from_payload(
    payload: dict[str, Any],
) -> TypedPropertyTarget | TypedRepeatedCardinality | TypedUnionArm:
    """Read one record back, dispatching on its EXACT key set.

    Exact rather than "has a `count` key": a payload carrying an
    unexpected extra key is a shape this module did not write, and
    guessing which record it meant is how a reader silently drops a field
    a later revision added.
    """
    keys = frozenset(payload)
    if keys == frozenset(_PAYLOAD_KEYS[TypedPropertyTarget]):
        return TypedPropertyTarget(**payload)
    if keys == frozenset(_PAYLOAD_KEYS[TypedRepeatedCardinality]):
        return TypedRepeatedCardinality(**payload)
    if keys == frozenset(_PAYLOAD_KEYS[TypedUnionArm]):
        return TypedUnionArm(**{**payload, "arm_domain": tuple(payload["arm_domain"])})
    raise TypedTargetError(
        f"payload with keys {sorted(keys)} matches none of the three typed "
        "target record shapes — it was not written by this module"
    )


_PAYLOAD_KEYS: dict[type, tuple[str, ...]] = {
    TypedPropertyTarget: (
        "node_id",
        "node_type",
        "node_property_pointer",
        "value_type",
        "value",
        "head_id",
        "known_mask",
        "observable_mask",
        "supervision_mask",
        "anchor_token_index",
        "origin",
        "verification",
    ),
    TypedRepeatedCardinality: (
        "node_id",
        "container_pointer",
        "count",
        "known_mask",
        "observable_mask",
        "supervision_mask",
        "anchor_token_index",
    ),
    TypedUnionArm: (
        "node_id",
        "container_pointer",
        "arm",
        "arm_domain",
        "known_mask",
        "observable_mask",
        "supervision_mask",
        "anchor_token_index",
    ),
}


# ---------------------------------------------------------------------------
# Node selection and anchors
# ---------------------------------------------------------------------------


def _owned_node_ids(
    masked_typed: dict[str, Any], fragment: dict[str, Any] | None
) -> list[str]:
    """The typed-bearing nodes whose typed targets belong to this page.

    Document-level (`fragment is None`): all of them. Otherwise the
    fragment's `typed_target_owner` slices decide — a non-owning slice
    supervises no typed property in the first place
    (`masks.unsupervised_placements` reads the same field).
    """
    if fragment is None:
        return list(masked_typed)
    seen: dict[str, None] = {}
    for entry in fragment.get("slices", ()):
        node_id = entry["node_id"]
        if entry.get("typed_target_owner") and node_id in masked_typed:
            seen.setdefault(node_id, None)
    return list(seen)


def _present_supervised_node_ids(
    masked_typed: dict[str, Any], fragment: dict[str, Any] | None
) -> list[str]:
    """DEC-033 §10 / RFC003-ERRATA-002 E2: under `STANDALONE_V1_PROFILE_ID`
    ownership is not the criterion — each page independently supervises
    whatever it observes, with no cross-page deduplication. A node
    contributes typed targets iff this page's fragment carries a slice
    for it with `supervision_mask` true.

    A node `withdraw_page_piece_typed` dropped (§4's page projection) is
    simply absent from `masked_typed` by the time this runs, so it is
    never selected here -- this function does not re-check `dropped`
    itself.
    """
    if fragment is None:
        return list(masked_typed)
    seen: dict[str, None] = {}
    for entry in fragment.get("slices", ()):
        node_id = entry["node_id"]
        if entry.get("supervision_mask") and node_id in masked_typed:
            seen.setdefault(node_id, None)
    return list(seen)


def _anchor_token_indices(sequence: Any, structural: dict[str, Any]) -> dict[str, int]:
    """`node_id -> token position of the node's OWN opening event`.

    A node appears here only if this page's token stream carries the token
    its open/leaf event produced. The fragment encoder emits a resumed
    node's frame with no origin at all, and emits its CLOSE token carrying
    the *opening* event's origin — so the first token matching the pair
    `(ownership node_id, own open event index)` is the opening one, and a
    resumed node matches nothing.
    """
    opening: dict[str, int] = {}
    for event in structural["events"]:
        if event["op"] == "open":
            opening.setdefault(event["node_id"], event["index"])

    anchors: dict[str, int] = {}
    for token_index, unit in enumerate(sequence.loss_units):
        if not isinstance(unit.locator, EventRangeLocator):
            continue
        node_id = unit.ownership_proof.node_id
        if opening.get(node_id) == unit.locator.event_start:
            anchors.setdefault(node_id, token_index)
    return anchors


# ---------------------------------------------------------------------------
# Leaf targets
# ---------------------------------------------------------------------------


def _target_for_field(
    node_id: str, node_type: str, field: dict[str, Any], anchor: int
) -> list[TypedPropertyTarget]:
    """The record for one field, or `[]` when the pointer carries no head
    by an explicit registry decision."""
    pointer = field["node_property_pointer"]
    key = (node_type, normalize_pointer(pointer))
    if key not in FAMILY and key not in DEFERRED:
        if key in EXCLUDED_PROPERTY_POINTERS:
            return []
        raise TypedTargetError(
            f"node {node_id!r} carries the pointer {pointer!r} on {node_type!r}, which "
            "is neither an assigned typed-property pointer nor an explicitly excluded "
            "one — a registry-completeness bug, not a pointer to route around"
        )
    known, observable, supervision = effective_typed_masks(field)
    return [
        TypedPropertyTarget(
            node_id=node_id,
            node_type=node_type,
            node_property_pointer=pointer,
            value_type=field["value_type"],
            value=field["value"],
            head_id=head_for_pointer(*key),
            known_mask=known,
            observable_mask=observable,
            supervision_mask=supervision,
            anchor_token_index=anchor,
            origin=_required(field, "origin", node_id),
            verification=_required(field, "verification", node_id),
        )
    ]


def _required(field: dict[str, Any], key: str, node_id: str) -> str:
    value = field.get(key)
    if not isinstance(value, str):
        raise TypedTargetError(
            f"node {node_id!r} pointer {field['node_property_pointer']!r} has no "
            f"{key!r} — every typed label carries its provenance (RFC003 §29)"
        )
    return value


# ---------------------------------------------------------------------------
# Repeated containers
# ---------------------------------------------------------------------------


def _cardinalities(
    node_id: str, node_type: str, fields: list[dict[str, Any]], anchor: int
) -> list[TypedRepeatedCardinality]:
    """One record per repeated container instance the node emitted.

    The container set is the pinned `repeated_containers()` for this node
    type, not the set of pointers that happen to carry a `*` segment: an
    array pointer whose container is not in the pinned set supervises no
    count, so the write path consults the pinned set rather than
    re-deriving membership from the pointer.
    """
    repeated = repeated_containers()
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for field in fields:
        split = _split_container_index(field["node_property_pointer"])
        if split is None:
            continue
        container, index = split
        if (node_type, container) not in repeated:
            continue
        grouped.setdefault(container, {}).setdefault(index, []).append(field)

    records = []
    for container, by_index in grouped.items():
        indices = sorted(by_index)
        if indices != list(range(len(indices))):
            raise TypedTargetError(
                f"node {node_id!r} container {container!r} carries the indices "
                f"{', '.join(str(i) for i in indices)} — a repeated container must be "
                "exactly {0..count-1}; a gap is refused, never filled"
            )
        leaves = [field for index in indices for field in by_index[index]]
        known, observable, supervision = _conjunction(leaves)
        records.append(
            TypedRepeatedCardinality(
                node_id=node_id,
                container_pointer=container,
                count=len(indices),
                known_mask=known,
                observable_mask=observable,
                supervision_mask=supervision,
                anchor_token_index=anchor,
            )
        )
    return records


def _split_container_index(pointer: str) -> tuple[str, int] | None:
    """`("/layout/tabs", 1)` for `/layout/tabs/1/position_mpt`; `None` for
    a pointer with no array segment. The container pointer carries NO
    trailing `/*` — RFC004 §26 and DEC-020 name these containers without
    it."""
    raw = pointer.lstrip("/").split("/")
    normalized = normalize_pointer(pointer).lstrip("/").split("/")
    for position, segment in enumerate(normalized):
        if segment == "*":
            return "/" + "/".join(raw[:position]), int(raw[position])
    return None


def _conjunction(fields: list[dict[str, Any]]) -> tuple[bool, bool, bool]:
    """The masks of a container are the conjunction over its leaves: a
    partially withdrawn list has an unsupervised count."""
    known = observable = supervision = True
    for field in fields:
        leaf = effective_typed_masks(field)
        known &= leaf[0]
        observable &= leaf[1]
        supervision &= leaf[2]
    return known, observable, supervision


# ---------------------------------------------------------------------------
# Union containers
# ---------------------------------------------------------------------------


def _union_arms(
    node_id: str, node_type: str, fields: list[dict[str, Any]], anchor: int
) -> list[TypedUnionArm]:
    """One record per union container instance the node emitted.

    Arm domains come from `typed_property_pointer_map.union_containers()`
    in schema order — re-deriving them here would be a second mechanism
    for a mapping that already works.
    """
    by_pointer = {field["node_property_pointer"]: field for field in fields}
    records = []
    for (owner, container), arm_domain in union_containers().items():
        if owner != node_type:
            continue
        leaves = [p for p in by_pointer if p.startswith(container + "/")]
        if not leaves:
            continue
        mode = by_pointer.get(f"{container}/{_UNION_DISCRIMINANT}")
        if mode is None:
            raise TypedTargetError(
                f"node {node_id!r} emitted {len(leaves)} leaves under the union "
                f"container {container!r} but no {_UNION_DISCRIMINANT!r} leaf — the "
                "active arm cannot be named, and a union never gets a default arm"
            )
        known, observable, supervision = effective_typed_masks(mode)
        if not known:
            # No arm to record: the mode itself is unknown, and the leaf
            # target already carries `known_mask: false`. Refusing here
            # would abort the build on a legitimately partial label.
            continue
        if mode["value"] not in arm_domain:
            raise TypedTargetError(
                f"node {node_id!r} union {container!r} has arm {mode['value']!r}, "
                f"outside its schema arm domain {arm_domain}"
            )
        records.append(
            TypedUnionArm(
                node_id=node_id,
                container_pointer=container,
                arm=mode["value"],
                arm_domain=arm_domain,
                known_mask=known,
                observable_mask=observable,
                supervision_mask=supervision,
                anchor_token_index=anchor,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Page-piece withdrawal (DEC-033 §10, RFC003-ERRATA-002 E2)
# ---------------------------------------------------------------------------

#: The DEC-033 §10 lists and their lookup live in the dataset-free
#: `piece_withdrawal` (assembler design §3), re-imported above under
#: their original private names.


#: The `typed_withdrawn` pointer key for a whole-node drop (§4's page
#: projection): the node carries no single withdrawn pointer, its entire
#: typed surface is gone, so there is no real pointer to key the ledger
#: entry by. `*` is a sentinel, never a real `node_property_pointer`.
DROPPED_NODE_LEDGER_KEY = "*"


def withdraw_page_piece_typed(
    typed: dict[str, Any],
    pieces: dict[str, PieceKind],
    dropped: frozenset[str],
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    """DEC-033 §10 / RFC003-ERRATA-002 E2: withdraw the typed properties a
    page-split piece cannot observe, and remove nodes `page_projection`
    dropped from this page's presentation entirely (§4).

    Applied to the WHOLE page-masked typed surface (`masks.
    withdraw_placements_typed`'s output) before node selection: a dropped
    node is not merely masked off here, it carries NO typed target at all
    -- `_present_supervised_node_ids` never sees it, because it is simply
    absent from the returned dict.

    A withdrawal only sets `observable_mask`/`supervision_mask` False; it
    never writes a value (a missing label is never numeric zero). `kind`
    for a `row` node is its enclosing TABLE's piece kind, not its own --
    `pieces` is expected to already carry that
    (`page_pieces.piece_kinds_for_page`; a row's own `continuation` never
    says so, since a row is never itself split across pages).

    Returns `(copy, typed_withdrawn)` where `typed_withdrawn` is
    `{reason_code: {pointer: count}}` (design §4's ledger shape); a
    whole-node drop counts under `DROPPED_NODE_LEDGER_KEY`.
    """
    counts: dict[str, dict[str, int]] = {code: {} for code in WITHDRAWAL_REASON_CODES}
    result: dict[str, Any] = {}
    for node_id, entry in typed.items():
        if node_id in dropped:
            by_pointer = counts["PAGE_PROJECTION_DROPPED"]
            by_pointer[DROPPED_NODE_LEDGER_KEY] = by_pointer.get(DROPPED_NODE_LEDGER_KEY, 0) + 1
            continue
        withdrawn_pointers = _withdrawn_pointers_for_piece(
            entry["node_type"], pieces.get(node_id, "whole")
        )
        if not withdrawn_pointers:
            result[node_id] = entry
            continue
        new_fields = []
        for field in entry["fields"]:
            pointer = normalize_pointer(field["node_property_pointer"])
            if pointer in withdrawn_pointers:
                field = {**field, "observable_mask": False, "supervision_mask": False}
                by_pointer = counts["CONTINUATION_PIECE_UNOBSERVABLE"]
                by_pointer[pointer] = by_pointer.get(pointer, 0) + 1
            new_fields.append(field)
        result[node_id] = {**entry, "fields": new_fields}
    return result, counts


def page_piece_kinds(
    fragment: dict[str, Any], structural: dict[str, Any]
) -> dict[str, PieceKind]:
    """`node_id -> piece_kind` for every node with a slice on this page,
    for `withdraw_page_piece_typed` -- `page_pieces.piece_kinds_for_page`
    under this module's established name (kept here as a thin alias so
    the T7 call sites and tests need no rename)."""
    return piece_kinds_for_page(fragment, structural)
