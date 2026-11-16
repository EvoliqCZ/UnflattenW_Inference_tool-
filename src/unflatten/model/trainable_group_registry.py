"""RFC004 §24 / packet R4-9 closed trainable-parameter-group registry.

Closes `OPEN-005-016` at the code level: DEC-005 (`docs/decisions/
DEC-005-dense-v0-contract-and-s1-canary.md`) and its authoritative packet
(`docs/plans/dense-v0-ratification-packet.md` §4, R4-9) closed the blocker at
governance level but left "machine-readable registry and the proof of
completeness/disjointness" outstanding (packet §4, last paragraph; §7 row
`trainable-group registry`). This module is that registry; the disjointness
and completeness proofs live in `tests/model/test_trainable_group_registry.py`
because the packet asks for them as *evidence*, not as a docstring claim.

Group taxonomy
--------------
Every group is exactly one of:

- ``trainable_primary_s1`` — one of the four dense V0 primary-S1-trainable
  groups (R5-3): ``adapter.projection``, ``document_context_encoder``,
  ``tree_text_decoder``, ``expert_adapter.dense``.
- ``frozen_base`` — a base (non-LoRA) group that owns real tensors/outputs in
  the dense V0 architecture but is frozen or diagnostic-only in primary S1:
  the three backbone slices, the three ownership-table typed-head/table-plan
  groups, ``resource_recovery`` and ``node_visual_aligner`` (the S2
  region/visual aligner, DEC-027 — a real, instantiated child, frozen until
  S2).
- ``not_applicable`` — a group RFC004 names but dense V0 does not instantiate
  at all (``visual_compressor``, ``speculators``, ``confidence_heads``).
  Their state is *always* ``not_applicable``, not a
  degenerate case of ``frozen`` — this mirrors the already-ratified
  :data:`unflatten.model.capability_profile.PRIMARY_S1_CAPABILITY_PROFILE`,
  whose payload pins ``sparse_moe`` / ``visual_compressor`` / ``speculators``
  / ``typed_properties`` to the literal string ``"not_applicable"``.
- ``lora`` — one of the four closed LoRA delta groups (R4-9). A LoRA group's
  ``parent_group_id`` names its base group, but a LoRA delta is never a
  member of that base group's owned-tensor set (packet §4: "LoRA delta nie
  je zároveň členom parent base group").

Two ambiguities in the source packet, resolved here rather than guessed
silently (see the module docstring cross-reference in the delivery report):

1. R5-3's "Frozen:" list folds together groups that hold real frozen tensors
   in dense V0 (backbone.*, typed_heads.*, table_plan, resource_recovery,
   node_visual_aligner) and groups RFC004 names for the *target*
   architecture but dense V0 never instantiates (visual_compressor,
   speculators, confidence_heads, sparse-MoE). This registry follows the
   more specific, already-ratified evidence in ``capability_profile.py``
   and treats the second set as ``not_applicable``, never ``frozen``.
2. R5-3 also says "all sparse-MoE groups" are frozen, but no packet clause
   anywhere names an exact sparse-MoE group ID, and R4-9 requires the
   enumeration to be exact with no wildcard. Since dense V0's own
   capability profile pins ``expert_adapter_mode = "dense"`` and
   ``sparse_moe = "not_applicable"`` (no router, no specialists
   instantiated), there is no tensor for a sparse-MoE group to own in this
   architecture, so none is registered. Inventing a placeholder ID here
   would violate the "no wildcard, exact ID" instruction more than omitting
   it does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from unflatten.model.hashing import domain_hash

_REGISTRY_DOMAIN = "rfc004.trainable-group-registry.v0"

#: Disposition values a group may be assigned during a run.
S1State = str  # "trainable" | "frozen" | "diagnostic_only" | "not_applicable"

#: The four exact LoRA group IDs (packet §4, R4-9). Closed — no other
#: `lora.*` ID may ever validate.
CLOSED_LORA_GROUP_IDS: frozenset[str] = frozenset(
    {
        "lora.backbone.visual",
        "lora.backbone.language",
        "lora.backbone.projector",
        "lora.tree_text_decoder",
    }
)


@dataclass(frozen=True)
class TrainableGroup:
    """One row of the closed parameter-group registry."""

    group_id: str
    kind: str  # "trainable_primary_s1" | "frozen_base" | "not_applicable" | "lora"
    parent_group_id: str | None
    s1_allowed_states: frozenset[str]
    owned_tensor_identities: tuple[str, ...]


def _group(
    group_id: str,
    kind: str,
    s1_allowed_states: frozenset[str],
    owned_tensor_identities: tuple[str, ...] = (),
    parent_group_id: str | None = None,
) -> TrainableGroup:
    return TrainableGroup(
        group_id=group_id,
        kind=kind,
        parent_group_id=parent_group_id,
        s1_allowed_states=frozenset(s1_allowed_states),
        owned_tensor_identities=owned_tensor_identities,
    )


#: The closed registry itself, keyed by exact group ID. Ordering here is the
#: canonical enumeration order used by :func:`registry_projection`.
_REGISTRY: dict[str, TrainableGroup] = {
    g.group_id: g
    for g in (
        # --- R5-3 primary S1 trainable groups (exactly four) -------------
        _group(
            "adapter.projection",
            "trainable_primary_s1",
            frozenset({"trainable"}),
            ("adapter.projection/base_weights",),
        ),
        _group(
            "document_context_encoder",
            "trainable_primary_s1",
            frozenset({"trainable"}),
            ("document_context_encoder/base_weights",),
        ),
        _group(
            "tree_text_decoder",
            "trainable_primary_s1",
            frozenset({"trainable"}),
            (
                "tree_text_decoder/structural_additions",
                "tree_text_decoder/text_additions",
                "tree_text_decoder/base_weights",
            ),
        ),
        _group(
            "expert_adapter.dense",
            "trainable_primary_s1",
            frozenset({"trainable"}),
            ("expert_adapter.dense/base_weights",),
        ),
        # --- frozen-in-S1 base groups that own real dense V0 tensors -----
        _group(
            "backbone.visual",
            "frozen_base",
            frozenset({"frozen"}),
            ("backbone.visual/base_weights",),
        ),
        _group(
            "backbone.language",
            "frozen_base",
            frozenset({"frozen"}),
            ("backbone.language/base_weights",),
        ),
        _group(
            "backbone.projector",
            "frozen_base",
            frozenset({"frozen"}),
            ("backbone.projector/base_weights",),
        ),
        _group(
            # RFC004-ERRATA-001 E4: floating_image geometry SHARES this
            # tensor identity — the same property family must not be
            # trainable or frozen depending on whether the image floats.
            # E7.3 removed `alt_text` from this group's ownership: alt text
            # never reaches the rendered page, so no head owns it.
            "typed_heads.inline_image",
            "frozen_base",
            frozenset({"frozen", "not_applicable"}),
            (
                "inline_image/size",
                "inline_image/crop",
                "inline_image/rotation_mdeg",
                "floating_image/size",
                "floating_image/crop",
                "floating_image/rotation_mdeg",
            ),
        ),
        _group(
            # RFC004-ERRATA-001 E3: ONE anchor head for floating_image and
            # text_frame together. `$defs/anchor` is a single ratified
            # contract; two heads would duplicate it at parameter level,
            # which RFC001-ERRATA-004 E1 already rejected once. `z_order`
            # is deliberately NOT owned here (E7.1) — it is unobservable
            # when floating objects do not overlap.
            "typed_heads.anchor",
            "frozen_base",
            frozenset({"frozen", "not_applicable"}),
            (
                "anchor/placement",
                "anchor/wrap",
                "anchor/overlap_and_depth",
            ),
        ),
        _group(
            "typed_heads.table",
            "frozen_base",
            frozenset({"frozen", "not_applicable"}),
            ("table/layout_style_except_column_widths",),
        ),
        _group(
            "typed_heads.row_cell",
            "frozen_base",
            frozenset({"frozen", "not_applicable"}),
            (
                "row/layout",
                "cell/layout_style_except_grid_span",
            ),
        ),
        _group(
            "table_plan",
            "frozen_base",
            frozenset({"frozen", "diagnostic_only"}),
            (
                "table_plan/grid_column_count",
                "table_plan/row_refs",
                "table/columns/*/width_mpt",
                "cell/grid_column",
                "cell/col_span",
                "cell/row_span",
                "table_plan/boundary_claims",
            ),
        ),
        _group(
            "resource_recovery",
            "frozen_base",
            frozenset({"frozen", "not_applicable"}),
            (
                "resource_recovery/crop_bytes",
                "resource_recovery/raster_bytes",
                "resource_recovery/mime_validation",
                "resource_recovery/dimensions_validation",
                "resource_recovery/slot_resolution",
            ),
        ),
        _group(
            # DEC-027: the S2 region/visual aligner. A real, instantiated
            # child of the candidate model (unlike the not_applicable
            # groups below), frozen until S2, so it is a frozen_base group
            # owning its four submodule tensor identities.
            "node_visual_aligner",
            "frozen_base",
            frozenset({"frozen", "not_applicable"}),
            (
                "node_visual_aligner/query",
                "node_visual_aligner/attention",
                "node_visual_aligner/bbox_head",
                "node_visual_aligner/status_head",
            ),
        ),
        # --- groups RFC004 names but dense V0 never instantiates ---------
        _group(
            "visual_compressor",
            "not_applicable",
            frozenset({"not_applicable"}),
        ),
        _group(
            "speculators",
            "not_applicable",
            frozenset({"not_applicable"}),
        ),
        _group(
            "confidence_heads",
            "not_applicable",
            frozenset({"not_applicable"}),
        ),
        # --- the four closed LoRA delta groups ----------------------------
        _group(
            "lora.backbone.visual",
            "lora",
            frozenset({"not_applicable"}),
            ("lora.backbone.visual/delta",),
            parent_group_id="backbone.visual",
        ),
        _group(
            "lora.backbone.language",
            "lora",
            frozenset({"not_applicable"}),
            ("lora.backbone.language/delta",),
            parent_group_id="backbone.language",
        ),
        _group(
            "lora.backbone.projector",
            "lora",
            frozenset({"not_applicable"}),
            ("lora.backbone.projector/delta",),
            parent_group_id="backbone.projector",
        ),
        _group(
            "lora.tree_text_decoder",
            "lora",
            frozenset({"not_applicable"}),
            ("lora.tree_text_decoder/delta",),
            parent_group_id="tree_text_decoder",
        ),
    )
}


def trainable_group_registry() -> dict[str, TrainableGroup]:
    """Return the closed registry, keyed by exact group ID.

    A copy is returned so callers cannot mutate the module-level singleton.
    """
    return dict(_REGISTRY)


def registry_projection() -> dict[str, Any]:
    """Deterministic hash projection of the registry (RFC004 §3.1 shape)."""
    return {
        "trainable_group_registry_schema_version": "0.1.0",
        "groups": [
            {
                "group_id": g.group_id,
                "kind": g.kind,
                "parent_group_id": g.parent_group_id,
                "s1_allowed_states": sorted(g.s1_allowed_states),
                "owned_tensor_identities": list(g.owned_tensor_identities),
            }
            for g in sorted(_REGISTRY.values(), key=lambda g: g.group_id)
        ],
    }


TRAINABLE_GROUP_REGISTRY_HASH: str = domain_hash(_REGISTRY_DOMAIN, registry_projection())


def is_known_group_id(group_id: str) -> bool:
    """True iff `group_id` is an exact member of the closed registry."""
    return group_id in _REGISTRY


def validate_lora_group_id(group_id: str) -> None:
    """Reject any `lora.*` ID outside :data:`CLOSED_LORA_GROUP_IDS`.

    Raises ``ValueError`` for an unknown `lora.*` ID. Non-LoRA IDs are not
    this function's concern and are accepted without a lookup here.
    """
    if group_id.startswith("lora.") and group_id not in CLOSED_LORA_GROUP_IDS:
        raise ValueError(
            f"{group_id!r} is not one of the closed dense-V0 LoRA group IDs "
            f"{sorted(CLOSED_LORA_GROUP_IDS)}; RFC004 packet R4-9 forbids "
            "any other lora.* group, it does not implicitly allow it"
        )


def validate_s1_state(group_id: str, state: S1State) -> None:
    """Raise ``ValueError`` if `state` is not allowed for `group_id` in S1.

    Raises ``KeyError`` if `group_id` is not in the closed registry at all.
    """
    group = _REGISTRY[group_id]
    if state not in group.s1_allowed_states:
        raise ValueError(
            f"group {group_id!r} may not be assigned S1 state {state!r}; "
            f"allowed states are {sorted(group.s1_allowed_states)}"
        )


def tensor_identity_owners() -> dict[str, str]:
    """Map every owned tensor identity to its single owning group ID.

    Raises ``ValueError`` if any tensor identity is owned by more than one
    group — this is the disjointness proof's core primitive, not just a
    convenience accessor.
    """
    owners: dict[str, str] = {}
    for group in _REGISTRY.values():
        for identity in group.owned_tensor_identities:
            if identity in owners:
                raise ValueError(
                    f"tensor identity {identity!r} is owned by both "
                    f"{owners[identity]!r} and {group.group_id!r}"
                )
            owners[identity] = group.group_id
    return owners
