"""Sparse Word Expert Adapter topology manifest -- RFC004 §22.2-22.4, §23.

DEC-047 dormant-code condition 4 ("topológia a prahy sú parametre"): the
number of physical specialists, `top_k`, logical role names, the insertion
point and the grammar eligibility table are all fields of
:class:`SparseExpertAdapterManifest`, never module-level constants. Building
this dataclass does not itself select a topology -- nothing in the
production path constructs one (DEC-047 condition 1, 5; only tests do).

§22.2's seven logical roles are the reference vocabulary
(:data:`REFERENCE_LOGICAL_ROLES`); a manifest may reuse them for its
specialists or name additional roles -- §22.2 states a logical role need not
mean exactly one physical FFN, so the manifest does not enforce the
seven-role set as a closed enum.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

#: §22.2 reference logical expert registry, for documentation/reuse only --
#: not a closed enum a manifest must draw from (§22.2 last sentence).
REFERENCE_LOGICAL_ROLES: frozenset[str] = frozenset(
    {
        "shared_document",
        "structure_flow",
        "text_ocr",
        "paragraph_run",
        "table_grid",
        "section_layout",
        "image_resource",
    }
)

#: §23 lists three integration points; the RFC's reference V0 profile picks
#: the third (node-level bank before TypedHeadBank), but the manifest keeps
#: this a validated field rather than hardcoding it (DEC-047 condition 4).
INSERTION_POINTS: frozenset[str] = frozenset(
    {"adapter_stack", "decoder_ffn_layer", "node_level_bank"}
)


class SparseExpertAdapterManifestError(ValueError):
    """A manifest field violates an RFC004 §22/§23 invariant."""


@dataclass(frozen=True)
class ExpertRole:
    """One physical specialist slot.

    `expert_id` is the stable identifier used for deterministic tie-break
    ordering in routing (§22.3: "stable_expert_id_tie_break") and for
    telemetry/registry keys -- it is never re-derived from array position.
    """

    expert_id: str
    logical_role: str


@dataclass(frozen=True)
class SparseExpertAdapterManifest:
    """Validated, immutable sparse Word Expert Adapter topology.

    `eligible_logical_roles_by_context` implements §22.5's grammar
    eligibility mask: a mapping from a grammar-confirmed context key (e.g.
    ``"TABLE_OPEN"``, ``"TEXT"``, ``"SECTION"``, ``"INLINE_IMAGE"``) to the
    set of logical roles allowed in that context. §22.4 forbids the router
    from ever seeing dataset split, source, task-view id or target values --
    this manifest carries none of those fields, and the context keys this
    mapping is built from must come only from grammar-confirmed node
    type/decoder mode, never from those forbidden signals (enforced by the
    caller, not representable here).
    """

    d_model: int
    hidden_multiplier: int
    layers: int
    dropout: float
    top_k: int
    alpha_shared: float
    specialist_roles: tuple[ExpertRole, ...]
    insertion_point: str
    eligible_logical_roles_by_context: Mapping[str, frozenset[str]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise SparseExpertAdapterManifestError(f"d_model must be >= 1, got {self.d_model}")
        if self.hidden_multiplier < 1:
            raise SparseExpertAdapterManifestError(
                f"hidden_multiplier must be >= 1, got {self.hidden_multiplier}"
            )
        if self.layers < 1:
            raise SparseExpertAdapterManifestError(f"layers must be >= 1, got {self.layers}")
        if not len(self.specialist_roles):
            raise SparseExpertAdapterManifestError(
                "specialist_roles must be non-empty: a sparse adapter with "
                "zero specialists is dense mode, build DenseExpertAdapter instead"
            )
        expert_ids = [role.expert_id for role in self.specialist_roles]
        if len(expert_ids) != len(set(expert_ids)):
            raise SparseExpertAdapterManifestError(
                f"specialist_roles expert_id values must be unique, got {expert_ids}"
            )
        if self.top_k < 1 or self.top_k > len(self.specialist_roles):
            raise SparseExpertAdapterManifestError(
                f"top_k must be in [1, {len(self.specialist_roles)}], got {self.top_k}"
            )
        if self.insertion_point not in INSERTION_POINTS:
            raise SparseExpertAdapterManifestError(
                f"insertion_point {self.insertion_point!r} not one of {sorted(INSERTION_POINTS)}"
            )
        known_roles = {role.logical_role for role in self.specialist_roles}
        for context, roles in self.eligible_logical_roles_by_context.items():
            unknown = roles - known_roles
            if unknown:
                raise SparseExpertAdapterManifestError(
                    f"eligible_logical_roles_by_context[{context!r}] names "
                    f"unknown logical roles {sorted(unknown)}; known roles "
                    f"are {sorted(known_roles)}"
                )

    def eligibility_mask_for_context(self, context: str | None) -> tuple[bool, ...]:
        """Boolean mask over `specialist_roles`, in manifest order.

        `context is None` (or a context key absent from the table) means
        "no grammar-confirmed restriction": all specialists are eligible.
        A context present in the table but naming an empty role set makes
        every specialist ineligible for that context.
        """
        if context is None or context not in self.eligible_logical_roles_by_context:
            return tuple(True for _ in self.specialist_roles)
        allowed = self.eligible_logical_roles_by_context[context]
        return tuple(role.logical_role in allowed for role in self.specialist_roles)
