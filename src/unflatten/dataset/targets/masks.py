"""Mask projection — observability vs. supervision scopes (RFC003 §29).

`known_mask` (the value is known; it lives in the Canonical Target
Bundle) is not `observable_mask` (the value is visible in the rendered
pixels) and neither is `supervision_mask` (the value may be trained on).
The §29 invariant is one-directional:

    supervision_mask = true  ⇒  known_mask = true AND observable_mask = true

The projection carries two producers.

**1. Font substitution.** The RFC002
renderer pins a font pack, so a run requesting a face outside the pack is
rendered in a substitute (`Arial → Liberation Sans`, an unmapped face →
the pinned default `DejaVu Sans`). The AST keeps the *requested* family —
correctly, the source really did say "Arial" — but the page shows the
substitute.

DEC-042 §1 (2026-09-23): a substituted run's requested name IS
supervised, and `observable_mask` for `/style/font_family` means exactly
this from here on: **the substitute the pixels show is a deterministic
function of `(requested family, FontPolicy)`**, not an arbitrary one, so
the requested name is exactly as recoverable from those pixels as it
would be from the requested face itself — the pack, not a guess, fixes
the association. `reason_code: "FONT_SUBSTITUTED"` (`FONT_FALLBACK`,
whether the substitute came from `fallback_map` or the pinned
`fallback_default`) therefore now carries `known_mask: true,
observable_mask: true, supervision_mask: true`: the entry stays in the
projection as a record that a substitution happened (telemetry,
`counts.affected_runs`), but it no longer withdraws supervision.
Measured motivation: before this fix, `/style/font_family` had 21 563
targets in the training cache and only 3 were supervised — this producer
was the reason (`docs/decisions/DEC-042-chybajuce-typove-rodiny.md` §1).

`reason_code: "FONT_UNAVAILABLE"` is different in kind, not degree, and
stays known-but-not-observable: it fires only when the compile is
non-strict and the requested face is in NEITHER `fallback_map` nor
covered by `fallback_default` (see below) — LibreOffice then picks an
arbitrary, unpinned system font the policy does not name. Nothing fixes
that association, so the requested name is not recoverable from those
pixels and training it would teach a guess.

The projection is derived WITHOUT rendering: it is a pure function of the
canonical AST and the pinned `FontPolicy`, computed by asking
:func:`unflatten.renderer.fonts.resolve_font` what the compiler will do.

Because the projection *is* a function of the policy, it carries the
policy's identity: `font_pack_hash` (the same digest the render profile
and the package manifest record). A projection derived under one policy
says nothing about another, and `ast_hash` cannot see the difference —
the same AST under an older pack matches every label and applies
silently, wrong in both directions (a run that now renders in its
requested face stays masked; a newly substituted run stays unmasked).

That check is therefore ENFORCED, not advised: :func:`apply_supervision_mask`
requires the `font_pack_hash` of the render whose pixels the caller
holds and refuses anything else, including a caller that cannot name it.
A guard nobody is obliged to call is a guard nobody calls — this one was
documented as a consumer duty and had no consumer for as long as it
existed. :func:`projection_is_stale` is the DERIVATION-side predicate,
for a caller that holds a live `FontPolicy` rather than a recorded
digest — `m2b_masks._policy_failure` asks it before backfilling a mask.
It is NOT the consumer check and must not be offered as one.

The field is part of the hashed payload, so `mask_projection_hash`
(and through it `task_view_id`) is exactly recomputable from
`(AST, policy)` alone, as RFC003 §12.1 requires.

`FONT_NOT_IN_PACK` (`require_exact`, or an unmapped face with no
`fallback_default`) depends on the compile's strictness:

* `strict=True` — NOT a mask entry. No substitution happens; the compile
  fails, so the document never reaches a render, a target bundle or a
  training example at all. Masking presupposes pixels to disagree with;
  a document that does not compile has none.
* `strict=False` — a mask entry with `reason_code: "FONT_UNAVAILABLE"`.
  The compile keeps going, the DOCX ships a family the pack does not
  carry, and LibreOffice renders an arbitrary unpinned system font. The
  requested family is known but not observable.

**2. No model channel** (DEC-003 §1.3). `hyperlink /target/uri` and
`note_ref /marker_text` are visible strings the model has no way to
produce: RFC004 §18.3 confines `TEXT_BEGIN…TEXT_END` to Run, and neither
value is an integer or an enum a typed head covers. They are `known` —
the round trip keeps them, so the rendered Word document has a live link
and a correct marker — but supervising them would train the model to
guess a URI from pixels that do not contain it. The two differ in
observability and the entries say so: the marker IS ink on the page
(`observable_mask: true`, withheld only for want of a channel), the URI
is NOT (`observable_mask: false`, it is never rendered as glyphs). Both
carry `supervision_mask: false`, and unlike the font entries this
withdrawal is a property of the ALPHABET, not of the render — it does
not depend on the font pack, but it travels in the same projection so
the one enforced application path (:func:`apply_masks`) applies it.

`header.kind` is the third string DEC-003 §1.3 names and is deliberately
NOT here: it is an enum, so a typed head covers it
(`properties.PROPERTY_FIELDS["header"]`), and it must be supervised —
`canonicalizer._STORY_KIND_ORDER` sorts stories by kind, so an
unsupervised kind is an `ast_hash` divergence waiting to happen.

Observability is assessed for the pointers in :data:`ASSESSED_POINTERS`
only. Every other property is *undetermined*:
:func:`apply_supervision_mask` leaves it without scope keys rather than
stamping "observable" on a question nobody asked.

Determinism: entries are sorted by ``(node_id, node_property_pointer)``
as plain strings, values are strings/booleans/integers only (no floats),
so `to_canonical_json` of the projection is byte-stable.
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

from unflatten.ast.hashing import ast_hash
from unflatten.ast.issues import Issue
from unflatten.dataset.identity import domain_hash
from unflatten.dataset.targets.bundle import target_bundle_hash
from unflatten.dataset.targets.events import preorder_nodes
from unflatten.renderer.fonts import resolve_font
from unflatten.renderer.manifest import compute_font_pack_hash

#: DEC-043 §5: bumped from "0.2.0" for the DEC-042 §1 font-substitution
#: semantics change below — the two versions disagree on what
#: `observable_mask`/`supervision_mask` mean for a `FONT_SUBSTITUTED`
#: entry, so a store mixing them must be distinguishable at a glance.
MASK_PROJECTION_SCHEMA_VERSION = "0.3.0"

#: Node-relative pointer (RFC003 §27 scope) of the masked run property.
FONT_FAMILY_POINTER = "/style/font_family"

#: Node-relative pointers of the visible strings with no model channel,
#: by node type (DEC-003 §1.3). The value is `(pointer, observable)`:
#: the note marker is ink on the page, the URI is not.
NO_CHANNEL_POINTERS: dict[str, tuple[str, bool]] = {
    "hyperlink": ("/target/uri", False),
    "note_ref": ("/marker_text", True),
}

#: Pointers whose observability the projection actually assesses. A
#: pointer outside this tuple is undetermined, NOT observable — see
#: `apply_supervision_mask`.
ASSESSED_POINTERS: tuple[str, ...] = (
    FONT_FAMILY_POINTER,
    *sorted(pointer for pointer, _ in NO_CHANNEL_POINTERS.values()),
)

class MaskAstMismatchError(ValueError):
    """The projection and the target bundle describe different ASTs.

    A mask entry names a `(node_id, pointer)` the bundle has no label
    for. Distinct from :class:`StaleMaskProjectionError` because the two
    say different things about the store: this is the known two-identity
    condition (review finding #11) and must be ZERO after the
    re-canonicalization, while a stale pack fires on nothing today.
    Both abort — the classes exist so a stopped build names its cause
    without anyone reading a traceback, not so either can be swallowed.
    """


class UnassessedMaskPointerError(ValueError):
    """A mask entry names a pointer the projection never assessed.

    A projection/policy contract violation, distinct from the other two:
    the projection and the bundle may describe the same AST under the
    right font pack and this still means the entry masks a property
    whose observability nobody determined. Dropping it silently would
    leak unobservable supervision.
    """


class StaleMaskProjectionError(ValueError):
    """The projection was not derived under the render's font pack.

    See :func:`apply_supervision_mask`. Also raised when the caller
    cannot name its render at all: unprovable is refused, not assumed.
    """


#: Identity domain of the projection hash (RFC003 §12.1 hash discipline).
MASK_PROJECTION_HASH_DOMAIN = "rfc003.mask-projection/v1"

_SCHEMA_PATH = Path(__file__).parent / "schemas" / "mask_projection.schema.json"


@lru_cache(maxsize=1)
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def build_mask_projection(
    canonical_ast: dict[str, Any], font_policy: Any, *, strict: bool = True
) -> dict[str, Any]:
    """The §29 mask projection: font substitution + no-channel strings.

    `font_policy` is the `RendererOptions.font_policy` used for the
    compile — the mask is only meaningful against the policy the pixels
    were produced under, and the policy's `font_pack_hash` is recorded in
    the projection so a consumer can prove that. Proving it is not
    optional: :func:`apply_supervision_mask` requires the render's
    recorded pack hash and refuses a projection from any other.

    `strict` is the `RendererOptions.strict` of that same compile: it
    decides whether a `FONT_NOT_IN_PACK` face aborts the compile (no
    entry) or ships unavailable into the render (`FONT_UNAVAILABLE`
    entry). See the module docstring.
    """
    entries: list[dict[str, Any]] = []
    total_runs = 0
    for node in preorder_nodes(canonical_ast):
        node_type = node.get("type")
        channel = NO_CHANNEL_POINTERS.get(node_type)
        if channel is not None:
            pointer, observable = channel
            entries.append(
                {
                    "node_id": node["id"],
                    "node_property_pointer": pointer,
                    "known_mask": True,
                    "observable_mask": observable,
                    "supervision_mask": False,
                    "reason_code": "NO_MODEL_CHANNEL",
                    "details": {"node_type": node_type},
                }
            )
            continue
        if node_type != "run":
            continue
        total_runs += 1
        requested = node["style"]["font_family"]
        _used, issue = resolve_font(
            requested,
            node["id"],
            FONT_FAMILY_POINTER,
            font_policy.mode,
            font_policy.fallback_map,
            font_policy.fallback_default,
        )
        if issue is None:
            # The face is in the pack, the pixels show it.
            continue
        if issue.code == "FONT_FALLBACK":
            # DEC-042 §1: the substitute is a deterministic function of
            # (requested, FontPolicy), so the requested name stays
            # supervised — see the module docstring.
            entry_fields = {
                "known_mask": True,
                "observable_mask": True,
                "supervision_mask": True,
                "reason_code": "FONT_SUBSTITUTED",
                "details": {
                    "requested": issue.details["requested"],
                    "used": issue.details["used"],
                    "via": issue.details["via"],
                },
            }
        elif not strict:
            # FONT_NOT_IN_PACK under a non-strict compile: the package
            # ships a family the pack does not carry, so the renderer
            # picks an arbitrary unpinned system font nothing determines.
            entry_fields = {
                "known_mask": True,
                "observable_mask": False,
                "supervision_mask": False,
                "reason_code": "FONT_UNAVAILABLE",
                "details": {"requested": issue.details["requested"]},
            }
        else:
            # Strict FONT_NOT_IN_PACK: the compile fails, no pixels exist.
            continue
        entries.append(
            {
                "node_id": node["id"],
                "node_property_pointer": FONT_FAMILY_POINTER,
                **entry_fields,
            }
        )
    font_entries = [e for e in entries if e["node_property_pointer"] == FONT_FAMILY_POINTER]
    entries.sort(key=lambda e: (e["node_id"], e["node_property_pointer"]))
    return {
        "mask_projection_schema_version": MASK_PROJECTION_SCHEMA_VERSION,
        "ast_hash": ast_hash(canonical_ast),
        "font_pack_hash": compute_font_pack_hash(font_policy),
        # `strict` decides whether a FONT_NOT_IN_PACK face is omitted
        # (strict compile aborts, no pixels exist) or masked as
        # FONT_UNAVAILABLE (ships unavailable into the render). Recording
        # it makes the derivation context self-describing: a consumer
        # holding only the projection can tell which rule produced it.
        "strict_compile": bool(strict),
        "assessed_pointers": list(ASSESSED_POINTERS),
        "entries": entries,
        "counts": {
            "entries": len(entries),
            # runs whose FACE the pack will not show — the font
            # producer's own count, not a count of every entry: a
            # no-channel entry is never a run.
            "affected_runs": len({e["node_id"] for e in font_entries}),
            "total_runs": total_runs,
        },
    }


def mask_projection_hash(projection: dict[str, Any]) -> str:
    """`H("rfc003.mask-projection/v1", projection)` as `sha256:<64hex>`.

    The projection hashes ITSELF minus any `mask_projection_hash` field,
    so embedding the result back into the record is a fixed point.
    """
    payload = {k: v for k, v in projection.items() if k != "mask_projection_hash"}
    return domain_hash(MASK_PROJECTION_HASH_DOMAIN, payload)


def projection_is_stale(record: dict[str, Any], font_policy: Any) -> bool:
    """True when `record` was NOT produced under `font_policy`.

    `record` is anything carrying a `font_pack_hash`: a mask projection,
    or an M2b `render_profile` (`m2b_masks._policy_failure` asks it of
    the row it is about to backfill a mask for). A record with no
    `font_pack_hash` at all compares unequal — but callers should treat
    "cannot name its pack" as its own outcome rather than folding it in
    here, which is what that one does.

    The DERIVATION-side predicate, for a caller that holds a live policy
    object and is deciding whether to rebuild. It is the WEAKER question
    and it is not the consumer check: it asks whether the projection matches the policy
    THIS process would construct, not whether it came from the pack that
    rendered the pixels. Those differ whenever the stored pixels and
    their projection are mutually consistent but both older than this
    code. A consumer holds pixels, so its check is enforced inside
    :func:`apply_supervision_mask` against the render's recorded
    `font_pack_hash` (the Observation Sidecar's
    `renderer.font_pack_hash`).
    """
    return record.get("font_pack_hash") != compute_font_pack_hash(font_policy)


def validate_mask_projection(projection: Any) -> list[Issue]:
    """Validate a mask projection; returns Issues, never raises."""
    issues = [
        Issue(
            code="MASK_SCHEMA_VIOLATION",
            severity="error",
            node_id=None,
            path="/" + "/".join(str(p) for p in error.absolute_path),
            message=error.message,
        )
        for error in sorted(
            _validator().iter_errors(projection), key=lambda e: str(e.absolute_path)
        )
    ]
    if issues:
        return issues
    issues.extend(_order_issues(projection))
    issues.extend(_scope_issues(projection))
    issues.extend(_count_issues(projection))
    return issues


def _order_issues(projection: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    keys = [(e["node_id"], e["node_property_pointer"]) for e in projection["entries"]]
    if keys != sorted(keys):
        issues.append(
            Issue(
                code="MASK_ENTRY_ORDER_VIOLATION",
                severity="error",
                node_id=None,
                path="/entries",
                message="entries are not sorted by (node_id, node_property_pointer).",
            )
        )
    if len(set(keys)) != len(keys):
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        issues.append(
            Issue(
                code="MASK_ENTRY_DUPLICATE",
                severity="error",
                node_id=None,
                path="/entries",
                message=f"duplicate (node_id, pointer) entries: {duplicates}.",
            )
        )
    return issues


def _scope_issues(projection: dict[str, Any]) -> list[Issue]:
    return [
        Issue(
            code="MASK_SCOPE_INVARIANT_VIOLATION",
            severity="error",
            node_id=entry["node_id"],
            path=f"/entries/{position}",
            message="supervision_mask=true requires known_mask and observable_mask "
            "(RFC003 §29).",
        )
        for position, entry in enumerate(projection["entries"])
        if entry["supervision_mask"]
        and not (entry["known_mask"] and entry["observable_mask"])
    ]


def _count_issues(projection: dict[str, Any]) -> list[Issue]:
    entries = projection["entries"]
    expected = {
        "entries": len(entries),
        "affected_runs": len(
            {
                e["node_id"]
                for e in entries
                if e["node_property_pointer"] == FONT_FAMILY_POINTER
            }
        ),
    }
    counts = projection["counts"]
    issues = [
        Issue(
            code="MASK_COUNT_MISMATCH",
            severity="error",
            node_id=None,
            path=f"/counts/{key}",
            message=f"counts.{key} is {counts[key]}, derived value is {value}.",
        )
        for key, value in expected.items()
        if counts[key] != value
    ]
    if counts["total_runs"] < counts["affected_runs"]:
        issues.append(
            Issue(
                code="MASK_COUNT_MISMATCH",
                severity="error",
                node_id=None,
                path="/counts/total_runs",
                message=f"counts.total_runs {counts['total_runs']} is below "
                f"affected_runs {counts['affected_runs']}.",
            )
        )
    return issues


def apply_supervision_mask(
    target_bundle: dict[str, Any],
    projection: dict[str, Any],
    *,
    render_font_pack_hash: str | None,
) -> dict[str, Any]:
    """Reference implementation of the §29 font withdrawal, kept as the
    readable statement of what the scopes mean. Production applies both
    withdrawals through :func:`apply_masks`, which is the only caller
    path; this function has none and is not one to add.

    It returns a COPY of `target_bundle` with supervision scopes stamped.

    `render_font_pack_hash` is the `font_pack_hash` the render recorded
    for the pixels this bundle is being masked FOR (the Observation
    Sidecar's `renderer.font_pack_hash`, or the M2b render profile).
    It is required, with no default: the projection is only meaningful
    against the render it was derived for, and a mismatch — or a caller
    that cannot name its render — is a ``ValueError``, never a silent
    application and never a silent skip. See the module docstring.

    Scopes are stamped ONLY on fields whose pointer the projection
    actually assessed (``projection["assessed_pointers"]``, V0
    ``/style/font_family``): a masked ``(node_id, pointer)`` gets
    ``observable_mask: false`` (its ``known_mask`` is untouched — the
    value IS known), an assessed-but-unmasked one ``observable_mask:
    true``, and ``supervision_mask`` is derived, never assumed:
    ``known_mask AND observable_mask`` (§29).

    Every other field is left exactly as it came: absence of the keys
    means *observability undetermined*, not observable. Stamping
    ``observable_mask: true`` on a property nobody assessed would be a
    default posing as ground truth — the very failure this projection
    exists to prevent.

    Three conditions abort, each with its own class so a stopped build
    names its cause (all three subclass ``ValueError``, so an existing
    broad catch keeps working, but a narrow catch must list all three):

    * :class:`StaleMaskProjectionError` — the projection was not derived
      under ``render_font_pack_hash``, or the caller passed ``None`` and
      cannot name its render at all;
    * :class:`MaskAstMismatchError` — a mask entry matches no typed
      label, so the projection and the bundle describe different ASTs;
    * :class:`UnassessedMaskPointerError` — an entry names a pointer
      outside ``assessed_pointers``.

    The strictness is that of
    :func:`unflatten.dataset.targets.properties.mask_unknown`: a
    silently dropped mask would leak unobservable supervision.
    """
    result = copy.deepcopy(target_bundle)
    _stamp_supervision_scopes(
        result, projection, render_font_pack_hash=render_font_pack_hash
    )
    # the stamped copy is a TaskViewProjection-scoped derivative, not the
    # canonical bundle: its §27 identity is recomputed so it stays
    # self-consistent, and it differs from the canonical one by design
    result["target_bundle_hash"] = target_bundle_hash(result)
    return result


def _stamp_supervision_scopes(
    result: dict[str, Any],
    projection: dict[str, Any],
    *,
    render_font_pack_hash: str | None,
) -> None:
    """Stamp the §29 scopes on `result` IN PLACE. See the callers."""
    carried = projection.get("font_pack_hash")
    if render_font_pack_hash is None or carried != render_font_pack_hash:
        raise StaleMaskProjectionError(
            f"mask projection was derived under font pack {carried!r} but the "
            f"render it is being applied to reports {render_font_pack_hash!r} — "
            "the projection says nothing about this render; re-derive it "
            "(RFC003 §29)"
        )
    assessed = set(projection["assessed_pointers"])
    masked = {
        (e["node_id"], e["node_property_pointer"]): e for e in projection["entries"]
    }
    unassessed_entries = sorted(key for key in masked if key[1] not in assessed)
    if unassessed_entries:
        raise UnassessedMaskPointerError(
            f"mask entries name pointers outside assessed_pointers: {unassessed_entries} — "
            "an entry can only mask a property the projection assessed"
        )
    matched: set[tuple[str, str]] = set()
    for node_id, entry in result["typed"].items():
        for field in entry["fields"]:
            pointer = field["node_property_pointer"]
            if pointer not in assessed:
                continue
            key = (node_id, pointer)
            entry = masked.get(key)
            if entry is None:
                # assessed and unmasked: observable, and supervision
                # follows from `known AND observable` (§29).
                field["observable_mask"] = True
                field["supervision_mask"] = bool(field["known_mask"])
                continue
            matched.add(key)
            # a masked field takes the ENTRY's scopes verbatim. Re-deriving
            # `known AND observable` here would restore supervision for a
            # NO_MODEL_CHANNEL marker, which is observable ink that the
            # model still has no way to emit.
            field["observable_mask"] = entry["observable_mask"]
            field["supervision_mask"] = entry["supervision_mask"]
    unmatched = sorted(set(masked) - matched)
    if unmatched:
        raise MaskAstMismatchError(
            f"mask entries match no typed label: {unmatched} — the projection and "
            "the target bundle are not derived from the same AST"
        )


#: Reason recorded for a node whose page placement is `unresolved`.
NO_PLACEMENT_SUPERVISION = "no_supervision"


def unsupervised_placements(fragments: list[dict[str, Any]]) -> dict[str, str]:
    """``node_id -> "no_supervision"`` for every unresolved placement.

    A slice is `unresolved` (RFC003 §26 `page_evidence`) when the node
    carries real text that alignment never found: it was placed on a
    borrowed page which may be the wrong page entirely, so its typed
    properties describe pixels nobody can point at. Only the node's
    `typed_target_owner` slice decides — a non-owning slice supervises
    no typed property in the first place.
    """
    return {
        entry["node_id"]: NO_PLACEMENT_SUPERVISION
        for fragment in fragments
        for entry in fragment["slices"]
        if entry.get("typed_target_owner") and entry.get("page_evidence") == "unresolved"
    }


def apply_placement_mask(
    target_bundle: dict[str, Any], fragments: list[dict[str, Any]]
) -> dict[str, Any]:
    """Reference implementation of the §29 placement withdrawal, kept as
    the readable statement of what it withdraws. Production applies both
    withdrawals through :func:`apply_masks`, which is the only caller
    path; this function has none and is not one to add.

    It returns a COPY of `target_bundle` with typed supervision withdrawn
    for every node whose placement is `unresolved`.

    All of the node's typed fields get ``observable_mask: false`` and
    ``supervision_mask: false`` — the value stays exactly as it is
    (`known_mask` is untouched: the label IS known). That is the
    explicit "no supervision" RFC005 requires; encoding the loss as a
    numeric zero would teach the model the borrowed page is right.

    The withdrawal is ordered after the font one — :func:`apply_masks`
    runs them in that order — because :func:`apply_supervision_mask`
    re-derives `supervision_mask` from `known AND observable` for the
    pointers it assesses and would otherwise restore what this one
    withdrew. The result is a TaskViewProjection-scoped derivative, so
    its `target_bundle_hash` is recomputed and differs from the canonical
    bundle's by design.

    Structural supervision is withdrawn on the FRAGMENT side, by
    `supervision_mask` on the slice — see
    :func:`unflatten.dataset.fragments.envelope.supervised_event_ranges`.
    """
    result = copy.deepcopy(target_bundle)
    _withdraw_unresolved_placements(result, fragments)
    result["target_bundle_hash"] = target_bundle_hash(result)
    return result


def _withdraw_unresolved_placements(
    result: dict[str, Any], fragments: list[dict[str, Any]]
) -> int:
    """Withdraw typed supervision on `result` IN PLACE; return the flips.

    The count is what this pass actually changed — a field already
    withdrawn by the font projection is not counted a second time — so it
    goes to zero if the withdrawal is ever removed, which a count
    re-derived from the fragment would not.
    """
    flipped = 0
    for node_id in unsupervised_placements(fragments):
        entry = result["typed"].get(node_id)
        if entry is None:  # a node type with no typed properties
            continue
        for field in entry["fields"]:
            flipped += field.get("supervision_mask") is not False
            field["observable_mask"] = False
            field["supervision_mask"] = False
    return flipped


def apply_masks(
    target_bundle: dict[str, Any],
    projection: dict[str, Any] | None,
    fragments: list[dict[str, Any]],
    *,
    render_font_pack_hash: str | None,
) -> tuple[dict[str, Any], int]:
    """Both §29 withdrawals in one pass: `(masked_bundle, placement_flips)`.

    Byte-identical to `apply_placement_mask(apply_supervision_mask(...))`
    — including `target_bundle_hash` — for one deep copy and one hash
    instead of two of each. The intermediate bundle's hash was never
    consumed by anyone, and recomputing it costs more than everything
    else these two functions do put together.

    The order inside stays load-bearing:
    :func:`apply_supervision_mask` re-derives `supervision_mask` from
    `known AND observable` for the pointers it assesses, so the placement
    withdrawal must come last or it would be undone. `projection` is
    `None` for a page the store has no mask projection for; then only the
    placement withdrawal applies, exactly as the two-call form did.

    The second element is the number of typed fields the placement pass
    withdrew — the caller that reports it (`model.build_cache`) can no
    longer diff against the intermediate, so the pass that performs the
    flips counts them.
    """
    result = copy.deepcopy(target_bundle)
    if projection is not None:
        _stamp_supervision_scopes(
            result, projection, render_font_pack_hash=render_font_pack_hash
        )
    flipped = _withdraw_unresolved_placements(result, fragments)
    result["target_bundle_hash"] = target_bundle_hash(result)
    return result, flipped


def masked_typed_for_document(
    target_bundle: dict[str, Any],
    projection: dict[str, Any] | None,
    *,
    render_font_pack_hash: str | None,
) -> dict[str, Any]:
    """The document-scoped half of §29 masking: the font-projection stamp
    over the WHOLE typed surface, with no per-page placement withdrawal
    applied. `_stamp_supervision_scopes` walks `target_bundle["typed"]`
    in full and never reads `fragments`, so this result is identical for
    every page of a document — a caller processing many pages
    (`model.build_cache`, `model.build_exact_text_cache`) computes it
    ONCE per document and reuses it, instead of paying `apply_masks`'
    whole-bundle deep copy and `target_bundle_hash` recompute on every
    page.

    Returns a fresh copy of `target_bundle["typed"]` only — never the
    whole bundle, and never its hash, which no caller of this function
    needs (the canonical `target_bundle["target_bundle_hash"]` is what
    identity checks bind to; the masked bundle's own hash is a
    TaskViewProjection-scoped derivative that `apply_masks`' callers use
    and this one's do not). Callers must copy again before mutating what
    they get back — `withdraw_placement_counts` only reads it,
    `withdraw_placements_typed` copies it again for its caller.
    """
    typed = copy.deepcopy(target_bundle.get("typed", {}))
    if projection is not None:
        _stamp_supervision_scopes(
            {"typed": typed}, projection, render_font_pack_hash=render_font_pack_hash
        )
    return typed


def withdraw_placement_counts(
    typed: dict[str, Any], fragment: dict[str, Any]
) -> tuple[int, int]:
    """`(nodes, flipped)` for this page's placement withdrawal against an
    already font-stamped `typed` surface (`masked_typed_for_document`),
    without mutating or copying it.

    A caller that only needs the COUNTS `withdraw_placements` reports
    (`model.build_cache.prepare_page`) never needs the withdrawn values
    themselves, so this reads what `_withdraw_unresolved_placements`
    would overwrite instead of mutating a copy of it: same numbers,
    `nodes` from the fragment alone and `flipped` from each unresolved
    node's CURRENT `supervision_mask`.
    """
    unresolved = unsupervised_placements([fragment])
    flipped = 0
    for node_id in unresolved:
        entry = typed.get(node_id)
        if entry is None:  # a node type with no typed properties
            continue
        for field in entry["fields"]:
            flipped += field.get("supervision_mask") is not False
    return len(unresolved), flipped


def withdraw_placements_typed(
    typed: dict[str, Any], fragment: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    """Placement withdrawal over an already font-stamped `typed` surface
    (`masked_typed_for_document`), returning a COPY plus `flipped` — for
    a caller that needs the withdrawn typed values themselves, not just
    their count (`build_exact_text_cache.row_for_page`'s typed-property
    targets read `masked["typed"]`).

    Byte-identical, for the `typed` section, to calling `apply_masks`
    with the same document and fragment and reading `result["typed"]`
    off it — this just skips recomputing the (discarded) whole-bundle
    hash and the (already cached, document-scoped) font stamp.

    Copy-on-write: only the entries this page withdraws are new objects;
    every other entry is SHARED with `typed`, so neither the argument nor
    the result may be mutated. A per-page deep copy of the whole
    document's surface was 63 % of exact-text cache build time (py-spy,
    2026-10-05), quadratic in document length.
    """
    result = dict(typed)
    flipped = 0
    for node_id in unsupervised_placements([fragment]):
        entry = result.get(node_id)
        if entry is None:  # a node type with no typed properties
            continue
        fields = []
        for field in entry["fields"]:
            flipped += field.get("supervision_mask") is not False
            fields.append({**field, "observable_mask": False, "supervision_mask": False})
        result[node_id] = {**entry, "fields": fields}
    return result, flipped

