"""The S0 admission report — model-side half (RFC005 §9.1/§9.5).

Moved out of `report.py` on 2026-09-13 as a behavior-identical split:
the §9.1 artifact probes, the §9.5 blocker matrix, and the derived
model-admission verdict now live here. `report.py` keeps re-exporting
`MODEL_BLOCKERS` and `MODEL_ARTIFACT_PROBES`, reads those (potentially
patched) report-module attributes, and passes them to
`build_model_admission` — so the S0 test suite's
`monkeypatch.setattr(report, "MODEL_BLOCKERS", ...)` pattern still
observes the change.

Two invariants carried over from `report.py`:

- This module never imports `unflatten.model` (that package pulls in
  `torch`, and admission has to stay importable without it).
- The probe excludes the whole admission package directory — the same
  scope as when the probe lived in `report.py`, both modules sitting in
  the same package — which is why the move is behavior-identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: §9.5 ReleaseBlockerMatrix rows that bear on S0/S1 admission.
MODEL_BLOCKERS: tuple[dict[str, Any], ...] = (
    {
        "blocker": "OPEN-005-016",
        "blocker_class": "contract_integrity_blocker",
        "block_before_stage": "S1",
        "required_resolution": (
            "complete, non-overlapping parameter ownership for InlineImage "
            "typed heads, learned TablePlan and every LoRA delta tensor, in a "
            "ratified RFC004-compatible trainable-group registry amendment"
        ),
        "allowed_release_modes_if_unresolved": "none",
        # RESOLVED 2026-09-04. The amendment this blocker demanded is
        # RFC004-ERRATA-001, ratified by the owner that day and implemented
        # the same day. The row is KEPT rather than deleted so a later reader
        # can see what permitted the first gradient update, and re-derive it
        # instead of trusting this comment:
        #   tests/model/test_property_head_registry.py — 183 mapped + 7
        #     excluded = 190, every pointer in exactly one of the two tables;
        #   tests/model/test_trainable_group_registry.py — 37 tensor
        #     identities, none owned twice (image heads 6, table plan 5,
        #     LoRA 4, each with exactly one owner).
        "resolved": True,
        "resolved_on": "2026-09-04",
        "resolution_evidence": (
            "RFC004-ERRATA-001 RATIFIED by owner 2026-09-04 and implemented: "
            "property_head_registry 125->183 mapped / 65->7 excluded; "
            "trainable_group_registry gained typed_heads.anchor (E3) and "
            "extended typed_heads.inline_image ownership to floating_image "
            "geometry (E4); measured 0 doubly-owned tensor identities of 37"
        ),
    },
    {
        "blocker": "RFC004 OPEN-004-013",
        "blocker_class": "contract_integrity_blocker",
        "block_before_stage": "S8",
        "required_resolution": "ratified RFC001 AST hash-projection algorithm/erratum",
        "allowed_release_modes_if_unresolved": "none",
    },
    {
        "blocker": "Rights/privacy",
        "blocker_class": "contract_integrity_blocker",
        "block_before_stage": "S1/S8",
        "required_resolution": "approved transitive checkpoint/data status",
        "allowed_release_modes_if_unresolved": "none",
        # RESOLVED 2026-09-04 by the owner — DEC-006, in full, on the ground
        # that the project is internal and no checkpoint is distributed.
        #
        # The row is KEPT because what it rested on matters more than that it
        # closed. DEC-002 §2.2 already records that no content scan was ever
        # run: `public_non_personal` is assigned from the `source` field, not
        # from the documents. The scan was offered a second time before this
        # decision and again not run. And S1 is an `exact_text` canary, so
        # memorization is the objective, not a side effect.
        #
        # DEC-002's escape hatch — narrow the corpus with one filter over the
        # lineage graph — does NOT survive into weights. Read DEC-006 before
        # citing this row.
        "resolved": True,
        "resolved_on": "2026-09-04",
        "resolution_evidence": (
            "DEC-006: owner approved the transitive checkpoint/data status in "
            "full (internal project). At that time DEC-001 "
            "redistribution_allowed=false was unchanged and distributing a "
            "checkpoint was a new decision, not a consequence of this one; "
            "that decision was made as DEC-011 (2026-09-06), which approved "
            "distribution of checkpoints trained on the ratified corpus at "
            "the decision level. Store records keep the historical DEC-001 "
            "value (no recompute; the field feeds no identity) and C6 "
            "derives distribution from the typed DEC-011 reference"
        ),
    },
)

#: §9.1 inputs whose absence blocks the model half. Each is PROBED, with
#: the pattern and the matching files reported — no claim of absence is
#: made without the search that supports it.
MODEL_ARTIFACT_PROBES: tuple[tuple[str, str, str], ...] = (
    ("trainable_group_registry", "trainable_group", "RFC005 §9.1 item 7"),
    ("property_head_registry", "property_head_registry", "RFC005 §9.1 item 8"),
    ("dense_expert_adapter_baseline", "ExpertAdapter", "RFC005 §9.1 item 10"),
    ("trainability_smoke_suite", "trainability_smoke", "RFC005 §9.1 item 11"),
)


def _probe_source(pattern: str, repo_root: Path) -> dict[str, Any]:
    """Search the package source for `pattern`, reporting what matched.

    The admission package is excluded: the probe patterns are literals
    in its modules, so a naive search finds the searcher and reports the
    artifact as present.
    """
    root = repo_root / "src" / "unflatten"
    excluded = Path(__file__).resolve().parent
    matches = sorted(
        str(path.relative_to(repo_root))
        for path in root.rglob("*.py")
        if excluded not in path.resolve().parents
        and pattern in path.read_text(encoding="utf-8", errors="ignore")
    )
    return {
        "pattern": pattern,
        "root": str(root),
        "excluded": str(excluded),
        "matched_files": matches,
    }


def build_model_admission(
    repo_root: Path,
    *,
    blockers: tuple[dict[str, Any], ...],
    probes: tuple[tuple[str, str, str], ...],
) -> dict[str, Any]:
    missing = []
    present = []
    for name, pattern, source in probes:
        evidence = _probe_source(pattern, repo_root)
        entry = {"artifact": name, "required_by": source, "evidence": evidence}
        (present if evidence["matched_files"] else missing).append(entry)
    missing_names = ", ".join(sorted(entry["artifact"] for entry in missing))
    # FAIL-CLOSED, and derived rather than declared.
    #
    # `outcome` used to be the literal "blocked", paired with a fixed reason
    # naming the §9.1 artifacts as absent. Once those artifacts landed the
    # report rendered a sentence that asserted their absence and then listed
    # none — a verdict that had stopped describing what the probe found. A
    # gate that cannot change its mind is not a gate; it is a comment.
    #
    # "admitted" is now reachable ONLY through an empty `reasons` list, so
    # every unenumerated path still ends in "blocked".
    unresolved = [b for b in blockers if not b.get("resolved")]
    blocking_s1 = [
        b for b in unresolved if "S1" in str(b.get("block_before_stage", ""))
    ]
    reasons: list[str] = []
    if missing:
        reasons.append(
            "§13.3's 'all mandatory registries resolve' cannot hold while "
            f"these §9.1 artifacts are absent: {missing_names}"
        )
    if blocking_s1:
        ids = ", ".join(sorted(str(b["blocker"]) for b in blocking_s1))
        reasons.append(f"unresolved blockers that gate S1: {ids}")
    return {
        "outcome": "blocked" if reasons else "admitted",
        "reason": "; ".join(reasons)
        if reasons
        else (
            "every §9.1 model-side artifact is present and every blocker that "
            "gates S1 is resolved; this admits the DENSE S1 contract canary "
            "ONLY (RFC004 is ratified just for dense_contract_canary_ready / "
            "dense_vertical_slice_ready under DEC-005) and authorizes no "
            "later stage — S2+ each need separate ratification"
        ),
        "admitted_tracks": [] if reasons else ["dense_contract_canary"],
        "blockers": list(blockers),
        "unresolved_blockers_gating_s1": [b["blocker"] for b in blocking_s1],
        "missing_artifacts": missing,
        "present_artifacts": present,
    }
