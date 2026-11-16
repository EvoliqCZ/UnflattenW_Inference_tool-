"""S2 exit-gate measurement, phase A (design §3, §6): cohort -> sealed
hypotheses. RFC005 §36.1.

**Phase A never touches the reference text.** It reads visual features and
the frozen (or freshly-initialized, for the baseline arm) controller only.
`test_decode.py::test_decode_module_never_imports_reference_text_machinery`
parses this file's own AST and refuses an `exact_text`/`exact_text_cache`
import or a `decode_tokens` reference anywhere in it -- the separation is
structural, not a convention a later edit could quietly erode.

Phase A reuses, rather than reimplements, three pieces already built and
mutation-proven for RFC005 §14.4's own gates 7/13:

- `s1_runner.build_gate13_decode_kwargs` -- the real decode's own kwargs
  over one cached page (the caller builds this per page; this module never
  constructs a `DecodeConfig` itself so it cannot silently disagree with
  gate 13's).
- `s1_runner_inputs.HypothesisCapture` -- wraps `decode_fn`, stores the
  `RawDocumentHypothesis` it returns, re-raises every typed failure
  untouched.
- `prediction_ir.seal_fragment` -- `RawDocumentHypothesis -> FragmentHypothesis`,
  refusing anything that did not reach a valid mode terminator.

**Five recorded outcomes (M1, design §12), not four.** RFC005 §14.4 gate 13
already defines "no silent truncation"; this module does not redefine it,
it calls `s1_runner.gate_13_free_running_no_silent_truncation(decode_fn,
decode_kwargs)` per page and reads which branch fired. That gate's own
`except` clause catches only `DecodeMaxStepsExceeded`/
`DecodeVocabularyExhausted` -- it does NOT catch
`node_lifecycle.UnsupportedResumeInV0`, which the untrained arm raised on a
real page (design §12): the grammar permitted `RESUME_NODE(document)` and
the V0 lifecycle tracker refused it. Uncaught, that ends a whole cohort at
page one. `decode_page` below adds the fifth branch on top of gate 13's
own two, rather than editing gate 13 to know about a failure mode that is
this module's problem, not gate 13's:

1. `terminated` -- gate 13 branch `"terminated"`; sealable.
2. `max_steps_exceeded` -- gate 13 branch `"typed_failure"`,
   `DecodeMaxStepsExceeded`.
3. `vocabulary_exhausted` -- gate 13 branch `"typed_failure"`,
   `DecodeVocabularyExhausted`.
4. `unsupported_resume` -- `node_lifecycle.UnsupportedResumeInV0`, caught
   here, never by gate 13.
5. `not_terminated` -- gate 13 branch `"silent_truncation"` (`status ==
   "fail"`): a hypothesis was returned but never terminated. Gate 13
   itself reports this as a `fail`, not an exception; recorded as its own
   outcome rather than folded into another one, so a caller that swallows
   a typed failure and returns a partial hypothesis instead is visible in
   the outcome distribution, not merged into "terminated" or an error.
6. `empty_hypothesis` (DEC-030 plan task 1) -- gate 13 branch
   `"terminated"`, but `hypothesis.node_states` is empty:
   `prediction_ir.seal_fragment` refuses to seal a
   `RawDocumentHypothesis` with no node states ("a RawDocumentHypothesis
   with no node_states is not sealable"), and the frozen-init baseline
   arm can terminate after only a few tokens with no nodes at all. Not a
   crash, not a sealed success -- carried the same way `not_terminated`
   is: `inference_mode`/`input_parity_proof_hash` present,
   `prediction_ir_hash` absent.

**M4 seal (RFC005 §36.1:2392/2394/2400).** A `terminated` page's record
carries `prediction_ir_hash`, `inference_mode="free_running"` and
`input_parity_proof_hash` -- both of the latter two are mandatory on every
prediction manifest per §36.1 and their absence is itself a refusal at
score time (governance review M4), never a default. `scope="single_page"`
is licensed by §36.1:2398 because every cohort bundle is a genuinely
single-page document. §36.1:2400 defines InputParityProof as binding FOUR
things -- original pages, the production-preprocessor Component Lock, the
window/page-selection algorithm, and the resulting input hashes --
`input_parity_proof_hash` binds all four: the page identity, an explicit
required `component_lock_hash` (a proof that omits a named component does
not prove what its name claims -- this is the one place that shows the
Observation Sidecar and target-derived coordinates stayed out of input
selection), the cohort manifest's own `policy_hash` as the page-selection
rule for this cohort, and a hash of the actual visual-features tensor
decoded ("the resulting input hashes"). `eval_cache_identity` (minus
`split`, which is expected to differ from the training cache by
construction) is folded in too, so a caller cannot silently substitute a
different page's features under the same `document_bundle_id` without the
hash moving.

**M6 denominator (design §12 table, governance M6 correction).** Phase A
iterates the **cohort manifest**, never the cache rows: every eligible
page gets a record, including the pre-decode typed-error classes
(`capability_excluded`, `grammar_refused`, `text_encoding_failed`,
`over_length`) that never had a cached feature entry to begin with, and
the one ratified exclusion (`withdrawn_text_coverage_floor`, DEC-009
dodatok D) that never gets attempted. `CohortManifest.expected_accounting`
is a *declared* count baked into the manifest (and covered by its sha256),
checked against what a live scan of `pages` actually produces --
`verify_cohort_accounting` raises `CohortDenominatorNarrowed` on any
disagreement, including one caused by silently dropping a single page
from the manifest (design §9 guard 7).

**The frozen-init baseline arm (task 7, DEC-009 dodatok C).** DEC-009
dodatok C re-bases §15.4's "frozen initialization baseline" as Qwen's
native reading after the VisualMemory identity fix -- not the retracted
random-projection defect (memory `random-projection-destroys-native-signal`).
`build_frozen_init_model` is exactly `s1_runner.build_context_from_loaded`'s
own model construction (`s1_runner.py:2687-2701`) with the checkpoint load
omitted: a fresh `QwenDenseCandidateModel` wrapping the same frozen
candidate embedding/lm-head weights, never loaded from a trained
checkpoint. `arm` is recorded on the sealed artefact (not per page) so
phase B can never compare an arm's hypotheses against themselves.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from unflatten.eval.s2_gate import sealed
from unflatten.eval.s2_gate.sealed import (
    ATTEMPT_DISPOSITION,
    DecodeArm,
    DecodeOutcome,
    NodeStateProjection,
    PageDecodeResult,
    SealedCohortArtefact,
    UnterminatedPrefix,
)
from unflatten.eval.s2_gate.sealed import CohortAccounting as SealedCohortAccounting
from unflatten.model import s1_runner
from unflatten.model.candidate_model import QwenDenseCandidateModel
from unflatten.model.composite_vocab import packed_ranges
from unflatten.model.free_running_decode import (
    DecodeMaxStepsExceeded,
    DecodeVocabularyExhausted,
)
from unflatten.model.hashing import domain_hash
from unflatten.model.node_lifecycle import UnsupportedResumeInV0
from unflatten.model.prediction_ir import (
    FragmentHypothesis,
    InferenceScope,
    RawDocumentHypothesis,
    assemble,
    prediction_ir_projection,
    region_prediction_projection,
    seal_fragment,
    typed_prediction_projection,
)
from unflatten.model.prediction_ir import (
    prediction_ir_hash as compute_prediction_ir_hash,
)
from unflatten.model.s1_runner_inputs import HypothesisCapture

#: `DecodeArm`/`PageDecodeResult`/`SealedCohortArtefact`/`DecodeOutcome` are
#: defined in `sealed.py` (the phase A -> phase B contract) and re-exported
#: here for every existing caller/test that imports them from this module;
#: listed in `__all__` so ruff does not read the re-export as dead code.
__all__ = [
    "ATTEMPT_DISPOSITION",
    "CohortAccounting",
    "CohortDenominatorNarrowed",
    "CohortDispositionRefused",
    "CohortManifest",
    "CohortManifestRefused",
    "CohortPage",
    "CohortSelectionRuleRefused",
    "DecodeArm",
    "DecodeOutcome",
    "EvalCacheIdentityRefused",
    "PageDecodeResult",
    "PageProgress",
    "SealedCohortArtefact",
    "build_frozen_init_model",
    "cohort_manifest_sha256",
    "compute_cohort_accounting",
    "compute_input_parity_proof_hash",
    "decode_cohort",
    "decode_page",
    "default_page_progress",
    "load_cohort_manifest",
    "refuse_unless_eval_cache_identity_ok",
    "requested_bundles_sha256",
    "seal_page",
    "selection_rule_policy_hash",
    "verify_cohort_accounting",
]

#: `RawDocumentHypothesis.terminated == True` reaches this scope kind only
#: (design §6, "Reference text"; §36.1:2398 licenses `single_page`). Under
#: DEC-033 every page is a standalone document, so this holds for every
#: cohort page regardless of slice (`s2_gate.cohort`'s `single_page_docs`
#: and `multi_page_pages` slices alike) -- not only pages from a
#: single-page document. MUST equal `sealed._SCOPE_KIND` -- both name the
#: one V0 `InferenceScope.kind` value.
_SCOPE_KIND = "single_page"

#: RFC004 inference-mode literal §36.1 makes mandatory on every prediction
#: manifest. MUST equal `sealed.INFERENCE_MODE`.
_INFERENCE_MODE = sealed.INFERENCE_MODE

_INPUT_PARITY_DOMAIN = "rfc005.s2-gate.input-parity-proof/v1"


class EvalCacheIdentityRefused(ValueError):
    """The eval cache is not what it must be to measure the S2 exit gate
    (design §7, governance M3): its identity differs from the run
    manifest's bindings somewhere other than `split`, or it claims
    `split="train"` outright."""


class CohortManifestRefused(ValueError):
    """The cohort manifest is missing, or its recomputed sha256 does not
    match the claimed one (governance M3/M4: membership is bound by the
    cohort manifest, since the cohort pin is not an eval-cache identity
    field)."""


class CohortDenominatorNarrowed(ValueError):
    """A live scan of `CohortManifest.pages` disagrees with the manifest's
    own declared `expected_accounting` (governance M6, design §9 guard 7).
    Raised on ANY disagreement, including a single page silently dropped
    from the manifest -- the report must fail, not shrink."""


class CohortSelectionRuleRefused(ValueError):
    """The cohort manifest's `selection_rule` is missing, its declared
    `policy_hash` does not match a recomputed hash over `selection_rule`
    itself (raised by `load_cohort_manifest`), or `selection_rule`'s own
    `requested_bundles_sha256` does not match a recomputed hash over the
    sorted bundle ids `pages` actually lists (raised by
    `verify_cohort_accounting` -- guard-7 follow-up). `verify_cohort_
    accounting`'s `expected_accounting` check alone is self-consistency
    only: a manifest can drop one page AND adjust `expected_accounting` to
    match, staying internally consistent while still narrowing the
    denominator. `selection_rule.requested_bundles_sha256` is pinned to the
    ORIGINAL request and is not recomputed alongside such an edit, so
    recomputing it here is what catches that remaining hole."""


class CohortDispositionRefused(ValueError):
    """A page's declared disposition disagrees with reality: it claims a
    feature-cache row it does not have, or claims to have none while one
    exists (governance follow-up to M6 -- "derived, not declared". A
    manifest can pass `verify_cohort_accounting`'s self-consistency check
    (declared counts match the pages listed) while still being simply
    wrong about what those pages are; a disposition that disagrees with
    the cache is a refusal, not a warning)."""


# ---------------------------------------------------------------------------
# Cohort manifest (M3/M4/M6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CohortPage:
    """One eligible page named by the cohort manifest (design §7,
    governance M3's `s2_exit_iid_provisional` role). `disposition` is
    either `"attempt"` (phase A must decode it) or one of the five typed
    exclusions design §12/governance M6 name -- never inferred here from
    cache contents, because the pre-decode error classes have no
    cache row to infer anything from."""

    document_bundle_id: str
    lineage_root_id: str
    leakage_component_id: str
    disposition: str

    def __post_init__(self) -> None:
        if self.disposition not in sealed.ALL_DISPOSITIONS:
            raise ValueError(
                f"{self.document_bundle_id}: disposition {self.disposition!r} is not "
                f"one of {sorted(sealed.ALL_DISPOSITIONS)}"
            )


@dataclass(frozen=True, slots=True)
class CohortManifest:
    """Design §7/governance M3's `s2_exit_iid_provisional` cohort manifest,
    the only place cohort membership is bound (the cohort pin is not an
    eval-cache identity field). `policy_hash` is a hash over `selection_rule`
    (split, single-page, 300 DPI, feature availability, the dodatok-D floor,
    and `requested_bundles_sha256` -- a hash over the sorted bundle ids the
    selection actually requested); `load_cohort_manifest` recomputes and
    refuses on mismatch (governance guard-7 follow-up), it does not merely
    carry it through unchecked. `expected_accounting` is the manifest's own
    declared per-disposition counts, checked against a live scan of `pages`
    by `verify_cohort_accounting` -- a declared deviation from ratified
    §10.4 role manifests (which this named role is not one of), not a
    silent substitute for one. `selection_rule` is required, not optional:
    a manifest without one cannot be checked against its own selection
    rule, and an optional check is the inert-guard shape."""

    role: str
    policy_hash: str
    selection_rule: Mapping[str, Any]
    expected_accounting: Mapping[str, int]
    pages: tuple[CohortPage, ...]


#: `CohortAccounting` lives in `sealed.py` -- `SealedCohortArtefact.accounting`
#: needs it, so it is part of the phase A -> phase B contract, not a
#: decode.py-only type.
CohortAccounting = SealedCohortAccounting


def compute_cohort_accounting(pages: Sequence[CohortPage]) -> CohortAccounting:
    """A live scan of `pages` -- never trusted on its own (see
    `verify_cohort_accounting`), but the only source of truth for what the
    manifest actually contains."""
    counts: dict[str, int] = dict.fromkeys(sealed.ALL_DISPOSITIONS, 0)
    for page in pages:
        counts[page.disposition] += 1
    withdrawn = counts[sealed.WITHDRAWN_DISPOSITION]
    eligible = len(pages)
    return CohortAccounting(
        eligible=eligible,
        withdrawn_text_coverage_floor=withdrawn,
        attempted=eligible - withdrawn,
        decoded=counts[sealed.ATTEMPT_DISPOSITION],
        capability_excluded=counts["capability_excluded"],
        grammar_refused=counts["grammar_refused"],
        text_encoding_failed=counts["text_encoding_failed"],
        over_length=counts["over_length"],
        **{key: counts[key] for key in sealed.ADDITIVE_DISPOSITIONS},
    )


def verify_cohort_accounting(
    manifest: CohortManifest,
    *,
    feature_presence: Callable[[str], bool] | None = None,
) -> CohortAccounting:
    """Raise `CohortDenominatorNarrowed` unless a live scan of
    `manifest.pages` matches `manifest.expected_accounting` exactly
    (governance M6, design §9 guard 7). Moving one page out of `pages`
    without also editing `expected_accounting` -- the mutation the guard
    names -- fails here rather than silently shrinking a denominator
    downstream.

    That check alone is self-consistency only: `expected_accounting` and
    `pages` are both inside the same sha256-covered document, so a
    manifest that DECLARES 218 capability-excluded pages and lists 218
    pages so labelled passes it even if those 218 pages are not actually
    capability-excluded, AND a manifest that drops one page while also
    adjusting `expected_accounting` to match stays self-consistent while
    still narrowing the denominator. The second recomputation below closes
    that hole: `manifest.selection_rule.requested_bundles_sha256` is pinned
    to the bundle ids ORIGINALLY requested and is never edited alongside a
    page drop, so recomputing it over the live `pages` list and refusing on
    mismatch (`CohortSelectionRuleRefused`) catches exactly that mutation.
    `feature_presence`, if supplied, is the derived cross-check against
    reality this project's rule requires: every page whose disposition is
    in `sealed.NO_FEATURES_DISPOSITIONS` must genuinely be absent from the
    feature cache, and every page in `sealed.HAS_FEATURES_DISPOSITIONS`
    must genuinely be present. A disagreement raises
    `CohortDispositionRefused` -- a refusal, not a warning."""
    actual = compute_cohort_accounting(manifest.pages)
    actual_dict = actual.as_dict()
    expected = dict(manifest.expected_accounting)
    if actual_dict != expected:
        raise CohortDenominatorNarrowed(
            f"cohort manifest {manifest.role!r}: declared expected_accounting "
            f"{expected} does not match a live scan of its {len(manifest.pages)} "
            f"page(s), {actual_dict} -- a page was added, removed, or "
            "reclassified without the manifest's own accounting being updated"
        )
    recomputed_bundles_hash = requested_bundles_sha256(
        page.document_bundle_id for page in manifest.pages
    )
    claimed_bundles_hash = manifest.selection_rule.get("requested_bundles_sha256")
    if recomputed_bundles_hash != claimed_bundles_hash:
        raise CohortSelectionRuleRefused(
            f"cohort manifest {manifest.role!r}: selection_rule.requested_bundles_sha256 "
            f"{claimed_bundles_hash!r} does not match a recomputed hash over the "
            f"{len(manifest.pages)} page(s) actually listed, {recomputed_bundles_hash!r} -- "
            "a page was added or removed without updating the requested-bundles hash "
            "(guard-7 follow-up)"
        )
    if feature_presence is not None:
        for page in manifest.pages:
            present = feature_presence(page.document_bundle_id)
            should_be_present = page.disposition in sealed.HAS_FEATURES_DISPOSITIONS
            if present != should_be_present:
                raise CohortDispositionRefused(
                    f"{page.document_bundle_id}: disposition {page.disposition!r} implies "
                    f"feature-cache presence={should_be_present}, but the feature cache "
                    f"reports presence={present} -- the manifest disagrees with the cache "
                    "it was built from"
                )
    return actual


def _cohort_manifest_hash_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in raw.items() if k != "sha256"}


def cohort_manifest_sha256(raw: Mapping[str, Any]) -> str:
    """sha256 hex digest over every key of `raw` except `sha256` itself,
    as canonical (sorted-key) JSON."""
    payload = _cohort_manifest_hash_payload(raw)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def selection_rule_policy_hash(selection_rule: Mapping[str, Any]) -> str:
    """`sha256:<hex>` over `selection_rule` as canonical (sorted-key) JSON --
    matches the real `cohort-s2-exit-iid-provisional.json`'s `policy_hash`
    over its `selection_rule` block exactly (verified against the file on
    disk: recomputing this over its `selection_rule` reproduces its
    `policy_hash` byte for byte)."""
    canonical = json.dumps(dict(selection_rule), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def requested_bundles_sha256(bundle_ids: Iterable[str]) -> str:
    """Bare sha256 hex digest (no `sha256:` prefix, matching the real file's
    own spelling) over the sorted bundle ids, newline-joined -- matches the
    real `cohort-s2-exit-iid-provisional.json`'s
    `selection_rule.requested_bundles_sha256` exactly (verified against the
    file on disk). `bundle_ids` is every page's `document_bundle_id`
    regardless of disposition: this hashes what was REQUESTED, before any
    disposition filtering."""
    ids = sorted(bundle_ids)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def load_cohort_manifest(raw: Mapping[str, Any] | None) -> CohortManifest:
    """`raw` is the on-disk cohort manifest payload (a `dict` with
    `role`/`policy_hash`/`selection_rule`/`expected_accounting`/`pages`/
    `sha256` keys). Refuses (`CohortManifestRefused`) if `raw` is `None`
    (no manifest supplied), if the recomputed sha256 does not match the
    claimed one, if `selection_rule` is absent, or if the recomputed
    `selection_rule_policy_hash` does not match the manifest's own
    `policy_hash` -- membership is bound by this manifest and nothing else
    (governance M3/M4), and `selection_rule` is required, not optional
    (guard-7 follow-up: an optional check is the inert-guard shape)."""
    if raw is None:
        raise CohortManifestRefused("no cohort manifest supplied -- refusing to run without one")
    claimed = raw.get("sha256")
    if not isinstance(claimed, str) or not claimed:
        raise CohortManifestRefused("cohort manifest carries no 'sha256' field")
    actual = cohort_manifest_sha256(raw)
    if actual != claimed:
        raise CohortManifestRefused(
            f"cohort manifest sha256 mismatch: recomputed {actual}, manifest claims "
            f"{claimed} -- the manifest was edited after it was sealed"
        )
    selection_rule = raw.get("selection_rule")
    if not isinstance(selection_rule, Mapping) or not selection_rule:
        raise CohortManifestRefused(
            "cohort manifest carries no 'selection_rule' block -- required, not optional "
            "(guard-7 follow-up): a manifest without one cannot be checked against its own "
            "selection rule"
        )
    recomputed_policy_hash = selection_rule_policy_hash(selection_rule)
    claimed_policy_hash = raw.get("policy_hash")
    if recomputed_policy_hash != claimed_policy_hash:
        raise CohortManifestRefused(
            f"cohort manifest policy_hash mismatch: recomputed {recomputed_policy_hash} over "
            f"selection_rule, manifest claims {claimed_policy_hash!r} -- selection_rule does "
            "not match its own declared policy_hash"
        )
    pages = tuple(
        CohortPage(
            document_bundle_id=p["document_bundle_id"],
            lineage_root_id=p["lineage_root_id"],
            leakage_component_id=p["leakage_component_id"],
            disposition=p["disposition"],
        )
        for p in raw["pages"]
    )
    return CohortManifest(
        role=raw["role"],
        policy_hash=raw["policy_hash"],
        selection_rule=dict(selection_rule),
        expected_accounting=dict(raw["expected_accounting"]),
        pages=pages,
    )


# ---------------------------------------------------------------------------
# Eval-cache identity refusal (design §7, governance M3)
# ---------------------------------------------------------------------------


def refuse_unless_eval_cache_identity_ok(
    *,
    eval_cache_identity: Mapping[str, Any],
    run_manifest_bindings: Mapping[str, Any],
) -> None:
    """Raise `EvalCacheIdentityRefused` unless `eval_cache_identity` agrees
    with `run_manifest_bindings` on every field the latter names, except
    `split` -- which is EXPECTED to differ (the eval cache's `val_iid`
    against the training cache's `train`, design §7). Also refuses a
    `split == "train"` eval cache outright: a cache claiming `train` is
    the training cache, not an eval one, regardless of what else matches.

    Compares every key `run_manifest_bindings` carries (mirrors
    `train.SHARED_LINEAGE_FIELDS`'s shape without hardcoding that specific
    field list here, so a binding set change is inherited rather than
    silently under-checked).
    """
    if eval_cache_identity.get("split") == "train":
        raise EvalCacheIdentityRefused(
            "eval cache records split='train' -- that is the training cache, not "
            "an eval one, regardless of any other field"
        )
    mismatches = {
        key: (expected, eval_cache_identity.get(key))
        for key, expected in run_manifest_bindings.items()
        if key != "split" and eval_cache_identity.get(key) != expected
    }
    if mismatches:
        raise EvalCacheIdentityRefused(
            "eval cache identity disagrees with the run manifest bindings on "
            f"{sorted(mismatches)}: {mismatches} -- the two would not be "
            "measuring the same model (design §7)"
        )


# ---------------------------------------------------------------------------
# Per-page decode (M1's five outcomes)
# ---------------------------------------------------------------------------


#: `PageDecodeResult`/`SealedCohortArtefact`/`DecodeArm`/`DecodeOutcome`
#: live in `sealed.py` (the phase A -> phase B contract) -- imported above,
#: not redefined here, so `score.py` and this module share exactly one
#: definition of each.


def _inner_decode_fn_identity(capture: HypothesisCapture) -> str | None:
    """The evidence one level below `capture`'s own identity (coordinator
    follow-up, second round): `HypothesisCapture.__module__`/`__qualname__`
    name the WRAPPER, which is identical for a real `decode_free_running`
    run and for every test's fake decode function -- true of both, so it
    cannot distinguish them, the same defect as the retired
    `no_grammar_bypass = True` literal one level down. `HypothesisCapture`
    always stores what it wraps on `self._decode_fn` (defaulting to
    `decode_free_running` itself), so reading that attribute names the
    thing that actually has no bypass mode -- `decode_free_running` has no
    flag or parameter that disables the grammar constraint anywhere in its
    module, so naming IT is the whole proof; naming the wrapper is not.
    `None` only if `capture` (or whatever is passed) is not a wrapper of
    this shape at all -- distinguishes "unwrapped" from "wrapped by
    something that hides its inner function", never conflating the two."""
    inner = getattr(capture, "_decode_fn", None)
    if inner is None:
        return None
    return f"{inner.__module__}.{inner.__qualname__}"


def _loop_guard_counters_from_capture(capture: HypothesisCapture) -> dict[str, int] | None:
    """DEC-031 task 2: this page's `LoopGuardV1.counters`, read from
    wherever `decode_free_running` actually left them -- `capture.
    hypothesis.issues` (a `{"kind": "loop_guard_v1", "counters": ...}`
    entry `decode_free_running` appends whenever `config.loop_guard ==
    "v1"`, never touching `prediction_ir`'s own schema to do it) for a
    terminated page, or `capture.exception.loop_guard_counters` for a
    `max_steps_exceeded`/`vocabulary_exhausted` page (both typed failures
    carry it directly, `free_running_decode.py`). `None` when the guard
    was off, or this page's outcome carries neither (e.g.
    `unsupported_resume`, `not_terminated`)."""
    hypothesis = getattr(capture, "hypothesis", None)
    if hypothesis is not None:
        for issue in getattr(hypothesis, "issues", ()):
            if isinstance(issue, Mapping) and issue.get("kind") in _LOOP_GUARD_ISSUE_KINDS:
                return dict(issue["counters"])
        return None
    counters = getattr(getattr(capture, "exception", None), "loop_guard_counters", None)
    return dict(counters) if counters is not None else None


#: `loop_guard_v2` (DEC-036) nests v1's counters under `counters["v1"]`.
_LOOP_GUARD_ISSUE_KINDS = frozenset({"loop_guard_v1", "loop_guard_v2"})


def _loop_guard_v2_record_from_capture(capture: HypothesisCapture) -> Mapping[str, Any] | None:
    """DEC-036: the page's `loop_guard_v2` issue entry -- from the
    terminated hypothesis, or `loop_guard_record` on a typed failure."""
    hypothesis = getattr(capture, "hypothesis", None)
    if hypothesis is not None:
        for issue in getattr(hypothesis, "issues", ()):
            if isinstance(issue, Mapping) and issue.get("kind") == "loop_guard_v2":
                return issue
        return None
    return getattr(getattr(capture, "exception", None), "loop_guard_record", None)


def _page_start_record_from_capture(capture: HypothesisCapture) -> Mapping[str, Any] | None:
    """DEC-061 (PROPOSED): the page's `page_start_policy` issue entry -- from
    the terminated hypothesis, or `page_start_record` on a typed failure.
    Mirrors `_loop_guard_v2_record_from_capture`."""
    hypothesis = getattr(capture, "hypothesis", None)
    if hypothesis is not None:
        for issue in getattr(hypothesis, "issues", ()):
            if isinstance(issue, Mapping) and issue.get("kind") == "page_start_policy":
                return issue
        return None
    return getattr(getattr(capture, "exception", None), "page_start_record", None)


def _story_length_cap_record_from_capture(capture: HypothesisCapture) -> Mapping[str, Any] | None:
    """`header-length-cap.v0` (PROPOSED): the page's `header_length_cap` issue entry --
    from the terminated hypothesis, or `story_length_cap_record` on a typed failure."""
    hypothesis = getattr(capture, "hypothesis", None)
    if hypothesis is not None:
        for issue in getattr(hypothesis, "issues", ()):
            if isinstance(issue, Mapping) and issue.get("kind") == "header_length_cap":
                return issue
        return None
    return getattr(getattr(capture, "exception", None), "story_length_cap_record", None)


def _story_retry_record_from_capture(capture: HypothesisCapture) -> Mapping[str, Any] | None:
    """`story-retry.v0` (PROPOSED): the page's `story_retry` issue entry (terminated
    hypotheses only: a failed first decode never reaches the retry)."""
    hypothesis = getattr(capture, "hypothesis", None)
    if hypothesis is None:
        return None
    for issue in getattr(hypothesis, "issues", ()):
        if isinstance(issue, Mapping) and issue.get("kind") == "story_retry":
            return issue
    return None


def _loop_guard_detail(wiring: Mapping[str, Any]) -> str | None:
    """DEC-031 task 2: JSON-encode `wiring`'s `loop_guard`/`loop_guard_
    counters` into `PageDecodeResult.detail` -- the least invasive carrier
    (an existing, otherwise-unused field) rather than growing the sealed
    schema for one diagnostic. `None` whenever the guard was off, so a
    `loop_guard=None` decode's sealed pages are byte-identical to before
    this task."""
    payload = _loop_guard_payload(wiring)
    return None if payload is None else json.dumps(payload, sort_keys=True)


def _loop_guard_payload(wiring: Mapping[str, Any]) -> dict[str, Any] | None:
    loop_guard = wiring.get("loop_guard")
    if loop_guard is None:
        return None
    payload = {"loop_guard": loop_guard, "loop_guard_counters": wiring.get("loop_guard_counters")}
    for key in _LOOP_GUARD_V2_WIRING_KEYS:
        if key in wiring:
            payload[key] = wiring[key]
    return payload


def _page_start_payload(wiring: Mapping[str, Any]) -> dict[str, Any] | None:
    """DEC-061: the `page_start_*` block of `detail`. Present only when the
    decode asked for the policy (so every other decode's detail is
    unchanged). The sealed `score.verify_seal` never trusts the status
    written here -- it re-reads the committed policy file."""
    policy = wiring.get("page_start_policy")
    if policy is None:
        return None
    record = wiring.get("page_start_record") or {}
    return {
        "page_start_policy": policy,
        "page_start_policy_sha256": record.get("policy_sha256"),
        "page_start_policy_status": record.get("policy_status"),
        "page_start_pattern": record.get("pattern"),
        "page_start_pattern_source": record.get("pattern_source"),
        "page_start_applied": record.get("applied"),
        "page_start_counters": record.get("counters"),
    }


def _story_length_cap_payload(wiring: Mapping[str, Any]) -> dict[str, Any] | None:
    """The `story_length_cap_*` block of `detail`; present only when the decode asked
    for the policy. `score.verify_seal` re-reads the committed policy for its status."""
    policy = wiring.get("story_length_cap")
    if policy is None:
        return None
    record = wiring.get("story_length_cap_record") or {}
    return {
        "story_length_cap": policy,
        "story_length_cap_sha256": record.get("policy_sha256"),
        "story_length_cap_status": record.get("policy_status"),
        "story_length_cap_header_tokens": record.get("cap_header_tokens"),
        "story_length_cap_footer_tokens": record.get("cap_footer_tokens"),
        "story_length_cap_counters": record.get("counters"),
    }


def _story_retry_payload(wiring: Mapping[str, Any]) -> dict[str, Any] | None:
    """The `story_retry_*` block of `detail`; present only when the decode asked for the policy.
    `score.verify_seal` re-reads the committed policy for its status."""
    policy = wiring.get("story_retry")
    if policy is None:
        return None
    record = wiring.get("story_retry_record") or {}
    return {
        "story_retry": policy,
        "story_retry_sha256": record.get("policy_sha256"),
        "story_retry_status": record.get("policy_status"),
        "story_retry_threshold_tokens": record.get("threshold_tokens"),
        "story_retry_outcome": record.get("outcome"),
        "story_retry_pattern": record.get("pattern"),
        "story_retry_first_decode_stories": record.get("first_decode_stories"),
    }


def _decode_detail(wiring: Mapping[str, Any]) -> str | None:
    """The loop-guard block (as `_loop_guard_detail`) merged with the page-start and
    story-length-cap blocks; `None` only when all are off."""
    blocks = (
        _loop_guard_payload(wiring),
        _page_start_payload(wiring),
        _story_length_cap_payload(wiring),
        _story_retry_payload(wiring),
    )
    if all(block is None for block in blocks):
        return None
    return json.dumps({k: v for block in blocks for k, v in (block or {}).items()}, sort_keys=True)


#: Present in `wiring` (and so in `detail`) only for a `"v2"` decode, so
#: the `None`/`"v1"` detail strings stay byte-identical (DEC-036 §5).
_LOOP_GUARD_V2_WIRING_KEYS = ("loop_guard_caps_sha256", "loop_guard_interventions")


def _capture_wiring(capture: HypothesisCapture, decode_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Mirrors `s1_runner.gate_13_free_running_no_silent_truncation`'s own
    `wiring` dict (`s1_runner.py:1568-1571`) -- computed independently here
    (from the same `capture`/`decode_kwargs` gate 13 itself is given, so
    the values agree) rather than read off `gate13.evidence`, because the
    `UnsupportedResumeInV0` branch never gets a `GateResult` to read one
    from at all (the exception propagates out of gate 13 before it builds
    one). `decode_fn` names `capture` (`HypothesisCapture`), not the raw
    `decode_fn` passed to `decode_page` -- `HypothesisCapture` sets its own
    `__qualname__` for exactly this reason (its own docstring): it IS the
    callee gate 13 (and this function) actually invokes. `decode_fn_inner_
    identity` is the second, distinguishing layer -- see
    `_inner_decode_fn_identity`. `unterminated_prefix` (bug `max-steps-
    discards-the-evidence`) is `_unterminated_prefix_from_capture`'s
    result -- `None` unless `capture` caught a typed failure that carries
    the tokens it accumulated before its cap. `loop_guard`/`loop_guard_
    counters` (DEC-031 task 2): the decode's own `config.loop_guard`
    identity, read straight off `decode_kwargs` so it is present for
    EVERY outcome including `unsupported_resume` (where `capture` carries
    no counters), and `_loop_guard_counters_from_capture`'s result.
    A `"v2"` decode (DEC-036) adds `loop_guard_caps_sha256`/`loop_guard_
    interventions` from its `loop_guard_v2` record (`None` when the page
    carries none, e.g. `unsupported_resume`)."""
    wiring = {
        "decode_fn": f"{capture.__module__}.{capture.__qualname__}",
        "max_steps": getattr(decode_kwargs.get("config"), "max_steps", None),
        "decode_fn_inner_identity": _inner_decode_fn_identity(capture),
        "unterminated_prefix": _unterminated_prefix_from_capture(capture),
        "loop_guard": getattr(decode_kwargs.get("config"), "loop_guard", None),
        "loop_guard_counters": _loop_guard_counters_from_capture(capture),
    }
    if wiring["loop_guard"] == "v2":
        record = _loop_guard_v2_record_from_capture(capture) or {}
        wiring["loop_guard_caps_sha256"] = record.get("caps_sha256")
        wiring["loop_guard_interventions"] = record.get("interventions")
    page_start_policy = getattr(decode_kwargs.get("config"), "page_start_policy", None)
    if page_start_policy is not None:
        wiring["page_start_policy"] = page_start_policy
        wiring["page_start_record"] = _page_start_record_from_capture(capture)
    story_length_cap = getattr(decode_kwargs.get("config"), "story_length_cap", None)
    if story_length_cap is not None:
        wiring["story_length_cap"] = story_length_cap
        wiring["story_length_cap_record"] = _story_length_cap_record_from_capture(capture)
    story_retry_id = getattr(decode_kwargs.get("config"), "story_retry", None)
    if story_retry_id is not None:
        wiring["story_retry"] = story_retry_id
        wiring["story_retry_record"] = _story_retry_record_from_capture(capture)
    return wiring


def decode_page(
    decode_fn: Callable[..., RawDocumentHypothesis],
    decode_kwargs: Mapping[str, Any],
) -> tuple[DecodeOutcome, RawDocumentHypothesis | None, dict[str, Any]]:
    """Run one page's free-running decode through gate 13's own logic
    (`s1_runner.gate_13_free_running_no_silent_truncation`), reused rather
    than reimplemented (M1), plus the fifth branch gate 13 does not catch:
    `node_lifecycle.UnsupportedResumeInV0` (design §12). `decode_fn` is
    wrapped in `s1_runner_inputs.HypothesisCapture` so a `"terminated"`
    outcome carries the actual `RawDocumentHypothesis` to seal.

    Returns `(outcome, hypothesis_or_none, wiring)` -- `hypothesis` is
    populated only for `"terminated"` (the only sealable outcome).
    `wiring` (`{"decode_fn": ..., "max_steps": ..., "decode_fn_inner_
    identity": ...}`) is ALWAYS populated, for every outcome: it is the
    evidence of which callee actually ran (code review follow-up -- the
    proof that no unconstrained decode path was substituted for the
    constrained one; report.py derives `no_grammar_bypass` from this
    rather than asserting a boolean). `decode_fn` alone names the WRAPPER
    (`HypothesisCapture`), true of every real run and every test's fake
    decode function alike -- `decode_fn_inner_identity` is what actually
    distinguishes them, naming what `HypothesisCapture` wraps."""
    capture = HypothesisCapture(decode_fn)
    try:
        gate13 = s1_runner.gate_13_free_running_no_silent_truncation(
            capture, dict(decode_kwargs)
        )
    except UnsupportedResumeInV0:
        # `_capture_wiring` must run AFTER the call, not before: it reads
        # `capture.exception`/`capture.hypothesis`, which the call above is
        # what populates (bug `max-steps-discards-the-evidence` -- computing
        # it before the call always saw a fresh, empty `capture`).
        wiring = _capture_wiring(capture, decode_kwargs)
        return "unsupported_resume", None, wiring

    wiring = _capture_wiring(capture, decode_kwargs)
    branch = gate13.evidence.get("branch")
    if branch == "terminated":
        return "terminated", capture.hypothesis, wiring
    if branch == "typed_failure":
        typed_failure = gate13.evidence.get("typed_failure")
        if typed_failure == DecodeMaxStepsExceeded.__name__:
            return "max_steps_exceeded", None, wiring
        if typed_failure == DecodeVocabularyExhausted.__name__:
            return "vocabulary_exhausted", None, wiring
        raise AssertionError(
            f"gate 13 reported an unrecognized typed_failure {typed_failure!r}"
        )
    if branch == "silent_truncation":
        return "not_terminated", None, wiring
    raise AssertionError(f"gate 13 reported an unrecognized branch {branch!r}")


def _tensor_sha256(t: torch.Tensor) -> str:
    """Hash `t`'s exact bytes, dtype included in the hashed payload so two
    runs that fed the model different dtypes over identical values (e.g.
    bfloat16 zeros vs float32 zeros) never collide (must_fix 2, code
    review). `Tensor.numpy()` raises `TypeError` on bfloat16 --
    `build_gate13_decode_kwargs` produces visual features in bfloat16 by
    default (`s1_runner.py:2539,2549`), so every terminated page crashed
    here. bfloat16 is reinterpreted as `uint16` (same 2-byte width, exact
    bit pattern preserved) before going to numpy; every other dtype numpy
    already supports natively. Never upcast to float32: that would hash a
    value that was never actually fed to the model."""
    t = t.detach().to("cpu").contiguous()
    if t.dtype == torch.bfloat16:
        raw_bytes = t.view(torch.uint16).numpy().tobytes()
    else:
        raw_bytes = t.numpy().tobytes()
    payload = f"{t.dtype}:".encode() + raw_bytes
    return hashlib.sha256(payload).hexdigest()


def compute_input_parity_proof_hash(
    *,
    document_bundle_id: str,
    eval_cache_identity: Mapping[str, Any],
    visual_features: torch.Tensor,
    component_lock_hash: str,
    selection_policy_hash: str,
) -> str:
    """RFC005 §36.1:2400's mandatory `input_parity_proof_hash`, binding all
    four things InputParityProof names: the page identity, the production-
    preprocessor **Component Lock** (`component_lock_hash` -- required, not
    folded optionally into `eval_cache_identity`: an optional binding a
    caller may forget to supply is the same failure as not having one), the
    window/page-selection algorithm (`selection_policy_hash` -- for this
    cohort, the cohort manifest's own `policy_hash`), and the resulting
    input hashes (a hash of the exact visual-features tensor decoded).
    `eval_cache_identity` (`split` excluded -- it is expected to differ
    across arms of the same run measured against the same page) is folded
    in too, so a caller cannot silently substitute another page's features
    under the same `document_bundle_id`, another Component Lock, or another
    selection rule without this hash moving."""
    identity_without_split = {k: v for k, v in eval_cache_identity.items() if k != "split"}
    payload = {
        "document_bundle_id": document_bundle_id,
        "eval_cache_identity": identity_without_split,
        "component_lock_hash": component_lock_hash,
        "selection_policy_hash": selection_policy_hash,
        "visual_features_sha256": _tensor_sha256(visual_features),
    }
    return domain_hash(_INPUT_PARITY_DOMAIN, payload)


def _project_node_states(node_states: Iterable[Any]) -> tuple[NodeStateProjection, ...]:
    """Real `node_lifecycle.NodeState` objects (which carry torch tensor
    fields `anchor`/`content_summary`) projected into the torch-free
    `sealed.NodeStateProjection` shape -- exactly the fields `prediction_ir.
    _node_state_projection` keeps, nothing else, so phase B (which never
    imports torch) can carry and rehash them."""
    return tuple(
        NodeStateProjection(
            local_node_id=node.local_node_id,
            slot_kind=node.slot_kind,
            node_type=node.node_type,
            parent_local_node_id=node.parent_local_node_id,
            open_event_index=node.open_event_index,
            leaf_event_index=node.leaf_event_index,
            resume_event_index=node.resume_event_index,
            close_event_index=node.close_event_index,
        )
        for node in node_states
    )


def _node_state_projections(
    hypothesis: RawDocumentHypothesis,
) -> tuple[NodeStateProjection, ...]:
    """`hypothesis.node_states` projected via `_project_node_states` -- see
    that function for what is kept and why."""
    return _project_node_states(hypothesis.node_states)


def _unterminated_prefix_from_capture(capture: HypothesisCapture) -> UnterminatedPrefix | None:
    """Diagnostic evidence a `max_steps_exceeded`/`vocabulary_exhausted`
    page's decode accumulated before its cap (bug `max-steps-discards-
    the-evidence`): `HypothesisCapture.exception` is the typed failure
    `free_running_decode` raised, storing `accepted_tokens`/`node_states`
    at the moment it stopped. `None` whenever `capture` recorded no
    exception (a `"terminated"` or `"not_terminated"` page) or the
    exception is not one of the two typed failures that carry these
    fields (e.g. `node_lifecycle.UnsupportedResumeInV0`, which does not)."""
    exc = getattr(capture, "exception", None)
    if exc is None:
        return None
    accepted_tokens = getattr(exc, "accepted_tokens", None)
    node_states = getattr(exc, "node_states", None)
    if accepted_tokens is None or node_states is None:
        return None
    return UnterminatedPrefix(
        accepted_tokens=tuple(accepted_tokens),
        node_states=_project_node_states(node_states),
    )


def seal_page(
    *,
    document_bundle_id: str,
    hypothesis: RawDocumentHypothesis,
    eval_cache_identity: Mapping[str, Any],
    visual_features: torch.Tensor,
    component_lock_hash: str,
    selection_policy_hash: str,
    decode_fn_identity: str | None = None,
    max_steps: int | None = None,
    decode_fn_inner_identity: str | None = None,
    detail: str | None = None,
) -> PageDecodeResult:
    """Seal a `"terminated"` hypothesis into a `PageDecodeResult` carrying
    all three §36.1-mandatory fields (M4), PLUS the hypothesis fields
    `prediction_ir_hash` was computed over (`accepted_tokens`/
    `node_states`/`typed_predictions`/`resource_predictions`/`issues`) --
    without them phase B has a hash but nothing to recompute it from,
    which defeats the seal (M4 follow-up).

    `decode_fn_identity`/`max_steps`/`decode_fn_inner_identity` (code
    review follow-up, "carry gate 13's decode_fn identity through to the
    artefact", two rounds): `decode_page`'s `wiring` return value -- which
    callee actually ran the decode, both the wrapper (`decode_fn_identity`)
    and, distinguishingly, what it wraps (`decode_fn_inner_identity`).
    Optional here only because `seal_page` is also exercised directly
    against a hand-built hypothesis in tests that predate this evidence;
    `decode_cohort` always supplies all three.

    `detail` (DEC-031 task 2): `_loop_guard_detail`'s JSON string, or
    `None` when the guard was off -- see that function."""
    fragment: FragmentHypothesis = seal_fragment(hypothesis, InferenceScope(kind=_SCOPE_KIND))
    ir = assemble((fragment,))
    ir_hash = compute_prediction_ir_hash(prediction_ir_projection(ir))
    parity_hash = compute_input_parity_proof_hash(
        document_bundle_id=document_bundle_id,
        eval_cache_identity=eval_cache_identity,
        visual_features=visual_features,
        component_lock_hash=component_lock_hash,
        selection_policy_hash=selection_policy_hash,
    )
    return PageDecodeResult(
        document_bundle_id=document_bundle_id,
        outcome="terminated",
        prediction_ir_hash=ir_hash,
        inference_mode=_INFERENCE_MODE,
        input_parity_proof_hash=parity_hash,
        scope=_SCOPE_KIND,
        accepted_tokens=tuple(hypothesis.accepted_tokens),
        node_states=_node_state_projections(hypothesis),
        typed_predictions=tuple(
            typed_prediction_projection(t) for t in hypothesis.typed_predictions
        ),
        region_predictions=sealed._sealed_region_predictions(
            [region_prediction_projection(r) for r in hypothesis.region_predictions]
        ),
        resource_predictions=tuple(hypothesis.resource_predictions),
        issues=tuple(hypothesis.issues),
        decode_fn_identity=decode_fn_identity,
        max_steps=max_steps,
        decode_fn_inner_identity=decode_fn_inner_identity,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Cohort orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PageProgress:
    """One page's progress report (bug `decode-reports-no-progress`) --
    purely a human-observability signal, never part of the sealed artefact:
    `decode_cohort` builds one of these per page and hands it to the
    `on_page` callback, but nothing in `SealedCohortArtefact`/
    `PageDecodeResult` reads it back. `position` is 1-based (`"12/30"`
    reads naturally); `elapsed_seconds`/`mean_seconds_per_page` are plain
    wall-clock `time.monotonic()` deltas around `decode_page` (or ~0.0 for
    a page whose disposition never reaches it), never a GPU-synchronising
    measurement -- the GOAL constraint is "no per-step reporting, nothing
    that touches or synchronises the GPU", and `decode_page` already forces
    the sync a caller needs by returning a Python outcome string."""

    position: int
    total: int
    document_bundle_id: str
    outcome: str
    token_count: int
    elapsed_seconds: float
    mean_seconds_per_page: float


def default_page_progress(progress: PageProgress) -> None:
    """The default `on_page` callback (GOAL: "defaulting to something that
    writes a line to stderr, injectable so tests can capture it"). Writes
    to `sys.stderr` (not stdout) so it lands in a log the caller `tee`s
    without being confused for the artefact itself, which `run_decode`
    writes separately to `--output`. One `print` call, one line -- no
    per-step reporting, no GPU touch."""
    bundle_abbrev = progress.document_bundle_id[:12]
    remaining = progress.total - progress.position
    eta_seconds = remaining * progress.mean_seconds_per_page
    print(
        f"[decode] {progress.position}/{progress.total} "
        f"bundle={bundle_abbrev} outcome={progress.outcome} "
        f"tokens={progress.token_count} "
        f"page_seconds={progress.elapsed_seconds:.1f} "
        f"mean_seconds={progress.mean_seconds_per_page:.1f} "
        f"eta_seconds={eta_seconds:.0f}",
        file=sys.stderr,
    )


def _token_count_for_progress(
    outcome: DecodeOutcome,
    hypothesis: RawDocumentHypothesis | None,
    wiring: Mapping[str, Any],
) -> int:
    """How many tokens a page's decode actually emitted, for the progress
    line only -- `terminated` reads the sealed hypothesis, a typed failure
    that carries an `UnterminatedPrefix` (`max_steps_exceeded`/
    `vocabulary_exhausted`) reads its `accepted_tokens`, everything else
    (including pages that never reach `decode_page` at all) reports 0."""
    if outcome == "terminated" and hypothesis is not None:
        return len(hypothesis.accepted_tokens)
    prefix = wiring.get("unterminated_prefix")
    if prefix is not None:
        return len(prefix.accepted_tokens)
    return 0


def _report_progress(
    on_page: Callable[[PageProgress], None],
    *,
    position: int,
    total: int,
    document_bundle_id: str,
    outcome: str,
    token_count: int,
    page_elapsed: float,
    total_elapsed: float,
) -> None:
    """Build one `PageProgress` and hand it to `on_page`, swallowing (not
    propagating) any exception `on_page` raises: a progress reporter must
    never take down a multi-hour decode over a bad `on_page` (task decision
    -- see `decode_cohort`'s own docstring for the reasoning). Logged to
    stderr, once, so a broken callback is still visible rather than
    silently inert."""
    progress = PageProgress(
        position=position,
        total=total,
        document_bundle_id=document_bundle_id,
        outcome=outcome,
        token_count=token_count,
        elapsed_seconds=page_elapsed,
        mean_seconds_per_page=total_elapsed / position,
    )
    try:
        on_page(progress)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        print(f"[decode] on_page progress callback raised: {exc!r}", file=sys.stderr)


def decode_cohort(
    *,
    manifest: CohortManifest,
    cohort_manifest_sha256_value: str,
    arm: str,
    eval_cache_identity: Mapping[str, Any],
    run_manifest_bindings: Mapping[str, Any],
    component_lock_hash: str,
    feature_lookup: Callable[[str], Any],
    build_decode_kwargs: Callable[[Any], Mapping[str, Any]],
    decode_fn: Callable[..., RawDocumentHypothesis],
    feature_presence: Callable[[str], bool] | None = None,
    on_page: Callable[[PageProgress], None] = default_page_progress,
) -> SealedCohortArtefact:
    """Phase A's orchestrator: cohort manifest in, sealed artefact out.

    `feature_lookup(document_bundle_id)` returns whatever
    `build_decode_kwargs` needs to build one page's decode kwargs (a real
    `feature_cache.CacheEntry` or a test double) -- called only for pages
    whose `disposition == "attempt"`; the pre-decode typed-error
    classes and the ratified withdrawal never reach it, because none of
    them has a feature entry to look up. `decode_fn` is the real
    `free_running_decode.decode_free_running` (or a fake, in every test --
    free-running decode does not fit the 4070, design §12/GOAL).

    `component_lock_hash` is the run's production-preprocessor Component
    Lock hash (e.g. read from `component-lock-s1-c2.json` / the run
    manifest); folded into every sealed page's `input_parity_proof_hash`
    together with `manifest.policy_hash` as the page-selection rule for
    this cohort (§36.1:2400's other two mandatory bindings, M4 follow-up).

    `feature_presence`, if supplied, cross-checks every page's declared
    disposition against the real feature cache (governance follow-up:
    "derived, not declared" -- `verify_cohort_accounting`'s manifest-only
    check cannot catch a manifest that is simply wrong about which pages
    it says have features).

    `on_page` (bug `decode-reports-no-progress`) fires exactly once per
    `manifest.pages` entry, in order, AFTER that page's record is built --
    including the pre-decode typed-error dispositions and the
    withdrawn one, which never reach `decode_page` at all (a watcher must
    see the count advance on those too, not wonder why it jumped).
    Defaults to `default_page_progress` (writes one line to stderr);
    injectable so tests can capture it. A raising `on_page` is caught and
    swallowed here, not propagated: a broken progress reporter is a
    watcher's problem, never a reason to lose hours of real decode work,
    so it is logged to stderr once and decoding continues."""
    if arm not in sealed.VALID_ARMS:
        raise ValueError(f"arm must be one of {sorted(sealed.VALID_ARMS)}, got {arm!r}")
    refuse_unless_eval_cache_identity_ok(
        eval_cache_identity=eval_cache_identity,
        run_manifest_bindings=run_manifest_bindings,
    )
    accounting = verify_cohort_accounting(manifest, feature_presence=feature_presence)

    total_pages = len(manifest.pages)
    total_elapsed = 0.0
    records: list[PageDecodeResult] = []
    for position, page in enumerate(manifest.pages, start=1):
        page_started = time.monotonic()
        if page.disposition != ATTEMPT_DISPOSITION:
            records.append(
                PageDecodeResult(
                    document_bundle_id=page.document_bundle_id, outcome=page.disposition
                )
            )
            page_elapsed = time.monotonic() - page_started
            total_elapsed += page_elapsed
            _report_progress(
                on_page,
                position=position,
                total=total_pages,
                document_bundle_id=page.document_bundle_id,
                outcome=page.disposition,
                token_count=0,
                page_elapsed=page_elapsed,
                total_elapsed=total_elapsed,
            )
            continue
        entry = feature_lookup(page.document_bundle_id)
        if entry is None:
            raise ValueError(
                f"{page.document_bundle_id}: disposition is 'attempt' but no feature "
                "cache entry was found -- an 'attempt' page must have one"
            )
        decode_kwargs = build_decode_kwargs(entry)
        outcome, hypothesis, wiring = decode_page(decode_fn, decode_kwargs)
        if outcome == "terminated" and hypothesis is not None and hypothesis.node_states:
            records.append(
                seal_page(
                    document_bundle_id=page.document_bundle_id,
                    hypothesis=hypothesis,
                    eval_cache_identity=eval_cache_identity,
                    visual_features=decode_kwargs["visual_features"],
                    component_lock_hash=component_lock_hash,
                    selection_policy_hash=manifest.policy_hash,
                    decode_fn_identity=wiring.get("decode_fn"),
                    max_steps=wiring.get("max_steps"),
                    decode_fn_inner_identity=wiring.get("decode_fn_inner_identity"),
                    detail=_decode_detail(wiring),
                )
            )
        else:
            # must_fix 1 (code review): a non-terminated "attempt" page WAS
            # decoded under free_running against a real input -- it lacks a
            # hypothesis, not an inference mode or an input parity proof.
            # score.py:129 (`verify_seal`) refuses any attempted page
            # missing either of these two unconditionally, regardless of
            # outcome, so they must be carried here too or a single
            # max_steps_exceeded/unsupported_resume page kills phase B for
            # the whole cohort. `prediction_ir_hash` stays `None`: there is
            # no sealed hypothesis to hash.
            if outcome == "terminated":
                # DEC-030 plan task 1: `hypothesis.node_states` is empty --
                # `prediction_ir.seal_fragment` refuses to seal this
                # ("a RawDocumentHypothesis with no node_states is not
                # sealable"), and the frozen-init baseline arm can
                # terminate after only a few tokens with no nodes at all
                # (real traceback, 16.9.). Recorded as its own typed,
                # non-sealed outcome -- not a crash, not a sealed success --
                # carried exactly like every other non-sealed outcome below.
                outcome = "empty_hypothesis"
            parity_hash = compute_input_parity_proof_hash(
                document_bundle_id=page.document_bundle_id,
                eval_cache_identity=eval_cache_identity,
                visual_features=decode_kwargs["visual_features"],
                component_lock_hash=component_lock_hash,
                selection_policy_hash=manifest.policy_hash,
            )
            records.append(
                PageDecodeResult(
                    document_bundle_id=page.document_bundle_id,
                    outcome=outcome,
                    inference_mode=_INFERENCE_MODE,
                    input_parity_proof_hash=parity_hash,
                    scope=_SCOPE_KIND,
                    decode_fn_identity=wiring.get("decode_fn"),
                    max_steps=wiring.get("max_steps"),
                    decode_fn_inner_identity=wiring.get("decode_fn_inner_identity"),
                    unterminated_prefix=wiring.get("unterminated_prefix"),
                    detail=_decode_detail(wiring),
                )
            )
        page_elapsed = time.monotonic() - page_started
        total_elapsed += page_elapsed
        _report_progress(
            on_page,
            position=position,
            total=total_pages,
            document_bundle_id=page.document_bundle_id,
            outcome=outcome,
            token_count=_token_count_for_progress(outcome, hypothesis, wiring),
            page_elapsed=page_elapsed,
            total_elapsed=total_elapsed,
        )

    return SealedCohortArtefact(
        role=manifest.role,
        arm=arm,
        eval_cache_identity=eval_cache_identity,
        cohort_manifest_sha256=cohort_manifest_sha256_value,
        accounting=accounting,
        pages=tuple(records),
    )


# ---------------------------------------------------------------------------
# Task 7: the frozen-init baseline arm (DEC-009 dodatok C)
# ---------------------------------------------------------------------------


def build_frozen_init_model(
    *,
    sdpa_candidate: Any,
    device: torch.device,
    deepstack: bool = False,
) -> QwenDenseCandidateModel:
    """DEC-009 dodatok C's frozen-initialization baseline: exactly
    `s1_runner.build_context_from_loaded`'s own model construction
    (`s1_runner.py:2687-2701`) with the checkpoint load omitted -- Qwen's
    native reading after the VisualMemory identity fix, never the retracted
    random-projection defect (memory
    `random-projection-destroys-native-signal`). `sdpa_candidate` is the
    same `_CandidateLike` (real `QwenCandidate` or
    `candidate_loader.build_stub_qwen_candidate`) the trained arm's
    construction reads.

    `deepstack` (fresh S4 selection, default off): the DEC-056 DeepStack
    model for a checkpoint trained with it. It adds no parameter, so the
    state dict is the plain model's; only the forward reads the levels.
    """
    ranges = packed_ranges(sdpa_candidate.vocab_size)
    model = QwenDenseCandidateModel(
        decode=sdpa_candidate.decode,
        inject_visual=sdpa_candidate.inject_visual,
        visual_dim=sdpa_candidate.visual_hidden_size,
        d_model=sdpa_candidate.hidden_size,
        ranges=ranges,
        qwen_embedding_weight=sdpa_candidate.embed_tokens.weight,
        qwen_lm_head_weight=sdpa_candidate.lm_head_weight,
        **(
            {"deepstack": True, "image_token_id": sdpa_candidate.image_token_id}
            if deepstack
            else {}
        ),
    )
    model.to(device)
    model.eval()
    return model
