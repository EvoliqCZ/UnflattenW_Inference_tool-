"""RFC005 §12.2/§14.4 Stage Gate Report + Stage Gate Policy (plan
`spec-stage-gate-report-c5`).

A `StageGatePolicy` is a frozen, closed document naming the fifteen S1/S2
contract gates (`s1_runner.py`'s own `GateResult` name literals) and pinning
`missing_evidence_action_policy_wide = "blocked"`. Like `runtime_limits_policy`
it carries **no embedded self-hash** -- its identity
(`rfc005.stage-gate-policy/v1`) is a `domain_hash` over the whole validated
document, computed here, never stored inside it.

A `StageGateReport` is the closed, hash-bound artifact this module builds
from a `s1_runner.py` run: the runner's own fifteen `GateResult`s verbatim,
plus every C3/C4/C6/R1/S0 binding a promotion decision needs, cross-checked
against the real documents rather than accepted as caller-declared strings.
Unlike the policy, it DOES carry a self-hash (`stage_gate_report_hash`,
domain `rfc005.stage-gate-report/v1`) over the projection of every other
field -- mirrors `checkpoint_ancestry`/`candidate_track`, not
`runtime_limits_policy`.

`s1_runner.py` (3071 lines) is never imported or edited by this module: it
reads only the JSON payload `stage_gate_report_payload` already writes.

Torch-free by construction: nothing here imports torch.
"""

from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema.protocols
import jsonschema.validators

from unflatten.model import candidate_track as ct
from unflatten.model import exact_text_roundtrip_corpus as roundtrip
from unflatten.model import stage_policy as sp
from unflatten.model.capability_profile import (
    CapabilityProfileHashUnresolved,
    resolve_capability_profile_by_hash,
)
from unflatten.model.exact_text_cache import CacheExpectation, ExactTextCacheIncompatible
from unflatten.model.hashing import domain_hash
from unflatten.model.manifests import component_lock_hash
from unflatten.model.run_binding import (
    RunManifestBindingVerifier,
    compute_exact_text_cache_identity_hash,
)

_SCHEMA_DIR = Path(__file__).parent / "schemas"

#: Design registry row `rfc005.stage-gate-policy/v1`. Owned here: no other
#: module computes a stage gate policy's identity hash.
_POLICY_DOMAIN = "rfc005.stage-gate-policy/v1"

#: Design registry row `rfc005.stage-gate-report/v1`. Owned here.
_REPORT_DOMAIN = "rfc005.stage-gate-report/v1"

STAGE_GATE_POLICY_SCHEMA_VERSION = "1.0.0"
STAGE_GATE_REPORT_SCHEMA_VERSION = "1.0.0"

#: `s1_runner.S1_RUNNER_SCHEMA_VERSION` values this module accepts as
#: evidence. Read as a plain string constant, never by importing
#: `s1_runner` -- this module must stay independent of the 3071-line
#: read-only contract.
RUNNER_REPORT_SCHEMA_VERSIONS: frozenset[str] = frozenset({"0.1.0"})

#: The one frozen policy this plan ships (2026-09-07 owner decision).
FROZEN_POLICY_PATH = Path(__file__).parent / "policies" / "s1_stage_gate_policy.json"

#: Exactly fifteen distinct gate numbers, 1..15 (RFC005 §14.4).
_EXPECTED_GATE_NUMBERS = frozenset(range(1, 16))


class StageGateReportInvalid(ValueError):
    """A stage gate policy or report is malformed, or a report's bindings
    disagree with the real documents they claim to summarize.

    M1-internal (design §9.3); not an RFC004/RFC005 registry code.
    """


@cache
def _validator(name: str) -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / f"{name}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _validate(name: str, doc: dict[str, Any], *, what: str) -> None:
    errors = sorted(_validator(name).iter_errors(doc), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        pointer = "/" + "/".join(str(p) for p in first.absolute_path)
        raise StageGateReportInvalid(f"{what} invalid at {pointer or '/'}: {first.message}")


def validate_stage_gate_policy(doc: dict[str, Any]) -> None:
    _validate("stage_gate_policy", doc, what="stage gate policy")
    if doc["stage"] not in ("S1", "S2"):
        raise StageGateReportInvalid(f"stage gate policy: unknown stage {doc['stage']!r}")
    if doc["quality"]["status"] == "applies":
        raise StageGateReportInvalid(
            "stage gate policy: quality.status='applies' is refused at S1/S2 "
            "(owner R5 2026-09-07, D4) -- only 'not_applicable' may be frozen here"
        )


def validate_stage_gate_report(doc: dict[str, Any]) -> None:
    _validate("stage_gate_report", doc, what="stage gate report")


def stage_gate_policy_hash(policy: dict[str, Any]) -> str:
    """`policy_hash` for `policy`: `domain_hash` over the whole validated
    document (module docstring) -- no embedded self-hash to project away."""
    validate_stage_gate_policy(policy)
    return domain_hash(_POLICY_DOMAIN, policy)


def load_stage_gate_policy(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise StageGateReportInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise StageGateReportInvalid(f"{path} is not valid JSON: {exc}") from exc
    validate_stage_gate_policy(payload)
    return payload


def _hashable(value: Any) -> Any:
    """`value` with every float replaced by its exact, tagged hex form.

    The hash profile (`rfc8949-core-det-nfc-int-v0`) has no float, but a real
    runner report's gate evidence does (gate 3 losses, gate 12 logit diffs --
    2026-09-14, the first report frozen from real data). `float.hex` is exact
    and deterministic, so the hash still binds the precise value; the tag
    keeps a float from colliding with an equal int or with a string."""
    if isinstance(value, float):
        return {"float64_hex": value.hex()}
    if isinstance(value, dict):
        return {k: _hashable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_hashable(v) for v in value]
    return value


def report_projection(doc: dict[str, Any]) -> dict[str, Any]:
    """The payload `stage_gate_report_hash` is computed over: `doc` with its
    own `stage_gate_report_hash` removed, floats in their exact tagged hex
    form (`_hashable`). A float-free document projects unchanged."""
    if "stage_gate_report_hash" not in doc:
        raise StageGateReportInvalid(
            "doc has no 'stage_gate_report_hash' key -- the self-hash slot must "
            "be present (even as None) before it can be projected away"
        )
    return _hashable({k: v for k, v in doc.items() if k != "stage_gate_report_hash"})


def compute_stage_gate_report_hash(doc: dict[str, Any]) -> str:
    return domain_hash(_REPORT_DOMAIN, report_projection(doc))


# ---------------------------------------------------------------------------
# Derived projections from the runner payload
# ---------------------------------------------------------------------------


def _gate_numbers(gates: list[dict[str, Any]]) -> set[int]:
    return {g["number"] for g in gates}


def _gate_by_number(gates: list[dict[str, Any]], number: int) -> dict[str, Any] | None:
    for g in gates:
        if g["number"] == number:
            return g
    return None


def _gate_12_negative_control(gates: list[dict[str, Any]]) -> dict[str, Any]:
    gate12 = _gate_by_number(gates, 12)
    if gate12 is None:
        raise StageGateReportInvalid("runner payload has no gate 12")
    evidence = gate12.get("evidence", {})
    ran = "negative_control_status" in evidence
    return {"ran": ran, "status": evidence.get("negative_control_status") if ran else None}


def _lineage_evidence(gates: list[dict[str, Any]]) -> dict[str, Any]:
    gate9 = _gate_by_number(gates, 9)
    if gate9 is None:
        raise StageGateReportInvalid("runner payload has no gate 9")
    evidence = gate9.get("evidence", {})
    required = (
        "overlap_count",
        "cohort_lineage_root_count",
        "requested_document_count",
        "resolved_document_count",
    )
    if all(key in evidence for key in required):
        return {"status": "evidenced", **{key: evidence[key] for key in required}}
    return {
        "status": "not_evidenced",
        "reason": "gate 9 evidence carries no lineage counts (blocked or absent)",
    }


def _missing_evidence_outcomes(gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"gate": g["number"], "action": "blocked"}
        for g in sorted(gates, key=lambda g: g["number"])
        if g["status"] == "blocked"
    ]


def _promotion_blockers(
    *,
    gates: list[dict[str, Any]],
    run_class: str,
    run_id: str,
    evaluated_candidate: dict[str, Any],
    ancestry_rights_status: str,
    ancestry_privacy_status: str,
    gate_12_negative_control: dict[str, Any],
) -> list[str]:
    """Every closed-enum promotion disqualifier, in order -- shared by the
    builder and the verifier so there is exactly one place that decides what
    blocks promotion."""
    blockers: list[str] = []
    for gate in sorted(gates, key=lambda g: g["number"]):
        if gate["status"] != "pass":
            blockers.append(f"gate_not_pass:{gate['number']}")
    if run_class == "discovery":
        blockers.append("run_class_discovery")
    if not run_id.startswith("run_"):
        blockers.append("run_id_not_rfc005_44_1")
    if evaluated_candidate["kind"] == "base_weights_only":
        blockers.append("evaluated_candidate_not_run_output")
    if ancestry_rights_status != "approved":
        blockers.append("ancestry_rights_status_not_approved")
    if ancestry_privacy_status != "approved":
        blockers.append("ancestry_privacy_status_not_approved")
    gate12 = _gate_by_number(gates, 12)
    if gate12 is not None and gate12["status"] == "pass" and not gate_12_negative_control["ran"]:
        blockers.append("gate_12_no_negative_control")
    return blockers


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


# ---------------------------------------------------------------------------
# build_stage_gate_report
# ---------------------------------------------------------------------------


def build_stage_gate_report(
    *,
    policy: dict[str, Any],
    runner_payload: dict[str, Any],
    runner_report_path: Path,
    run_manifest: dict[str, Any],
    component_lock: dict[str, Any],
    cache_identity_payload: dict[str, Any],
    candidate_track: dict[str, Any],
    ancestry: dict[str, Any],
    s0_report: dict[str, Any],
    s0_report_path: Path,
    s0_report_relpath: str,
    stage_policy_doc: dict[str, Any],
    corpus_descriptor: dict[str, Any],
    corpus_check: dict[str, Any],
    evaluated_candidate: dict[str, Any],
    run_stage: str,
    allowlist: dict[str, Any] | None = None,
    previous_corpus_descriptor: dict[str, Any] | None = None,
    expectation: CacheExpectation | None = None,
) -> dict[str, Any]:
    """Build one Stage Gate Report, refusing (`StageGateReportInvalid`) on
    every disagreement decision 4 names -- see the module docstring and the
    plan for the full rationale of each check."""
    validate_stage_gate_policy(policy)

    gates: list[dict[str, Any]] = runner_payload["gates"]
    numbers = _gate_numbers(gates)
    if numbers != _EXPECTED_GATE_NUMBERS:
        raise StageGateReportInvalid(
            f"runner payload gates are {sorted(numbers)}, expected exactly 1..15"
        )

    if runner_payload["schema_version"] not in RUNNER_REPORT_SCHEMA_VERSIONS:
        raise StageGateReportInvalid(
            f"runner payload schema_version {runner_payload['schema_version']!r} is not in "
            f"the pinned set {sorted(RUNNER_REPORT_SCHEMA_VERSIONS)}"
        )

    policy_gate_names = {g["gate"]: g["name"] for g in policy["gates"]}
    for gate in gates:
        expected_name = policy_gate_names.get(gate["number"])
        if expected_name != gate["name"]:
            raise StageGateReportInvalid(
                f"gate {gate['number']} name {gate['name']!r} does not match the policy's "
                f"{expected_name!r}"
            )

    gates_all_pass = all(g["status"] == "pass" for g in gates)
    if runner_payload["promoted"] != gates_all_pass:
        raise StageGateReportInvalid(
            "runner payload's own 'promoted' does not agree with its own gates -- "
            f"promoted={runner_payload['promoted']!r}, "
            f"all_pass={gates_all_pass!r}"
        )

    if runner_payload["run_class"] != run_manifest["run_class"]:
        raise StageGateReportInvalid(
            f"runner payload run_class {runner_payload['run_class']!r} != "
            f"run manifest run_class {run_manifest['run_class']!r}"
        )
    run_class = run_manifest["run_class"]
    run_id = run_manifest["run_id"]

    # -- component lock: run manifest, candidate track, and a fresh recompute
    #    over `component_lock` must all agree (decision 4: "lock hash run
    #    manifestu != track != prepocitany").
    recomputed_lock_hash = component_lock_hash(component_lock)
    if run_manifest["component_lock_hash"] != recomputed_lock_hash:
        raise StageGateReportInvalid(
            f"run manifest component_lock_hash {run_manifest['component_lock_hash']!r} != "
            f"component_lock_hash(component_lock) {recomputed_lock_hash!r}"
        )
    if candidate_track["model_component_lock_hash"] != recomputed_lock_hash:
        raise StageGateReportInvalid(
            f"candidate track model_component_lock_hash "
            f"{candidate_track['model_component_lock_hash']!r} != "
            f"component_lock_hash(component_lock) {recomputed_lock_hash!r}"
        )

    # -- candidate track must independently verify against the same lock and
    #    ancestry (decision 4: "track neoveri").
    try:
        ct.verify_candidate_track(candidate_track, component_lock=component_lock, ancestry=ancestry)
    except ct.CandidateTrackInvalid as exc:
        raise StageGateReportInvalid(f"candidate track failed verification: {exc}") from exc

    # -- cache identity vs run manifest: reuse the real verifier rather than
    #    a private field-by-field comparison (decision 4: "identita cache !=
    #    manifest").
    # Resolved from the cache identity's own `capability_profile_hash`
    # (DEC-024 wiring), never an import-time global: the profile a report
    # is checked against must be the one its cache actually names.
    try:
        capability_profile = resolve_capability_profile_by_hash(
            cache_identity_payload.get("capability_profile_hash")
        )
    except CapabilityProfileHashUnresolved as exc:
        raise StageGateReportInvalid(f"cache identity capability profile refused: {exc}") from exc
    try:
        RunManifestBindingVerifier().verify(
            run_manifest,
            cache_identity_payload,
            allowlist,
            capability_profile=capability_profile,
            # DEC-MP-4: a gradient-bearing manifest is verified only under the
            # cohort it was frozen with (2026-09-14: this call omitted it).
            expectation=expectation,
        )
    except ExactTextCacheIncompatible as exc:
        raise StageGateReportInvalid(f"run manifest/cache identity binding failed: {exc}") from exc
    cache_identity_hash = compute_exact_text_cache_identity_hash(cache_identity_payload)

    # -- admission track name must be one S0 actually admitted.
    admitted_tracks = s0_report["model_admission"]["admitted_tracks"]
    admission_track_name = candidate_track["admission_track_name"]
    if admission_track_name not in admitted_tracks:
        raise StageGateReportInvalid(
            f"admission_track_name {admission_track_name!r} not in "
            f"s0.model_admission.admitted_tracks {admitted_tracks!r}"
        )

    if s0_report["outcome"] != "promoted":
        raise StageGateReportInvalid(
            f"s0 report outcome is {s0_report['outcome']!r}, expected 'promoted'"
        )
    if "report_schema_version" not in s0_report:
        raise StageGateReportInvalid("s0 report has no 'report_schema_version' key")

    # -- S0 <-> cache identity / ancestry cross-links.
    s0_release_root_hash = s0_report["inputs"]["release_root_hash"]
    if s0_release_root_hash != cache_identity_payload["dataset_release_root_hash"]:
        raise StageGateReportInvalid(
            f"s0 report inputs.release_root_hash {s0_release_root_hash!r} != "
            f"cache identity dataset_release_root_hash "
            f"{cache_identity_payload['dataset_release_root_hash']!r}"
        )
    s0_file_sha256 = _file_sha256(Path(s0_report_path))
    ancestry_s0_sha256 = ancestry["legacy_ancestry"]["evidence"]["s0_report"]["sha256"]
    if s0_file_sha256 != ancestry_s0_sha256:
        raise StageGateReportInvalid(
            f"s0 report file sha256 {s0_file_sha256!r} != ancestry "
            f"legacy_ancestry.evidence.s0_report.sha256 {ancestry_s0_sha256!r}"
        )

    # -- roundtrip corpus: must be Qwen-path evidence, passing, byte-fallback
    #    routed for the byte-length buckets, and bound to the SAME codec
    #    identity as the cache (never the run manifest -- decision 4).
    if not corpus_check["passed"]:
        raise StageGateReportInvalid("roundtrip corpus check did not pass")
    if corpus_check["codec_mode"] != "qwen":
        raise StageGateReportInvalid(
            f"roundtrip corpus codec_mode is {corpus_check['codec_mode']!r}, expected 'qwen'"
        )
    if corpus_check["byte_len_route"] != "byte_fallback":
        raise StageGateReportInvalid(
            f"roundtrip corpus byte_len_route is {corpus_check['byte_len_route']!r}, "
            "expected 'byte_fallback'"
        )
    if corpus_check["tokenizer_artifact_set_hash"] != cache_identity_payload[
        "tokenizer_artifact_set_hash"
    ]:
        raise StageGateReportInvalid(
            "roundtrip corpus check tokenizer_artifact_set_hash does not match the "
            "cache identity's tokenizer_artifact_set_hash"
        )
    if corpus_check["composite_vocabulary_hash"] != cache_identity_payload[
        "composite_vocabulary_hash"
    ]:
        raise StageGateReportInvalid(
            "roundtrip corpus check composite_vocabulary_hash does not match the "
            "cache identity's composite_vocabulary_hash"
        )
    try:
        roundtrip.validate_corpus_descriptor(corpus_descriptor)
        if previous_corpus_descriptor is not None:
            roundtrip.assert_monotonic(previous_corpus_descriptor, corpus_descriptor)
    except roundtrip.RoundtripCorpusInvalid as exc:
        raise StageGateReportInvalid(str(exc)) from exc

    # -- loss policy stage binding: the frozen stage_policy.json's own
    #    stage_id, cross-checked by hash against the run manifest, must
    #    equal this gate policy's own `stage` (governance 2026-09-07 item 1).
    if sp.stage_policy_hash(stage_policy_doc) != run_manifest["stage_policy_hash"]:
        raise StageGateReportInvalid(
            "stage_policy_doc does not hash to the run manifest's stage_policy_hash"
        )
    loss_policy_stage_id = stage_policy_doc["stage_id"]
    if loss_policy_stage_id != policy["stage"]:
        raise StageGateReportInvalid(
            f"loss policy stage_id {loss_policy_stage_id!r} != stage gate policy "
            f"stage {policy['stage']!r}"
        )

    if run_stage not in ("S1", "S2"):
        raise StageGateReportInvalid(f"run_stage must be 'S1' or 'S2', got {run_stage!r}")
    notes: tuple[str, ...]
    if run_stage != loss_policy_stage_id:
        notes = (
            f"DEC-009: this {run_stage} run is evaluated under the frozen "
            f"{loss_policy_stage_id} loss policy (stage_policy.json "
            f"stage_id={loss_policy_stage_id})",
        )
    else:
        notes = ()

    ancestry_rights_status = ancestry["checkpoint_rights_status"]
    ancestry_privacy_status = ancestry["checkpoint_privacy_status"]

    gate_12_negative_control = _gate_12_negative_control(gates)
    lineage_evidence = _lineage_evidence(gates)

    # -- promotion_blockers: every closed-enum disqualifier, in order.
    promotion_blockers = _promotion_blockers(
        gates=gates,
        run_class=run_class,
        run_id=run_id,
        evaluated_candidate=evaluated_candidate,
        ancestry_rights_status=ancestry_rights_status,
        ancestry_privacy_status=ancestry_privacy_status,
        gate_12_negative_control=gate_12_negative_control,
    )

    missing_evidence_outcomes = _missing_evidence_outcomes(gates)

    promoted = not promotion_blockers

    runner_report_sha256 = _file_sha256(Path(runner_report_path))

    doc: dict[str, Any] = {
        "stage_gate_report_schema_version": STAGE_GATE_REPORT_SCHEMA_VERSION,
        "stage": loss_policy_stage_id,
        "run_stage": run_stage,
        "notes": list(notes),
        "run_class": run_class,
        "runner_report_sha256": runner_report_sha256,
        "bindings": {
            "policy_hash": stage_gate_policy_hash(policy),
            "run_id": run_id,
            "s1_run_manifest_hash": run_manifest["s1_run_manifest_hash"],
            "component_lock_hash": recomputed_lock_hash,
            "candidate_track_id": candidate_track["candidate_track_id"],
            "admission_track_name": admission_track_name,
            "ancestry_hash": ancestry["ancestry_hash"],
            "exact_text_cache_identity_hash": cache_identity_hash,
            "dataset_release_root_hash": cache_identity_payload["dataset_release_root_hash"],
            "loss_policy_stage_id": loss_policy_stage_id,
            "stage_policy_hash": run_manifest["stage_policy_hash"],
            "ancestry_rights_status": ancestry_rights_status,
            "ancestry_privacy_status": ancestry_privacy_status,
            "exposure_ledger": ancestry["exposure_ledger"],
            "s0_report": {
                "path": s0_report_relpath,
                "sha256": s0_file_sha256,
                "report_schema_version": s0_report["report_schema_version"],
                "admitted_tracks": list(admitted_tracks),
            },
            "roundtrip_corpus": {
                "descriptor": corpus_descriptor,
                "check": {
                    "passed": corpus_check["passed"],
                    "checked_case_count": corpus_check["checked_case_count"],
                    "codec_mode": corpus_check["codec_mode"],
                    "byte_len_route": corpus_check["byte_len_route"],
                    "budget_ops": corpus_check["budget_ops"],
                    "tokenizer_artifact_set_hash": corpus_check["tokenizer_artifact_set_hash"],
                    "composite_vocabulary_hash": corpus_check["composite_vocabulary_hash"],
                },
            },
        },
        "evaluated_candidate": evaluated_candidate,
        "gates": gates,
        "gate_12_negative_control": gate_12_negative_control,
        "lineage_evidence": lineage_evidence,
        "gates_all_pass": gates_all_pass,
        "promotion_blockers": promotion_blockers,
        "missing_evidence_outcomes": missing_evidence_outcomes,
        "promoted": promoted,
        "stage_gate_report_hash": None,
    }
    doc["stage_gate_report_hash"] = compute_stage_gate_report_hash(doc)

    validate_stage_gate_report(doc)
    verify_stage_gate_report(doc, policy=policy)
    return doc


# ---------------------------------------------------------------------------
# verify_stage_gate_report
# ---------------------------------------------------------------------------


def verify_stage_gate_report(doc: dict[str, Any], *, policy: dict[str, Any] | None = None) -> None:
    """Verify a previously-built stage gate report: schema, self-hash, and
    every field this module derives -- re-derived and compared, not merely
    schema-checked. When `policy` is supplied, its hash and its gate names
    are also cross-checked against `doc`."""
    validate_stage_gate_report(doc)

    expected_hash = compute_stage_gate_report_hash(doc)
    if doc["stage_gate_report_hash"] != expected_hash:
        raise StageGateReportInvalid(
            "stage_gate_report_hash does not match its own payload -- the "
            "document was mutated after freezing"
        )

    gates = doc["gates"]
    numbers = _gate_numbers(gates)
    if numbers != _EXPECTED_GATE_NUMBERS:
        raise StageGateReportInvalid(f"report gates are {sorted(numbers)}, expected exactly 1..15")

    gates_all_pass = all(g["status"] == "pass" for g in gates)
    if doc["gates_all_pass"] != gates_all_pass:
        raise StageGateReportInvalid(
            f"gates_all_pass {doc['gates_all_pass']!r} does not match a recompute "
            f"over doc['gates'] ({gates_all_pass!r})"
        )

    if doc["gate_12_negative_control"] != _gate_12_negative_control(gates):
        raise StageGateReportInvalid("gate_12_negative_control does not match doc['gates']")

    if doc["lineage_evidence"] != _lineage_evidence(gates):
        raise StageGateReportInvalid("lineage_evidence does not match doc['gates']")

    missing_evidence_outcomes = _missing_evidence_outcomes(gates)
    if doc["missing_evidence_outcomes"] != missing_evidence_outcomes:
        raise StageGateReportInvalid("missing_evidence_outcomes does not match doc['gates']")

    run_id = doc["bindings"]["run_id"]
    evaluated_candidate = doc["evaluated_candidate"]
    ancestry_rights_status = doc["bindings"]["ancestry_rights_status"]
    ancestry_privacy_status = doc["bindings"]["ancestry_privacy_status"]

    expected_blockers = _promotion_blockers(
        gates=gates,
        run_class=doc["run_class"],
        run_id=run_id,
        evaluated_candidate=evaluated_candidate,
        ancestry_rights_status=ancestry_rights_status,
        ancestry_privacy_status=ancestry_privacy_status,
        gate_12_negative_control=doc["gate_12_negative_control"],
    )
    if doc["promotion_blockers"] != expected_blockers:
        raise StageGateReportInvalid(
            f"promotion_blockers {doc['promotion_blockers']!r} does not match a "
            f"recompute {expected_blockers!r}"
        )

    expected_promoted = not expected_blockers
    if doc["promoted"] != expected_promoted:
        raise StageGateReportInvalid(
            f"promoted {doc['promoted']!r} does not match a recompute {expected_promoted!r}"
        )

    if policy is not None:
        validate_stage_gate_policy(policy)
        expected_policy_hash = stage_gate_policy_hash(policy)
        if doc["bindings"]["policy_hash"] != expected_policy_hash:
            raise StageGateReportInvalid(
                f"bindings.policy_hash {doc['bindings']['policy_hash']!r} != "
                f"stage_gate_policy_hash(policy) {expected_policy_hash!r}"
            )
        policy_gate_names = {g["gate"]: g["name"] for g in policy["gates"]}
        for gate in gates:
            expected_name = policy_gate_names.get(gate["number"])
            if expected_name != gate["name"]:
                raise StageGateReportInvalid(
                    f"gate {gate['number']} name {gate['name']!r} does not match the "
                    f"policy's {expected_name!r}"
                )
        if policy["stage"] != doc["stage"]:
            raise StageGateReportInvalid(
                f"policy stage {policy['stage']!r} != report stage {doc['stage']!r}"
            )
