"""Assembles the RFC005 §9.3 `ModelTargetCoverageReport` over a real dataset
release: the four bindings, the ten axes, and the CLI that writes one.

Bindings are **recomputed, never read as a claim** (§9.3 binds the report to
them, and a copied hash binds nothing):

* `release_root_hash` -- from `admission.evidence.load_release`, which
  recomputes it from the manifest projection;
* `component_lock_hash` -- `manifests.component_lock_hash` over the lock file;
* `training_target_materializer_hash` -- the **descriptor** hash
  (`rfc005.training-target-materializer/v1`), rebuilt from the lock's pinned
  `composite_vocabulary` / `model_capability_profile` / `grammar` component
  hashes and `loss_unit.loss_unit_schema_hash()`. Not the implementation
  hash: `...-impl/v1` moves with `MATERIALIZER_VERSION` alone and says
  nothing about which vocabulary or profile the targets were built under;
* `stage_policy_hash` -- over the loaded, validated policy document.

Five of the ten axes come out entirely zero (down from seven: since S3 the
two typed-property axes are `unmeasured`, not a derived zero), and naming
*why* is the work. Since S3 (RFC005 §27.5)
`training_materializer.SUPPORTED_LOSS_FAMILIES` is `("L_structure",
"L_text", "L_typed")`: `L_typed` declares a supervision channel -- per-row
`known_mask=false` is `no_supervision`, never a numeric zero -- that the
release walk has no counter for. The remaining zeros are named, and RFC003
label presence in the release does not change them -- §9.3's own last
sentence is that AST coverage does not imply model token/head coverage,
and reporting the presence counts here would be exactly that mistake.

Five axes have no domain in code to enumerate at all (numeric bins,
repeated-field variants, TablePlan variants, resource recovery modes, and
the prediction-status half of §9.3's eighth bullet). They are
`domain_undefined` with the missing artefact named, because an axis quietly
omitted is the hole this report exists to close.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from unflatten.dataset.admission.evidence import load_release
from unflatten.dataset.enums import LABEL_ORIGIN
from unflatten.dataset.targets.properties import FIELD_VERIFICATION
from unflatten.model import events_vocab
from unflatten.model.loss_unit import loss_unit_schema_hash
from unflatten.model.manifests import component_lock_hash
from unflatten.model.model_target_coverage import build_model_target_coverage
from unflatten.model.model_target_coverage_axes import (
    categorical_value_axis,
    counted_entry,
    excluded_entry,
    make_axis,
    property_pointer_axis,
    undefined_domain_axes,
    zero_entry,
)
from unflatten.model.model_target_coverage_counts import (
    TRAIN_READY,
    ModelTargetCountingError,
    ReleaseTally,
    count_release_targets,
    count_release_targets_resumable,
)
from unflatten.model.stage_policy import load_stage_policy, stage_policy_hash
from unflatten.model.training_materializer import (
    build_materializer_descriptor,
    materializer_descriptor_hash,
)

_NO_TEXT_ROUTE_MEASUREMENT = (
    "text routes are decided by exact_text.encode_text, which needs the pinned "
    "Qwen tokenizer and is quadratic in tokenizer probes per span; no "
    "release-wide materialisation of it exists. The only materialised surface "
    "is the S1 cohort exact-text cache, whose scope is a page cohort and not "
    "this report's scope, so counting it here would misstate the scope"
)


def _token_node_type() -> dict[str, str]:
    """token name -> the node type it claims, for the tokens that claim one."""
    mapping: dict[str, str] = {}
    for table in (
        events_vocab.OPEN_TOKEN,
        events_vocab.CLOSE_TOKEN,
        events_vocab.LEAF_TOKEN,
        events_vocab.RESUME_TOKEN,
        events_vocab.SUSPEND_TOKEN,
    ):
        for node_type, token in table.items():
            mapping[token] = node_type
    return mapping


#: The three symbols a page-level target can never emit, and why. Kept apart
#: from the generic reason because "the corpus happens not to contain it" and
#: "the target unit structurally cannot emit it" are different findings, and
#: only the second one stays true for any release.
_DOCUMENT_LEVEL_ONLY: Mapping[str, str] = {
    "DOCUMENT_OPEN": "a fragment is a window INTO a document, so RFC004 §18.4 "
    "makes DOCUMENT_OPEN illegal as a fragment-root child and a page target "
    "resumes the document frame instead",
    "DOCUMENT_CLOSE": "the counterpart of DOCUMENT_OPEN: a page target suspends "
    "the document frame rather than closing it (RFC004 §18.4)",
    "END_DOCUMENT": "a page target ends with END_FRAGMENT; END_DOCUMENT is "
    "emitted only by encode_structural over a whole document, and is used as BOS "
    "priming in model/targets.py",
}


def _uncovered_symbol_reason(token: str) -> tuple[str, str]:
    """`(missing artefact, zero_class)` for a symbol no target emits.

    The two kinds are returned as data, not only as prose: a wider release
    closes an `absent_from_release` zero and can never close a
    `target_unit_incompatible` one.
    """
    document_level = _DOCUMENT_LEVEL_ONLY.get(token)
    if document_level is not None:
        return (
            f"{document_level}. Covering {token} needs a document-level target "
            "stream, which no stage materialises today",
            "target_unit_incompatible",
        )
    return (
        f"no page fragment in this scope emits {token}: the counted release "
        "contains no supervised page carrying the construct (for a RESUME/SUSPEND "
        "symbol, no container of that type crosses a page boundary). Covering it "
        "needs a dataset release that does",
        "absent_from_release",
    )


def structural_symbol_axis(tally: ReleaseTally, profiles: Sequence[str]) -> dict[str, Any]:
    """Axis 1: every structural and fragment symbol of the 71-token alphabet.

    A symbol whose node type no declared capability profile admits is
    capability-excluded; every other zero is a zero the release simply does
    not contain, and names that.
    """
    eligible: set[str] = set()
    for profile in profiles:
        eligible |= set(events_vocab.node_types(profile))
    node_type_of = _token_node_type()
    entries: list[dict[str, Any]] = []
    for token in sorted(events_vocab.TOKEN_IDS):
        node_type = node_type_of.get(token)
        counted = tally.symbols.get(token)
        if node_type is not None and node_type not in eligible:
            if counted:
                raise ModelTargetCountingError(
                    f"token {token} occurs {counted[0]} times but its node type "
                    f"{node_type!r} is outside the declared profiles "
                    f"{list(profiles)} -- the lock and the release disagree"
                )
            entries.append(
                excluded_entry(
                    token,
                    f"node type {node_type!r} is not admitted by any capability "
                    f"profile the Component Lock declares ({list(profiles)}); "
                    "events_vocab.PROFILE_NODE_TYPES",
                )
            )
        elif counted:
            entries.append(counted_entry(token, tally.symbols))
        else:
            entries.append(zero_entry(token, *_uncovered_symbol_reason(token)))
    return make_axis(
        "structural_symbols",
        "kazdy structural a fragment symbol",
        "unflatten.model.events_vocab.TOKEN_IDS via encode_fragment over the "
        "release's pinned page fragments",
        "token_occurrence",
        entries,
    )


def tokenizer_route_axis(tally: ReleaseTally) -> dict[str, Any]:
    """Axis 2: the four reversible composite-vocabulary routes."""
    entries = [counted_entry("structural", tally.routes)]
    entries += [
        {
            "feature_key": route,
            "state": "unmeasured",
            "unmeasured_reason": _NO_TEXT_ROUTE_MEASUREMENT,
        }
        for route in ("qwen_text", "byte_control", "byte_value")
    ]
    return make_axis(
        "tokenizer_routes",
        "kazdy reversible tokenizer route vratane scalar/byte fallbacku",
        "unflatten.model.composite_vocab.packed_ranges (structural | qwen | "
        "byte_control | byte_value)",
        "token_occurrence",
        entries,
    )


def label_authority_axis(tally: ReleaseTally) -> dict[str, Any]:
    """Axis 8a: the label authority behind what is actually supervised.

    The domain is the full `label_origin x verification` product, so an
    authority the release contains none of is visible as a zero rather than
    as an absent row. Field-level `origin`/`verification` are not counted:
    they belong to typed labels, which nothing supervises (axis 3).
    """
    entries: list[dict[str, Any]] = []
    for origin in sorted(LABEL_ORIGIN):
        for verification in sorted(FIELD_VERIFICATION):
            key = f"{origin}/{verification}"
            if key in tally.label_authority:
                entries.append(counted_entry(key, tally.label_authority))
            else:
                entries.append(
                    zero_entry(
                        key,
                        "no counted target bundle declares this label authority in "
                        "its `provenance`; covering it needs a dataset release "
                        f"carrying {origin!r} labels verified as {verification!r}",
                        "absent_from_release",
                    )
                )
    return make_axis(
        "label_authority",
        "kazdy supervised prediction status s explicitnou label autoritou "
        "(the label-authority half)",
        "unflatten.dataset.enums.LABEL_ORIGIN x "
        "unflatten.dataset.targets.properties.FIELD_VERIFICATION, counted from "
        "each pinned target bundle's `provenance`",
        "document_bundle",
        entries,
    )


def _read_component_lock(path: Path | str) -> dict[str, Any]:
    """Read the Component Lock, wrapping both failure modes (repo rule:
    every error names the operation that failed)."""
    try:
        lock = json.loads(Path(path).read_bytes())
    except OSError as exc:
        raise ModelTargetCountingError(f"cannot read component lock {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ModelTargetCountingError(
            f"component lock {path} is not valid JSON: {exc}"
        ) from exc
    absent = [key for key in ("components", "profiles") if key not in lock]
    if absent:
        raise ModelTargetCountingError(
            f"component lock {path} declares no {', '.join(absent)} -- the "
            "bindings and the capability exclusions both read them"
        )
    return lock


def resolve_bindings(
    *,
    release_dir: Path | str,
    component_lock_path: Path | str,
    stage_policy_path: Path | str,
    exact_text_cache_identity_hash: str | None,
) -> dict[str, Any]:
    """The four §9.3 bindings, each recomputed from its artefact."""
    lock = _read_component_lock(component_lock_path)
    components = {
        component["kind"]: component.get("semantic_hash")
        or component.get("artifact_sha256")
        for component in lock["components"]
    }
    try:
        descriptor = build_materializer_descriptor(
            loss_unit_schema_hash=loss_unit_schema_hash(),
            composite_vocabulary_hash=components["composite_vocabulary"],
            composite_grammar_hash=components["grammar"],
            capability_profile_hash=components["model_capability_profile"],
        )
    except KeyError as exc:
        raise ModelTargetCountingError(
            f"component lock {component_lock_path} pins no {exc.args[0]!r} "
            "component -- the materializer descriptor cannot be rebuilt from it, "
            "and a binding that is guessed binds nothing"
        ) from exc
    return {
        "release_root_hash": load_release(release_dir).manifest["release_root_hash"],
        "component_lock_hash": component_lock_hash(lock),
        "training_target_materializer_hash": materializer_descriptor_hash(descriptor),
        "stage_policy_hash": stage_policy_hash(load_stage_policy(stage_policy_path)),
        "exact_text_cache_identity_hash": exact_text_cache_identity_hash,
    }


def build_release_report(
    *,
    release_dir: Path | str,
    store_dir: Path | str,
    component_lock_path: Path | str,
    stage_policy_path: Path | str,
    splits: Sequence[str],
    eligibility: str,
    stage_id: str,
    report_version: str,
    exact_text_cache_identity_hash: str | None,
    run_dir: Path | str | None = None,
    workers: int | None = None,
    checkpoint_every: int = 25,
) -> dict[str, Any]:
    """Count one release and assemble the validated §9.3 report over it.

    `run_dir` selects the counting path: `None` runs the small,
    single-process `count_release_targets` (used by the tests here);
    a path runs the parallel, resumable `count_release_targets_resumable`
    (DEC-023 -- the real release run's `encode_fragment` pass takes well
    over 20 minutes).
    """
    lock = _read_component_lock(component_lock_path)
    if run_dir is None:
        tally = count_release_targets(
            release_dir=release_dir,
            store_dir=store_dir,
            splits=splits,
            eligibility=eligibility,
        )
    else:
        tally = count_release_targets_resumable(
            release_dir=release_dir,
            store_dir=store_dir,
            splits=splits,
            eligibility=eligibility,
            run_dir=run_dir,
            workers=workers,
            checkpoint_every=checkpoint_every,
        )
    axes = [
        structural_symbol_axis(tally, lock["profiles"]),
        tokenizer_route_axis(tally),
        property_pointer_axis(),
        categorical_value_axis(),
        *undefined_domain_axes(),
        label_authority_axis(tally),
    ]
    return build_model_target_coverage(
        report_version=report_version,
        stage_id=stage_id,
        bindings=resolve_bindings(
            release_dir=release_dir,
            component_lock_path=component_lock_path,
            stage_policy_path=stage_policy_path,
            exact_text_cache_identity_hash=exact_text_cache_identity_hash,
        ),
        scope={
            "splits": list(tally.splits),
            "eligibility": tally.eligibility,
            "bundles_counted": tally.bundles,
            "fragments_counted": tally.fragments,
            "lineage_roots_counted": tally.lineage_roots,
        },
        axes=axes,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--component-lock", required=True)
    parser.add_argument("--stage-policy", required=True)
    parser.add_argument("--split", action="append", required=True)
    parser.add_argument("--eligibility", default=TRAIN_READY)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--report-version", required=True)
    parser.add_argument("--exact-text-cache-identity-hash", default=None)
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "checkpoint/resume directory for the parallel counting pass over the "
            "whole release (DEC-023); omit to run the small, single-process path"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="worker processes for --run-dir; default is os.cpu_count() minus two",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="write the checkpoint after this many completed bundles (--run-dir only)",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    if out_path.exists():
        print(
            f"refusing to overwrite existing report {out_path} -- the artefact is "
            "written as a new file with the new binding and is never overwritten "
            "in place (design §7.3); write to a different --out path",
            file=sys.stderr,
        )
        return 1

    report = build_release_report(
        release_dir=args.release,
        store_dir=args.store,
        component_lock_path=args.component_lock,
        stage_policy_path=args.stage_policy,
        splits=args.split,
        eligibility=args.eligibility,
        stage_id=args.stage_id,
        report_version=args.report_version,
        exact_text_cache_identity_hash=args.exact_text_cache_identity_hash,
        run_dir=args.run_dir,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
    )
    out_path.write_text(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
