"""Model-free preparation CLI for S4.5 (DEC-047 addendum A v3, item 7; read-only).

``python -m unflatten.model.rl.data_prep <command> ... --out DIR``

Reads a real RFC003 release, the store index, failure-taxonomy reports, the
S4 ``stage_data_matrix.json``, a cache ``manifest.json`` (its ``identity``
object, as JSON) and a checkpoint sidecar; writes JSON artefacts into ``--out``
only and never overwrites one. There is no ``torch.load``, no LibreOffice, no
GPU and no ratification record here, and the module is import-light: ``torch``,
``candidate_loader``, ``renderer.probe`` stay out of ``sys.modules`` at import.

Admission uses the cache identity from JSON: ``exact_text_cache`` imports
``torch``, so no reader is constructed. ``freeze-ancestry-base`` takes
``--expected-revision`` and ``--model-id`` explicitly so ``candidate_loader`` is
never reached. ``build-dataset-manifest`` resolves the bundle source through
``adapters.release.load_bundle_source``, a torch-free re-implementation of the
store-index rule (parity-tested against ``s4_plan``); no command imports ``torch``.

Gold screen: ``--gold-screen-report`` is JSON written elsewhere
(:func:`data_prep_support.gold_screen_document`); without it the manifest is refused unless
``--allow-unscreened``, and the preparation record then says ``not_run``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from unflatten.model.checkpoint_ancestry import CheckpointAncestryInvalid
from unflatten.model.rl import data_prep_support as sup
from unflatten.model.rl.adapters import exclusions as ex
from unflatten.model.rl.adapters.admission import cache_identity_hash, rights_privacy_refs
from unflatten.model.rl.adapters.release import read_split_records_verified
from unflatten.model.rl.adapters.taxonomy_assembly import bundle_lineage_from_split_records
from unflatten.model.rl.policies import FORBIDDEN_SOURCES, build_dataset_manifest
from unflatten.model.rl.selection import (
    FrozenSelection,
    build_selection,
    selection_document,
    selection_from_document,
    selection_hash,
)

SOURCES: Final = ("docxcorp", "synthetic")
REGISTRY_FILE: Final = "exclusion_registry.json"
SELECTION_FILE: Final = "selection.json"
MANIFEST_FILE: Final = "dataset_manifest.json"
PREP_FILE: Final = "dataset_manifest_prep.json"
PLAN_FILE: Final = "rl_data_plan.json"
ADMISSION_FILE: Final = "admission_check.json"
_REFUSALS: Final = (ValueError, OSError, KeyError, CheckpointAncestryInvalid)


# --- shared readers ---


def _verified_registry(args: argparse.Namespace, release: Path) -> dict[str, Any]:
    from unflatten.dataset.samples import release_root_hash

    doc = sup.read_json(args.registry, "exclusion registry")
    ex.verify_exclusion_registry(
        doc, release_root_hash=release_root_hash(release), release_dir=release,
        artefact_root=args.artefact_root, decisions_dir=args.decisions_dir,
    )  # fmt: skip
    return doc


def _selection(path: str) -> FrozenSelection:
    return selection_from_document(sup.read_json(path, "selection"))


# --- commands ---------------------------------------------------------------------------


def _cmd_admission_check(args: argparse.Namespace) -> dict[str, Any]:
    from unflatten.dataset.samples import release_root_hash

    out = sup.out_dir(args.out)
    sup.require_free(out, [ADMISSION_FILE])
    records = read_split_records_verified(args.release)
    identity = sup.cache_identity(args.cache_dir)
    lineages = (
        sorted(_selection(args.selection).lineage_ids())
        if args.selection
        else sorted(sup.train_lineages(records))
    )
    doc = {
        "admission": "strict",
        "release_root_hash": release_root_hash(Path(args.release)),
        "cache_identity_hash": cache_identity_hash(identity),
        "lineage_count": len(lineages),
        "rights_privacy_refs": rights_privacy_refs(args.release, identity, lineages),
    }
    return {
        "command": "admission-check",
        "written": sup.write_docs(out, {ADMISSION_FILE: doc}),
        **doc,
    }


def _entry_specs(args: argparse.Namespace) -> dict[str, ex.EntrySpec]:
    specs: dict[str, ex.EntrySpec] = {}
    for raw in args.artefact:
        label, _, rest = raw.partition("=")
        kind, _, ref = rest.partition(":")
        specs[label] = ex.ArtefactSpec(kind, ref)
    for raw in args.absent:
        label, _, ref = raw.partition("=")
        specs[label] = ex.DeclaredAbsentSpec(ref or None)
    if len(specs) != len(args.artefact) + len(args.absent):
        raise sup.DataPrepRefused("a label is given more than once (--artefact / --absent)")
    return specs


def _cmd_build_exclusions(args: argparse.Namespace) -> dict[str, Any]:
    out = sup.out_dir(args.out)
    sup.require_free(out, [REGISTRY_FILE])
    doc = ex.build_exclusion_registry(
        args.release, _entry_specs(args), cross_release_dirs=args.cross_release,
        artefact_root=args.artefact_root, decisions_dir=args.decisions_dir,
    )  # fmt: skip
    return {
        "command": "build-exclusions",
        "written": sup.write_docs(out, {REGISTRY_FILE: doc}),
        "release_root_hash": doc["release_root_hash"],
        "registry_hash": doc["registry_hash"],
        "label_statuses": {e["label"]: e["status"] for e in doc["entries"]},
        "exclude_lineage_count": len(ex.exclude_lineage_ids(doc)),
    }


def _read_drop_lineages(path: str) -> list[str]:
    ids = sup.read_json(path, "drop-lineages file")
    if not isinstance(ids, list) or not all(isinstance(i, str) and i for i in ids):
        raise sup.DataPrepRefused(
            f"--drop-lineages {path}: must hold a JSON list of non-empty lineage root id strings"
        )
    return ids


def _cmd_build_selection(args: argparse.Namespace) -> dict[str, Any]:
    out = sup.out_dir(args.out)
    sup.require_free(out, [SELECTION_FILE])
    release = Path(args.release)
    registry = _verified_registry(args, release)
    records = read_split_records_verified(release)
    selection = build_selection(
        [sup.read_json(p, "taxonomy report") for p in args.report],
        policy_parent_checkpoint_id=args.parent_checkpoint_id,
        allowed_lineage_ids=sup.train_lineages(records) - ex.exclude_lineage_ids(registry),
        bundle_lineage=bundle_lineage_from_split_records(records),
        budget=args.budget,
        bundles_from=args.bundles_from,
        classes=args.classes,
        drop_lineages=_read_drop_lineages(args.drop_lineages) if args.drop_lineages else None,
    )
    if not selection.ready:
        raise sup.DataPrepRefused(
            f"selection is insufficient_evidence: no lineage survived (dropped_outside_allowed="
            f"{selection.dropped_outside_allowed}, unmapped_bundles={selection.unmapped_bundles}"
            + (
                f", dropped_lineages={len(selection.dropped_lineages)}"
                if args.drop_lineages
                else ""
            )
            + ")"
        )
    return {
        "command": "build-selection",
        "written": sup.write_docs(out, {SELECTION_FILE: selection_document(selection)}),
        "selection_hash": selection_hash(selection),
        "status": selection.status,
        "budget": selection.budget,
        "selected_lineages": len(selection.entries),
        "truncated_by_budget": selection.truncated_by_budget,
        "dropped_outside_allowed": selection.dropped_outside_allowed,
        "unmapped_bundles": selection.unmapped_bundles,
        "report_hashes": list(selection.all_report_hashes()),
        "checkpoint_id": selection.checkpoint_id,
        **({"dropped_lineages": len(selection.dropped_lineages)} if args.drop_lineages else {}),
        **(
            {"selected_pages": len(selection.selected_pages()), "classes": list(selection.classes)}
            if selection.classes
            else {}
        ),
    }


def _data_plan(args: argparse.Namespace, manifest: Mapping[str, Any], reg_hash: str) -> Any:
    from unflatten.model.rl.data import build_rl_data_plan
    from unflatten.model.stage_data_matrix import load_stage_data_matrix, stage_data_matrix_hash

    plan = build_rl_data_plan(
        stage_data_matrix_hash=stage_data_matrix_hash(
            load_stage_data_matrix(Path(args.stage_data_matrix))
        ),
        dataset_manifest=manifest,
        exclusion_registry_hash=reg_hash,
    )
    return {
        "stage_data_matrix_hash": plan.stage_data_matrix_hash,
        "selection_hash": plan.selection_hash,
        "dataset_manifest_hash": plan.dataset_manifest_hash,
        "exclusion_registry_hash": plan.exclusion_registry_hash,
        "plan_hash": plan.plan_hash,
    }


def _cmd_build_dataset_manifest(args: argparse.Namespace) -> dict[str, Any]:
    from unflatten.model.rl.adapters.release import load_bundle_source
    from unflatten.model.rl.data import load_train_pages

    if args.gold_screen_report is None and not args.allow_unscreened:
        raise sup.DataPrepRefused(
            "no --gold-screen-report: the gold chain was not screened; pass "
            "--allow-unscreened to record gold_screen 'not_run' explicitly"
        )
    out = sup.out_dir(args.out)
    sup.require_free(
        out, [MANIFEST_FILE, PREP_FILE, *([PLAN_FILE] if args.stage_data_matrix else [])]
    )
    release = Path(args.release)
    registry = _verified_registry(args, release)
    selection = _selection(args.selection)
    records = read_split_records_verified(release)
    lineage_ids = sorted(selection.lineage_ids())
    exclude = ex.exclude_lineage_ids(registry)
    manifest = build_dataset_manifest(
        split="train",
        sources=sorted(set(args.source)),
        rights_privacy_refs=rights_privacy_refs(
            release, sup.cache_identity(args.cache_dir), lineage_ids
        ),
        lineage_ids=lineage_ids,
        selection_hash=selection_hash(selection),
        exclude_lineage_ids=exclude,
    )
    pages = load_train_pages(
        release, manifest, bundle_source=load_bundle_source(args.store),
        page_keys=[(r["authority_id"], 0) for r in records],
        registry_exclude_lineage_ids=exclude,
    )  # fmt: skip
    prep = {
        "dataset_manifest_hash": manifest["dataset_manifest_hash"],
        "selection_hash": manifest["selection_hash"],
        "exclusion_registry_hash": registry["registry_hash"],
        "gold_screen": sup.gold_screen(
            args.gold_screen_report, bundle_lineage_from_split_records(records), set(lineage_ids)
        ),
        "load_summary": pages.summary(),
    }
    docs: dict[str, Any] = {MANIFEST_FILE: manifest, PREP_FILE: prep}
    if args.stage_data_matrix:
        docs[PLAN_FILE] = _data_plan(args, manifest, registry["registry_hash"])
    return {
        "command": "build-dataset-manifest",
        "written": sup.write_docs(out, docs),
        "label_statuses": {e["label"]: e["status"] for e in registry["entries"]},
        "forbidden_sources": list(FORBIDDEN_SOURCES),
        "selection_budget": selection.budget,
        **prep,
    }


def _also_exposed(raw: Sequence[str]) -> list[tuple[Path, str, str]]:
    out: list[tuple[Path, str, str]] = []
    for item in raw:
        parts = item.split(":")
        if len(parts) != 3 or not all(parts):
            raise sup.DataPrepRefused(
                f"--also-exposed {item!r} is not RELEASE_DIR:RELEASE_NAME:STORE"
            )
        out.append((Path(parts[0]), parts[1], parts[2]))
    return out


def _cmd_freeze_ancestry_base(args: argparse.Namespace) -> dict[str, Any]:
    from unflatten.model.freeze_checkpoint_ancestry import CHECKPOINT_ANCESTRY_BASE_FILENAME as BASE
    from unflatten.model.rl.adapters.ancestry import freeze_ancestry_base

    out = sup.out_dir(args.out)
    sup.require_free(out, [BASE])
    out.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ancestry-", dir=out))
    try:
        doc = freeze_ancestry_base(
            Path(args.parent_checkpoint) if args.parent_checkpoint else None,
            Path(args.parent_run_manifest) if args.parent_run_manifest else None,
            staging,
            pinned_hashes=Path(args.pinned_hashes), component_lock=Path(args.component_lock),
            s0_report=Path(args.s0_report), release_manifest=Path(args.release_manifest),
            release_name=args.release_name, store_name=args.store_name,
            base_model_decision=Path(args.base_model_decision),
            dataset_decision=Path(args.dataset_decision),
            distribution_decision=(
                Path(args.distribution_decision) if args.distribution_decision else None
            ),
            also_exposed=_also_exposed(args.also_exposed),
            expected_revision=args.expected_revision, model_id=args.model_id,
        )  # fmt: skip
        sup.publish(out, BASE, (staging / BASE).read_bytes())
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "command": "freeze-ancestry-base",
        "written": [BASE],
        "ancestry_hash": doc["ancestry_hash"],
        "initialization_kind": doc.get("initialization_kind"),
        "dataset_exposures": len(doc.get("dataset_exposures", ())),
    }


# --- argparse (closed surface) ---------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m unflatten.model.rl.data_prep")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, run: Callable[[argparse.Namespace], dict[str, Any]]):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="output directory (never overwritten)")
        p.set_defaults(run=run)
        return p

    def release_inputs(p: argparse.ArgumentParser) -> None:
        p.add_argument("--release", required=True)
        p.add_argument("--artefact-root", default=None)
        p.add_argument("--decisions-dir", default=ex.DEFAULT_DECISIONS_DIR)

    p = add("admission-check", _cmd_admission_check)
    p.add_argument("--release", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--selection", default=None)

    p = add("build-exclusions", _cmd_build_exclusions)
    release_inputs(p)
    p.add_argument("--artefact", action="append", default=[], metavar="LABEL=KIND[:REF]")
    p.add_argument("--absent", action="append", default=[], metavar="LABEL=DECISION_REF")
    p.add_argument("--cross-release", action="append", default=[])

    p = add("build-selection", _cmd_build_selection)
    release_inputs(p)
    p.add_argument("--registry", required=True)
    p.add_argument("--report", action="append", required=True)
    p.add_argument("--parent-checkpoint-id", required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument(
        "--bundles-from", choices=("examples", "affected", "affected-pages"), default="examples",
        help="affected = every bundle of a class's affected_slice, not its <= 5 examples; "
        "affected-pages = only the pages hit by --classes",
    )  # fmt: skip
    p.add_argument(
        "--classes", nargs="+", default=None, metavar="C",
        help="failure classes to select on (required with, and only valid for, affected-pages)",
    )  # fmt: skip

    p.add_argument(
        "--drop-lineages", default=None, metavar="JSON",
        help="JSON list of lineage root ids removed after the budget cut (no refill); every "
        "id must be in the selection",
    )  # fmt: skip

    p = add("build-dataset-manifest", _cmd_build_dataset_manifest)
    release_inputs(p)
    p.add_argument("--store", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--registry", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--source", action="append", required=True, choices=SOURCES)
    p.add_argument("--gold-screen-report", default=None)
    p.add_argument("--allow-unscreened", action="store_true")
    p.add_argument("--stage-data-matrix", default=None)

    p = add("freeze-ancestry-base", _cmd_freeze_ancestry_base)
    for flag in (
        "pinned-hashes", "component-lock", "s0-report", "release-manifest", "release-name",
        "store-name", "base-model-decision", "dataset-decision", "expected-revision", "model-id",
    ):  # fmt: skip
        p.add_argument(f"--{flag}", required=True)
    for flag in ("distribution-decision", "parent-checkpoint", "parent-run-manifest"):
        p.add_argument(f"--{flag}", default=None)
    p.add_argument("--also-exposed", action="append", default=[], metavar="DIR:RELEASE:STORE")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    try:
        summary = args.run(args)
    except _REFUSALS as exc:
        text = " ".join(str(exc).split()) or repr(exc)
        print(f"{args.command}: refused: {type(exc).__name__}: {text}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
