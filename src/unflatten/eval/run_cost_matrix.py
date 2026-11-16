"""CLI: measure the `/layout/spacing/after_mpt` confusion cost matrix.

Three subcommands, deliberately separate because they cost three orders
of magnitude apart:

    vocab     read ASTs, report the observed vocabulary (seconds)
    units     units-derived matrix from a recorded vocabulary (instant)
    rendered  V LibreOffice renders per document, then V^2 comparisons

Nothing here supplies a vocabulary, a DPI or a sample size: all three are
arguments, and all three are written into the output next to the numbers
they produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from unflatten.ast.canonicalizer import CanonicalizationError, canonicalize
from unflatten.dataset.gates.store import read_object
from unflatten.eval.units_cost import units_cost_matrix
from unflatten.eval.vocabulary import measure_vocabulary

Bundle = tuple[str, dict[str, Any], dict[str, bytes]]


def load_bundles(store_dir: Path, release_dir: Path, split: str, limit: int) -> list[Bundle]:
    """Load the first `limit` bundles of `split` as canonicalized ASTs.

    "First" is the split file's own order, which is sorted by bundle id —
    deterministic and not a sample the caller can nudge.
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    split_path = Path(release_dir) / "splits" / f"{split}.jsonl"
    if not split_path.is_file():
        raise FileNotFoundError(f"split {split!r} not found at {split_path}")
    wanted: list[str] = []
    with split_path.open(encoding="utf-8") as handle:
        for line in handle:
            wanted.append(json.loads(line)["authority_id"])
            if len(wanted) == limit:
                break
    envelopes = _index_envelopes(Path(store_dir), set(wanted))
    bundles: list[Bundle] = []
    for bundle_id in wanted:
        envelope = envelopes.get(bundle_id)
        if envelope is None:
            raise KeyError(f"bundle {bundle_id!r} is in the split but not in the store index")
        ast = json.loads(read_object(Path(store_dir), envelope["ast"]))
        manifest = json.loads(read_object(Path(store_dir), envelope["resources_manifest"]))
        # M2a manifest rows carry no mime_type; ArtifactRef needs one and
        # rejects the extra `path` key (m2b does the same rebuild)
        resources = {
            entry["path"]: read_object(
                Path(store_dir),
                {
                    "ref": entry["ref"],
                    "sha256": entry["sha256"],
                    "size_bytes": entry["size_bytes"],
                    "mime_type": entry.get("mime_type", "application/octet-stream"),
                },
            )
            for entry in manifest
        }
        try:
            ast = canonicalize(ast)
        except CanonicalizationError as exc:
            raise ValueError(
                f"bundle {bundle_id!r} does not canonicalize: "
                + ", ".join(issue.code for issue in exc.issues)
            ) from exc
        bundles.append((bundle_id, ast, resources))
    return bundles


def _index_envelopes(store_dir: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    envelopes: dict[str, dict[str, Any]] = {}
    with (store_dir / "index.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            envelope = json.loads(line)["envelope"]
            if envelope["document_bundle_id"] in wanted:
                envelopes[envelope["document_bundle_id"]] = envelope
    return envelopes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_cost_matrix", description=__doc__)
    subparsers = parser.add_subparsers(dest="arm", required=True)

    def add_store_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--store", type=Path, default=Path.home() / "unflatten-data/m2a-v2")
        sub.add_argument("--release", type=Path, default=Path.home() / "unflatten-data/release-v6")
        sub.add_argument("--split", default="train")
        sub.add_argument("--limit", type=int, required=True, help="documents to read")
        sub.add_argument("--out", type=Path, required=True)

    vocab = subparsers.add_parser("vocab", help="measure the observed vocabulary")
    add_store_args(vocab)

    units = subparsers.add_parser("units", help="units-derived matrix (no rendering)")
    units.add_argument("--vocabulary", type=Path, required=True, help="a `vocab` output file")
    units.add_argument("--dpi", type=int, required=True)
    units.add_argument("--out", type=Path, required=True)

    rendered = subparsers.add_parser("rendered", help="V renders per document, V^2 comparisons")
    add_store_args(rendered)
    rendered.add_argument("--vocabulary", type=Path, required=True)
    rendered.add_argument("--dpi", type=int, required=True)
    rendered.add_argument("--profile-slot", type=int, default=0)
    return parser


def _read_vocabulary(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("vocabulary_mpt")
    if not values:
        raise ValueError(f"{path} carries no `vocabulary_mpt`")
    return list(values)


def _run_vocab(args: argparse.Namespace) -> dict[str, Any]:
    bundles = load_bundles(args.store, args.release, args.split, args.limit)
    measurement = measure_vocabulary([(bundle_id, ast) for bundle_id, ast, _ in bundles])
    payload = measurement.to_dict()
    payload["sample"] = {"release": str(args.release), "split": args.split, "limit": args.limit}
    return payload


def _run_rendered(args: argparse.Namespace) -> dict[str, Any]:
    # imported here: the rendered arm needs Pillow/numpy, the other two do not
    from unflatten.eval.render_pipeline import (
        ClassRenderError,
        cost_matrix_from_renders,
        poppler_version,
        render_classes,
    )

    values = _read_vocabulary(args.vocabulary)
    bundles = load_bundles(args.store, args.release, args.split, args.limit)
    documents = []
    for bundle_id, ast, resources in bundles:
        try:
            renders = render_classes(
                ast,
                values,
                dpi=args.dpi,
                resources=resources,
                profile_slot=args.profile_slot,
            )
        except ClassRenderError as exc:
            documents.append({"document_bundle_id": bundle_id, "error": str(exc)})
            print(f"[skip] {bundle_id}: {exc}", file=sys.stderr)
            continue
        matrix = cost_matrix_from_renders(
            renders,
            provenance={
                "render_dpi": args.dpi,
                "poppler_version": poppler_version(),
                "document_bundle_id": bundle_id,
            },
        )
        documents.append(matrix)
        print(f"[done] {bundle_id}: {len(renders)} renders", file=sys.stderr)
    return {"arm": "rendered", "vocabulary_mpt": values, "documents": documents}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.arm == "vocab":
        payload = _run_vocab(args)
    elif args.arm == "units":
        payload = units_cost_matrix(_read_vocabulary(args.vocabulary), args.dpi).to_dict()
    else:
        payload = _run_rendered(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
