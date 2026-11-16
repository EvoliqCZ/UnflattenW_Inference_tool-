"""RFC003-ERRATA... `spec-page-standalone-targets` design §8 / strategy §2,
T11 -- proves that pages a v12 feature cache already holds still name the
SAME page image and the SAME split in a new release before the exact-text
builder is allowed to read that cache as though its features belonged to
that new release.

Re-extracting the backbone over the new release would be the honest
alternative, but the whole point of reusing a feature cache is that the
pixels did not change (`--align-only` refuses a bundle rewrite,
`m2b.py:577-581`). This tool makes that claim checkable per page instead of
assumed: it walks the `to` release's page index and, for every key a named
v12 feature cache indexes, checks that the `from` release names the same
page image sha256 and the same split. Any mismatch, or a key that left the
`to` release outright, refuses -- no proof file is written on refusal, so a
half-checked proof can never be mistaken for a passing one.

`build_exact_text_cache.verify_feature_cache_identity` is the acceptor: a
source feature cache whose `release_root_hash` names the `from` release
here is admitted in place of the build's own release iff this proof's `to`
is that build's release, its `split` is the build's split, its `proof_hash`
recomputes, and the source's own `manifest.json` sha256 matches one entry
of `feature_cache_dirs`, at which that entry's own `keys_checked` equals
the source's own key count. One proof built over several feature cache
directories therefore admits every one of them, each checked against its
own entry -- never against the proof-wide total, which is a UNION and
would wrongly refuse every source once more than one dir is named.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from unflatten.dataset.identity import domain_hash
from unflatten.dataset.samples import _pinned_rows, _release_manifest, _verify_ref
from unflatten.model.feature_cache import validate_manifest
from unflatten.model.s2_plan import load_split_records

RELEASE_LINEAGE_PROOF_SCHEMA_VERSION = "1.0.0"

#: `proof_hash = H(PROOF_DOMAIN, proof-without-its-own-hash)`.
PROOF_DOMAIN = "unflatten.release-lineage-proof/v1"


class ReleaseLineageProofRefused(ValueError):
    """The `from` and `to` releases disagree on at least one checked page's
    image sha256 or split, or a key the named feature cache(s) hold left
    the `to` release outright. Raised before anything is written -- a
    proof file only ever exists when every key checked clean."""


def _release_page_index(
    release_dir: Path | str, store_dir: Path | str
) -> dict[tuple[str, int], dict[str, str | None]]:
    """`(document_bundle_id, page_index) -> {"sha256", "split"}` over every
    page a release's artifact index pins.

    Independent of task eligibility on purpose: a lineage proof must be
    able to name a page that left `train_ready` in the new release just as
    readily as one that stayed in it -- filtering by eligibility first
    would turn "moved to another split/eligibility" into a silent "missing
    key", which is a different, coarser fact.
    """
    release_dir, store_dir = Path(release_dir), Path(store_dir)
    manifest = _release_manifest(release_dir)
    split_by_bundle = {
        record["authority_id"]: record.get("split_name")
        for record in load_split_records(release_dir)
    }
    index: dict[tuple[str, int], dict[str, str | None]] = {}
    for row in _pinned_rows(release_dir, manifest, "artifact_index"):
        bundle_id = row["document_bundle_id"]
        split = split_by_bundle.get(bundle_id)
        pages = (row.get("artifacts") or {}).get("pages") or []
        for page_index, image_ref in enumerate(pages):
            ref = _verify_ref(store_dir, image_ref, f"{bundle_id}/pages/{page_index}")
            index[(bundle_id, page_index)] = {"sha256": ref["sha256"], "split": split}
    return index


def feature_cache_manifest_sha256(feature_cache_dir: Path | str) -> str:
    """sha256 of a feature cache's `manifest.json` raw bytes.

    The identity `feature_cache_dirs[i].manifest_sha256` pins, and what a
    builder recomputes (`build_exact_text_cache.open_feature_sources`) to
    confirm a source really is the directory a proof entry names -- not
    merely a cache with the same key COUNT (T11 follow-up: a proof used to
    admit any same-sized cache, swapped or not).
    """
    return hashlib.sha256((Path(feature_cache_dir) / "manifest.json").read_bytes()).hexdigest()


def _feature_cache_keys(feature_cache_dir: Path | str) -> tuple[list[tuple[str, int]], str]:
    """Every key a feature cache manifest indexes, plus the manifest file's
    own sha256 -- the pin recorded as `feature_cache_dirs[i].manifest_sha256`.

    Reads only `manifest.json` (never a shard), the same discipline
    `open_feature_sources` uses for its index (`build_exact_text_cache.py`).
    """
    path = Path(feature_cache_dir) / "manifest.json"
    raw = path.read_bytes()
    manifest = json.loads(raw)
    validate_manifest(manifest)
    keys = sorted(
        {
            (bundle, int(page))
            for shard in manifest["shard_index"]
            for bundle, page in shard["rows"]
        }
    )
    return keys, hashlib.sha256(raw).hexdigest()


def _proof_projection(proof: Mapping[str, Any]) -> dict[str, Any]:
    """The proof with `proof_hash` removed -- what the hash is computed
    over, so the field can never cover itself."""
    return {key: value for key, value in proof.items() if key != "proof_hash"}


def compute_proof_hash(proof: Mapping[str, Any]) -> str:
    """`H(PROOF_DOMAIN, proof-without-its-own-hash)`."""
    return domain_hash(PROOF_DOMAIN, _proof_projection(proof))


def verify_proof_hash(proof: Mapping[str, Any]) -> bool:
    """Does `proof["proof_hash"]` recompute from the rest of `proof`?"""
    return compute_proof_hash(proof) == proof.get("proof_hash")


def build_release_lineage_proof(
    *,
    from_release_dir: Path | str,
    from_store_dir: Path | str,
    to_release_dir: Path | str,
    to_store_dir: Path | str,
    split: str,
    task_kind: str,
    feature_cache_dirs: Sequence[Path | str],
) -> dict[str, Any]:
    """Prove every key the named feature cache(s) index still names the
    same page image and split in `to_release_dir` as in `from_release_dir`.

    Raises :class:`ReleaseLineageProofRefused` on any mismatch or missing
    key; the caller (here, or `main`) must not write a proof file in that
    case. Returns the proof dict, `proof_hash` included, only when every
    key checked clean.
    """
    if not feature_cache_dirs:
        raise ValueError(
            "feature_cache_dirs must be non-empty -- a proof over zero keys "
            "proves nothing"
        )
    from_manifest = _release_manifest(Path(from_release_dir))
    to_manifest = _release_manifest(Path(to_release_dir))
    from_index = _release_page_index(from_release_dir, from_store_dir)
    to_index = _release_page_index(to_release_dir, to_store_dir)

    all_keys: set[tuple[str, int]] = set()
    dirs_record: list[dict[str, Any]] = []
    for feature_cache_dir in feature_cache_dirs:
        keys, manifest_sha256 = _feature_cache_keys(feature_cache_dir)
        all_keys.update(keys)
        dirs_record.append(
            {
                "dir": str(feature_cache_dir),
                "manifest_sha256": manifest_sha256,
                # This DIR's own key count -- what a builder compares its
                # own index length against (never the proof-wide total
                # below, which is a UNION across every named dir and would
                # wrongly refuse every source in a multi-dir proof).
                "keys_checked": len(keys),
            }
        )

    keys_missing_in_to = 0
    mismatched_image_sha256 = 0
    mismatched_split = 0
    for key in sorted(all_keys):
        from_entry = from_index.get(key)
        if from_entry is None:
            raise ReleaseLineageProofRefused(
                f"{key}: not present in {from_release_dir} (the 'from' release) "
                "at all -- the named feature cache(s) do not resolve against "
                "this release"
            )
        to_entry = to_index.get(key)
        if to_entry is None:
            keys_missing_in_to += 1
            continue
        if to_entry["sha256"] != from_entry["sha256"]:
            mismatched_image_sha256 += 1
        if to_entry["split"] != from_entry["split"]:
            mismatched_split += 1

    proof: dict[str, Any] = {
        "release_lineage_proof_schema_version": RELEASE_LINEAGE_PROOF_SCHEMA_VERSION,
        "from_release_root_hash": from_manifest.get("release_root_hash"),
        "to_release_root_hash": to_manifest.get("release_root_hash"),
        "split": split,
        "task_kind": task_kind,
        "feature_cache_dirs": dirs_record,
        "keys_checked": len(all_keys),
        "keys_missing_in_to": keys_missing_in_to,
        "mismatched_image_sha256": mismatched_image_sha256,
        "mismatched_split": mismatched_split,
    }
    if keys_missing_in_to or mismatched_image_sha256 or mismatched_split:
        raise ReleaseLineageProofRefused(
            "release lineage proof refused: "
            f"keys_missing_in_to={keys_missing_in_to}, "
            f"mismatched_image_sha256={mismatched_image_sha256}, "
            f"mismatched_split={mismatched_split} "
            f"(of {len(all_keys)} keys checked) -- no proof file written"
        )
    proof["proof_hash"] = compute_proof_hash(proof)
    return proof


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.release_lineage_proof",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("from_release_dir", type=Path)
    parser.add_argument("from_store_dir", type=Path)
    parser.add_argument("to_release_dir", type=Path)
    parser.add_argument("to_store_dir", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument(
        "--feature-cache-dir",
        dest="feature_cache_dirs",
        type=Path,
        action="append",
        required=True,
        help="a feature cache directory whose keys this proof covers. "
        "Repeatable -- every key across every named directory is checked.",
    )
    parser.add_argument("--split", required=True)
    parser.add_argument("--task-kind", default="structure_v0")
    args = parser.parse_args(argv)

    proof = build_release_lineage_proof(
        from_release_dir=args.from_release_dir,
        from_store_dir=args.from_store_dir,
        to_release_dir=args.to_release_dir,
        to_store_dir=args.to_store_dir,
        split=args.split,
        task_kind=args.task_kind,
        feature_cache_dirs=args.feature_cache_dirs,
    )
    rendered = json.dumps(proof, indent=1, sort_keys=True)
    args.out_path.write_text(rendered)
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
