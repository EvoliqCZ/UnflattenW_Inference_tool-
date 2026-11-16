"""S4 assembly evaluation CLI (Track A): ``eval | summary | acceptance``.

- ``eval``    -- design §8 per-document pipeline over a pinned bundle
  manifest of one release split (JSONL ledger, resume, progress, B0 and
  B00 scored in the same pass). ``train`` needs no flag. A
  ``val_iid``/``val_template_ood`` run is refused unless ``--acceptance``
  is given, so the acceptance set is never spent unrecorded; on success it
  appends one entry to the M2 ledger (design §8). ``test`` is refused
  outright, ``--acceptance`` or not.
- ``summary`` -- the design §10 tables from an existing ledger, plus the
  ledger and summary hashes (G-A2: two runs must agree on both).
- ``acceptance`` -- reads the frozen G-A1 policy (DEC-049 point 7, item 9.1,
  `ga1_acceptance_policy.py`) and evaluates the M2 ledger entry
  (`acceptance_ledger.py`) for the current code's `assembly_policy_hash()`
  and the given ``--split``, printing PASS / FAIL / `insufficient_evidence`
  per gated class plus the hard zeros. ``eval --acceptance`` on a
  ``val_iid``/``val_template_ood`` split appends the measurement this reads
  (design §8); that ledger records measurements only, this prints the verdict.

``--workers`` determinism: bundles are processed and appended in sorted
order, so the ledger hash is invariant to the worker count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from unflatten.assembly.policy import assembly_policy_hash
from unflatten.eval.assembly import acceptance_ledger as al
from unflatten.eval.assembly import ga1_acceptance_policy as gap
from unflatten.eval.assembly import harness as h
from unflatten.model.capability_profile import STANDALONE_V1_PROFILE_ID

DEFAULT_PROFILE_ID = STANDALONE_V1_PROFILE_ID
TASK_KIND = "structure_v0"
_ACCEPTANCE_SPLITS = ("val_iid", "val_template_ood", "test")

# Set in the parent before the fork pool exists, inherited copy-on-write.
_CTX: dict[str, Any] = {}


def _refuse(message: str) -> None:
    raise SystemExit(f"refused: {message}")


def _resolve_policies(spec: str) -> dict[str, Any]:
    names = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = sorted(set(names) - set(h.DEFAULT_POLICIES))
    if unknown or not names:
        _refuse(f"unknown policies {unknown or [spec]!r}; known: {sorted(h.DEFAULT_POLICIES)}")
    return {name: h.DEFAULT_POLICIES[name] for name in names}


def _load_population(release: Path, store: Path, split: str) -> dict[str, list[Any]]:
    from unflatten.dataset.samples import iter_page_samples

    by_doc: dict[str, list[Any]] = {}
    for sample in iter_page_samples(release, store, split, TASK_KIND):
        by_doc.setdefault(sample.document_bundle_id, []).append(sample)
    for samples in by_doc.values():
        samples.sort(key=lambda s: s.page_index)
    return by_doc


def _ast_refs(store: Path) -> dict[str, dict[str, Any]]:
    index = store / "index.jsonl"
    if not index.exists():
        raise h.HarnessError(f"store index not found: {index}")
    refs: dict[str, dict[str, Any]] = {}
    with open(index) as handle:
        for line in handle:
            if line.strip():
                envelope = json.loads(line)["envelope"]
                refs[envelope["document_bundle_id"]] = envelope["ast"]
    return refs


def _evaluate_one(bundle_id: str) -> h.DocumentResult:
    import torch

    from unflatten.dataset.gates.store import read_object

    torch.set_num_threads(1)
    ctx = _CTX
    ast = json.loads(read_object(ctx["store"], ctx["ast_refs"][bundle_id]))
    return h.evaluate_document(
        store_dir=ctx["store"],
        samples=ctx["population"][bundle_id],
        ast=ast,
        bound=ctx["bound"],
        tokenizer=ctx["tokenizer"],
        policies=ctx["policies"],
        profile=ctx["profile"],
    )


def run_eval(args: argparse.Namespace) -> int:
    acceptance_flow = False
    policy_hash = None
    release_dir = Path(args.release)
    ledger_file = None
    release_root = None
    if args.split in _ACCEPTANCE_SPLITS:
        if not args.acceptance:
            _refuse(
                f"split {args.split!r} is an acceptance split; pass --acceptance to run "
                "it against the M2 ledger (design §8), or use 'train'"
            )
        if args.split == "test":
            _refuse("split 'test' is refused outright in V0 (design §8)")
        acceptance_flow = True
        policy_hash = assembly_policy_hash()
        release_root = _release_root_hash_or_refuse(release_dir)
        ledger_file = al.ledger_path(release_dir)
        try:
            if al.has_entry(ledger_file, policy_hash, args.split):
                _refuse(
                    f"the M2 ledger already holds an entry for policy_hash={policy_hash!r} "
                    f"split={args.split!r} ({ledger_file})"
                )
        except al.AcceptanceLedgerError as exc:
            _refuse(str(exc))

    policies = _resolve_policies(args.policies)
    out_path = Path(args.out)
    if args.fresh:
        h.backup_ledger(out_path)
        out_path.unlink(missing_ok=True)
    bundle_ids = sorted(
        h.pending_bundle_ids(
            Path(args.manifest), out_path if out_path.exists() else None, limit=args.limit
        )
    )
    if not bundle_ids:
        print("No pending bundle ids to evaluate.", file=sys.stderr)
        if not acceptance_flow:
            return 0
        return _finish_acceptance_and_summary(out_path, args.json, ledger_file, policy_hash,
                                               release_root, args.split)

    from unflatten.model.capability_profile import resolve_capability_profile_by_id

    profile = resolve_capability_profile_by_id(args.profile_id)
    started = time.monotonic()
    population = _load_population(Path(args.release), Path(args.store), args.split)
    missing = [bid for bid in bundle_ids if bid not in population]
    if missing:
        _refuse(
            f"{len(missing)} manifest bundle(s) are not {args.split} documents of this "
            f"release (first: {missing[0]})"
        )
    bound, tokenizer = h.build_bound(profile)
    _CTX.update(
        store=Path(args.store),
        population=population,
        ast_refs=_ast_refs(Path(args.store)),
        bound=bound,
        tokenizer=tokenizer,
        policies=policies,
        profile=profile.ast_profile,
    )
    print(
        f"[assembly-eval] {len(bundle_ids)} documents, "
        f"{sum(len(population[b]) for b in bundle_ids)} pages, policies "
        f"{list(policies)}, workers {args.workers}, setup {time.monotonic() - started:.1f}s",
        flush=True,
    )
    results: list[h.DocumentResult] = []
    t0 = time.monotonic()
    if args.workers == 1:
        for n, bid in enumerate(bundle_ids, 1):
            result = _evaluate_one(bid)
            results.append(result)
            h.append_result(out_path, result)
            _progress(n, len(bundle_ids), t0, args.progress_every)
    else:
        context = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            futures = {bid: pool.submit(_evaluate_one, bid) for bid in bundle_ids}
            for n, bid in enumerate(bundle_ids, 1):
                result = futures[bid].result()
                results.append(result)
                h.append_result(out_path, result)
                _progress(n, len(bundle_ids), t0, args.progress_every)
    elapsed = time.monotonic() - t0
    pages = sum(r.n_pages for r in results)
    print(
        f"[assembly-eval] done: {len(results)} documents, {pages} pages in {elapsed:.1f}s "
        f"({pages / elapsed if elapsed else 0:.2f} pages/s)",
        flush=True,
    )
    if not acceptance_flow:
        return _print_summary(out_path, args.json)
    return _finish_acceptance_and_summary(
        out_path, args.json, ledger_file, policy_hash, release_root, args.split
    )


def _progress(n: int, total: int, t0: float, every: int) -> None:
    if every and (n % every == 0 or n == total):
        elapsed = time.monotonic() - t0
        print(f"[assembly-eval] {n}/{total} documents, {elapsed:.0f}s", flush=True)


def _release_root_hash_or_refuse(release_dir: Path) -> str:
    from unflatten.dataset.samples import SampleSelectionError, release_root_hash

    try:
        return release_root_hash(release_dir)
    except SampleSelectionError as exc:
        _refuse(str(exc))
    return ""  # pragma: no cover -- _refuse always raises


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise h.HarnessError(f"git rev-parse HEAD failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _finish_acceptance_and_summary(
    out_path: Path,
    json_path: str | None,
    ledger_file: Path,
    policy_hash: str,
    release_root: str,
    split: str,
) -> int:
    """Append one M2 ledger entry (design §8) for this run, then print the
    usual eval summary. Refuses rather than appending if the run's own
    per-document ledger has vanished, or the M2 ledger already holds this
    (policy_hash, split)."""
    rows = h.read_ledger(out_path)
    summary = h.compute_summary(rows.values())
    digest = h.summary_hash(summary)
    report_sha256 = "sha256:" + hashlib.sha256(out_path.read_bytes()).hexdigest()
    entry = al.AcceptanceEntry(
        policy_hash=policy_hash,
        release_root_hash=release_root,
        split=split,
        summary=summary,
        summary_hash=digest,
        report_sha256=report_sha256,
        code_commit=_git_head(),
        timestamp=datetime.now(UTC).isoformat(),
    )
    try:
        al.append_entry(ledger_file, entry)
    except al.AcceptanceLedgerError as exc:
        _refuse(str(exc))
    print(f"[assembly-eval] M2 acceptance entry appended: {ledger_file}", flush=True)
    return _print_summary(out_path, json_path)


def _print_summary(ledger_path: Path, json_path: str | None) -> int:
    rows = h.read_ledger(ledger_path)
    summary = h.compute_summary(rows.values())
    digest = h.summary_hash(summary)
    ledger_digest = h.ledger_hash(ledger_path)
    print(h.format_summary(summary))
    print(f"ledger_hash {ledger_digest}")
    print(f"summary_hash {digest}")
    if json_path:
        Path(json_path).write_text(
            json.dumps(
                {"summary": summary, "summary_hash": digest, "ledger_hash": ledger_digest,
                 "ledger": str(ledger_path)},
                indent=1,
                sort_keys=True,
            )
            + "\n"
        )
    return 0


def run_summary(args: argparse.Namespace) -> int:
    return _print_summary(Path(args.ledger), args.json)


def run_acceptance(args: argparse.Namespace) -> int:
    ledger_file = Path(args.ledger)
    try:
        entries = al.read_entries(ledger_file)
    except al.AcceptanceLedgerError as exc:
        _refuse(str(exc))
        return 2  # pragma: no cover -- _refuse always raises
    current_hash = assembly_policy_hash()
    matches = [e for e in entries if e["policy_hash"] == current_hash and e["split"] == args.split]
    if not matches:
        _refuse(
            f"no acceptance-ledger entry for policy_hash={current_hash!r} split={args.split!r} "
            f"in {ledger_file}; run `eval --acceptance` first (design §8)"
        )
        return 2  # pragma: no cover -- _refuse always raises
    if len(matches) > 1:
        _refuse(
            f"{ledger_file}: {len(matches)} entries for policy_hash={current_hash!r} "
            f"split={args.split!r}; the ledger forbids more than one, refused as corrupted"
        )
        return 2  # pragma: no cover -- _refuse always raises
    policy = gap.load_ga1_acceptance_policy(args.policy)
    verdict = gap.evaluate_acceptance(matches[0]["summary"], policy)
    print(json.dumps(verdict, indent=1, sort_keys=True))
    if args.json:
        Path(args.json).write_text(json.dumps(verdict, indent=1, sort_keys=True) + "\n")
    return 0 if verdict["verdict"] == "PASS" else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.eval.assembly",
        description="S4 assembly evaluation (Track A): eval | summary | acceptance",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="per-document evaluation over a pinned manifest")
    ev.add_argument("--release", required=True, help="RFC003 release directory")
    ev.add_argument("--store", required=True, help="M2a store root directory")
    ev.add_argument("--split", default="train", help="release split (only 'train' is accepted)")
    ev.add_argument("--manifest", required=True, help="JSON with `bundle_ids`")
    ev.add_argument("--out", required=True, help="output JSONL ledger (append, resume)")
    ev.add_argument("--json", default=None, help="also write the summary JSON here")
    ev.add_argument("--workers", type=int, default=1)
    ev.add_argument("--limit", type=int, default=None, help="max pending documents")
    ev.add_argument("--fresh", action="store_true", help="back up and clear the ledger first")
    ev.add_argument(
        "--acceptance", action="store_true",
        help="append one M2 ledger entry (design §8); required for val/test splits",
    )
    ev.add_argument("--policies", default="B0,B00", help="comma-separated policy names")
    ev.add_argument("--profile-id", default=DEFAULT_PROFILE_ID, help="capability profile id")
    ev.add_argument("--progress-every", type=int, default=20)

    su = sub.add_parser("summary", help="design §10 tables from an existing ledger")
    su.add_argument("--ledger", required=True)
    su.add_argument("--json", default=None, help="also write the summary JSON here")

    ac = sub.add_parser("acceptance", help="M2 acceptance verdict (design §8, DEC-049 9.1)")
    ac.add_argument(
        "--ledger", required=True, help="the M2 acceptance ledger (acceptance_ledger.py)",
    )
    ac.add_argument(
        "--split", required=True, choices=("val_iid", "val_template_ood"),
        help="which acceptance-ledger entry to evaluate",
    )
    ac.add_argument(
        "--policy", default=None,
        help="G-A1 acceptance policy JSON (default: the frozen DEC-049 9.1 policy)",
    )
    ac.add_argument("--json", default=None, help="also write the verdict JSON here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "eval":
        return run_eval(args)
    if args.command == "summary":
        return run_summary(args)
    return run_acceptance(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
