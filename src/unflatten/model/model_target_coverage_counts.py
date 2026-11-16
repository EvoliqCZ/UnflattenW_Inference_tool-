"""Counting side of the RFC005 §9.3 `ModelTargetCoverageReport`: what a
dataset release's *model targets* actually contain, raw and per lineage root.

The counted surface is the one a training sample really is. A sample is one
page (`model/targets.py`: "using the whole document would ask page 1 to
predict page 7"), so this counts `encode_fragment` over every page fragment
the release pins, not `encode_structural` over the document. That choice is
visible in the result: `END_DOCUMENT` never occurs, and `FRAGMENT_BEGIN` /
`RESUME_NODE` / `SUSPEND_NODE` do -- the opposite of what a document-level
count would report, for the same release.

Everything read here is pinned by the release and verified against its
pinned `sha256` before it is parsed. The release identity itself comes from
:func:`unflatten.dataset.admission.evidence.load_release`, which recomputes
`release_root_hash` rather than reading the key -- reading the key would
skip exactly the check that makes binding to it worth anything.

Membership and lineage come from `ReleaseEvidence.membership`
(`bundle_id -> (split, lineage_root_id)`); no second counter is written for
lineage. `lineage_count` is therefore a count of distinct lineage *roots*.
RFC003 §42 additionally asks for `independent_lineage_families`, and no
family field exists on a document bundle -- so nothing here is labelled as
families.

Only the axes a release can answer are counted (structural symbols, the
structural tokenizer route, and label authority). The axes that are zero or
undefined because the *model* has no such target yet are assembled in
:mod:`unflatten.model.model_target_coverage_report`, which is where their
missing upstream artefacts are named.

Ships no `nn.Module` and authorizes no gradient update.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unflatten.dataset.admission.evidence import ReleaseEvidence, load_release
from unflatten.model.encode_fragment import encode_fragment

#: Release eligibility that trains by default (CLAUDE.md: "Only
#: `TRAIN_READY` trains by default"). Passed explicitly, never assumed.
TRAIN_READY = "train_ready"

#: Schema of the resumable-count checkpoint (DEC-023). Bumping this is a
#: breaking change to any checkpoint already on disk.
_CHECKPOINT_SCHEMA_VERSION = "1.0.0"


class ModelTargetCountingError(ValueError):
    """A release cannot be counted: a pinned artefact is missing, does not
    match its pinned digest, or a fragment cannot be encoded.

    Counting stops rather than skipping the document: a silently skipped
    bundle lowers every count in the report by an amount nobody records.
    """


class _Tally:
    """`key -> (raw occurrences, distinct lineage roots)` for one axis."""

    def __init__(self) -> None:
        self._raw: dict[str, int] = {}
        self._lineages: dict[str, set[str]] = {}

    def add(self, key: str, count: int, lineage_root_id: str) -> None:
        self._raw[key] = self._raw.get(key, 0) + count
        self._lineages.setdefault(key, set()).add(lineage_root_id)

    def resolve(self) -> dict[str, tuple[int, int]]:
        return {
            key: (raw, len(self._lineages[key])) for key, raw in sorted(self._raw.items())
        }


#: `key -> {"raw": int, "lineages": [lineage_root_id, ...]}`, the JSON-safe
#: shape a `_Tally` is checkpointed in (a `set` is not JSON-serialisable).
_TallyState = dict[str, dict[str, Any]]


def _merge_tally_state(state: _TallyState, counts: Mapping[str, int], lineage_root_id: str) -> None:
    """Fold one bundle's `key -> count` contribution into a checkpointed
    tally state, in place."""
    for key, count in counts.items():
        entry = state.setdefault(key, {"raw": 0, "lineages": []})
        entry["raw"] += count
        if lineage_root_id not in entry["lineages"]:
            entry["lineages"].append(lineage_root_id)


def _resolve_tally_state(state: _TallyState) -> dict[str, tuple[int, int]]:
    return {key: (entry["raw"], len(entry["lineages"])) for key, entry in sorted(state.items())}


@dataclass(frozen=True)
class ReleaseTally:
    """Everything counted over one release, with the scope it was counted in.

    The scope travels with the numbers deliberately: "raw_count 12" is a
    different claim over one split of one release than over all of them.
    """

    splits: tuple[str, ...]
    eligibility: str
    bundles: int
    fragments: int
    lineage_roots: int
    #: token name -> (occurrences, distinct lineage roots)
    symbols: dict[str, tuple[int, int]]
    #: composite-vocabulary route -> (occurrences, distinct lineage roots)
    routes: dict[str, tuple[int, int]]
    #: `<label_origin>/<verification>` -> (bundles, distinct lineage roots)
    label_authority: dict[str, tuple[int, int]]


def _read_pinned(store_dir: Path, entry: Mapping[str, Any], what: str) -> dict[str, Any]:
    """Read one release-pinned object, refusing anything that does not match
    the digest the release pinned for it."""
    path = store_dir / entry["ref"]
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ModelTargetCountingError(f"cannot read {what} at {path}: {exc}") from exc
    digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
    if digest != entry.get("sha256"):
        raise ModelTargetCountingError(
            f"{what} at {path} hashes to {digest}, but the release pins "
            f"{entry.get('sha256')} -- the counting surface is not the artefact "
            "the release identity covers"
        )
    try:
        return json.loads(data)
    except json.JSONDecodeError as exc:
        raise ModelTargetCountingError(f"{what} at {path} is not valid JSON: {exc}") from exc


def _selected_bundles(
    evidence: ReleaseEvidence, splits: Iterable[str], eligibility: str
) -> list[tuple[str, str]]:
    """`(bundle_id, lineage_root_id)` for the counted cohort.

    Membership and lineage are the release's (`ReleaseEvidence.membership`);
    the eligibility filter is the release's own gate row, not a re-decision.
    """
    wanted = set(splits)
    gate_rows = evidence.gate_rows_by_bundle
    selected: list[tuple[str, str]] = []
    for bundle_id, (split, lineage_root_id) in sorted(evidence.membership.items()):
        if split not in wanted:
            continue
        row = gate_rows.get(bundle_id)
        if row is None:
            raise ModelTargetCountingError(
                f"bundle {bundle_id} is in split {split!r} but has no gate report "
                "row; its eligibility cannot be established and it can neither be "
                "counted nor silently dropped"
            )
        if row.get("eligibility") == eligibility:
            selected.append((bundle_id, lineage_root_id))
    return selected


def _process_bundle(
    bundle_id: str,
    lineage_root_id: str,
    store_dir: Path,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Count one bundle's fragments. The unit both the sequential and the
    parallel/resumable counter share, so the two paths cannot silently
    diverge on what counting a bundle means."""
    target_bundle = _read_pinned(
        store_dir, artifacts["target_bundle"], f"target bundle of {bundle_id}"
    )
    index = _read_pinned(
        store_dir, artifacts["fragment_index"], f"fragment index of {bundle_id}"
    )
    provenance = target_bundle["provenance"]
    structural = target_bundle["structural"]
    symbols: dict[str, int] = {}
    fragments = 0
    route_tokens = 0
    for entry in index["entries"]:
        fragment = _read_pinned(store_dir, entry["fragment"], f"fragment {entry['fragment_id']}")
        try:
            tokens = encode_fragment(fragment, structural)
        except (ValueError, KeyError) as exc:
            raise ModelTargetCountingError(
                f"fragment {entry['fragment_id']} of {bundle_id} does not encode: {exc}"
            ) from exc
        fragments += 1
        route_tokens += len(tokens)
        for token in tokens:
            symbols[token] = symbols.get(token, 0) + 1
    return {
        "bundle_id": bundle_id,
        "lineage_root_id": lineage_root_id,
        "fragments": fragments,
        "symbols": symbols,
        "route_tokens": route_tokens,
        "label_authority_key": f"{provenance['label_origin']}/{provenance['verification']}",
    }


def _require_artifacts(
    evidence: ReleaseEvidence, selected: Sequence[tuple[str, str]]
) -> dict[str, Mapping[str, Any]]:
    by_bundle: dict[str, Mapping[str, Any]] = {}
    for bundle_id, _lineage_root_id in selected:
        artifacts = evidence.artifacts.get(bundle_id)
        if not artifacts:
            raise ModelTargetCountingError(
                f"bundle {bundle_id} is selected for counting but the release's "
                "artifact index pins no artefacts for it"
            )
        by_bundle[bundle_id] = artifacts
    return by_bundle


def count_release_targets(
    *,
    release_dir: Path | str,
    store_dir: Path | str,
    splits: Sequence[str],
    eligibility: str,
) -> ReleaseTally:
    """Count model target features over one release's pinned page fragments.

    `store_dir` is where the release's artefact `ref`s resolve (the factory
    object store). Every object is digest-checked before it is parsed.

    Single-process; for a whole real release use
    :func:`count_release_targets_resumable` instead (DEC-023 -- `encode_fragment`
    over every fragment of a release runs well past 20 minutes).
    """
    store_dir = Path(store_dir)
    evidence = load_release(release_dir)
    selected = _selected_bundles(evidence, splits, eligibility)
    artifacts_by_bundle = _require_artifacts(evidence, selected)

    symbols, routes, authority = _Tally(), _Tally(), _Tally()
    fragments = 0
    lineage_roots: set[str] = set()

    for bundle_id, lineage_root_id in selected:
        lineage_roots.add(lineage_root_id)
        result = _process_bundle(
            bundle_id, lineage_root_id, store_dir, artifacts_by_bundle[bundle_id]
        )
        fragments += result["fragments"]
        for token, count in result["symbols"].items():
            symbols.add(token, count, lineage_root_id)
        routes.add("structural", result["route_tokens"], lineage_root_id)
        authority.add(result["label_authority_key"], 1, lineage_root_id)

    return ReleaseTally(
        splits=tuple(splits),
        eligibility=eligibility,
        bundles=len(selected),
        fragments=fragments,
        lineage_roots=len(lineage_roots),
        symbols=symbols.resolve(),
        routes=routes.resolve(),
        label_authority=authority.resolve(),
    )


# --- resumable / parallel counting (DEC-023) --------------------------------
#
# `encode_fragment` over every fragment of a whole release (release-v11:
# ~87k fragments) runs well past DEC-023's 20-minute threshold, so this path
# is a process pool by bundle with a parent-owned atomic checkpoint,
# identity refusal on resume, and unbuffered progress -- the same shape as
# `scripts/scan_multipage_reconcile.py` and
# `scripts/census/capability_exclusion_agreement.py`.

#: Set once in the parent process, before the fork pool is created, so every
#: worker inherits them via copy-on-write instead of re-loading the release
#: view for itself.
_SELECTED: list[tuple[str, str]] | None = None
_ARTIFACTS_BY_BUNDLE: dict[str, Mapping[str, Any]] | None = None
_STORE_DIR: Path | None = None


def _worker_init() -> None:
    """Tie a worker's life to its parent's: the parent cannot reap its pool
    on `SIGKILL` (it never runs again to do so), so without this a killed
    parent leaves orphaned workers still burning CPU. `PR_SET_PDEATHSIG`
    makes the kernel deliver `SIGKILL` to this worker the moment its parent
    exits, by any means. Linux-only; best-effort no-op elsewhere."""
    try:
        import ctypes
        import signal

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    except OSError:
        pass


def _worker_process_bundle(index: int) -> tuple[int, dict[str, Any]]:
    """Run in a forked worker. Reads `_SELECTED`/`_ARTIFACTS_BY_BUNDLE`/
    `_STORE_DIR` as inherited from the parent at fork time."""
    bundle_id, lineage_root_id = _SELECTED[index]
    return index, _process_bundle(
        bundle_id, lineage_root_id, _STORE_DIR, _ARTIFACTS_BY_BUNDLE[bundle_id]
    )


def _default_worker_count() -> int:
    cpus = os.cpu_count() or 1
    return max(1, cpus - 2)


#: A tail bundle is worth naming once it has run at least this long. Fixed,
#: not derived from completed-bundle timings: the tail check only fires when
#: `remaining <= resolved_workers` (a handful of checkpoints at most), too
#: few completions to make a median meaningful.
_TAIL_MIN_SECONDS = 60.0
#: Cap on how many long-running bundles are named; the rest are folded into
#: one count so a wide worker pool cannot flood the log (this is the defect
#: R-7 reported: up to 94 names per checkpoint line).
_TAIL_MAX_NAMES = 5


def _tail_line(
    now: float, in_flight: dict[int, float], selected: list[tuple[str, str]]
) -> str | None:
    """The "in flight (tail)" progress line, or `None` if nothing has run
    long enough to be worth naming. Only bundles running at least
    `_TAIL_MIN_SECONDS`, oldest first, capped at `_TAIL_MAX_NAMES`; any
    remainder is reported as a count, not dropped silently."""
    long_running = sorted(
        (
            (i, now - started)
            for i, started in in_flight.items()
            if now - started >= _TAIL_MIN_SECONDS
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )
    if not long_running:
        return None
    shown, rest = long_running[:_TAIL_MAX_NAMES], long_running[_TAIL_MAX_NAMES:]
    names = ", ".join(f"{selected[i][0]}({elapsed:.0f}s)" for i, elapsed in shown)
    if rest:
        names += f", +{len(rest)} more"
    return f"  in flight (tail, >={_TAIL_MIN_SECONDS:.0f}s): {names}"


def _new_checkpoint(
    *,
    release_dir: Path,
    store_dir: Path,
    splits: Sequence[str],
    eligibility: str,
    bundles_total: int,
) -> dict[str, Any]:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "started_at_utc": now,
        "updated_at_utc": now,
        "release_dir": str(release_dir),
        "store_dir": str(store_dir),
        "splits": sorted(splits),
        "eligibility": eligibility,
        "bundles_total": bundles_total,
        "completed_bundle_indices": [],
        "fragments": 0,
        "lineage_roots": [],
        "symbols": {},
        "routes": {},
        "label_authority": {},
        "done": False,
    }


def _load_checkpoint(
    checkpoint_path: Path,
    *,
    release_dir: Path,
    store_dir: Path,
    splits: Sequence[str],
    eligibility: str,
    bundles_total: int,
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        return _new_checkpoint(
            release_dir=release_dir,
            store_dir=store_dir,
            splits=splits,
            eligibility=eligibility,
            bundles_total=bundles_total,
        )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
        raise ModelTargetCountingError(
            f"checkpoint {checkpoint_path} has schema_version "
            f"{checkpoint.get('schema_version')!r}, expected {_CHECKPOINT_SCHEMA_VERSION!r}"
        )
    identity = {
        "release_dir": str(release_dir),
        "store_dir": str(store_dir),
        "splits": sorted(splits),
        "eligibility": eligibility,
        "bundles_total": bundles_total,
    }
    mismatched = {
        key: (checkpoint.get(key), value)
        for key, value in identity.items()
        if checkpoint.get(key) != value
    }
    if mismatched:
        raise ModelTargetCountingError(
            f"checkpoint {checkpoint_path} identity mismatch {mismatched} -- resuming "
            "onto a run over a different release/store/scope is worse than starting "
            "fresh, because the result would look complete (DEC-023). Move the old "
            "checkpoint aside to start over."
        )
    return checkpoint


def _write_checkpoint(checkpoint_path: Path, checkpoint: dict[str, Any]) -> None:
    checkpoint["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp_path = checkpoint_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=False), encoding="utf-8")
    os.replace(tmp_path, checkpoint_path)


def _merge_result(checkpoint: dict[str, Any], result: dict[str, Any]) -> None:
    lineage_root_id = result["lineage_root_id"]
    checkpoint["fragments"] += result["fragments"]
    if lineage_root_id not in checkpoint["lineage_roots"]:
        checkpoint["lineage_roots"].append(lineage_root_id)
    _merge_tally_state(checkpoint["symbols"], result["symbols"], lineage_root_id)
    _merge_tally_state(
        checkpoint["routes"], {"structural": result["route_tokens"]}, lineage_root_id
    )
    _merge_tally_state(
        checkpoint["label_authority"], {result["label_authority_key"]: 1}, lineage_root_id
    )


def count_release_targets_resumable(
    *,
    release_dir: Path | str,
    store_dir: Path | str,
    splits: Sequence[str],
    eligibility: str,
    run_dir: Path | str,
    workers: int | None = None,
    checkpoint_every: int = 25,
) -> ReleaseTally:
    """`count_release_targets`, parallel by bundle and resumable (DEC-023).

    State lives under `run_dir` (`checkpoint.json`), atomically rewritten
    every `checkpoint_every` completed bundles. A crash loses at most the
    bundles counted since the last checkpoint, never the ones already in
    it. Resuming onto a checkpoint whose release/store/scope has changed is
    refused (`ModelTargetCountingError`), not silently accepted.
    """
    global _SELECTED, _ARTIFACTS_BY_BUNDLE, _STORE_DIR

    release_dir = Path(release_dir)
    store_dir = Path(store_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.json"

    evidence = load_release(release_dir)
    selected = _selected_bundles(evidence, splits, eligibility)
    artifacts_by_bundle = _require_artifacts(evidence, selected)
    bundles_total = len(selected)

    checkpoint = _load_checkpoint(
        checkpoint_path,
        release_dir=release_dir,
        store_dir=store_dir,
        splits=splits,
        eligibility=eligibility,
        bundles_total=bundles_total,
    )
    completed: set[int] = set(checkpoint["completed_bundle_indices"])
    pending = [i for i in range(bundles_total) if i not in completed]
    print(
        f"model-target-coverage counting: {len(completed)}/{bundles_total} bundles "
        f"already counted, {len(pending)} pending",
        flush=True,
    )

    if pending:
        _SELECTED = selected
        _ARTIFACTS_BY_BUNDLE = artifacts_by_bundle
        _STORE_DIR = store_dir
        resolved_workers = workers if workers is not None else _default_worker_count()
        print(f"model-target-coverage counting: {resolved_workers} workers", flush=True)

        start_time = time.monotonic()
        total_start = len(completed)
        done_since_checkpoint = 0
        in_flight: dict[int, float] = {}
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=resolved_workers, mp_context=ctx, initializer=_worker_init
        ) as executor:
            futures = {}
            now = time.monotonic()
            for i in pending:
                futures[executor.submit(_worker_process_bundle, i)] = i
                in_flight[i] = now
            for future in as_completed(futures):
                index, result = future.result()
                in_flight.pop(index, None)
                _merge_result(checkpoint, result)
                completed.add(index)
                done_since_checkpoint += 1

                remaining = bundles_total - len(completed)
                if done_since_checkpoint >= checkpoint_every or remaining == 0:
                    checkpoint["completed_bundle_indices"] = sorted(completed)
                    _write_checkpoint(checkpoint_path, checkpoint)
                    done_since_checkpoint = 0
                    elapsed = time.monotonic() - start_time
                    rate = (len(completed) - total_start) / elapsed if elapsed > 0 else 0.0
                    eta_s = remaining / rate if rate > 0 else float("inf")
                    print(
                        f"progress: {len(completed)}/{bundles_total} bundles "
                        f"({checkpoint['fragments']} fragments) "
                        f"rate={rate:.2f} bundles/s elapsed={elapsed:.0f}s eta={eta_s:.0f}s",
                        flush=True,
                    )
                    if 0 < remaining <= resolved_workers and in_flight:
                        # The tail: fewer bundles left than workers, so some
                        # workers sit idle while the rest may be grinding
                        # through one very large document. Name what is
                        # still running and for how long, so an operator can
                        # tell "working on a huge document" from "hung" --
                        # but only the ones actually worth naming (see
                        # `_tail_line`).
                        line = _tail_line(time.monotonic(), in_flight, selected)
                        if line is not None:
                            print(line, flush=True)

    checkpoint["completed_bundle_indices"] = sorted(completed)
    checkpoint["done"] = True
    _write_checkpoint(checkpoint_path, checkpoint)

    return ReleaseTally(
        splits=tuple(splits),
        eligibility=eligibility,
        bundles=bundles_total,
        fragments=checkpoint["fragments"],
        lineage_roots=len(checkpoint["lineage_roots"]),
        symbols=_resolve_tally_state(checkpoint["symbols"]),
        routes=_resolve_tally_state(checkpoint["routes"]),
        label_authority=_resolve_tally_state(checkpoint["label_authority"]),
    )
