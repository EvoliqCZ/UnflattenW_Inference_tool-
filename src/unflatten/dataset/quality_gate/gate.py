"""The DEC-057 data-quality gate: one pass over every page of the train and
val caches (G-A + G-B), the cache manifests' own counters, and the G-C
register -- one sealed result, one verdict.

Chain position (agreed with v16-data-fixes, `docs/plans/v16-data-fixes-
design.md` §4 step 5b / `docs/runs/rebuild-plan-dec049.md` step 6a): after
the exact-text caches are built, before the loop-guard caps (V4), the run
freeze and the pilot. The gate is read-only; on a violation it exits
non-zero, names the pages, and writes `gate-exclusions.tsv` in the
`--page-exclusions` format (`v16_target_policy.load_page_exclusions`), so
the operator adds it to the cache rebuild and runs the gate again.

Binding: the result's `gate_result_hash` covers the policy, every
detector version, both cache identities, the counts, the proposed
exclusions and the register hash. `require_data_quality_gate` is the one
check a trainer calls: for a cache built under a v16 target policy it
refuses unless a PASSING, complete result binds exactly those caches, and
returns the hash the run manifest records; for every earlier cache it
returns `None` and nothing changes.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import Any

from unflatten.dataset.identity import domain_hash, lineage_root_id_real
from unflatten.dataset.quality_gate import ga, gb, gc
from unflatten.dataset.quality_gate.pages import (
    CacheShard,
    GatePage,
    iter_cache_shard,
    skipped_keys,
)

GATE_SCHEMA = "data-quality-gate/v1"
RESULT_DOMAIN = "unflatten.data-quality-gate/v1"
POLICY_DOMAIN = "unflatten.data-quality-gate-policy/v1"
DEFAULT_POLICY = Path(__file__).with_name("policy.v16.json")
RESULT_FILENAME = "data-quality-gate.json"
EXCLUSIONS_FILENAME = "gate-exclusions.tsv"

#: the cache manifest key the v16 target policy writes (v16_target_policy)
V16_POLICY_PREFIX = "v16-targets."


class DataQualityGateRefused(ValueError):
    """A v16 run was asked to start without a passing gate result that
    binds its caches, or a gate result does not verify."""


# -- policy --------------------------------------------------------------------


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    policy = json.loads(Path(path).read_text())
    for key in ("policy_id", "status", "splits", "ga", "manifest", "gb", "gc"):
        if key not in policy:
            raise ValueError(f"{path}: data-quality-gate policy has no {key!r}")
    missing = sorted(set(ga.DETECTORS) - set(policy["ga"]))
    if missing:
        raise ValueError(f"{path}: policy has no G-A threshold for {missing}")
    return policy


def policy_hash(policy: Mapping[str, Any]) -> str:
    return domain_hash(POLICY_DOMAIN, policy)


def is_v16_cache(manifest: Mapping[str, Any]) -> bool:
    """A cache built under a v16 target policy (DEC-055): the manifest's
    top-level `target_policy.descriptor.policy_id` (absent before v16)."""
    policy_id = ((manifest.get("target_policy") or {}).get("descriptor") or {}).get("policy_id")
    return isinstance(policy_id, str) and policy_id.startswith(V16_POLICY_PREFIX)


def cache_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    from unflatten.model.run_binding import compute_exact_text_cache_identity_hash

    target_policy = manifest.get("target_policy") or {}
    return {
        "cache_identity_hash": compute_exact_text_cache_identity_hash(dict(manifest["identity"])),
        "page_key_set_sha256": manifest["scope"]["page_key_set_sha256"],
        "entries": manifest["entries"],
        "target_policy_id": (target_policy.get("descriptor") or {}).get("policy_id"),
        "page_exclusions_sha256": (target_policy.get("descriptor") or {}).get(
            "page_exclusions_sha256"
        ),
    }


def sample_cache_binding_violation(
    sample: Mapping[str, Any], manifests: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any] | None:
    """The G-C spot check must be drawn from the train cache this gate
    reads (DEC-057 addendum A: every finding round repeats the gate on a
    NEW sample). `cmd_sample` seals the train manifest's `cache_binding`
    into the sample body; a sample drawn before the last rebuild, or one
    without the record, is a violation -- never a pass."""
    if "train" not in manifests:
        return None
    recorded = sample.get("train_cache_binding")
    if recorded == cache_binding(manifests["train"]):
        return None
    return {
        "part": "G-C",
        "check": "sample_cache_binding",
        "problem": "missing" if recorded is None else "stale",
    }


# -- page objects (G-B) --------------------------------------------------------


def store_index(store_dir: Path, bundles: set[str]) -> dict[str, dict[str, str]]:
    """`bundle -> {source_sha256, lineage_root_id}` from the store index."""
    out: dict[str, dict[str, str]] = {}
    with (Path(store_dir) / "index.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            bundle = row["envelope"]["document_bundle_id"]
            if bundle in bundles:
                out[bundle] = {
                    "source_sha256": row.get("source_sha256") or "",
                    "lineage_root_id": row["envelope"]["lineage_root_id"],
                }
    return out


def store_objects(store_dir: Path, bundles: set[str]) -> dict[str, dict[str, Any]]:
    """`bundle -> {"pages": [png path...], "pdf": pdf path}` from the store's
    `renders.jsonl` (the PNGs are the ones the release pins, the PDF is
    what they were rasterised from)."""
    root = Path(store_dir)
    out: dict[str, dict[str, Any]] = {}
    with (root / "renders.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["document_bundle_id"] not in bundles:
                continue
            artifacts = row["artifacts"]
            if "pdf" not in artifacts or not artifacts.get("pages"):
                continue
            out[row["document_bundle_id"]] = {
                "pages": [root / p["ref"] for p in artifacts["pages"]],
                "pdf": root / artifacts["pdf"]["ref"],
            }
    return out


# -- the page pass ---------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _init_worker(
    skipped: set[tuple[str, int]], tokenizer_factory: Callable[[], Any] | None,
    objects: Callable[[str, int], dict[str, str] | None],
) -> None:
    _WORKER["skipped"] = skipped
    _WORKER["tokenizer"] = tokenizer_factory() if tokenizer_factory else None
    _WORKER["objects"] = objects


def _pinned_tokenizer() -> Any:
    from unflatten.model.qwen_tokenizer_artifacts import load_pinned_tokenizer

    return load_pinned_tokenizer()[0]


def page_work(
    item: tuple[str, int, list[int], dict[str, str] | None, Mapping[str, Any]], min_row_milli: int
) -> tuple[str, dict[str, int], dict[str, int] | None]:
    bundle, page, tokens, objects, extra = item
    values = ga.page_values(tokens, bundle, page, _WORKER["skipped"], extra=extra)
    measured = None
    if objects is not None:
        facts = gb.target_facts(tokens, _WORKER["tokenizer"])
        measured = gb.measure(
            facts,
            gb.load_gray(Path(objects["png"])),
            gb.text_layer(Path(objects["pdf"]), page),
            min_row_milli=min_row_milli,
        )
    return f"{bundle}#{page}", values, measured


def _work_chunk(args: tuple[list[Any] | CacheShard, int]) -> list[Any]:
    items, min_row_milli = args
    if isinstance(items, CacheShard):
        pages = iter_cache_shard(items.path, with_extra=items.with_extra)
        return [
            page_work(
                (p.bundle, p.page, p.tokens, _WORKER["objects"](p.bundle, p.page), p.extra),
                min_row_milli,
            )
            for p in islice(pages, items.max_pages)
        ]
    return [page_work(item, min_row_milli) for item in items]


def _ordered_work(pool: ProcessPoolExecutor, tasks: Iterable[Any], pending_limit: int):
    """Bound submitted work and yield in source order, even for slow shards."""
    tasks = iter(tasks)
    pending: deque[Any] = deque()
    try:
        for task in islice(tasks, pending_limit):
            pending.append(pool.submit(_work_chunk, task))
        while pending:
            yield pending.popleft().result()
            for task in islice(tasks, 1):
                pending.append(pool.submit(_work_chunk, task))
    finally:
        for future in pending:
            future.cancel()


def scan_split(
    pages: Iterable[GatePage | CacheShard],
    manifest: Mapping[str, Any],
    objects: Callable[[str, int], dict[str, str] | None],
    *,
    min_row_milli: int,
    workers: int = 4,
    tokenizer_factory: Callable[[], Any] | None = _pinned_tokenizer,
    chunk: int = 64,
    limit_pages: int | None = None,
    progress: Callable[[int], None] | None = None,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]], list[str]]:
    """G-A values for every page, G-B values where `objects` resolves the
    page (a page without objects is reported, never silently passed)."""
    skipped = skipped_keys(manifest)
    ga_values: dict[str, dict[str, int]] = {}
    gb_values: dict[str, dict[str, int]] = {}
    no_objects: list[str] = []
    completed = 0
    if limit_pages is not None and limit_pages < 0:
        raise ValueError("limit_pages must be non-negative")

    def chunks() -> Iterator[tuple[list[Any] | CacheShard, int]]:
        buf: list[Any] = []
        page_iter = iter(pages)
        while not limit_pages or completed + len(buf) < limit_pages:
            try:
                p = next(page_iter)
            except StopIteration:
                break
            if isinstance(p, CacheShard):
                if buf:
                    yield buf, min_row_milli
                    buf = []
                # Partial scans use one outstanding shard so the next limit
                # is based on actual completed rows, never manifest counts.
                if limit_pages:
                    remaining = limit_pages - completed
                    if p.max_pages is not None:
                        remaining = min(remaining, p.max_pages)
                    p = replace(p, max_pages=remaining)
                yield p, min_row_milli
                continue
            buf.append((p.bundle, p.page, p.tokens, objects(p.bundle, p.page), p.extra))
            if len(buf) == chunk:
                yield buf, min_row_milli
                buf = []
        if buf:
            yield buf, min_row_milli

    with ProcessPoolExecutor(
        workers, initializer=_init_worker, initargs=(skipped, tokenizer_factory, objects)
    ) as pool:
        for rows in _ordered_work(pool, chunks(), 1 if limit_pages else workers * 2):
            for page_id, values, measured in rows:
                ga_values[page_id] = values
                if measured is None:
                    no_objects.append(page_id)
                else:
                    gb_values[page_id] = measured
            completed += len(rows)
            if progress is not None:
                progress(completed)
    return ga_values, gb_values, no_objects


def evaluate_gb(
    values: Mapping[str, Mapping[str, int]],
    policy: Mapping[str, Any],
    split: str,
    waived: frozenset[str],
    no_objects: Sequence[str],
) -> dict[str, Any]:
    flagged: dict[str, list[str]] = {check: [] for check in gb.GB_CHECKS}
    for page_id, row in sorted(values.items()):
        for check in gb.flags(row, policy):
            flagged[check].append(page_id)
    out: dict[str, Any] = {"version": gb.GB_VERSION, "pages": len(values), "checks": {}}
    for check in gb.GB_CHECKS:
        rule = policy[check]
        hits = flagged[check]
        open_hits = [p for p in hits if p not in waived]
        out["checks"][check] = {
            "rule": {k: v for k, v in rule.items() if k != "basis"},
            "flagged": len(hits),
            "waived": len(hits) - len(open_hits),
            "violation": len(open_hits) > rule["max_flagged_pages"][split],
            "pages": [{"page": p, "values": dict(values[p])} for p in open_hits],
        }
    out["pages_without_objects"] = len(no_objects)
    out["pages_without_objects_ids"] = list(no_objects[:50])
    out["violation"] = any(c["violation"] for c in out["checks"].values())
    return out


def _ga_policy(policy: Mapping[str, Any], split: str) -> dict[str, dict[str, int]]:
    return {
        name: {"page_max": rule["page_max"], "max_flagged_pages": rule["max_flagged_pages"][split]}
        for name, rule in policy["ga"].items()
    }


# -- result ------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_result(
    *,
    policy: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
    scans: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any], Sequence[str]]],
    register: Sequence[gc.Finding],
    register_sha256: str,
    sample: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, str]],
    partial: bool = False,
) -> dict[str, Any]:
    """Assemble and seal the gate result. `scans[split]` is `scan_split`'s
    output; `identities` maps `bundle -> {source_sha256, lineage_root_id}`.
    `sample` must carry the train manifest's `cache_binding`
    (`sample_cache_binding_violation`)."""
    present: dict[str, str] = {}
    for split in manifests:
        for page_id in scans[split][0]:
            bundle, page = page_id.rsplit("#", 1)
            present[page_id] = page_id
            lineage = (identities.get(bundle) or {}).get("lineage_root_id")
            if lineage:
                present[f"{lineage}#{page}"] = page_id
    gc_result = gc.evaluate_register(register, sample, policy["gc"], present)
    waived = frozenset(gc_result["waived_pages"])
    splits: dict[str, Any] = {}
    violations: list[dict[str, Any]] = []
    proposed: dict[str, set[str]] = {}
    for split, manifest in manifests.items():
        ga_values, gb_values, no_objects = scans[split]
        ga_result = ga.evaluate(ga_values.items(), _ga_policy(policy, split), waived)
        gb_result = evaluate_gb(gb_values, policy["gb"], split, waived, no_objects)
        checks = ga.manifest_checks(manifest, policy["manifest"][split])
        skip_counts = Counter(row["reason"] for row in manifest.get("page_skip_rows", ()))
        splits[split] = {
            "cache": cache_binding(manifest),
            "excluded_by_build": {
                "page_exclusions": (manifest.get("target_policy") or {}).get("page_exclusions"),
                "skipped_defect_listed": skip_counts.get("skipped_defect_listed", 0),
                "skipped_defect_spill": skip_counts.get("skipped_defect_spill", 0),
                "skipped_unowned_ink": skip_counts.get("skipped_unowned_ink", 0),
            },
            "ga": ga_result,
            "gb": gb_result,
            "manifest_checks": checks,
        }
        for name, det in ga_result["detectors"].items():
            if det["violation"]:
                violations.append(
                    {
                        "split": split,
                        "part": "G-A",
                        "check": name,
                        "flagged": len(det["pages"]),
                        "allowed": det["max_flagged_pages"],
                    }
                )
            if det["violation"] and len(det["pages"]) <= policy["ga"][name]["exclude_up_to_pages"]:
                for p in det["pages"]:
                    proposed.setdefault(p["page"], set()).add(f"dqg:ga:{name}")
        for name, check in gb_result["checks"].items():
            if check["violation"]:
                violations.append(
                    {
                        "split": split,
                        "part": "G-B",
                        "check": name,
                        "flagged": len(check["pages"]),
                        "allowed": check["rule"]["max_flagged_pages"][split],
                    }
                )
            if (
                check["violation"]
                and len(check["pages"]) <= policy["gb"][name]["exclude_up_to_pages"]
            ):
                for p in check["pages"]:
                    proposed.setdefault(p["page"], set()).add(f"dqg:gb:{name}")
        if no_objects:  # a page G-B could not image never passes silently
            violations.append(
                {
                    "split": split,
                    "part": "G-B",
                    "check": "pages_without_objects",
                    "flagged": len(no_objects),
                    "allowed": 0,
                }
            )
        for name, check in checks.items():
            if check["violation"]:
                violations.append(
                    {
                        "split": split,
                        "part": "manifest",
                        "check": name,
                        "value": check["value"],
                        "limit": check["limit"],
                    }
                )
    for finding in gc_result["open_blocking"]:
        target = present.get(finding["page"])
        if target:
            proposed.setdefault(target, set()).add(f"dqg:gc:{finding['class']}")
    if gc_result["violation"]:
        violations.append(
            {
                "part": "G-C",
                "check": "register",
                "open_blocking": len(gc_result["open_blocking"]),
                "unreviewed": len(gc_result["unreviewed_sample_pages"]),
            }
        )
    sample_violation = sample_cache_binding_violation(sample, manifests)
    if sample_violation:
        violations.append(sample_violation)
    if partial:
        violations.append({"part": "gate", "check": "partial_run"})
    exclusions, unexcludable = exclusion_rows(proposed, identities)
    body = {
        "schema": GATE_SCHEMA,
        "policy": {
            "policy_id": policy["policy_id"],
            "status": policy["status"],
            "policy_hash": policy_hash(policy),
        },
        "detector_versions": {
            "ga": {name: version for name, (version, _field) in ga.DETECTORS.items()},
            "gb": gb.GB_VERSION,
            "gc": gc.GC_VERSION,
        },
        "dataset_release_root_hash": sorted(
            {m["identity"]["dataset_release_root_hash"] for m in manifests.values()}
        ),
        "register_sha256": register_sha256,
        "splits": splits,
        "gc": gc_result,
        "proposed_exclusions": exclusions,
        "unexcludable_pages": unexcludable,
        "violations": violations,
        "partial": partial,
        "verdict": "fail" if violations else "pass",
    }
    return {**body, "gate_result_hash": domain_hash(RESULT_DOMAIN, body)}


def exclusion_rows(
    proposed: Mapping[str, set[str]], identities: Mapping[str, Mapping[str, str]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Proposed exclusions in the `--page-exclusions` shape. A page whose
    lineage is not the real-source lineage of its `source_sha256`
    (synthetic documents) cannot be listed there and is returned apart."""
    rows, unexcludable = [], []
    for page_id in sorted(proposed):
        bundle, page = page_id.rsplit("#", 1)
        ident = identities.get(bundle) or {}
        sha, lineage = ident.get("source_sha256", ""), ident.get("lineage_root_id", "")
        if len(sha) != 64 or lineage != lineage_root_id_real(source_set_hash=f"sha256:{sha}"):
            unexcludable.append(page_id)
            continue
        rows.append(
            {
                "page_id": page_id,
                "source_sha256": sha,
                "lineage_root_id": lineage,
                "page_index": int(page),
                "reason": ",".join(sorted(proposed[page_id])),
            }
        )
    return rows, unexcludable


def exclusions_tsv(rows: Sequence[Mapping[str, Any]]) -> str:
    head = (
        "# DEC-057 data-quality gate: proposed exclusions (append to --page-exclusions,"
        " rebuild the caches, run the gate again)\n"
        "# page_id\tsource_sha256\tlineage_root_id\tpage_index\treason\n"
    )
    return head + "".join(
        f"{r['page_id']}\t{r['source_sha256']}\t{r['lineage_root_id']}\t{r['page_index']}\t{r['reason']}\n"
        for r in rows
    )


def verify_result(result: Mapping[str, Any]) -> None:
    body = {k: v for k, v in result.items() if k != "gate_result_hash"}
    if result.get("schema") != GATE_SCHEMA:
        raise DataQualityGateRefused(f"not a {GATE_SCHEMA} result")
    if domain_hash(RESULT_DOMAIN, body) != result.get("gate_result_hash"):
        raise DataQualityGateRefused("gate_result_hash does not verify against the result body")


def require_data_quality_gate(
    manifests: Mapping[str, Mapping[str, Any]], gate_path: Path | None
) -> str | None:
    """The trainer-side check (DEC-057 point 6). `None` when no cache was
    built under a v16 target policy (every existing release and run keeps
    its identity byte for byte); otherwise the passing, complete result's
    `gate_result_hash` -- or a refusal naming what is missing."""
    if not any(is_v16_cache(m) for m in manifests.values()):
        return None
    if gate_path is None:
        raise DataQualityGateRefused(
            "the caches were built under a v16 target policy: a passing DEC-057 "
            "data-quality gate result (--data-quality-gate) is required"
        )
    result = json.loads(Path(gate_path).read_text())
    verify_result(result)
    if result["verdict"] != "pass" or result.get("partial"):
        raise DataQualityGateRefused(
            f"data-quality gate result {gate_path} did not pass "
            f"({len(result.get('violations', []))} violation(s), partial={result.get('partial')})"
        )
    for split, manifest in manifests.items():
        bound = (result["splits"].get(split) or {}).get("cache")
        if bound != cache_binding(manifest):
            raise DataQualityGateRefused(
                f"data-quality gate result does not bind the {split} cache the run uses "
                "(cache identity, page key set or exclusions differ)"
            )
    return result["gate_result_hash"]
