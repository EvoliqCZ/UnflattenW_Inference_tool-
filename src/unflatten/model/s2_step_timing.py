"""CPU-only timing of the S2/S3 data path (spec-s2-training-throughput T0a).

Two modes, over real draws of a frozen ordered exposure plan:

- `serial`: per draw, how long the reader's own steps take on today's
  serial path -- shard sha256 (`hash_s`), `torch.load` (`load_s`), row
  decode + integrity of every row in the shard (`integrity_s`), the drawn
  row itself (`row_s`) and `collate_training_batch` (`collate_s`) -- and
  whether the shard was already in the reader's LRU. The reader's functions
  are wrapped for timing, never replaced.
- `supply`: samples/s delivered by `s2_train.plan_batches` at
  `--num-workers W` over draws `[S, S+N)`, then at 0 over the disjoint
  block `[S+N, S+2N)` (no warm-cache advantage for the new path; design
  §12 M2). `--verify-identical` then re-reads `[S, S+N)` at 0 and compares
  a per-sample digest of every tensor with the W run's.

No model is built and CUDA is never touched; the only diagnostic output is
one JSON line on stdout. Not a training or admission tool.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import statistics
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from unflatten.model import composite_vocab as cv
from unflatten.model import exact_text_cache as etc
from unflatten.model import s1_train
from unflatten.model import s2_plan as sp
from unflatten.model import s2_train as s2t
from unflatten.model import s3_typed_loss as stl

#: Design §12 M2: the new path must supply at least this many samples/s
#: and at least this multiple of the serial rate.
_GATE_MIN_RATE = 0.75
_GATE_MIN_SPEEDUP = 4.0


def _reader(dataset: Dataset) -> etc.ExactTextCacheReader:
    return getattr(dataset, "base", dataset).reader


def _base_sample(sample: Any) -> s1_train.ExactTextTrainingSample:
    return getattr(sample, "base", sample)


@contextmanager
def _timed_reader(reader: etc.ExactTextCacheReader, spent: dict[str, float]) -> Iterator[None]:
    """Accumulate time spent in the shard hash, `torch.load` and the whole
    `_shard_rows` call into `spent`, restoring every wrapped function on
    exit."""

    def timed(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                spent[name] += time.perf_counter() - t0

        return wrapper

    sha256, load = etc._sha256, torch.load
    etc._sha256 = timed("hash_s", sha256)
    torch.load = timed("load_s", load)
    reader._shard_rows = timed("shard_s", reader._shard_rows)
    try:
        yield
    finally:
        etc._sha256, torch.load = sha256, load
        del reader._shard_rows


def time_serial(
    dataset: Dataset,
    order: Sequence[int],
    *,
    ranges: cv.PackedRanges,
    image_token_id: int,
) -> list[dict[str, Any]]:
    """One record per draw of `order`, in order, on today's serial path."""
    reader = _reader(dataset)
    records: list[dict[str, Any]] = []
    for draw, index in enumerate(order):
        physical = index if reader._selection is None else reader._selection[index]
        shard = physical // reader.shard_size
        lru_hit = shard in reader._loaded
        spent = {"hash_s": 0.0, "load_s": 0.0, "shard_s": 0.0}
        with _timed_reader(reader, spent):
            t0 = time.perf_counter()
            sample = dataset[index]
            getitem_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        s1_train.collate_training_batch(
            [_base_sample(sample)], ranges=ranges, image_token_id=image_token_id
        )
        collate_s = time.perf_counter() - t0
        records.append(
            {
                "draw": draw,
                "index": index,
                "shard": shard,
                "lru_hit": lru_hit,
                "hash_s": spent["hash_s"],
                "load_s": spent["load_s"],
                "integrity_s": max(0.0, spent["shard_s"] - spent["hash_s"] - spent["load_s"]),
                "row_s": max(0.0, getitem_s - spent["shard_s"]),
                "collate_s": collate_s,
            }
        )
    return records


def sample_digest(sample: Any) -> str:
    """sha256 over every field of a (possibly nested) sample dataclass:
    dtype, shape and raw bytes for tensors, `repr` for everything else."""
    digest = hashlib.sha256()

    def feed(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            flat = value.detach().cpu().contiguous().flatten()
            digest.update(f"tensor:{value.dtype}:{tuple(value.shape)}:".encode())
            digest.update(flat.view(torch.uint8).numpy().tobytes())
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                digest.update(f"<{field.name}>".encode())
                feed(getattr(value, field.name))
        else:
            digest.update(repr(value).encode())

    feed(sample)
    return "sha256:" + digest.hexdigest()


def time_supply(
    dataset: Dataset, order: Sequence[int], *, num_workers: int, digest: bool = False
) -> dict[str, Any]:
    """Wall time for `s2_train.plan_batches` to deliver every draw of
    `order` (batch size 1), in total and after the first `num_workers`
    samples (worker start-up excluded)."""
    stamps: list[float] = []
    digests: list[str] = []
    t0 = time.perf_counter()
    for batch in s2t.plan_batches(
        dataset, order, start_step=0, steps=len(order), batch_size=1, num_workers=num_workers
    ):
        stamps.append(time.perf_counter() - t0)
        if digest:
            digests.append(sample_digest(batch[0]))
    total = stamps[-1]
    skip = num_workers if 0 < num_workers < len(stamps) else 0
    after = (len(stamps) - skip) / (total - stamps[skip - 1]) if skip else len(stamps) / total
    result: dict[str, Any] = {
        "num_workers": num_workers,
        "samples": len(stamps),
        "total_s": total,
        "samples_per_s": len(stamps) / total,
        "samples_per_s_after_first_w": after,
    }
    if digest:
        result["digests"] = digests
    return result


def _summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "draws": len(records),
        "lru_hits": sum(1 for r in records if r["lru_hit"]),
    }
    for name in ("hash_s", "load_s", "integrity_s", "row_s", "collate_s"):
        values = sorted(r[name] for r in records)
        summary[name] = {
            "median": statistics.median(values),
            "p90": values[min(len(values) - 1, int(0.9 * len(values)))],
        }
    return summary


def _qwen_collate_ids() -> tuple[cv.PackedRanges, int]:
    """The candidate's vocabulary size and image token id, read from its
    pinned config alone (no weights, no CUDA)."""
    from transformers import AutoConfig

    from unflatten.model.candidate_loader import QWEN_CANDIDATE_MODEL_ID, QWEN_CANDIDATE_REVISION

    config = AutoConfig.from_pretrained(QWEN_CANDIDATE_MODEL_ID, revision=QWEN_CANDIDATE_REVISION)
    return cv.packed_ranges(config.text_config.vocab_size), config.image_token_id


def _plan_order(run_dir: Path, release_dir: Path) -> tuple[list[int], etc.CacheExpectation]:
    expectation = etc.CacheExpectation.from_payload(
        json.loads((run_dir / "exact_text_cache_expectation.json").read_text())
    )
    plan = json.loads((run_dir / "ordered_exposure_plan.json").read_text())
    pages = s2t.resolve_plan_pages(
        plan=plan, expectation=expectation, split_records=sp.load_split_records(release_dir)
    )
    return s2t.plan_dataset_order(pages, expectation), expectation


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.s2_step_timing", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cache", type=Path, help="exact-text cache directory")
    parser.add_argument("--run-dir", type=Path, required=True, help="frozen S2/S3 run dir")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("serial", "supply"), required=True)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--start-draw", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--typed", action="store_true", help="wrap in S3TrainingDataset")
    parser.add_argument("--verify-identical", action="store_true")
    args = parser.parse_args(argv)

    order, expectation = _plan_order(args.run_dir, args.release_dir)
    blocks = 2 if args.mode == "supply" else 1
    end = args.start_draw + blocks * args.draws
    if args.draws < 1 or args.start_draw < 0 or end > len(order):
        parser.error(
            f"draws [{args.start_draw}, {end}) do not fit the plan's {len(order)} draws"
        )
    if args.mode == "supply" and args.num_workers < 1:
        parser.error("supply mode compares --num-workers W >= 1 against 0")

    def fresh_dataset() -> Dataset:
        base = s1_train.ExactTextTrainingDataset(
            args.cache, gradient_bearing=True, expectation=expectation
        )
        return stl.S3TrainingDataset(base) if args.typed else base

    a = (args.start_draw, args.start_draw + args.draws)
    if args.mode == "serial":
        ranges, image_token_id = _qwen_collate_ids()
        records = time_serial(
            fresh_dataset(), order[a[0] : a[1]], ranges=ranges, image_token_id=image_token_id
        )
        print(json.dumps({"mode": "serial", "draws": list(a), "summary": _summary(records),
                          "records": records}), flush=True)
        return 0

    b = (a[1], a[1] + args.draws)
    verify = args.verify_identical
    new = time_supply(fresh_dataset(), order[a[0] : a[1]],
                      num_workers=args.num_workers, digest=verify)
    old = time_supply(fresh_dataset(), order[b[0] : b[1]], num_workers=0, digest=verify)
    report: dict[str, Any] = {
        "mode": "supply",
        "typed": args.typed,
        "new": {"draws": list(a), **{k: v for k, v in new.items() if k != "digests"}},
        "old": {"draws": list(b), **{k: v for k, v in old.items() if k != "digests"}},
    }
    speedup = new["samples_per_s"] / old["samples_per_s"]
    gate = {
        "new_ge_0_75_per_s": new["samples_per_s"] >= _GATE_MIN_RATE,
        "new_ge_4x_old": speedup >= _GATE_MIN_SPEEDUP,
    }
    report["speedup"] = speedup
    report["gate_m2"] = {**gate, "pass": all(gate.values())}
    mismatches = 0
    if verify:
        replay = time_supply(fresh_dataset(), order[a[0] : a[1]], num_workers=0, digest=True)
        mismatches = sum(x != y for x, y in zip(new["digests"], replay["digests"]))
        report["identical"] = {"draws": list(a), "mismatches": mismatches}
    print(json.dumps(report), flush=True)
    return 1 if mismatches else 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
