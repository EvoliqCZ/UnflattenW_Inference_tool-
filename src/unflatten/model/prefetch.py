"""Run per-page CPU work ahead of the GPU, without reordering it.

Building a feature cache is two costs on two devices: ~360 ms of CPU per
page (label withdrawal, PNG decode, target encoding) and ~87 ms of GPU.
Run in sequence each waits for the other, and the measured result is a
GPU at 1-6 % — the CPU is the pipeline, the tower is idle.

This module is the overlap: chunks of consecutive items are prepared in
worker processes while the main process feeds the backbone.

Two properties are load-bearing and both are structural here, not
best-effort:

- **Order is the submission order.** `model/scaling.py` builds scaling
  curves from cache *prefixes*, so a reordered cache silently changes
  which pages a curve point contains. Results are yielded by walking a
  FIFO of futures, so a chunk that finishes early waits its turn.
- **Work in flight is bounded.** Each prepared page carries its decoded
  raster (~3 MB), so an unbounded pool would race ahead of the GPU and
  hold the whole split in RAM. At most `workers * window` chunks exist
  at once, and the source iterator is not advanced beyond them.

Chunks are consecutive on purpose: pages of one document share the
fragment index and the Observation Sidecar behind a per-process
`lru_cache`, and round-robin over single pages would turn one read per
document into one per page.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from itertools import islice


def _init_worker() -> None:
    """One thread per worker: N workers x N threads is contention."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dep of callers
        return
    torch.set_num_threads(1)


def _chunks[T](items: Iterable[T], size: int) -> Iterator[list[T]]:
    iterator = iter(items)
    while chunk := list(islice(iterator, size)):
        yield chunk


def ordered_prefetch[T, R](
    items: Iterable[T],
    work: Callable[[list[T]], list[R]],
    *,
    workers: int = 0,
    chunk: int = 8,
    window: int = 2,
    executor_factory: Callable[[int], object] | None = None,
) -> Iterator[R]:
    """Yield `work` applied to `items`, in order, computed ahead.

    With `workers <= 0` the work runs in this process, chunk by chunk and
    lazily — the same call sequence a plain loop would make, which is what
    keeps the serial path testable by monkeypatching what `work` calls.

    `work` and every item must be picklable: they cross a process
    boundary. `executor_factory` exists for tests that need a pool whose
    tasks share this process's memory.
    """
    batches = _chunks(items, max(1, chunk))
    if workers <= 0:
        for batch in batches:
            yield from work(batch)
        return

    in_flight = max(1, workers * max(1, window))
    executor = (
        executor_factory(workers)
        if executor_factory is not None
        else ProcessPoolExecutor(max_workers=workers, initializer=_init_worker)
    )
    pending: deque[Future[list[R]]] = deque()
    try:
        for batch in batches:
            pending.append(executor.submit(work, batch))
            if len(pending) < in_flight:
                continue
            # FIFO: the oldest chunk is the next in document order, even
            # if a later one finished first.
            yield from pending.popleft().result()
        while pending:
            yield from pending.popleft().result()
    finally:
        # Reached on a consumer that stops early (a page limit) as well as
        # on failure: unstarted chunks are dropped rather than computed
        # into a queue nobody reads.
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)  # type: ignore[attr-defined]
