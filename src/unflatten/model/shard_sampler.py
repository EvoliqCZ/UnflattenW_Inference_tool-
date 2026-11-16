"""Shuffled sampling that does not thrash a sharded cache.

Full shuffling over a sharded cache is a pathology, not a preference: a
39 GB cache in 79 shards of 512 MB serves one random sample per 512 MB
shard load, so an epoch of 2 500 samples reads ~1.3 TB — 32x the cache —
and the GPU waits on all of it. An LRU does not save it either, because
uniform random access over 79 shards hits a cache of 8 about a tenth of
the time.

`ShardWindowSampler` shuffles at two levels instead of one:

1. the **shard order** is permuted afresh every epoch;
2. the permuted shards are cut into windows of `window` shards, and the
   samples inside a window are permuted as a whole.

An epoch then loads each shard exactly once — one pass over the cache,
1.0x its size — while the order the model sees is random within a window
of `window * shard_size` pages built from randomly chosen shards.

**What this costs, stated plainly.** It is not i.i.d. sampling. Two
samples drawn close together in an epoch come from the same window, so a
batch is drawn from `window * shard_size` pages rather than from the
whole cache, and pages that share a window are correlated for the length
of that window. Two things bound the damage: the window is built from
shards picked at random (not from neighbouring ones, so build order does
not leak into it), and the partition is redrawn each epoch (so no pair of
pages is co-located for the whole run). This is the standard shuffle
buffer trade — with `window=8` and `shard_size=32` the buffer is 256
pages, i.e. 64 batches of 4 — and it is a trade, not a free win. A run
that needs true i.i.d. order can pass a window covering every shard, at
the cost of holding them all in RAM.

Indices are yielded in `[0, length)`, so a sampler built for a prefix of
a cache stays inside that prefix: `model/scaling.py` depends on subsets
being prefixes, and shuffling the *order* of a prefix does not change
*which* pages it contains.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class ShardWindowSampler(Sampler[int]):
    """Permute shards, then permute the samples within each window."""

    def __init__(
        self,
        length: int,
        *,
        shard_size: int,
        window: int,
        generator: torch.Generator | None = None,
    ) -> None:
        if length < 0:
            raise ValueError(f"length must not be negative, got {length}")
        if shard_size < 1:
            raise ValueError(f"shard_size must be positive, got {shard_size}")
        if window < 1:
            raise ValueError(f"window must be positive, got {window}")
        self.length = length
        self.shard_size = shard_size
        self.window = window
        self.generator = generator

    def __len__(self) -> int:
        return self.length

    def _shards(self) -> int:
        return (self.length + self.shard_size - 1) // self.shard_size

    def __iter__(self) -> Iterator[int]:
        order = torch.randperm(self._shards(), generator=self.generator).tolist()
        for start in range(0, len(order), self.window):
            indices: list[int] = []
            for shard in order[start : start + self.window]:
                low = shard * self.shard_size
                indices.extend(range(low, min(low + self.shard_size, self.length)))
            for position in torch.randperm(
                len(indices), generator=self.generator
            ).tolist():
                yield indices[position]
