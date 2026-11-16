"""Spatial probe: do frozen features know *where* things are?

The pooled probe answers "is document information present at all". It
cannot answer the question the tree decoder and every grounding target
actually depend on — whether an individual visual token knows what sits
under it — because mean-pooling discards position by construction. That
showed up in its own numbers: paragraph counts ranked at 0.40 while R²
stayed at -0.19, which is exactly what you see when the ordering survives
pooling and the cardinality does not.

This module builds the per-token test instead:

- the backbone reports a patch grid, and a merged token therefore covers
  a known rectangle of the page;
- the Observation Sidecar records every matched text node's box as
  `bbox_ppm` — millionths of a page, so resolution-independent;
- overlapping the two gives each token a label, and a linear head on that
  token's own feature vector either predicts it or does not.

`in_table` needs ancestry, and the target bundle's `parent_index` points
at the parent's *open event* rather than its ordinal — walking it as if
it were an ordinal silently produces wrong ancestors rather than an
error, so `ancestor_types` resolves the indirection explicitly.

Fitting streams: a page contributes thousands of token rows, so
materialising every row would need tens of GB. `RidgeAccumulator` keeps
only the normal equations, whose size depends on the feature width and
not on how many pages were seen.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

#: Page coordinates are millionths of a page (RFC003 `bbox_ppm`).
PPM = 1_000_000

#: Labels this module produces per token.
SPATIAL_LABELS: tuple[str, ...] = ("has_text", "in_table")


class MergeSizeError(ValueError):
    """A tower's row count matches no merge of the grid it reported.

    Typed so a caller can skip such a page without also swallowing the
    tiling errors that mean the feature tap itself is wrong.
    """


class AnyResGeometryError(MergeSizeError):
    """AnyRes tiles do not account for the page they claim to cover.

    The same refusal one grid up: a backbone that slices the page into
    tiles still has to produce a token count its reported geometry
    explains. Subclassing keeps that promise in one place — a caller that
    skips ungeometric pages skips these too, and nothing that catches
    `MergeSizeError` has to learn a second name to stay correct.
    """


def infer_merge_size(grid_thw: tuple[int, int, int], token_count: int) -> int:
    """How many patches per axis one output row covers.

    Towers disagree, and both of the ones measured here return one row per
    *patch* despite declaring `spatial_merge_size: 2` in their config —
    the merge happens in the projector that follows, which a vision-only
    load never reaches. Assuming the configured value silently misaligns
    every token from the page and the probe reports "no geometry" rather
    than an error, so the ratio is derived from what the tower actually
    produced.
    """
    _, height, width = grid_thw
    for merge in (1, 2, 4):
        if height % merge == 0 and width % merge == 0:
            if (height // merge) * (width // merge) == token_count:
                return merge
    raise MergeSizeError(
        f"grid {height}x{width} cannot produce {token_count} rows at any merge"
    )


def token_boxes(
    grid_thw: tuple[int, int, int], *, merge_size: int = 2
) -> torch.Tensor:
    """Page rectangle of every visual token, in ppm, row-major.

    `grid_thw` is the backbone's patch grid; `merge_size` is how many
    patches per axis one token covers.
    """
    _, height, width = grid_thw
    if height % merge_size or width % merge_size:
        raise ValueError(
            f"patch grid {height}x{width} is not divisible by merge {merge_size}"
        )
    rows, cols = height // merge_size, width // merge_size
    row_index = torch.arange(rows, dtype=torch.float32).repeat_interleave(cols)
    col_index = torch.arange(cols, dtype=torch.float32).repeat(rows)
    return torch.stack(
        [
            col_index / cols * PPM,
            row_index / rows * PPM,
            (col_index + 1) / cols * PPM,
            (row_index + 1) / rows * PPM,
        ],
        dim=1,
    )


def ancestor_types(structural: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """`node_id -> ancestor node types`, nearest ancestor first.

    `parent_index[i]` is the index of the parent's OPEN EVENT, not the
    parent's position in `node_order`; the two coincide only for trivial
    documents, so the mapping is resolved rather than assumed.
    """
    order = structural.get("node_order", [])
    types = structural.get("node_types", [])
    parents = structural.get("parent_index", [])
    events = structural.get("events", [])

    ordinal_of_open_event: dict[int, int] = {}
    ordinal = 0
    for event in events:
        if event.get("op") == "open":
            ordinal_of_open_event[event["index"]] = ordinal
            ordinal += 1

    ancestry: dict[str, tuple[str, ...]] = {}
    for index, node_id in enumerate(order):
        chain: list[str] = []
        cursor = index
        seen: set[int] = set()
        while True:
            parent_event = parents[cursor] if cursor < len(parents) else -1
            if parent_event < 0:
                break
            parent_ordinal = ordinal_of_open_event.get(parent_event)
            if parent_ordinal is None or parent_ordinal in seen:
                break
            seen.add(parent_ordinal)
            chain.append(types[parent_ordinal])
            cursor = parent_ordinal
        ancestry[node_id] = tuple(chain)
    return ancestry


def token_labels(
    boxes: torch.Tensor,
    observations: list[dict[str, Any]],
    ancestry: dict[str, tuple[str, ...]],
) -> dict[str, torch.Tensor]:
    """Per-token labels from the observation boxes covering each token.

    Every token starts at zero and only a covering observation lifts it,
    so `ancestry` must name every observed node: an observation whose
    ancestry is unknown would be labelled "not in a table" rather than
    "unknown". `page_spatial_labels` is what enforces that.
    """
    labels = {name: torch.zeros(boxes.shape[0]) for name in SPATIAL_LABELS}
    for observation in observations:
        box = observation.get("bbox_ppm")
        if not box or len(box) != 4:
            continue
        x0, y0, x1, y1 = (float(v) for v in box)
        overlaps = (
            (boxes[:, 0] < x1)
            & (boxes[:, 2] > x0)
            & (boxes[:, 1] < y1)
            & (boxes[:, 3] > y0)
        )
        if not bool(overlaps.any()):
            continue
        labels["has_text"][overlaps] = 1.0
        if "table" in ancestry.get(observation.get("node_id", ""), ()):
            labels["in_table"][overlaps] = 1.0
    return labels


class RidgeAccumulator:
    """Closed-form ridge over data too large to hold at once.

    Accumulates the normal equations `XᵀX` and `Xᵀy` (plus a bias column),
    so memory is O(d²) in the feature width no matter how many token rows
    are streamed through it.

    Two properties make this cheap, and missing either is expensive:

    - **`XᵀX` does not depend on the target.** Several labels over the
      same features share one Gram matrix; accumulating it per label costs
      a full k-times multiple of the dominant term for nothing.
    - **`XᵀX` is where all the work is** — 6.7 GFLOP per page at 5040x1153.
      In float64 on the CPU that is seconds per page; in float32 on the
      GPU it is under a millisecond. Only the final solve needs float64,
      and it runs once.
    """

    def __init__(
        self, features: int, targets: int = 1, *, device: str = "cpu"
    ) -> None:
        self.features = features
        self.targets = targets
        self.device = device
        size = features + 1
        self._gram = torch.zeros(size, size, dtype=torch.float32, device=device)
        self._moment = torch.zeros(size, targets, dtype=torch.float32, device=device)
        self.rows = 0

    def add(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate one chunk of rows.

        `targets` is (n,) for a single head or (n, k) for k heads sharing
        these features.
        """
        features = features.to(self.device, torch.float32)
        design = torch.cat(
            [features, torch.ones(features.shape[0], 1, device=self.device)], dim=1
        )
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        self._gram += design.T @ design
        self._moment += design.T @ targets.to(self.device, torch.float32)
        self.rows += int(features.shape[0])

    def column_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and sd of every accumulated feature column.

        Both come out of the normal equations themselves — the bias
        column of the design matrix carries the column sums and the row
        count — so the statistics cover EVERY row streamed through the
        accumulator, without a second pass over the pages. They are
        RECOVERED from those accumulated sums rather than recomputed from
        the rows: the sd in particular is a difference of large numbers,
        so it is accurate to the accumulation, not to the data.

        The sd uses the n-1 convention of `heads.standardize`, and a
        column with no variance keeps scale 1 there too, so it becomes
        all-zero rather than NaN.
        """
        if self.rows < 2:
            raise ValueError("column statistics need at least two rows")
        d = self.features
        gram = self._gram.double().cpu()
        n = float(self.rows)
        mean = gram[d, :d] / n
        variance = (torch.diagonal(gram)[:d] - n * mean**2) / (n - 1.0)
        std = torch.sqrt(torch.clamp(variance, min=0.0))
        std = torch.where(std > 1e-8, std, torch.ones_like(std))
        return mean, std

    def solve_standardized(
        self, *, alpha: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ridge weights over GLOBALLY standardised features, plus the
        mean and sd they were standardised by.

        Standardising with the first page's statistics makes the fitted
        head depend on which document happens to sort first, and a
        streaming fit cannot see the last page before it needs the first
        one's scale. It does not have to: standardising is affine, so the
        normal equations accumulated on raw rows transform algebraically
        into the ones a fully standardised fit would have produced.
        Centring also empties the bias column, which is why only the
        feature block is solved here.

        The algebra is exact; the arithmetic is not. Both the statistics
        and the transformed equations are recovered from sums accumulated
        in float32, not recomputed from the rows, so this agrees with a
        direct fit to floating-point tolerance rather than bit for bit —
        which is why `run` feeds it provisionally scaled rows.

        Returns `(weights, mean, std)`; apply the weights to
        `(rows - mean) / std`.
        """
        if not self.rows:
            raise ValueError("no rows accumulated")
        mean, std = self.column_stats()
        d, n = self.features, float(self.rows)
        gram = self._gram.double().cpu()
        moment = self._moment.double().cpu()
        # Z = (X - m)/s  =>  ZᵀZ = (XᵀX - n mmᵀ)/(s sᵀ), Zᵀ1 = 0
        core = (gram[:d, :d] - n * torch.outer(mean, mean)) / torch.outer(std, std)
        rhs = (moment[:d] - torch.outer(mean, moment[d])) / std[:, None]
        penalty = torch.eye(d, dtype=torch.float64) * alpha
        weights = torch.linalg.solve(core + penalty, rhs)
        bias = (moment[d] / n).unsqueeze(0)
        full = torch.cat([weights, bias], dim=0).float()
        return (
            full.squeeze(1) if self.targets == 1 else full,
            mean.float(),
            std.float(),
        )

    def solve(self, *, alpha: float = 1.0) -> torch.Tensor:
        """Ridge weights, bias last and unpenalised.

        Returns (d+1,) for a single head, (d+1, k) for several.
        """
        if not self.rows:
            raise ValueError("no rows accumulated")
        size = self.features + 1
        gram = self._gram.double().cpu()
        penalty = torch.eye(size, dtype=torch.float64) * alpha
        penalty[self.features, self.features] = 0.0
        weights = torch.linalg.solve(gram + penalty, self._moment.double().cpu())
        weights = weights.float()
        return weights.squeeze(1) if self.targets == 1 else weights


# ------------------------------------------------------------------ harness --


def page_spatial_labels(
    item: Any,
    page_index: int,
    grid_thw: tuple[int, int, int],
    token_count: int,
    *,
    task_kind: str,
) -> dict[str, torch.Tensor] | None:
    """Per-token labels for one page of a `TrainingItem`, or None.

    Returns None when the page carries no ground truth for a token:
    no observation geometry, no structural targets to resolve ancestry
    from, or an observed node the target bundle does not name. Every one
    of those would otherwise be labelled by omission — `has_text` 1 from
    the box, `in_table` 0 from an ancestry map that simply does not know
    — which is a fabricated negative, not a measurement.

    `task_kind` is not decoration. `token_labels` starts every token at
    zero and only the observed boxes lift it, so an UNMATCHED node reads
    as empty page. That is only defensible for the geometry task views,
    where `train_ready` requires every node mapped to rendered geometry
    (`gates.task_eligibility`); under any other task kind the premise is
    unverifiable here, and the refusal is loud rather than silent.
    """
    # private on purpose there, and the dependency is the point: this
    # module's zero-initialised labels are only sound inside that set
    from unflatten.dataset.gates.task_eligibility import _GEOMETRY_TASKS

    if task_kind not in _GEOMETRY_TASKS:
        raise ValueError(
            f"task kind {task_kind!r} does not require full geometry coverage "
            f"(one of {sorted(_GEOMETRY_TASKS)} does); an unmatched node would "
            "be labelled as empty page rather than as unknown"
        )
    sidecar = getattr(item, "observation_sidecar_data", None)
    if not sidecar:
        return None
    pages = sidecar.get("pages") or []
    page = next((p for p in pages if p.get("page_index") == page_index), None)
    if page is None:
        return None
    observations = page.get("observations") or []
    if not observations:
        return None
    ancestry = ancestor_types((item.targets or {}).get("structural", {}))
    if not ancestry:
        return None
    if any(
        observation.get("node_id") not in ancestry
        for observation in observations
        if observation.get("bbox_ppm")
    ):
        return None
    merge = infer_merge_size(grid_thw, token_count)
    return token_labels(token_boxes(grid_thw, merge_size=merge), observations, ancestry)


def evaluate_spatial(
    page_scores: Sequence[torch.Tensor],
    page_targets: Sequence[torch.Tensor],
    *,
    resamples: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Score a spatial head: AUROC plus its significance over PAGES.

    Both arguments are per-page tensors, and the grouping is the whole
    point. A page contributes thousands of tokens that are anything but
    independent — one table covers hundreds of them — so a noise sd
    derived from token counts is too small by roughly the square root of
    the tokens per page, and a 3-sigma threshold over it is no threshold
    at all. Significance is therefore estimated by resampling PAGES with
    replacement: the sd of the bootstrap AUROCs is the denominator.

    `sigma` is None when it is undefined — a single-class evaluation, or
    a bootstrap that never varies. Reporting 0.0 there would read as
    "measured, no signal".
    """
    from unflatten.probe.heads import roc_auc

    scores = torch.cat(list(page_scores)) if page_scores else torch.empty(0)
    targets = torch.cat(list(page_targets)) if page_targets else torch.empty(0)
    positives = int(targets.sum())
    area = roc_auc(targets, scores)
    result: dict[str, Any] = {
        "auroc": area,
        "tokens": int(targets.numel()),
        "pages": len(page_scores),
        "positive_rate": positives / max(1, targets.numel()),
        "sigma": None,
    }
    if area != area:
        return result

    generator = torch.Generator().manual_seed(seed)
    pages = len(page_scores)
    draws = []
    for _ in range(resamples):
        index = torch.randint(pages, (pages,), generator=generator).tolist()
        drawn = roc_auc(
            torch.cat([page_targets[i] for i in index]),
            torch.cat([page_scores[i] for i in index]),
        )
        if drawn == drawn:  # a resample can hold one class only
            draws.append(drawn)
    if len(draws) < 2:
        return result
    noise = float(torch.tensor(draws, dtype=torch.float64).std())
    if noise > 0.0:
        result["sigma"] = (area - 0.5) / noise
    return result
