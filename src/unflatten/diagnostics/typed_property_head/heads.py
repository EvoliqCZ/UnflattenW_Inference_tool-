"""Design §3/§4 — the three slice-1 typed property head families.

One `nn.Module` subclass per value-domain family, not per pointer (design
§3): a single `EnumPropertyHead(num_classes, d_model)`,
`BooleanPropertyHead(d_model)` or `IntegerPropertyHead(d_model, config)`
is instantiated once per `(node_type, pointer)` in scope; the family
module itself is pointer-agnostic, and `pointer_map.py` (design §3.1) is
what says which pointers exist.

`StringPropertyHead` is DEFERRED: no string pointer is in slice 1
(design §1.1), so building it now would be untested speculative code
(Karpathy simplicity-first).

RFC004 §25.3 hard constraint: the model never emits a floating-point
value -- `IntegerPropertyHead.predict_value` always returns a
`torch.long` tensor, under BOTH `bins_plus_residual` and
`direct_regression`.

`categorical-word-properties-design.md` hard constraint: `0` must be a
real class, never a code for "unknown" (48.7% of `spacing` is
legitimately zero). `IntegerHeadConfig` enforces this structurally for
`bins_plus_residual`: a caller who says the pointer's distribution is
zero-heavy (`zero_fraction >= zero_heavy_threshold`) may not pass
`bin_edges` that collapse `0` into a bin wider than `max_zero_bin_width`.
The "zero-heavy" threshold itself is OPEN (design §4) and is therefore a
caller-supplied parameter, never a hardcoded number.

Masking (design §6's masked-mean reduction) lives in `masking.py`, not
here -- every head family's `loss()` calls `masking.masked_mean` so the
reduction (and its zero-gradient guarantee, proved in
`tests/diagnostics/typed_property_head/test_masking.py`) is defined
exactly once.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from unflatten.diagnostics.typed_property_head.masking import masked_mean


class IntegerTargetOutOfRangeError(ValueError):
    """A target value fell below the first `bin_edges` entry. `bin_edges`
    are caller-supplied, so a target the configured bins cannot represent
    is a configuration error and must raise, never be clamped into bin 0
    (finding 6) -- clamping would silently merge an out-of-range value
    with the legitimate bin-0 class."""


class ZeroHeavyBinEdgesError(ValueError):
    """`bin_edges` collapses `0` into a bin spanning a range wider than
    `max_zero_bin_width` for a pointer whose measured distribution is
    zero-heavy -- the structural check design §4 requires so `0` never
    becomes a de facto code for "unknown" under a different name."""


@dataclass(frozen=True)
class IntegerHeadConfig:
    """Design §4. Both `bins_plus_residual` and `direct_regression` are
    fully expressible -- neither is pre-decided; the RFC004 §52.1 /
    RFC005 S3 ablation chooses between them later.

    `zero_fraction`/`zero_heavy_threshold`/`max_zero_bin_width` are all
    optional and only apply to `bins_plus_residual`: passing any of them
    without the others skips the zero-heavy check (there is nothing to
    compare), which is a deliberate opt-in, not a silent default that
    fires unexpectedly.
    """

    strategy: Literal["bins_plus_residual", "direct_regression"]
    unit: Literal["mpt"] = "mpt"
    bin_edges: tuple[int, ...] | None = None
    zero_fraction: float | None = None
    zero_heavy_threshold: float | None = None
    max_zero_bin_width: int | None = None

    def __post_init__(self) -> None:
        if self.strategy == "bins_plus_residual":
            if self.bin_edges is None or len(self.bin_edges) < 1:
                raise ValueError(
                    "bins_plus_residual requires at least one bin_edges entry"
                )
            if list(self.bin_edges) != sorted(set(self.bin_edges)):
                raise ValueError(
                    f"bin_edges must be strictly increasing, got {self.bin_edges!r}"
                )
            if len(self.bin_edges) < 2:
                raise ValueError(
                    "bins_plus_residual bounds its residual to the argmax bin's "
                    "width (RFC004 §25.3); the last bin is open-ended above, so "
                    "its width is defined as the previous bin's width -- a "
                    f"single-entry bin_edges={self.bin_edges!r} has no previous "
                    "bin to borrow a width from"
                )
            self._check_zero_heavy()
        elif self.strategy == "direct_regression":
            if self.bin_edges is not None:
                raise ValueError(
                    "direct_regression does not use bin_edges; got "
                    f"{self.bin_edges!r}"
                )
        else:
            raise ValueError(f"unknown strategy {self.strategy!r}")

    def _check_zero_heavy(self) -> None:
        if self.zero_fraction is None or self.zero_heavy_threshold is None:
            return
        if self.zero_fraction < self.zero_heavy_threshold:
            return
        # zero-heavy: 0 must be its own edge, and the bin it opens must
        # not span a large nonzero range.
        assert self.bin_edges is not None
        if 0 not in self.bin_edges:
            raise ZeroHeavyBinEdgesError(
                f"pointer is zero-heavy (zero_fraction={self.zero_fraction!r} >= "
                f"zero_heavy_threshold={self.zero_heavy_threshold!r}) but 0 is "
                f"not one of bin_edges={self.bin_edges!r}"
            )
        if self.max_zero_bin_width is None:
            raise ValueError(
                "zero_fraction/zero_heavy_threshold were given but "
                "max_zero_bin_width was not -- the zero-heavy check has "
                "nothing to compare the zero bin's width against"
            )
        zero_position = self.bin_edges.index(0)
        if zero_position + 1 < len(self.bin_edges):
            zero_bin_width = self.bin_edges[zero_position + 1] - 0
        else:
            # 0 is the last edge: the bin it opens is unbounded above,
            # which is unconditionally "a large nonzero range".
            zero_bin_width = None
        if zero_bin_width is None or zero_bin_width > self.max_zero_bin_width:
            raise ZeroHeavyBinEdgesError(
                f"bin_edges={self.bin_edges!r} collapses 0 into a bin of width "
                f"{zero_bin_width!r}, which exceeds max_zero_bin_width="
                f"{self.max_zero_bin_width!r} for a zero-heavy pointer"
            )


@dataclass
class IntegerBinsPrediction:
    bin_logits: torch.Tensor
    residual: torch.Tensor


class TypedPropertyHead(nn.Module, ABC):
    """Design §3 abstract base. `forward` returns a family-specific
    payload; `loss` returns a masked scalar (design §6)."""

    @abstractmethod
    def forward(self, anchor: torch.Tensor):  # noqa: ANN201 -- payload varies per family
        raise NotImplementedError

    @abstractmethod
    def loss(self, prediction, target: torch.Tensor, known_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BooleanPropertyHead(TypedPropertyHead):
    """True/false. `unknown` is carried entirely by `known_mask`, never a
    third class (design §3)."""

    def __init__(self, *, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        return self.linear(anchor).squeeze(-1)

    def loss(
        self, prediction: torch.Tensor, target: torch.Tensor, known_mask: torch.Tensor
    ) -> torch.Tensor:
        per_example = F.binary_cross_entropy_with_logits(
            prediction, target.to(prediction.dtype), reduction="none"
        )
        return masked_mean(per_example, known_mask)


class EnumPropertyHead(TypedPropertyHead):
    """A closed in-domain vocabulary, one instance per pointer's domain
    (design §3)."""

    def __init__(self, *, d_model: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        return self.linear(anchor)

    def loss(
        self, prediction: torch.Tensor, target: torch.Tensor, known_mask: torch.Tensor
    ) -> torch.Tensor:
        per_example = F.cross_entropy(prediction, target.long(), reduction="none")
        return masked_mean(per_example, known_mask)


class IntegerPropertyHead(TypedPropertyHead):
    """Design §4: `bins_plus_residual` or `direct_regression`, selected by
    `config.strategy` -- neither is pre-decided here."""

    def __init__(self, *, d_model: int, config: IntegerHeadConfig) -> None:
        super().__init__()
        self.config = config
        if config.strategy == "bins_plus_residual":
            assert config.bin_edges is not None
            n_bins = len(config.bin_edges)
            self.bin_logits_head = nn.Linear(d_model, n_bins)
            self.residual_head = nn.Linear(d_model, 1)
        else:
            self.regression_head = nn.Linear(d_model, 1)

    def forward(self, anchor: torch.Tensor):
        if self.config.strategy == "bins_plus_residual":
            return IntegerBinsPrediction(
                bin_logits=self.bin_logits_head(anchor),
                residual=self.residual_head(anchor).squeeze(-1),
            )
        return self.regression_head(anchor).squeeze(-1)

    def _bin_id_of(self, target: torch.Tensor) -> torch.Tensor:
        assert self.config.bin_edges is not None
        edges = torch.tensor(self.config.bin_edges, dtype=target.dtype, device=target.device)
        # rightmost edge <= target
        bin_id = torch.searchsorted(edges, target, right=True) - 1
        if bool((bin_id < 0).any()):
            offending = target[bin_id < 0]
            raise IntegerTargetOutOfRangeError(
                f"target value(s) {offending.tolist()!r} fall below the first "
                f"bin edge {self.config.bin_edges[0]!r}; bin_edges are caller-"
                "supplied, so this is a configuration error, not data to "
                "silently absorb into bin 0"
            )
        return bin_id

    def _bin_widths(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """One width per bin. Bin `i < n - 1` has width `edges[i+1] -
        edges[i]`. The last bin is open-ended above (design §4); its width
        is DEFINED as the previous bin's width, never an invented magic
        number -- `IntegerHeadConfig` refuses `bins_plus_residual` with
        fewer than two `bin_edges` entries precisely because this
        definition needs a previous bin to borrow from (finding 3)."""
        assert self.config.bin_edges is not None
        edges = torch.tensor(self.config.bin_edges, dtype=dtype, device=device)
        widths = edges[1:] - edges[:-1]
        return torch.cat([widths, widths[-1:]])

    def _reconstruct_value(
        self, bin_id: torch.Tensor, raw_residual: torch.Tensor
    ) -> torch.Tensor:
        """The continuous reconstructed value for `bin_id`.

        Bug fix (deep review, `bin_edges=(1, 2, 4)`): the previous version
        centered on the CONTINUOUS bin span (`edge_i + width/2`) and then
        tried to carve out a 0.5-plus-epsilon rounding margin from it. For
        a width-1 bin that center is `edge_i + 0.5` -- already an exact
        round-half-to-even tie -- and the margin's own clamp forced the
        residual scale to 0, so the tie was returned UNCONDITIONALLY and
        `torch.round` sent it to `edge_i + 1`, one bin over, on every
        input. No epsilon fixes a tie that sits ON the only valid value.

        The fix (option b, chosen over a floor-based option a): center on
        the DISCRETE set of integers the bin actually contains --
        `{edge_i, ..., edge_i + width - 1}` -- whose true midpoint is
        `edge_i + (width - 1) / 2`, not `edge_i + width / 2`. Scaling the
        residual by that same `half_span` means `center +/- half_span`
        lands EXACTLY on `edge_i` and `edge_i + width - 1` -- the bin's own
        integer boundaries -- even at `tanh` saturation (+/-1.0 in
        float32), with no epsilon needed. A floor-based option-a rewrite
        (`edge_i + floor(clamp(...))`) was considered instead but rejected:
        `_reconstruct_value` also runs inside `loss()`'s residual term, and
        `floor` kills the residual head's gradient almost everywhere -- the
        same problem `loss()`'s comment on `direct_regression` already
        flags for `round`. This form keeps the whole path differentiable
        (only `torch.tanh`, no floor/round), so it is preferred here.

        Proof the value can never leave the bin: for any `x` bounded to a
        CLOSED interval `[n, m]` with integer `n <= m`, `round(x)` is also
        in `[n, m]` (`round` is monotonic and `round(n) == n`,
        `round(m) == m`). Since `|tanh(raw_residual)| <= 1` for every
        FINITE `raw_residual`, `center +/- half_span` is always within
        `[edge_i, edge_i + width - 1]` by construction -- so is its
        round. For `width == 1`, `half_span == 0`, so the residual is
        multiplied away and `edge_i` is returned exactly, which is the
        bin's only valid value. This is the only bounding mechanism -- not
        a fallback clamp -- so it is what predict_value and loss() both go
        through.

        The contract is finite-in, not NaN-in: `tanh(nan) == nan`, so a
        NaN residual makes the reconstruction NaN and `predict_value`'s
        integer cast an out-of-contract value (INT64_MIN on x86 in
        practice). NaN is a model-divergence signal the production path
        never reaches -- masked loss keeps every supervised term finite --
        so the proof names finite inputs rather than claiming a NaN bound
        that does not hold."""
        assert self.config.bin_edges is not None
        edges = torch.tensor(
            self.config.bin_edges, dtype=raw_residual.dtype, device=raw_residual.device
        )
        widths = self._bin_widths(raw_residual.dtype, raw_residual.device)
        half_span = (widths[bin_id] - 1) / 2
        center = edges[bin_id] + half_span
        bounded_residual = torch.tanh(raw_residual) * half_span
        return center + bounded_residual

    def predict_value(self, prediction) -> torch.Tensor:
        """The reconstructed integer value -- always `torch.long`, in
        either strategy (RFC004 §25.3: no floating-point value emitted)."""
        if self.config.strategy == "bins_plus_residual":
            assert isinstance(prediction, IntegerBinsPrediction)
            bin_id = prediction.bin_logits.argmax(dim=-1)
            value = self._reconstruct_value(bin_id, prediction.residual)
            return torch.round(value).long()
        return torch.round(prediction).long()

    def loss(
        self, prediction, target: torch.Tensor, known_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.config.strategy == "bins_plus_residual":
            assert isinstance(prediction, IntegerBinsPrediction)
            bin_id = self._bin_id_of(target)
            classification = F.cross_entropy(
                prediction.bin_logits, bin_id, reduction="none"
            )
            value = self._reconstruct_value(bin_id, prediction.residual)
            residual_loss = F.smooth_l1_loss(
                value, target.to(value.dtype), reduction="none"
            )
            per_example = classification + residual_loss
            return masked_mean(per_example, known_mask)

        # direct_regression: continuous SmoothL1 on the raw prediction so
        # the loss stays differentiable; the "no float into AST" contract
        # is enforced at emission time by predict_value's rounding, not by
        # rounding inside the loss (rounding here would zero the gradient
        # almost everywhere).
        per_example = F.smooth_l1_loss(
            prediction, target.to(prediction.dtype), reduction="none"
        )
        return masked_mean(per_example, known_mask)
