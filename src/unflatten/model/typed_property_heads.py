"""The three typed property head families (RFC004 §24, §25.3).

Promoted verbatim-in-behaviour from
`unflatten.diagnostics.typed_property_head.heads` + `.masking`, whose
design was validated in the `spec-promotable-typed-head` ADR §4 and
carried, not re-derived, by that workflow's design §2.2/§2.4 (both under
`docs/plans/`; the paths are deliberately not spelled out -- no module in
`src/unflatten/model/` may name a design document).

One `nn.Module` subclass per value-domain family, not per pointer: a
single `EnumPropertyHead(num_classes, d_model)`, `BooleanPropertyHead
(d_model)` or `IntegerPropertyHead(d_model, config)` is instantiated once
per `(node_type, pointer)` in scope; the family module itself is
pointer-agnostic, and `typed_property_pointer_map.py` is what says which
pointers exist.

`StringPropertyHead` is DEFERRED: no string pointer is in the slice, so
building it now would be untested speculative code.

RFC004 §25.3 hard constraint: the model never emits a floating-point
value -- `IntegerPropertyHead.predict_value` always returns a
`torch.long` tensor, under BOTH `bins_plus_residual` and
`direct_regression`. Neither codec is pre-chosen here: which one S3 uses
is an open RFC005 ablation, so both stay config-selectable.

The `categorical-word-properties` design's hard constraint: `0` must be a
real class, never a code for "unknown" (48.7 % of `spacing` is
legitimately zero). `IntegerHeadConfig` enforces this structurally for
`bins_plus_residual`: a caller who says the pointer's distribution is
zero-heavy (`zero_fraction >= zero_heavy_threshold`) may not pass
`bin_edges` that collapse `0` into a bin wider than `max_zero_bin_width`.
The "zero-heavy" threshold itself is OPEN and is therefore a
caller-supplied parameter, never a hardcoded number.

This module ships **no optimizer** and authorizes no gradient update; it
neither calls the property head registry hash producer nor imports its
constant, and it mints no hash domain.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F


class IntegerTargetOutOfRangeError(ValueError):
    """A target value fell below the first `bin_edges` entry. `bin_edges`
    are caller-supplied, so a target the configured bins cannot represent
    is a configuration error and must raise, never be clamped into bin 0
    -- clamping would silently merge an out-of-range value with the
    legitimate bin-0 class."""


class ZeroHeavyBinEdgesError(ValueError):
    """`bin_edges` collapses `0` into a bin spanning a range wider than
    `max_zero_bin_width` for a pointer whose measured distribution is
    zero-heavy -- the structural check that keeps `0` from becoming a de
    facto code for "unknown" under a different name."""


def masked_mean(per_example_loss: torch.Tensor, known_mask: torch.Tensor) -> torch.Tensor:
    """The masked-mean reduction, shared verbatim across head families.

    `known_mask=False` means `no_supervision` (RFC003 §29.1/§29.5) -- it
    must NEVER be treated as a fabricated `0`, `abstained`,
    `not_applicable` or `conflict`. Every head family's `loss()` reduces
    its per-example loss through this one function, so the reduction (and
    its zero-gradient guarantee) is defined exactly once instead of
    re-derived, and possibly re-broken, per subclass.

    A batch with zero known examples for a pointer must return a defined
    `0`-valued loss -- via `clamp(min=1)` on the denominator -- rather
    than a `NaN` from a `0/0` divide: "silently NaN" is worse than
    "silently contributes nothing".

    The load-bearing property is stronger than "the forward value is
    correct at `known_mask=False`": a masked example must contribute
    EXACTLY ZERO gradient, not merely a zero-valued forward term that
    still backprops through some other tensor alias.
    `tests/model/test_typed_property_heads.py::test_masked_mean_adversarial_target_mutation_leaves_the_gradient_identical`
    is that proof, and it moved here with this function -- it was watched
    to fail against a plain, unmasked `.mean()` before being trusted.

    `known_mask` is required, not optional and not defaulted to `True` --
    callers must decide supervision explicitly for every example.
    """
    mask = known_mask.to(per_example_loss.dtype)
    return (per_example_loss * mask).sum() / mask.sum().clamp(min=1)


#: Trailing `/<index>` or `/*` segments of a repeated-field pointer.
_TRAILING_INDEX = re.compile(r"(?:/(?:\*|\d+))+$")


def scale_for_pointer(pointer: str) -> int:
    """DEC-032: the `direct_regression` learning-unit scale for a
    node-relative RFC001 pointer, derived from its unit suffix rather than
    listed -- a list drifts from the schema's own naming silently, a
    derivation cannot.

    `_mpt` (millipoints): 10_000. `_ppm` (parts-per-million multiplier):
    1_000_000. Anything else (counts, numbering starts, levels, spans):
    1 -- already learnable in raw units, DEC-032 does not touch them.

    Trailing index or wildcard segments (`/columns/widths_mpt/*`,
    `/columns/widths_mpt/3`) name elements of a repeated unit field; the
    unit is the suffix before them.
    """
    stem = _TRAILING_INDEX.sub("", pointer)
    if stem.endswith("_mpt"):
        return 10_000
    if stem.endswith("_ppm"):
        return 1_000_000
    return 1


@dataclass(frozen=True)
class IntegerHeadConfig:
    """Both `bins_plus_residual` and `direct_regression` are fully
    expressible -- neither is pre-decided; the RFC005 S3 ablation chooses
    between them later.

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
    #: DEC-032: `direct_regression`'s learning-unit scale. A head predicts
    #: `target / scale` and `predict_value` reconstructs `round(prediction
    #: * scale)`; without this, a `_mpt`-scale target (e.g. page width
    #: 612 000) leaves the linear head's SmoothL1 gradient effectively
    #: unlearnable at typical S3 learning rates. `bins_plus_residual`
    #: already operates directly in raw target units and refuses any
    #: `scale != 1` (DEC-032 item 4: unchanged).
    scale: int = 1
    #: R1 (RFC004-ERRATA-004 item R1, draft RFC004-ERRATA-006), opt-in and
    #: `bins_plus_residual`-only, default `False`. `False` is byte-identical
    #: to pre-R1 behaviour: `effective_bin_edges()` returns `bin_edges`
    #: unchanged. `True` prepends a dedicated width-1 bin for the value `0`
    #: -- the MEASURED bimodal `{0, W}` width distribution (table/cell
    #: width, RFC004-ERRATA-004 R1) means a caller-chosen bin that merges 0
    #: with nearby small nonzero values is exactly the representational
    #: limit R1 names, and a singleton bin structurally rules it out
    #: instead of relying on the caller to hand-tune `bin_edges` correctly.
    #: `bin_edges` itself must then describe ONLY the nonzero region
    #: (starting at `>= 1`, never containing `0`) -- the class inserts `0`
    #: itself, so a caller-supplied `0` would be ambiguous with the
    #: inserted bin.
    explicit_zero_bin: bool = False

    def __post_init__(self) -> None:
        if self.scale < 1:
            raise ValueError(f"scale must be >= 1, got {self.scale!r}")
        if self.strategy == "bins_plus_residual":
            if self.scale != 1:
                raise ValueError(
                    "bins_plus_residual operates directly in raw target "
                    f"units (DEC-032); got scale={self.scale!r}, expected 1"
                )
            if self.bin_edges is None or len(self.bin_edges) < 1:
                raise ValueError(
                    "bins_plus_residual requires at least one bin_edges entry"
                )
            if list(self.bin_edges) != sorted(set(self.bin_edges)):
                raise ValueError(
                    f"bin_edges must be strictly increasing, got {self.bin_edges!r}"
                )
            if self.explicit_zero_bin:
                if 0 in self.bin_edges:
                    raise ValueError(
                        "explicit_zero_bin=True prepends its own dedicated "
                        f"zero bin; bin_edges={self.bin_edges!r} must describe "
                        "only the nonzero region and must not itself contain 0"
                    )
                if self.bin_edges[0] < 1:
                    raise ValueError(
                        "explicit_zero_bin=True requires bin_edges to start "
                        f">= 1 (the nonzero region), got {self.bin_edges!r}"
                    )
            if len(self.effective_bin_edges()) < 2:
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
            if self.explicit_zero_bin:
                raise ValueError(
                    "explicit_zero_bin only applies to bins_plus_residual"
                )
        else:
            raise ValueError(f"unknown strategy {self.strategy!r}")

    def effective_bin_edges(self) -> tuple[int, ...]:
        """The bin edges the head actually builds/decodes against.

        Identical to `bin_edges` unless `explicit_zero_bin` is set, in
        which case a dedicated width-1 bin for the value `0` is prepended
        -- so `0` always decodes to that bin exactly, never merged with a
        caller-chosen range. `False` (the default) makes this an identity
        function, which is what keeps every existing `bin_edges`-based
        config byte-identical."""
        if not self.explicit_zero_bin:
            assert self.bin_edges is not None
            return self.bin_edges
        assert self.bin_edges is not None
        return (0, 1, *self.bin_edges)

    def _check_zero_heavy(self) -> None:
        if self.zero_fraction is None or self.zero_heavy_threshold is None:
            return
        if self.zero_fraction < self.zero_heavy_threshold:
            return
        # zero-heavy: 0 must be its own edge, and the bin it opens must
        # not span a large nonzero range.
        edges = self.effective_bin_edges()
        if 0 not in edges:
            raise ZeroHeavyBinEdgesError(
                f"pointer is zero-heavy (zero_fraction={self.zero_fraction!r} >= "
                f"zero_heavy_threshold={self.zero_heavy_threshold!r}) but 0 is "
                f"not one of bin_edges={edges!r}"
            )
        if self.max_zero_bin_width is None:
            raise ValueError(
                "zero_fraction/zero_heavy_threshold were given but "
                "max_zero_bin_width was not -- the zero-heavy check has "
                "nothing to compare the zero bin's width against"
            )
        zero_position = edges.index(0)
        if zero_position + 1 < len(edges):
            zero_bin_width = edges[zero_position + 1] - 0
        else:
            # 0 is the last edge: the bin it opens is unbounded above,
            # which is unconditionally "a large nonzero range".
            zero_bin_width = None
        if zero_bin_width is None or zero_bin_width > self.max_zero_bin_width:
            raise ZeroHeavyBinEdgesError(
                f"bin_edges={edges!r} collapses 0 into a bin of width "
                f"{zero_bin_width!r}, which exceeds max_zero_bin_width="
                f"{self.max_zero_bin_width!r} for a zero-heavy pointer"
            )


@dataclass
class IntegerBinsPrediction:
    bin_logits: torch.Tensor
    residual: torch.Tensor


class TypedPropertyHead(nn.Module, ABC):
    """Abstract base. `forward` returns a family-specific payload; `loss`
    returns a `known_mask`-weighted masked scalar."""

    @abstractmethod
    def forward(self, anchor: torch.Tensor):  # noqa: ANN201 -- payload varies per family
        raise NotImplementedError

    @abstractmethod
    def loss(self, prediction, target: torch.Tensor, known_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BooleanPropertyHead(TypedPropertyHead):
    """True/false. `unknown` is carried entirely by `known_mask`, never a
    third class."""

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
    """A closed in-domain vocabulary, one instance per pointer's domain."""

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


class CardinalityHead(EnumPropertyHead):
    """The element count of a repeated container (design `spec-typed-
    heads-missing-families` §4.2): classification over `0..k_max`, never a
    regression -- an off-by-one changes structure (a missing or invented
    element), so the distribution is over discrete counts, not a
    continuous value. Reuses `EnumPropertyHead`'s `nn.Linear` and masked-CE
    `loss()` verbatim (`num_classes = k_max + 1`); the only addition is
    `predict_count`, which names what the argmax class index MEANS here --
    the count itself, not a position in an unrelated schema enum.

    A missing count is never encoded as class `0` by this head -- that
    decision belongs to the caller deriving supervision (`zero_count.v0`,
    `s4_typed_loss.collate_typed`), not to the head, which has no way to
    tell "count is 0" from "count is unknown" on its own.
    """

    def __init__(self, *, d_model: int, k_max: int) -> None:
        super().__init__(d_model=d_model, num_classes=k_max + 1)
        self.k_max = k_max

    def predict_count(self, prediction: torch.Tensor) -> torch.Tensor:
        """The decoded element count -- `argmax` over the `k_max + 1`
        classes, always `torch.long`."""
        return prediction.argmax(dim=-1)


class PaletteHead(EnumPropertyHead):
    """The shared closed-palette colour classification head (design `spec-
    typed-heads-missing-families` §5.1/§10.1 T4), one instance per colour
    pointer: domain = `("auto", *COLOR_PALETTE_V0)` plus a trailing `"null"`
    class ONLY when `nullable` (`typed_head_config.palette_domain`, which
    this class does not import -- it only needs the CLASS COUNT, exactly
    like `CardinalityHead` only needs `k_max`; the label<->index mapping
    is the caller's job, the same split `EnumPropertyHead`/`enum_domain`
    already use).

    R9 (design §9): a non-nullable pointer's head has `palette_size + 1`
    outputs, never `palette_size + 2` -- one FEWER class, not a masked-out
    one, so `null` cannot be decoded for it structurally, not merely by
    convention.
    """

    def __init__(self, *, d_model: int, palette_size: int, nullable: bool) -> None:
        self.palette_size = palette_size
        self.nullable = nullable
        num_classes = palette_size + (2 if nullable else 1)
        super().__init__(d_model=d_model, num_classes=num_classes)


class IntegerPropertyHead(TypedPropertyHead):
    """`bins_plus_residual` or `direct_regression`, selected by
    `config.strategy` -- neither is pre-decided here."""

    def __init__(self, *, d_model: int, config: IntegerHeadConfig) -> None:
        super().__init__()
        self.config = config
        if config.strategy == "bins_plus_residual":
            n_bins = len(config.effective_bin_edges())
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
        effective_edges = self.config.effective_bin_edges()
        edges = torch.tensor(effective_edges, dtype=target.dtype, device=target.device)
        # rightmost edge <= target
        bin_id = torch.searchsorted(edges, target, right=True) - 1
        if bool((bin_id < 0).any()):
            offending = target[bin_id < 0]
            raise IntegerTargetOutOfRangeError(
                f"target value(s) {offending.tolist()!r} fall below the first "
                f"bin edge {effective_edges[0]!r}; bin_edges are caller-"
                "supplied, so this is a configuration error, not data to "
                "silently absorb into bin 0"
            )
        return bin_id

    def _bin_widths(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """One width per bin. Bin `i < n - 1` has width `edges[i+1] -
        edges[i]`. The last bin is open-ended above; its width is DEFINED
        as the previous bin's width, never an invented magic number --
        `IntegerHeadConfig` refuses `bins_plus_residual` with fewer than
        two effective bin edges precisely because this definition needs a
        previous bin to borrow from."""
        edges = torch.tensor(self.config.effective_bin_edges(), dtype=dtype, device=device)
        widths = edges[1:] - edges[:-1]
        return torch.cat([widths, widths[-1:]])

    def _reconstruct_value(
        self, bin_id: torch.Tensor, raw_residual: torch.Tensor
    ) -> torch.Tensor:
        """The continuous reconstructed value for `bin_id`.

        The tie fix, carried with the code it fixes (`bin_edges=(1, 2,
        4)`): an earlier version centered on the CONTINUOUS bin span
        (`edge_i + width/2`) and then tried to carve out a
        0.5-plus-epsilon rounding margin from it. For a width-1 bin that
        center is `edge_i + 0.5` -- already an exact round-half-to-even
        tie -- and the margin's own clamp forced the residual scale to 0,
        so the tie was returned UNCONDITIONALLY and `torch.round` sent it
        to `edge_i + 1`, one bin over, on every input. No epsilon fixes a
        tie that sits ON the only valid value.

        The fix (option b, chosen over a floor-based option a): center on
        the DISCRETE set of integers the bin actually contains --
        `{edge_i, ..., edge_i + width - 1}` -- whose true midpoint is
        `edge_i + (width - 1) / 2`, not `edge_i + width / 2`. Scaling the
        residual by that same `half_span` means `center +/- half_span`
        lands EXACTLY on `edge_i` and `edge_i + width - 1` -- the bin's own
        integer boundaries -- even at `tanh` saturation (+/-1.0 in
        float32), with no epsilon needed. A floor-based option-a rewrite
        (`edge_i + floor(clamp(...))`) was considered instead but
        rejected: `_reconstruct_value` also runs inside `loss()`'s
        residual term, and `floor` kills the residual head's gradient
        almost everywhere -- the same problem `loss()`'s comment on
        `direct_regression` already flags for `round`. This form keeps the
        whole path differentiable (only `torch.tanh`, no floor/round), so
        it is preferred here.

        Proof the value can never leave the bin: for any `x` bounded to a
        CLOSED interval `[n, m]` with integer `n <= m`, `round(x)` is also
        in `[n, m]` (`round` is monotonic and `round(n) == n`,
        `round(m) == m`). Since `|tanh(raw_residual)| <= 1` for every
        FINITE `raw_residual`, `center +/- half_span` is always within
        `[edge_i, edge_i + width - 1]` by construction -- so is its
        round. For `width == 1`, `half_span == 0`, so the residual is
        multiplied away and `edge_i` is returned exactly, which is the
        bin's only valid value. This is the only bounding mechanism -- not
        a fallback clamp -- so it is what `predict_value` and `loss()`
        both go through.

        The contract is finite-in, not NaN-in: `tanh(nan) == nan`, so a
        NaN residual makes the reconstruction NaN and `predict_value`'s
        integer cast an out-of-contract value (INT64_MIN on x86 in
        practice). NaN is a model-divergence signal the production path
        never reaches -- masked loss keeps every supervised term finite --
        so the contract names finite inputs rather than asserting a
        NaN bound that does not hold (R5, mirrored verbatim from the
        diagnostic copy)."""
        edges = torch.tensor(
            self.config.effective_bin_edges(),
            dtype=raw_residual.dtype,
            device=raw_residual.device,
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
        # direct_regression: the head learns in scaled units (DEC-032), so
        # reconstruction must undo the same scale before rounding to the
        # raw-unit integer RFC004 §25.3 requires.
        return torch.round(prediction * self.config.scale).long()

    def predict_continuous(self, prediction) -> torch.Tensor:
        """The differentiable reconstructed value `predict_value` rounds
        away -- kept as a separate method so a consistency term (R3,
        RFC004-ERRATA-004 item R3, draft RFC004-ERRATA-006) can backprop
        through a head's own prediction. `predict_value` itself must stay
        as-is: RFC004 §25.3 requires the emitted AST never sees a float,
        which is exactly what the `.long()` cast there enforces."""
        if self.config.strategy == "bins_plus_residual":
            assert isinstance(prediction, IntegerBinsPrediction)
            bin_id = prediction.bin_logits.argmax(dim=-1)
            return self._reconstruct_value(bin_id, prediction.residual)
        return prediction * self.config.scale

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
        # almost everywhere). The target is divided by `config.scale`
        # (DEC-032) so the head learns in scaled units instead of raw
        # milli-units, where SmoothL1's +/-1 gradient cap leaves AdamW
        # unable to move the prediction meaningfully.
        per_example = F.smooth_l1_loss(
            prediction, target.to(prediction.dtype) / self.config.scale, reduction="none"
        )
        return masked_mean(per_example, known_mask)


class SlotHead(TypedPropertyHead):
    """`k_slots` independent instances of ONE value-domain family, one per
    element position of a bounded repeated container (design `spec-typed-
    heads-missing-families` §4.3/§10.1 T3: `paragraph:/layout/tabs/*/...`,
    `table:/columns/*/width_mpt`, `section:/columns/widths_mpt/*`).

    `slots` is an `nn.ModuleList` named EXACTLY that -- `s4_train.
    replaced_keys`/`declared_new_keys` (design §7 rev 3, §9 R14a-d, §10.1
    T8) derive warm-start transplant targets PURELY from `model.
    state_dict()` key strings of the shape `"<...>.slots.<i>.<leaf-
    suffix>"`, without ever importing this class. Renaming this attribute,
    or building the slot list some other way, silently breaks that
    contract -- T8 landed first and this class only has to MATCH it.

    Slot `i`'s leaf module is built by calling `inner_factory()` once per
    slot: every slot shares the pointer's family/domain/config (the SAME
    class the pre-T3 single-instance head used for that pointer) but owns
    its own parameters. Slot `i` corresponds to the i-th element of the
    source list in importer order (design §3: RFC001 does not define an
    order for these containers; the importer's own `findall` order is the
    fact this project already committed to when it wrote the cache).
    """

    def __init__(self, *, inner_factory: Callable[[], TypedPropertyHead], k_slots: int) -> None:
        super().__init__()
        if k_slots < 1:
            raise ValueError(f"k_slots must be >= 1, got {k_slots!r}")
        self.k_slots = k_slots
        self.slots = nn.ModuleList(inner_factory() for _ in range(k_slots))

    def forward(self, anchor: torch.Tensor) -> list[Any]:
        """One payload per slot, in slot order -- each entry is exactly
        what that slot's own family `forward` would return for a single-
        instance head over the same anchor batch."""
        return [slot(anchor) for slot in self.slots]

    def loss(
        self, predictions: list[Any], target: torch.Tensor, known_mask: torch.Tensor
    ) -> torch.Tensor:
        """`target`/`known_mask` are `[N, k_slots]` -- one column per slot,
        `collate_typed` builds them from the container's element index,
        never from a slot's own prediction (design §4.4: "count zo
        záznamu kontajnera, nie z predikcie").

        Design §4.4's "kľúč = jeden hlas" macro-averaging rule applies a
        second time HERE, inside a single pointer key: this whole `SlotHead`
        casts exactly one vote in `s4_typed_loss`'s macro `L_typed` mean, so
        its own `k_slots` slots must not do so again -- summed only over
        slots with `>= 1` supervised example THIS batch, divided by that
        contributing count, never by `k_slots`. A slot with zero supervised
        examples this batch is excluded from both the sum and the divisor
        (R7): including it as a zero-loss term would still be mathematically
        a no-op for the mean's VALUE only because its own `masked_mean` is
        already exactly zero, but the caller-facing contract is symmetry
        with `masked_mean` itself -- a masked entry contributes to neither
        term, not just to the numerator.
        """
        per_slot_losses = [
            slot.loss(predictions[i], target[:, i], known_mask[:, i])
            for i, slot in enumerate(self.slots)
            if bool(known_mask[:, i].any())
        ]
        if not per_slot_losses:
            raise ValueError(
                "SlotHead.loss: no supervised slot column in this batch -- the "
                "caller must not invoke loss() on an all-unsupervised batch "
                "(the same contract s4_typed_loss.typed_losses already "
                "enforces generically via its supervised_count == 0 guard)"
            )
        return torch.stack(per_slot_losses).mean()
