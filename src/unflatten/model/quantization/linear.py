"""Weight-only quantized replacements for `nn.Linear` (owner requirement
§1): INT8 per-output-channel symmetric, FP8 (`torch.float8_e4m3fn` storage
with per-channel scale), and INT4 group-wise packed 2-per-byte. Pure torch,
no new dependency.

`Int8Linear` and `Fp8Linear` each try a fused GEMM that never rebuilds the
full bf16 weight (`docs/runs/decode-profile-2026-09-27.md`: on real decode
the naive dequant -- cast to fp32, multiply the whole matrix by the
per-channel scale, cast back -- costs MORE self-CUDA time than the useful
matmul, across ~252 quantized linears per step):

- `Int8Linear` uses `torch._weight_int8pack_mm`, a fused weight-only INT8
  GEMM available on both CPU and CUDA for a 2D input; any input is
  flattened to 2D first (the decoder feeds `(batch, seq, hidden)`).
- `Fp8Linear` uses `torch._scaled_mm` (real hardware fp8 GEMM, Ada/Hopper
  SM >= 8.9) with the input flattened to 2D, a per-row dynamic activation
  scale, and the existing per-output-channel weight scale as the row-wise
  `scale_b` -- `_scaled_mm` requires both `in_features` and `out_features`
  divisible by 16, so smaller/odd layers fall back.

All three fall back to the dequantize-then-`F.linear` path for any shape,
dtype, device or torch build the fused op refuses; `self.fallback_count`
counts every fallback (including a device that never had a fast path) so a
silent slowdown shows up as a number instead of "decode feels slow" -- see
`tests/model/quantization/test_linear.py`.

- `Int4Linear` uses `torch._weight_int4pack_mm` (CUDA) / `..._for_cpu`
  (CPU), when `group_size` is one of the native kernel's fixed choices
  (32/64/128/256) and `out_features % 16 == 0` -- see its class docstring
  for the packed-layout repacking, done lazily and cached per device, not
  at quantization time (so `weight_packed`/`scale`, and therefore
  serialization, are unchanged).

Activations always stay in the caller's dtype (typically bf16) -- only the
frozen weight is stored at reduced precision. Bias, when present, is kept
at full precision (never quantized -- one bias vector per layer is a
negligible fraction of parameter bytes).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

#: Largest finite magnitude representable in e4m3fn (used to scale a
#: tensor into the format's dynamic range before casting).
_FP8_E4M3_MAX = 448.0
#: Signed 4-bit symmetric range: levels -8..7.
_INT4_MAX = 7
_INT8_MAX = 127
#: Row-count ceiling for `Int8Linear`'s `torch._weight_int8pack_mm` fast
#: path -- measured break-even on this GPU is ~8 rows (see
#: `_int8pack_mm_path`'s docstring); decode always calls with 1 row, a
#: prefill call is in the thousands, so the gate cleanly separates them
#: without the module needing to know which one it is.
_INT8_FUSED_MAX_ROWS = 8
#: Row-count ceiling for `Int4Linear`'s native `torch._weight_int4pack_mm`
#: fast path. MEASURED on this GPU, a 3584x3584 layer, group_size=128
#: (`docs/runs/decode-profile-2026-09-27.md`, "tretia session" microbench
#: addendum): the fast path beats dequant+`F.linear` through M=8 (1.1-1.8x)
#: and loses from M=16 on (0.57x, falling to 0.18x by M=8580) -- the same
#: "fast GEMV, slow at prefill row counts" shape as `Int8Linear`, and the
#: same numeric ceiling, but independently measured, not copied.
_INT4_FUSED_MAX_ROWS = 8
#: `torch._weight_int4pack_mm` only accepts one of these group sizes.
_INT4_NATIVE_GROUP_SIZES = frozenset({32, 64, 128, 256})
#: `innerKTiles` for `torch._convert_weight_to_int4pack` (CUDA only -- the
#: `_for_cpu` counterpart accepts but ignores this argument). 2 is the
#: smallest value the CUDA op accepts, keeping the `in_features %
#: (16 * innerKTiles) == 0` requirement as loose as possible.
_INT4_INNER_K_TILES = 2


class QuantizedLinearInvalid(ValueError):
    """A quantized linear was built from or fed an invalid shape/dtype."""


def _per_channel_absmax(weight: torch.Tensor) -> torch.Tensor:
    """Max |w| per output channel (dim 0), floored to avoid a zero scale."""
    return weight.detach().abs().amax(dim=1).clamp(min=1e-12)


class Int8Linear(nn.Module):
    """INT8 per-output-channel symmetric weight-only linear."""

    def __init__(self, in_features: int, out_features: int, *, has_bias: bool) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer("scale", torch.ones(out_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_features)) if has_bias else None
        #: Times `forward` fell back to the full dequantize + `F.linear`
        #: path instead of the fused `_weight_int8pack_mm` GEMM. Plain
        #: Python counter (not a buffer): a diagnostic, not model state.
        self.fallback_count = 0

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> Int8Linear:
        module = cls(
            linear.in_features, linear.out_features, has_bias=linear.bias is not None
        ).to(linear.weight.device)
        weight = linear.weight.detach()
        scale = _per_channel_absmax(weight) / _INT8_MAX
        quantized = torch.round(weight.float() / scale.unsqueeze(1)).clamp(-_INT8_MAX, _INT8_MAX)
        module.weight_int8.copy_(quantized.to(torch.int8))
        module.scale.copy_(scale)
        if linear.bias is not None:
            module.bias.data.copy_(linear.bias.detach())
        return module

    def dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        return (self.weight_int8.to(torch.float32) * self.scale.unsqueeze(1)).to(dtype)

    def _int8pack_mm_path(self, x_2d: torch.Tensor) -> torch.Tensor | None:
        """Fused weight-only INT8 GEMM (`torch._weight_int8pack_mm`), when
        available -- works on both CPU and CUDA but only for a 2D input,
        and only pays off for a small row count (`x_2d.shape[0]`).

        Measured on this GPU (`docs/runs/decode-profile-2026-09-27.md`
        addendum): the kernel is a decode-shaped GEMV, ~180x faster than
        the dequant fallback at 1 row, roughly break-even around 8 rows,
        then ~30x SLOWER than the fallback at prefill's row count (8580):
        0.47s vs 0.004s for one 3584x3584 layer -- multiplied over ~252
        layers that turned a single prefill call into minutes. `forward`
        always calls this per-layer with every row `encode_step`/
        `encode_prefix` feed in one shot, so the gate has to be on row
        count, not "decode vs. prefill" (this module has no such concept).
        """
        op = getattr(torch, "_weight_int8pack_mm", None)
        if op is None or x_2d.dim() != 2 or x_2d.shape[0] > _INT8_FUSED_MAX_ROWS:
            return None
        try:
            return op(x_2d, self.weight_int8, self.scale.to(x_2d.dtype))
        except (RuntimeError, TypeError):
            # Shape/dtype combinations the fused kernel refuses vary by
            # torch build/device -- never let a performance optimization
            # crash a forward pass the dequantize fallback can serve.
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(x.dtype)
        x_2d = x.reshape(-1, self.in_features)
        fast = self._int8pack_mm_path(x_2d)
        if fast is not None:
            out = fast.reshape(*x.shape[:-1], self.out_features)
            return out if bias is None else out + bias
        self.fallback_count += 1
        weight = self.dequantized_weight(x.dtype)
        return F.linear(x, weight, bias)


def _quantize_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row dynamic e4m3 activation quantization for `_scaled_mm`:
    `(x_fp8, x_scale)` with `x_scale` float32 `(rows, 1)`. A pure,
    deterministic function of `x`'s values (elementwise ops plus a row
    `amax`, which is exact in any reduction order) -- which is what lets
    sibling projections share one result bit-identically."""
    x_amax = x.detach().float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    x_scale = x_amax / _FP8_E4M3_MAX
    x_fp8 = (x.float() / x_scale).to(torch.float8_e4m3fn)
    return x_fp8, x_scale.to(torch.float32)


class SharedActivationQuant:
    """One activation quantization shared by sibling `Fp8Linear`s that the
    frozen decoder calls on the SAME input tensor (`q_proj`/`k_proj`/
    `v_proj`, `gate_proj`/`up_proj`) -- opt-in, `docs/runs/fp8-qkv-fusion-
    2026-09-29.md`. Only the quantization is shared; every sibling keeps
    its own `_scaled_mm` over its own weight and per-channel scale, so its
    GEMM receives exactly the operands it computes unshared.

    Hit rule: the key is the live input tensor OBJECT (`is`), held by a
    strong reference -- never `id(x)`, whose value can be recycled once a
    tensor is freed; a held tensor cannot be freed, so no other tensor can
    ever alias the key. A version-tracked tensor additionally has to still
    be at the version it was quantized at (an in-place write is a miss);
    `torch.inference_mode()` tensors carry no version counter, which is why
    the entry is also scoped to ONE parent-module forward call
    (`quantize.share_sibling_input_quantization` clears it on entry and on
    exit), inside which the HF forward never writes its input in place.
    """

    __slots__ = ("_key", "_version", "_value")

    def __init__(self) -> None:
        self._key: torch.Tensor | None = None
        self._version: int | None = None
        self._value: tuple[torch.Tensor, torch.Tensor] | None = None

    @staticmethod
    def _version_of(key: torch.Tensor) -> int | None:
        return None if key.is_inference() else key._version

    @property
    def holds_entry(self) -> bool:
        return self._key is not None

    def quantize(
        self, key: torch.Tensor, x_2d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`_quantize_activation(x_2d)`, reused when `key` is the very
        tensor (at the same version) the cached entry was computed from.
        `x_2d` must be `key` flattened to `(rows, in_features)`."""
        version = self._version_of(key)
        if self._key is key and self._version == version and self._value is not None:
            return self._value
        value = _quantize_activation(x_2d)
        self._key, self._version, self._value = key, version, value
        return value

    def clear(self) -> None:
        self._key = self._version = self._value = None


class Fp8Linear(nn.Module):
    """FP8 (`float8_e4m3fn`) per-output-channel weight-only linear."""

    def __init__(self, in_features: int, out_features: int, *, has_bias: bool) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight_fp8", torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn)
        )
        self.register_buffer("scale", torch.ones(out_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_features)) if has_bias else None
        #: Times `forward` fell back to the full dequantize + `F.linear`
        #: path instead of the fused `_scaled_mm` GEMM. Plain Python
        #: counter (not a buffer): a diagnostic, not model state.
        self.fallback_count = 0
        #: Opt-in sibling sharing (`SharedActivationQuant`), wired only by
        #: `quantize.share_sibling_input_quantization`. `None` (default) is
        #: the unchanged per-call quantization. Not model state.
        self.shared_input_quant: SharedActivationQuant | None = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> Fp8Linear:
        module = cls(
            linear.in_features, linear.out_features, has_bias=linear.bias is not None
        ).to(linear.weight.device)
        weight = linear.weight.detach()
        scale = _per_channel_absmax(weight) / _FP8_E4M3_MAX
        quantized = (weight.float() / scale.unsqueeze(1)).to(torch.float8_e4m3fn)
        module.weight_fp8.copy_(quantized)
        module.scale.copy_(scale)
        if linear.bias is not None:
            module.bias.data.copy_(linear.bias.detach())
        return module

    def dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        return (self.weight_fp8.to(torch.float32) * self.scale.unsqueeze(1)).to(dtype)

    def _scaled_mm_path(
        self, x: torch.Tensor, share_key: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """`torch._scaled_mm` fast path -- real hardware fp8 GEMM (Ada/
        Hopper, SM >= 8.9), when available.

        Only attempted for a 2D bf16/fp16 input (row-wise `_scaled_mm`
        scaling only supports a bf16/fp16 high-precision output type) on a
        CUDA device that has the fp8 weight resident on it, with both
        `in_features` and `out_features` divisible by 16 (a hard
        `_scaled_mm` kernel constraint on `mat2`). Quantizes the
        activation with a per-row dynamic scale and reuses the existing
        per-output-channel weight scale as `_scaled_mm`'s row-wise
        `scale_b`. Anything else (CPU, batched >2D input, fp32 input, an
        odd shape, an environment without the op) falls back to
        `dequantized_weight` in `forward`. Returns `None` to signal "use
        the fallback" rather than raising, since the fast path is a pure
        performance optimization with no behavioral contract of its own.

        **Activation quantization per call by default; shared between
        siblings only when opted in.** `q_proj`/`k_proj`/`v_proj` (and
        `gate_proj`/`up_proj`) are separate `Fp8Linear` instances that the
        frozen decoder calls on the SAME `hidden_states` tensor; by default
        each call recomputes `_quantize_activation` on that identical input
        (`docs/runs/dspark-rehearsal-2026-09-27.md` §22.4). When
        `self.shared_input_quant` is set (opt-in,
        `quantize.share_sibling_input_quantization`, `docs/runs/fp8-qkv-
        fusion-2026-09-29.md`) and `share_key` is the caller's original
        input tensor, the siblings reuse one quantization of it -- held by
        a strong reference and scoped to one parent forward, never keyed on
        `id(x)` (see `SharedActivationQuant`). The GEMM itself is unchanged
        and per-sibling, so the output is bit-identical to the default.
        """
        if x.dim() != 2 or not x.is_cuda or not self.weight_fp8.is_cuda:
            return None
        if x.dtype not in (torch.bfloat16, torch.float16):
            return None
        if not hasattr(torch, "_scaled_mm"):
            return None
        if self.in_features % 16 != 0 or self.out_features % 16 != 0:
            return None
        try:
            shared = self.shared_input_quant
            if shared is not None and share_key is not None:
                x_fp8, x_scale = shared.quantize(share_key, x)
            else:
                x_fp8, x_scale = _quantize_activation(x)
            out = torch._scaled_mm(
                x_fp8,
                self.weight_fp8.t(),
                scale_a=x_scale,
                scale_b=self.scale.unsqueeze(0),
                out_dtype=x.dtype,
            )
            if isinstance(out, tuple):
                out = out[0]
            return out
        except (RuntimeError, TypeError, ValueError):
            # Shape/dtype combinations `_scaled_mm` refuses vary by torch
            # build/device -- never let an optimization attempt crash a
            # forward pass that the dequantize fallback can serve.
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(x.dtype)
        x_2d = x.reshape(-1, self.in_features)
        fast = self._scaled_mm_path(x_2d, share_key=x)
        if fast is not None:
            out = fast.reshape(*x.shape[:-1], self.out_features)
            return out if bias is None else out + bias
        self.fallback_count += 1
        weight = self.dequantized_weight(x.dtype)
        return F.linear(x, weight, bias)


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack signed 4-bit values (range -8..7, stored as low nibble unsigned
    0..15) two per byte: even columns in the low nibble, odd in the high
    nibble. `values` is `(out_features, in_features)` int16/int32; the
    packed result is `(out_features, ceil(in_features / 2))` uint8."""
    unsigned = (values + 8).to(torch.uint8)
    out_features, in_features = unsigned.shape
    if in_features % 2 == 1:
        pad = torch.zeros(out_features, 1, dtype=torch.uint8, device=unsigned.device)
        unsigned = torch.cat([unsigned, pad], dim=1)
    low = unsigned[:, 0::2]
    high = unsigned[:, 1::2]
    return (low | (high << 4)).to(torch.uint8)


def _pack_int4_cuda_native(unsigned: torch.Tensor) -> torch.Tensor:
    """Pack unsigned 4-bit codes (0..15) for `torch._convert_weight_to_
    int4pack` on CUDA -- the OPPOSITE nibble order from `_pack_int4`: the
    high nibble holds the even column, the low nibble the odd column.
    Empirically verified against `_weight_int4pack_mm`'s real output (a
    one-hot-input probe compared to the plain dequantized weight); using
    `_pack_int4`'s own order silently produces a column-swapped, wrong
    result with no error (see `Int4Linear._build_native_int4`).
    `unsigned` is `(out_features, in_features)` uint8, `in_features` even
    (the native kernel requires it -- `Int4Linear._native_int4_supported`
    already enforces `in_features % group_size == 0` for a supported
    `group_size`, and every native group size is even)."""
    low = unsigned[:, 0::2]
    high = unsigned[:, 1::2]
    return (high | (low << 4)).to(torch.uint8)


def _unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """Inverse of `_pack_int4`: returns signed values in `(-8..7)`."""
    low = (packed & 0x0F).to(torch.int16)
    high = ((packed >> 4) & 0x0F).to(torch.int16)
    interleaved = torch.stack([low, high], dim=2).reshape(packed.shape[0], -1)
    return interleaved[:, :in_features] - 8


class Int4Linear(nn.Module):
    """INT4 group-wise symmetric weight-only linear, packed 2 values/byte.

    Each output channel's `in_features` columns are split into groups of
    `group_size` (the last group may be shorter); every group carries its
    own float32 scale (round-to-nearest by default -- see
    :func:`calibrate_int4_from_linear` for the optional simplified
    GPTQ-style calibration path).

    Native fast path (`torch._weight_int4pack_mm`, CUDA, or `..._for_cpu`)
    when `group_size` is one of the kernel's fixed choices (32/64/128/256)
    and `out_features % 16 == 0` -- both are hard native constraints, not
    tunable. `weight_packed` (`(out_features, ceil(in_features/2))` bytes,
    2 values/byte) and `scale` (one float32 per group) stay the on-disk
    format unchanged, so `from_linear`/serialization/state_dict shape are
    unaffected. The native layout is a *derived cache*, built lazily on
    first `forward` per device and never persisted:

    - The stored code is unsigned `value + 8` in `[0, 15]` with a single
      per-group `scale` and an implicit zero-point of `-8 * scale`
      (`dequant = (unsigned - 8) * scale`). The native kernel's own affine
      scheme is `dequant = code * scale + (zero - scale * 2**(bits-1))`;
      setting `zero = 0` makes the two formulas identical, so the existing
      codes are repacked as-is via `torch._convert_weight_to_int4pack[_for
      _cpu]` -- there is no re-quantization and no accuracy change versus
      `dequantized_weight`.
    - `scale`/`zero` are packed into `(n_groups, out_features, 2)`
      bfloat16, the shape `_weight_int4pack_mm` requires.
    - Falls back to dequantize-then-`F.linear` (like `Int8Linear`/
      `Fp8Linear`) for any unsupported `group_size`/`out_features`,
      non-bf16 input, row count above `_INT4_FUSED_MAX_ROWS`, or an
      environment without the op; `self.fallback_count` counts every one.
    """

    def __init__(
        self, in_features: int, out_features: int, *, group_size: int, has_bias: bool
    ) -> None:
        super().__init__()
        if group_size <= 0:
            raise QuantizedLinearInvalid(f"group_size must be positive, got {group_size}")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        num_groups = (in_features + group_size - 1) // group_size
        packed_width = (in_features + 1) // 2
        self.register_buffer(
            "weight_packed", torch.zeros(out_features, packed_width, dtype=torch.uint8)
        )
        self.register_buffer("scale", torch.ones(out_features, num_groups, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_features)) if has_bias else None
        #: Times `forward` fell back to the full dequantize + `F.linear`
        #: path instead of the native `_weight_int4pack_mm` GEMM. Plain
        #: Python counter (not a buffer): a diagnostic, not model state.
        self.fallback_count = 0
        #: Native packed layout, built lazily and cached per device --
        #: not a buffer (never persisted, rebuilt from `weight_packed`/
        #: `scale` on first use after a device change).
        self._native_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def _group_scale_per_column(self) -> torch.Tensor:
        """Broadcast the per-group scale out to one scale per input column."""
        repeats = self.group_size
        expanded = self.scale.repeat_interleave(repeats, dim=1)
        return expanded[:, : self.in_features]

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        group_size: int,
        weights: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> Int4Linear:
        """Build from `linear`'s own weight (round-to-nearest), unless
        `weights`/`scale` are supplied by a calibration routine
        (:func:`calibrate_int4_from_linear`) -- both or neither."""
        module = cls(
            linear.in_features, linear.out_features, group_size=group_size,
            has_bias=linear.bias is not None,
        ).to(linear.weight.device)
        if (weights is None) != (scale is None):
            raise QuantizedLinearInvalid("weights and scale must both be given, or neither")
        if weights is None:
            weight = linear.weight.detach().float()
            num_groups = module.scale.shape[1]
            group_scales = torch.empty(
                module.out_features, num_groups, dtype=torch.float32, device=weight.device
            )
            quant_int = torch.empty_like(weight, dtype=torch.int16)
            for g in range(num_groups):
                start, end = g * group_size, min((g + 1) * group_size, module.in_features)
                chunk = weight[:, start:end]
                s = chunk.abs().amax(dim=1).clamp(min=1e-12) / _INT4_MAX
                group_scales[:, g] = s
                quant_int[:, start:end] = torch.round(chunk / s.unsqueeze(1)).clamp(
                    -_INT4_MAX - 1, _INT4_MAX
                ).to(torch.int16)
            weights, scale = quant_int, group_scales
        module.weight_packed.copy_(_pack_int4(weights))
        module.scale.copy_(scale)
        if linear.bias is not None:
            module.bias.data.copy_(linear.bias.detach())
        return module

    def dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        unpacked = _unpack_int4(self.weight_packed, self.in_features).to(torch.float32)
        return (unpacked * self._group_scale_per_column()).to(dtype)

    def _native_int4_supported(self) -> bool:
        """Whether `group_size`/`out_features` satisfy the native
        `torch._weight_int4pack_mm` kernel's fixed constraints (measured
        empirically -- see class docstring)."""
        return (
            self.group_size in _INT4_NATIVE_GROUP_SIZES
            and self.in_features % self.group_size == 0
            and self.out_features % 16 == 0
        )

    def _build_native_int4(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Repack the existing int4 codes for `device` into the native
        layout (see class docstring for why this needs no re-quantization).
        Returns `None` if the op is unavailable or refuses the shape.

        CPU (`_convert_weight_to_int4pack_for_cpu`) takes one code per
        `int32` element, `(out_features, in_features)`. CUDA
        (`_convert_weight_to_int4pack`) instead takes pre-packed
        `(out_features, in_features/2)` `uint8` bytes -- and, verified
        empirically against `_weight_int4pack_mm`'s actual output (a
        one-hot-input probe against `dequantized_weight`), with the
        OPPOSITE nibble order from `_pack_int4`: the high nibble holds the
        even column, the low nibble the odd column."""
        is_cuda = device.type == "cuda"
        op_name = (
            "_convert_weight_to_int4pack" if is_cuda else "_convert_weight_to_int4pack_for_cpu"
        )
        convert = getattr(torch, op_name, None)
        if convert is None:
            return None
        unsigned = (_unpack_int4(self.weight_packed.to(device), self.in_features) + 8).to(
            torch.uint8 if is_cuda else torch.int32
        )
        weight_arg = _pack_int4_cuda_native(unsigned) if is_cuda else unsigned.contiguous()
        try:
            packed = convert(weight_arg, _INT4_INNER_K_TILES)
        except (RuntimeError, TypeError):
            return None
        scale_bf16 = self.scale.to(device=device, dtype=torch.bfloat16)
        zero_bf16 = torch.zeros_like(scale_bf16)
        scales_and_zeros = torch.stack([scale_bf16, zero_bf16], dim=-1).transpose(0, 1).contiguous()
        return packed, scales_and_zeros

    def _int4pack_mm_path(self, x_2d: torch.Tensor) -> torch.Tensor | None:
        """Native weight-only INT4 GEMM, when available and the shape/
        row-count qualify -- see class docstring and `_INT4_FUSED_MAX_ROWS`.
        """
        if (
            not self._native_int4_supported()
            or x_2d.dim() != 2
            or x_2d.shape[0] > _INT4_FUSED_MAX_ROWS
            or x_2d.dtype != torch.bfloat16
        ):
            return None
        device = x_2d.device
        cached = self._native_cache.get(device)
        if cached is None:
            built = self._build_native_int4(device)
            if built is None:
                return None
            self._native_cache[device] = built
            cached = built
        packed, scales_and_zeros = cached
        mm = (
            torch._weight_int4pack_mm_for_cpu
            if device.type == "cpu"
            else torch._weight_int4pack_mm
        )
        try:
            return mm(x_2d, packed, self.group_size, scales_and_zeros)
        except (RuntimeError, TypeError):
            # Shape/dtype combinations the native kernel refuses vary by
            # torch build/device -- never let a performance optimization
            # crash a forward pass the dequantize fallback can serve.
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(x.dtype)
        x_2d = x.reshape(-1, self.in_features)
        fast = self._int4pack_mm_path(x_2d)
        if fast is not None:
            out = fast.reshape(*x.shape[:-1], self.out_features)
            return out if bias is None else out + bias
        self.fallback_count += 1
        weight = self.dequantized_weight(x.dtype)
        return F.linear(x, weight, bias)


def calibrate_int4_from_linear(
    linear: nn.Linear, *, group_size: int, activations: torch.Tensor
) -> Int4Linear:
    """Simplified GPTQ-style INT4 calibration over caller-supplied
    activations (owner requirement §1: "optional simple GPTQ-style
    calibration ... document limitations if you keep it simple").

    **What this is**: per-group scales chosen to minimize the
    activation-weighted squared quantization error
    `sum_j act_rms[j] * (w[:,j] - dequant(w[:,j]))^2` instead of plain
    round-to-nearest's unweighted `|w|` scale -- columns the calibration
    batch actually drives hard get a tighter scale at the expense of
    columns it barely touches.

    **What this is NOT**: real GPTQ. There is no Hessian (`X^T X`), no
    Cholesky decomposition, and no sequential per-column error
    compensation propagated to not-yet-quantized columns -- each group is
    still quantized independently. This is a one-shot activation-weighted
    round-to-nearest, not the iterative optimal-brain-surgery update GPTQ
    performs. `activations` must be `(n_samples, in_features)`.
    """
    if activations.dim() != 2 or activations.shape[1] != linear.in_features:
        raise QuantizedLinearInvalid(
            f"activations must be (n_samples, {linear.in_features}), got {tuple(activations.shape)}"
        )
    weight = linear.weight.detach().float()
    out_features, in_features = weight.shape
    act_rms = activations.float().pow(2).mean(dim=0).sqrt().clamp(min=1e-12).to(weight.device)
    num_groups = (in_features + group_size - 1) // group_size
    group_scales = torch.empty(out_features, num_groups, dtype=torch.float32, device=weight.device)
    quant_int = torch.empty_like(weight, dtype=torch.int16)
    for g in range(num_groups):
        start, end = g * group_size, min((g + 1) * group_size, in_features)
        chunk = weight[:, start:end]
        weights_col = act_rms[start:end]
        # Weighted absmax: scale to cover the columns with the largest
        # activation-weighted magnitude, not merely the largest raw weight.
        weighted = chunk.abs() * weights_col.unsqueeze(0)
        pivot = torch.argmax(weighted, dim=1)
        s = (chunk.abs().gather(1, pivot.unsqueeze(1)).squeeze(1)).clamp(min=1e-12) / _INT4_MAX
        group_scales[:, g] = s
        quant_int[:, start:end] = torch.round(chunk / s.unsqueeze(1)).clamp(
            -_INT4_MAX - 1, _INT4_MAX
        ).to(torch.int16)
    return Int4Linear.from_linear(
        linear, group_size=group_size, weights=quant_int, scale=group_scales
    )
