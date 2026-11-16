"""`quantize_candidate` -- walks a frozen Qwen candidate's backbone modules
and swaps `nn.Linear` children for the weight-only quantized replacements
in `linear.py`, per a `QuantizationSpec` (owner requirement §1).

Composition, not a `candidate_model.py`/`candidate_loader.py` edit: this
module never changes `QwenCandidate`'s shape. `decoder_module`/
`vision_module` recover the REAL `nn.Module` objects `QwenCandidate.decode`/
`vision_forward` close over (Python closures share one cell with their
defining scope, so quantizing the module found this way mutates exactly
what `decode`/`vision_forward` call -- no extra wiring needed). Our own
trainable heads (`QwenDenseCandidateModel.composite_head`, `dense_adapter`,
`node_visual_aligner`, adapters, ...) live in a completely different module
tree that is never reachable from a `QwenCandidate` at all, so they are
untouched by construction, not by an exclusion list.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from torch import nn

from unflatten.model.quantization.linear import (
    Fp8Linear,
    Int4Linear,
    Int8Linear,
    SharedActivationQuant,
    calibrate_int4_from_linear,
)
from unflatten.model.quantization.spec import QuantizationSpec


class QuantizationError(ValueError):
    """A candidate/module could not be quantized as requested."""


@dataclass(frozen=True, slots=True)
class QuantizedModuleRecord:
    qualified_name: str
    precision: str
    bytes_before: int
    bytes_after: int


@dataclass(frozen=True, slots=True)
class QuantizationReport:
    modules: tuple[QuantizedModuleRecord, ...] = field(default_factory=tuple)

    @property
    def bytes_before(self) -> int:
        return sum(m.bytes_before for m in self.modules)

    @property
    def bytes_after(self) -> int:
        return sum(m.bytes_after for m in self.modules)

    def __add__(self, other: QuantizationReport) -> QuantizationReport:
        return QuantizationReport(modules=self.modules + other.modules)


def _linear_bytes(linear: nn.Linear) -> int:
    total = linear.weight.numel() * linear.weight.element_size()
    if linear.bias is not None:
        total += linear.bias.numel() * linear.bias.element_size()
    return total


def _quantized_bytes(module: nn.Module) -> int:
    total = 0
    for buf in module.buffers():
        total += buf.numel() * buf.element_size()
    if getattr(module, "bias", None) is not None:
        total += module.bias.numel() * module.bias.element_size()
    return total


def _build_replacement(
    linear: nn.Linear,
    precision: str,
    *,
    group_size: int,
    activations: Any | None,
) -> nn.Module:
    if precision == "int8":
        return Int8Linear.from_linear(linear)
    if precision == "fp8":
        return Fp8Linear.from_linear(linear)
    if precision == "int4":
        if activations is not None:
            return calibrate_int4_from_linear(
                linear, group_size=group_size, activations=activations
            )
        return Int4Linear.from_linear(linear, group_size=group_size)
    raise QuantizationError(f"unsupported precision {precision!r} for a weight-only linear")


def quantize_module(
    root: nn.Module,
    precision: str,
    *,
    group_size: int = 128,
    calibration_activations: Mapping[str, Any] | None = None,
    name_prefix: str = "",
) -> QuantizationReport:
    """Replace every `nn.Linear` under `root` with `precision`'s quantized
    module, in place. `precision == "bf16"` is a no-op (returns an empty
    report) -- the default, untouched path.

    `calibration_activations`, when given, maps a linear's qualified name
    (relative to `root`) to its own `(n_samples, in_features)` calibration
    batch for the INT4 simplified-GPTQ path (`linear.py`'s
    `calibrate_int4_from_linear`); a name absent from the mapping falls back
    to plain round-to-nearest.
    """
    if precision == "bf16":
        return QuantizationReport()
    records: list[QuantizedModuleRecord] = []
    # Collect first: mutating `named_modules()` while iterating it is unsafe.
    targets = [
        (name, module) for name, module in root.named_modules() if isinstance(module, nn.Linear)
    ]
    for name, linear in targets:
        qualified = f"{name_prefix}{name}" if name_prefix else name
        activations = None
        if calibration_activations is not None:
            activations = calibration_activations.get(qualified) or calibration_activations.get(
                name
            )
        replacement = _build_replacement(
            linear, precision, group_size=group_size, activations=activations
        )
        parent_path, _, attr = name.rpartition(".")
        parent = root.get_submodule(parent_path) if parent_path else root
        setattr(parent, attr, replacement)
        records.append(
            QuantizedModuleRecord(
                qualified_name=qualified,
                precision=precision,
                bytes_before=_linear_bytes(linear),
                bytes_after=_quantized_bytes(replacement),
            )
        )
    return QuantizationReport(modules=tuple(records))


#: Direct-child `Fp8Linear` names the HF Qwen3-VL text decoder calls on ONE
#: shared input tensor (`Qwen3VLTextAttention.forward`: `q_proj`/`k_proj`/
#: `v_proj(hidden_states)`; `Qwen3VLTextMLP.forward`: `gate_proj(x)`/
#: `up_proj(x)`). `o_proj`/`down_proj` each see their own input, so they are
#: never grouped. A name set, not structural detection: which siblings share
#: an input is a fact about the parent's forward, not its children.
SIBLING_INPUT_GROUPS: tuple[tuple[str, ...], ...] = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
)


@dataclass(slots=True)
class SiblingQuantSharing:
    """Handle returned by :func:`share_sibling_input_quantization`."""

    group_count: int
    _linears: list[Fp8Linear] = field(default_factory=list)
    _hooks: list[Any] = field(default_factory=list)

    def remove(self) -> None:
        """Restore the default per-call quantization everywhere it was wired."""
        for handle in self._hooks:
            handle.remove()
        for linear in self._linears:
            linear.shared_input_quant = None
        self._hooks.clear()
        self._linears.clear()
        self.group_count = 0


def share_sibling_input_quantization(root: nn.Module) -> SiblingQuantSharing:
    """Opt-in (`docs/runs/fp8-qkv-fusion-2026-09-29.md`): let every
    :data:`SIBLING_INPUT_GROUPS` group of `Fp8Linear` children under `root`
    share ONE activation quantization of their common input, in place.

    Changes no module forward and no weight: each group gets one
    `SharedActivationQuant`, and the parent gets a forward pre-hook and an
    (always-called) forward hook that clear it, so an entry never outlives
    the parent forward that created it. Every sibling still runs its own
    `_scaled_mm`, so outputs are bit-identical to the default.

    Raises `QuantizationError` when nothing qualifies (e.g. the decoder is
    not fp8) -- an opt-in that silently wires nothing would read as "no
    speed-up" instead of "never enabled" -- or when a group's siblings
    disagree on `in_features` (they cannot share one input).
    """
    sharing = SiblingQuantSharing(group_count=0)
    for parent in root.modules():
        for names in SIBLING_INPUT_GROUPS:
            children = [getattr(parent, name, None) for name in names]
            if not all(isinstance(child, Fp8Linear) for child in children):
                continue
            if len({child.in_features for child in children}) != 1:
                raise QuantizationError(
                    f"sibling group {names} disagrees on in_features: "
                    f"{[child.in_features for child in children]}"
                )
            shared = SharedActivationQuant()
            for child in children:
                child.shared_input_quant = shared
            sharing._linears.extend(children)
            sharing._hooks.append(
                parent.register_forward_pre_hook(lambda _m, _a, s=shared: s.clear())
            )
            sharing._hooks.append(
                parent.register_forward_hook(
                    lambda _m, _a, _o, s=shared: s.clear(), always_call=True
                )
            )
            sharing.group_count += 1
    if sharing.group_count == 0:
        raise QuantizationError(
            "no fp8 sibling group (q/k/v or gate/up Fp8Linear children) under root -- "
            "shared input quantization needs a decoder quantized to fp8"
        )
    return sharing


def _module_from_closure(func: Any, *candidate_names: str) -> nn.Module:
    """Recover an `nn.Module` a closure (`QwenCandidate.decode`/
    `vision_forward`) captures over, by free-variable name. Raises if the
    function carries no closure or none of `candidate_names` is bound --
    this is the one place this package reaches past `QwenCandidate`'s public
    shape, so a caller handed something else gets a clear refusal instead of
    an `AttributeError` deep inside a quantized forward pass later."""
    code = getattr(func, "__code__", None)
    closure = getattr(func, "__closure__", None)
    if code is None or closure is None:
        raise QuantizationError(
            f"{func!r} carries no closure -- cannot recover its backing module"
        )
    bound = dict(zip(code.co_freevars, (cell.cell_contents for cell in closure), strict=True))
    for name in candidate_names:
        value = bound.get(name)
        if isinstance(value, nn.Module):
            return value
    raise QuantizationError(
        f"none of {candidate_names} is an nn.Module free variable of {func!r} "
        f"(found: {sorted(bound)})"
    )


def decoder_module(candidate: Any) -> nn.Module:
    """The real frozen language-decoder `nn.Module` behind `candidate.decode`."""
    return _module_from_closure(candidate.decode, "language_model")


def vision_module(candidate: Any) -> nn.Module:
    """The real frozen vision-tower `nn.Module` behind `candidate.vision_forward`.

    Real candidates (`load_qwen_candidate`) close over a variable named
    `visual`; `candidate_loader.build_stub_qwen_candidate`'s stub closes
    over `tower` instead -- both are tried."""
    return _module_from_closure(candidate.vision_forward, "visual", "tower")


def quantize_candidate(
    candidate: Any,
    spec: QuantizationSpec,
    *,
    calibration_activations: Mapping[str, Any] | None = None,
) -> QuantizationReport:
    """Quantize `candidate`'s frozen backbone per `spec`, in place.

    Only `spec.decoder_precision` is applied by default (`spec.vision_precision
    == "bf16"` is the default and a no-op). `candidate` must be a
    `QwenCandidate` (real or stub, from `candidate_loader.py`) -- its own
    trainable heads live outside this object entirely, so they can never be
    reached from here (module docstring).
    """
    report = quantize_module(
        decoder_module(candidate),
        spec.decoder_precision,
        group_size=spec.group_size,
        calibration_activations=calibration_activations,
        name_prefix="language_model.",
    )
    if spec.vision_precision != "bf16":
        report = report + quantize_module(
            vision_module(candidate),
            spec.vision_precision,
            group_size=spec.group_size,
            calibration_activations=calibration_activations,
            name_prefix="visual.",
        )
    return report
