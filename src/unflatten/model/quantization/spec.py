"""`QuantizationSpec` -- precision per component, group size, calibration
id, and KV-cache dtype, with a domain hash (RFC004 §3.1 style, mirroring
`s4_selection_policy.py`: no embedded self-hash, identity is computed by
:func:`quantization_spec_hash` over the CBOR of the payload).

This spec is what a model bundle's `precision` field pins (owner
requirement item 3): a bf16 bundle carries no spec at all (or the sentinel
`bf16_spec()`), so an unmodified bf16 bundle's identity is untouched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from unflatten.model.hashing import domain_hash

QUANTIZATION_SPEC_SCHEMA_VERSION = "0.1.0"

#: Weight precisions this package can produce (§1 of the owner requirement).
WEIGHT_PRECISIONS: frozenset[str] = frozenset({"bf16", "int8", "fp8", "int4"})
#: KV cache storage dtypes (§2).
KV_CACHE_DTYPES: frozenset[str] = frozenset({"bf16", "fp8"})


class QuantizationSpecInvalid(ValueError):
    """A `QuantizationSpec` field is outside its closed set."""


@dataclass(frozen=True, slots=True)
class QuantizationSpec:
    """One frozen precision assignment for a candidate model.

    `decoder_precision` covers the frozen Qwen language decoder's linears
    (the only component quantized by default). `vision_precision` is
    selectable independently (default `"bf16"`, i.e. untouched) -- the
    owner requirement's "optionally the vision encoder, selectable, default:
    decoder only". `group_size` only matters when a precision is `"int4"`.
    `calibration_id` names the calibration activation set used for INT4
    GPTQ-style calibration, or is `None` for plain round-to-nearest.
    """

    decoder_precision: str = "bf16"
    vision_precision: str = "bf16"
    group_size: int = 128
    calibration_id: str | None = None
    kv_cache_dtype: str = "bf16"
    schema_version: str = field(default=QUANTIZATION_SPEC_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        if self.decoder_precision not in WEIGHT_PRECISIONS:
            raise QuantizationSpecInvalid(
                f"decoder_precision must be one of {sorted(WEIGHT_PRECISIONS)}, "
                f"got {self.decoder_precision!r}"
            )
        if self.vision_precision not in WEIGHT_PRECISIONS:
            raise QuantizationSpecInvalid(
                f"vision_precision must be one of {sorted(WEIGHT_PRECISIONS)}, "
                f"got {self.vision_precision!r}"
            )
        if self.kv_cache_dtype not in KV_CACHE_DTYPES:
            raise QuantizationSpecInvalid(
                f"kv_cache_dtype must be one of {sorted(KV_CACHE_DTYPES)}, "
                f"got {self.kv_cache_dtype!r}"
            )
        if self.group_size <= 0:
            raise QuantizationSpecInvalid(f"group_size must be positive, got {self.group_size}")

    @property
    def is_bf16(self) -> bool:
        """True only for the fully-untouched bf16 configuration -- the
        sentinel a bundle carries when it was never quantized at all."""
        return (
            self.decoder_precision == "bf16"
            and self.vision_precision == "bf16"
            and self.kv_cache_dtype == "bf16"
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def spec_hash(self) -> str:
        """`sha256:<64 hex>` identity, RFC004 §3.1 domain-separated."""
        return domain_hash("unflatten.model.quantization/quantization-spec/v1", self.to_dict())


def bf16_spec() -> QuantizationSpec:
    """The sentinel spec an unmodified bf16 bundle is treated as carrying."""
    return QuantizationSpec()
