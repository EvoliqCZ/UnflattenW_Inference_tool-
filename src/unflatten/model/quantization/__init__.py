"""Post-training weight-only quantization for the frozen Qwen candidate
(owner requirement 2026-09-27, `docs/plans/code-ahead-plan-2026-09-26.md`
§10): INT8/FP8/INT4 weight-only linears, an FP8 KV cache, and a
`QuantizationSpec` identity so a quantized bundle can never be mistaken for
bf16. Post-training only -- no retraining, no new dependency (pure torch).
Our own heads (typed heads, composite head, node visual aligner, adapters)
are never touched by :func:`unflatten.model.quantization.quantize.quantize_candidate`.
"""

from __future__ import annotations
