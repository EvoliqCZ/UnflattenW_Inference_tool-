"""RFC004 §15.1 mandatory bypass: architecture must work with ``compressor = bypass``.

``BypassVisualCompressor`` is the control arm of every compression ablation
(RFC004-ERRATA-005 E5 rule 1: "step 0 == bypass") and today's product path.
It returns bit-identical tokens and an identity alignment — every original
token is its own compressed token, one-to-one, in the same order.
"""

from __future__ import annotations

import torch

from unflatten.model.visual_compression.contracts import (
    BudgetReport,
    CompressedVisualMemory,
    OriginalTokenAlignment,
    VisualMemory,
    retention_map_to_alignment,
)

#: RFC004-ERRATA-005 E5 "budget_report ... verzia politiky" — bypass has no
#: policy to version beyond identity.
BYPASS_POLICY_VERSION = "bypass-v0"


class BypassVisualCompressor:
    """RFC004 §15.2 compressor: identity everywhere, no learned parameters."""

    def compress(self, memory: VisualMemory) -> CompressedVisualMemory:
        batch, n_tokens, _ = memory.tokens.shape
        # 0 == "protected" risk class by convention.
        risk_map = torch.zeros((batch, n_tokens), dtype=torch.long)
        retention_map = torch.arange(n_tokens, dtype=torch.long)
        budget_report = BudgetReport(
            original_token_count=n_tokens,
            compressed_token_count=n_tokens,
            bypassed_region_count=n_tokens,
            policy_version=BYPASS_POLICY_VERSION,
        )
        return CompressedVisualMemory(
            coarse_memory=memory,
            protected_detail=memory,
            risk_map=risk_map,
            retention_map=retention_map,
            budget_report=budget_report,
        )

    def expand_alignment(self, retention_map: torch.Tensor) -> OriginalTokenAlignment:
        return retention_map_to_alignment(retention_map)
