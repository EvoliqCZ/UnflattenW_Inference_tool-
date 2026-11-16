"""RFC005 §21.2 architecture-contract gate for post-encoder compressors.

Every check below has a companion mutant in the test suite that flips it to
failing — a checker that cannot be made to fail is not a checker (see
project memory `guard-must-be-provably-failable`).

The alignment check (:func:`check_alignment_integrity`) is defined for a
single-page batch (``memory.tokens.shape[0] == 1``): a compressed token's
claim over multiple *pages* ("page swap", RFC005 §21.2) is a batch-level
concern this module does not attempt to check across samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from unflatten.model.visual_compression.bypass import BypassVisualCompressor
from unflatten.model.visual_compression.contracts import VisualCompressor, VisualMemory


@dataclass(frozen=True)
class ContractCheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ArchitectureContractReport:
    results: tuple[ContractCheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    def failures(self) -> tuple[ContractCheckResult, ...]:
        return tuple(result for result in self.results if not result.passed)


def check_bypass_identity(memory: VisualMemory) -> ContractCheckResult:
    """RFC005 §21.2: "bypass je output-identical s pre-S7 primary visual path"."""
    bypass = BypassVisualCompressor()
    compressed = bypass.compress(memory)
    n_tokens = memory.tokens.shape[1]
    identical = (
        torch.equal(compressed.coarse_memory.tokens, memory.tokens)
        and torch.equal(compressed.protected_detail.tokens, memory.tokens)
        and torch.equal(compressed.coarse_memory.key_padding_mask, memory.key_padding_mask)
        and torch.equal(compressed.retention_map, torch.arange(n_tokens, dtype=torch.long))
    )
    return ContractCheckResult(
        "bypass_identity", identical, "bypass must reproduce the uncompressed memory exactly"
    )


def check_page_order_preserved(
    compressor: VisualCompressor, memory: VisualMemory
) -> ContractCheckResult:
    compressed = compressor.compress(memory)
    ok = torch.equal(
        compressed.coarse_memory.page_ordinals, memory.page_ordinals
    ) and torch.equal(compressed.protected_detail.page_ordinals, memory.page_ordinals)
    return ContractCheckResult(
        "page_order_preserved", ok, "page_ordinals must survive compress() unchanged"
    )


def check_mask_semantics(compressor: VisualCompressor, memory: VisualMemory) -> ContractCheckResult:
    compressed = compressor.compress(memory)

    def _mask_matches(vm: VisualMemory) -> bool:
        return tuple(vm.key_padding_mask.shape) == (vm.tokens.shape[0], vm.tokens.shape[1])

    ok = _mask_matches(compressed.coarse_memory) and _mask_matches(compressed.protected_detail)
    return ContractCheckResult(
        "mask_semantics",
        ok,
        "key_padding_mask shape must track (batch, tokens) on every output memory",
    )


def check_deterministic_shape(
    compressor: VisualCompressor, memory: VisualMemory
) -> ContractCheckResult:
    first = compressor.compress(memory)
    second = compressor.compress(memory)
    ok = (
        first.coarse_memory.tokens.shape == second.coarse_memory.tokens.shape
        and first.protected_detail.tokens.shape == second.protected_detail.tokens.shape
    )
    return ContractCheckResult(
        "deterministic_shape",
        ok,
        "same input/config must produce the same output shape every call",
    )


def check_summary_path_present(
    compressor: VisualCompressor, memory: VisualMemory
) -> ContractCheckResult:
    compressed = compressor.compress(memory)
    ok = compressed.coarse_memory.tokens.shape[1] > 0
    return ContractCheckResult(
        "summary_path_present", ok, "coarse_memory (the global summary path) must not be empty"
    )


def check_alignment_integrity(
    compressor: VisualCompressor, memory: VisualMemory
) -> ContractCheckResult:
    if memory.tokens.shape[0] != 1:
        return ContractCheckResult(
            "alignment_integrity", False, "check_alignment_integrity requires a single-page batch"
        )
    compressed = compressor.compress(memory)
    try:
        alignment = compressor.expand_alignment(compressed.retention_map)
    except ValueError as exc:
        return ContractCheckResult("alignment_integrity", False, f"expand_alignment raised: {exc}")
    ok = alignment.original_token_count == memory.tokens.shape[1]
    return ContractCheckResult(
        "alignment_integrity",
        ok,
        "expand_alignment() must cover every original token exactly once (no gaps/orphans)",
    )


def check_hashes_present(implementation_hash: str, policy_hash: str) -> ContractCheckResult:
    ok = bool(implementation_hash) and bool(policy_hash)
    return ContractCheckResult(
        "hashes_present",
        ok,
        "compressor and alignment-expansion hashes must be present for the Component Lock",
    )


def check_architecture_contract(
    compressor: VisualCompressor,
    memory: VisualMemory,
    *,
    implementation_hash: str,
    policy_hash: str,
) -> ArchitectureContractReport:
    """Run the full RFC005 §21.2 gate over one compressor and one sample memory."""
    return ArchitectureContractReport(
        results=(
            check_bypass_identity(memory),
            check_page_order_preserved(compressor, memory),
            check_mask_semantics(compressor, memory),
            check_deterministic_shape(compressor, memory),
            check_summary_path_present(compressor, memory),
            check_alignment_integrity(compressor, memory),
            check_hashes_present(implementation_hash, policy_hash),
        )
    )
