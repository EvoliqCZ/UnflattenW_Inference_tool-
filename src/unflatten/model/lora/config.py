"""RFC005 §25.1 LoRA contract fields, as a plain config object.

§25.1 requires every LoRA branch to pin: target module names/patterns,
rank, alpha/scaling, dropout, initialization, precision, base checkpoint
hash, merge/unmerge algorithm, an optional input/warm-start adapter hash
(else explicit `none`), and trainable parameter count. This dataclass
carries the pre-run policy fields; `trainable_parameter_count` and the
produced adapter weight hash are evolving *outputs* (§25.1 last
paragraph) and are computed from a built `LoRALinear`, not stored here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LoRAConfig:
    """One LoRA branch's pre-run policy (RFC005 §25.1).

    `target_modules` is an explicit, closed tuple of module names/patterns
    — §25.1's "wildcard target that expands after a backbone config
    change is a hard compatibility error" is enforced by callers matching
    against this exact tuple, never a glob evaluated at match time.
    """

    rank: int
    alpha: float
    target_modules: tuple[str, ...]
    dropout: float = 0.0
    initialization: str = "kaiming_uniform_A_zero_B"
    precision: str = "fp32"
    base_checkpoint_hash: str = "none"
    merge_algorithm: str = "additive"
    warm_start_adapter_hash: str = "none"
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not self.target_modules:
            raise ValueError("target_modules must be a non-empty closed tuple")
        if self.initialization != "kaiming_uniform_A_zero_B":
            raise ValueError(
                "only 'kaiming_uniform_A_zero_B' is implemented — DEC-047 "
                "condition 2 requires step 0 == base output, which only a "
                "zero-initialized B factor guarantees"
            )

    @property
    def scaling(self) -> float:
        """`alpha / rank`, applied to the `A @ B` delta (§25.1 alpha/scaling)."""
        return self.alpha / self.rank
