"""Versioned explicit configuration; no backbone-specific defaults."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionHeadConfig:
    kind: str
    visual_dim: int
    text_dim: int
    hidden: int

    def __post_init__(self) -> None:
        if self.kind not in ("candidate_attention", "bilinear_moments"):
            raise ValueError(f"decision config: unsupported head kind {self.kind!r}")
        for name in ("visual_dim", "text_dim", "hidden"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"decision config: {name} must be a positive integer")
