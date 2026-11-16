"""Small shared-feature scorers, explicitly called outside default model paths.

Candidate vectors already represent question + answer. No tokenizer, encoder,
AST editing, calibration or reasoning-residual integration is performed here.
"""

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn

from .config import DecisionHeadConfig


@dataclass(frozen=True)
class DecisionResult:
    logits: Tensor
    scores: Tensor
    choice_index: Tensor
    abstained: Tensor


def _moments(visual: Tensor, mask: Tensor) -> Tensor:
    values = visual.float().masked_fill(~mask[..., None], 0)
    n = mask.sum(1).clamp_min(1).float()[:, None]
    mean = values.sum(1) / n
    centered = (values - mean[:, None]).masked_fill(~mask[..., None], 0)
    variance = centered.square().sum(1) / n
    # sqrt(0) has an infinite derivative; a zero-variance channel is constant.
    std = torch.where(variance > 0, variance.clamp_min(1e-30).sqrt(), 0)
    return torch.cat((mean, std), dim=-1)


class _CandidateAttention(nn.Module):
    def __init__(self, config: DecisionHeadConfig) -> None:
        super().__init__()
        h = config.hidden
        self.visual = nn.Sequential(
            nn.LayerNorm(config.visual_dim), nn.Linear(config.visual_dim, h)
        )
        self.text = nn.Sequential(nn.LayerNorm(config.text_dim), nn.Linear(config.text_dim, h))
        self.score = nn.Sequential(nn.Linear(h * 4, h), nn.GELU(), nn.Linear(h, 1))
        self.scale = sqrt(h)

    def forward(self, visual: Tensor, candidates: Tensor, mask: Tensor) -> Tensor:
        dtype = self.visual[1].weight.dtype
        v = self.visual(visual.to(dtype=dtype))
        q = self.text(candidates.to(dtype=dtype))
        attention = (q @ v.transpose(1, 2) / self.scale).float()
        attention = attention.masked_fill(~mask[:, None], float("-inf"))
        attended = attention.softmax(-1).to(dtype=dtype) @ v
        features = torch.cat((q, attended, q * attended, q - attended), dim=-1)
        return self.score(features).squeeze(-1)


class _BilinearMoments(nn.Module):
    def __init__(self, config: DecisionHeadConfig) -> None:
        super().__init__()
        h = config.hidden
        self.visual = nn.Linear(config.visual_dim * 2, h, bias=False)
        self.text = nn.Sequential(
            nn.LayerNorm(config.text_dim), nn.Linear(config.text_dim, h, bias=False)
        )
        self.register_buffer("mu", torch.zeros(config.visual_dim * 2))
        self.register_buffer("sd", torch.ones(config.visual_dim * 2))
        self.scale = sqrt(h)

    def forward(self, visual: Tensor, candidates: Tensor, mask: Tensor) -> Tensor:
        return self.forward_moments(_moments(visual, mask), candidates)

    def forward_moments(self, moments: Tensor, candidates: Tensor) -> Tensor:
        x = nn.functional.normalize((moments.float() - self.mu.float()) / self.sd.float(), dim=-1)
        dtype = self.visual.weight.dtype
        return (
            torch.einsum(
                "bd,bkd->bk", self.visual(x.to(dtype=dtype)), self.text(candidates.to(dtype=dtype))
            )
            / self.scale
        )


class DecisionHead(nn.Module):
    """Scorer primitive. Scores are uncalibrated; no default model imports it.

    All masks are bool and share the inputs' device. Inactive padding may
    contain arbitrary values. Invalid active inputs raise; unavailable evidence
    abstains. Candidate ids/names and feature-contract compliance belong to the
    caller. This object is not an ADRF ReasoningModule.
    """

    def __init__(self, config: DecisionHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.network = (
            _CandidateAttention(config)
            if config.kind == "candidate_attention"
            else _BilinearMoments(config)
        )

    def _validate_visual(self, visual: Tensor, mask: Tensor) -> None:
        if visual.ndim != 3 or visual.shape[-1] != self.config.visual_dim:
            raise ValueError("decision input: visual_features must be [B,N,visual_dim]")
        if not visual.is_floating_point():
            raise ValueError("decision input: visual_features must be floating point")
        if mask.shape != visual.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("decision input: visual_mask must be bool [B,N]")
        if mask.device != visual.device or visual.device != next(self.parameters()).device:
            raise ValueError("decision input: visual_features, visual_mask and head device differ")
        if not torch.isfinite(visual[mask]).all():
            raise ValueError("decision input: active visual_features contain nonfinite values")

    def forward(
        self,
        visual_features: Tensor,
        candidate_features: Tensor,
        *,
        visual_mask: Tensor,
        candidate_mask: Tensor,
        eligible: Tensor,
    ) -> DecisionResult:
        self._validate_visual(visual_features, visual_mask)
        v, c = visual_features, candidate_features
        self._validate_candidates(c, v.shape[0], v.device, candidate_mask, eligible)
        valid = eligible & visual_mask.any(1) & (candidate_mask.sum(1) >= 2)
        raw = None
        if valid.any():
            clean_v = v[valid].masked_fill(~visual_mask[valid, :, None], 0)
            clean_c = c[valid].masked_fill(~candidate_mask[valid, :, None], 0)
            raw = self.network(clean_v, clean_c, visual_mask[valid]).float()
        return self._result(c, candidate_mask, valid, raw)

    def _validate_candidates(
        self,
        c: Tensor,
        batch_size: int,
        device: torch.device,
        candidate_mask: Tensor,
        eligible: Tensor,
    ) -> None:
        if c.ndim != 3 or c.shape[0] != batch_size or c.shape[-1] != self.config.text_dim:
            raise ValueError("decision input: candidate_features must be [B,K,text_dim]")
        if not c.is_floating_point():
            raise ValueError("decision input: candidate_features must be floating point")
        if candidate_mask.shape != c.shape[:2] or candidate_mask.dtype != torch.bool:
            raise ValueError("decision input: candidate_mask must be bool [B,K]")
        if eligible.shape != (batch_size,) or eligible.dtype != torch.bool:
            raise ValueError("decision input: eligible must be bool [B]")
        if any(t.device != device for t in (c, candidate_mask, eligible)):
            raise ValueError(
                "decision input: candidate_features and masks must share the head device"
            )
        if not torch.isfinite(c[candidate_mask]).all():
            raise ValueError("decision input: active candidate_features contain nonfinite values")

    @staticmethod
    def _result(c: Tensor, mask: Tensor, valid: Tensor, raw: Tensor | None) -> DecisionResult:
        logits = torch.full(c.shape[:2], float("-inf"), device=c.device, dtype=torch.float32)
        scores = torch.zeros_like(logits)
        choice = torch.full((c.shape[0],), -1, device=c.device, dtype=torch.long)
        if raw is not None:
            if not torch.isfinite(raw[mask[valid]]).all():
                raise ValueError("decision output: active logits contain nonfinite values")
            active = raw.masked_fill(~mask[valid], float("-inf"))
            logits[valid] = active
            scores[valid] = active.softmax(-1)
            choice[valid] = active.argmax(-1)
        return DecisionResult(logits, scores, choice, ~valid)

    def _validate_moments(self, moments: Tensor, present: Tensor) -> None:
        if not isinstance(self.network, _BilinearMoments):
            raise ValueError("decision moments: only bilinear_moments supports cached moments")
        if moments.ndim != 2 or moments.shape[1] != self.config.visual_dim * 2:
            raise ValueError(
                "decision moments: expected [B,2*visual_dim], mean then population std"
            )
        if not moments.is_floating_point():
            raise ValueError("decision moments: features must be floating point")
        if present.shape != moments.shape[:1] or present.dtype != torch.bool:
            raise ValueError("decision moments: visual_present must be bool [B]")
        if present.device != moments.device or moments.device != next(self.parameters()).device:
            raise ValueError("decision moments: features, visual_present and head device differ")
        if not torch.isfinite(moments[present]).all():
            raise ValueError("decision moments: observed features contain nonfinite values")
        if (moments[present, self.config.visual_dim :] < 0).any():
            raise ValueError("decision moments: population standard deviations must be nonnegative")

    def forward_moments(
        self,
        moments: Tensor,
        candidate_features: Tensor,
        *,
        visual_present: Tensor,
        candidate_mask: Tensor,
        eligible: Tensor,
    ) -> DecisionResult:
        """Score cached mean/population-std from the same contracted visual tap.

        No token attention or alternate pooling is inferred. Missing visual rows
        can contain arbitrary padding. Caller proves cache and train provenance.
        """
        self._validate_moments(moments, visual_present)
        c = candidate_features
        self._validate_candidates(c, moments.shape[0], moments.device, candidate_mask, eligible)
        valid = eligible & visual_present & (candidate_mask.sum(1) >= 2)
        raw = None
        if valid.any():
            clean_c = c[valid].masked_fill(~candidate_mask[valid, :, None], 0)
            raw = self.network.forward_moments(moments[valid], clean_c).float()
        return self._result(c, candidate_mask, valid, raw)

    def set_train_moment_stats(self, moments: Tensor, *, visual_present: Tensor) -> None:
        """Update the bilinear normalizer from unique, observed train-view moments."""
        self._validate_moments(moments, visual_present)
        if not visual_present.any():
            raise ValueError("decision stats: at least one observed training sample is required")
        with torch.no_grad():
            x = moments[visual_present].float()
            self.network.mu.copy_(x.mean(0))
            self.network.sd.copy_(x.std(0, unbiased=False).clamp_min(1e-6))

    def set_train_stats(self, visual_features: Tensor, *, visual_mask: Tensor) -> None:
        """Explicit train-only normalizer update; caller supplies train evidence."""
        self._validate_visual(visual_features, visual_mask)
        if not isinstance(self.network, _BilinearMoments):
            raise ValueError("decision stats: only bilinear_moments supports train statistics")
        observed = visual_mask.any(1)
        if not observed.any():
            raise ValueError("decision stats: at least one observed training sample is required")
        with torch.no_grad():
            x = _moments(visual_features[observed], visual_mask[observed])
            self.network.mu.copy_(x.mean(0))
            self.network.sd.copy_(x.std(0, unbiased=False).clamp_min(1e-6))
