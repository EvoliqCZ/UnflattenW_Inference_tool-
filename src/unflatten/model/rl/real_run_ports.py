"""Injected ports of the S4.5 real run (design D3/D4/D7/D8/D9/D10; PROPOSED, dormant).

The composer (:mod:`unflatten.model.rl.real_run`) opens NOTHING itself. Every
real resource -- the candidate and its policy model, the frozen reference, the
train-only pages, gold, the SFT auxiliary batch, the ledger, the checkpoint
directory, the render directory -- arrives through a :class:`RealRunPorts`
callable. There is no default that opens anything: a port that is missing
raises :class:`RealRunPortMissing` (naming it) before any resource is touched,
and nothing is resolved lazily on first use either.

This module is import-light on purpose (typing and torch only): the dry run
and ``s4_5_train`` import it for the type of the ``ports`` keyword without
pulling in the chain adapter or the model stack.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import nn

    from unflatten.model.composite_vocab import CompositeVocabulary
    from unflatten.model.rl.auxiliary import SftAuxBatch
    from unflatten.model.rl.data import RlPage, RlTrainPages
    from unflatten.model.rl.exposure import RlExposureLedger
    from unflatten.model.rl.loss import RlLossProvenance
    from unflatten.model.rl.render_probe import RenderProbeRecord
    from unflatten.model.rl.reward_inputs import GoldReference
    from unflatten.model.rl.selection import FrozenSelection


class RealRunRefused(ValueError):
    """The real run's inputs break its contract; raised before any step."""


class RealRunPortMissing(RealRunRefused):
    """A required port was not injected (no default opens anything real)."""


class RegistryUnbound(RealRunRefused):
    """The opened pages carry no exclusion-registry hash: the data plan would be the
    legacy, registry-less one. Refused unless ``allow_unbound_registry`` (fixtures only)."""


class AllBuildersInvalid(RealRunRefused):
    """Every rollout of one update step ended as ``BuilderReportInvalid``: there is
    no learning signal, only a uniform penalty, so the update is refused (never
    stepped) rather than trained on."""


@dataclass(frozen=True)
class OpenedCandidate:
    """What ``open_candidate`` returns: the trainable policy model and everything
    the rollout / update path needs around it.

    ``candidate_parameters`` is ``QwenCandidate.named_parameters`` -- the frozen
    backbone tensors -- so the composer can re-check, independently of the port,
    that the reference shares them (``assert_reference_shares_frozen_backbone``).
    """

    policy: nn.Module
    vocabulary: CompositeVocabulary
    controller: Any
    tokenizer: Any
    capability_profile_id: str
    image_token_id: int
    candidate_parameters: Mapping[str, Any]


@dataclass(frozen=True)
class OpenedPages:
    """The C1 train-only loader result, the C2 frozen selection and the hash of
    the frozen S4 StageDataMatrix the data plan references."""

    train_pages: RlTrainPages
    selection: FrozenSelection
    stage_data_matrix_hash: str
    #: ``registry_hash`` of the verified exclusion registry the pages were opened
    #: under; bound into the data plan when set (None = legacy plan hash).
    exclusion_registry_hash: str | None = None


@dataclass(frozen=True)
class PageVisual:
    """Visual input of one page. ``deepstack_visual_embeds`` is passed to the
    sampler AND to the teacher-forced update forward (a DeepStack model refuses
    a batch without it); ``raster_source`` only feeds the builder chain."""

    visual_features: torch.Tensor
    image_grid_thw: torch.Tensor
    deepstack_visual_embeds: Sequence[torch.Tensor] | None = None
    raster_source: Any = None


@dataclass(frozen=True)
class PageGold:
    """Scorer-side gold of one page (never a model tensor) plus its ledger
    identity. ``gold_node_ids`` is parallel to ``gold.structure``."""

    gold: GoldReference
    gold_node_ids: tuple[str, ...]
    gold_profile: str
    document_bundle_id: str
    source: str
    cell: tuple[str, str, str, str]


@dataclass(frozen=True)
class RealRunPorts:
    """Every real resource of the run, injected. All fields default to ``None``;
    :meth:`require_complete` names the first missing one."""

    open_candidate: Callable[[], OpenedCandidate] | None = None
    open_reference: Callable[[OpenedCandidate], nn.Module] | None = None
    open_pages: Callable[[Mapping[str, Any]], OpenedPages] | None = None
    read_selection: Callable[[], FrozenSelection] | None = None
    build_visual: Callable[[RlPage], PageVisual] | None = None
    build_gold: Callable[[RlPage], PageGold] | None = None
    build_sft_batch: Callable[[RlPage], SftAuxBatch] | None = None
    open_ledger: Callable[[str, Mapping[str, Any], str], RlExposureLedger] | None = None
    write_ledger: Callable[[RlExposureLedger], None] | None = None
    ancestry_base: Callable[[], Mapping[str, Any]] | None = None
    checkpoint_dir: Path | None = None
    render_dir: Path | None = None
    feature_source: str | None = None
    stop_requested: Callable[[], bool] | None = None
    #: test seam, optional: a loop-guard factory handed to ``sample_rollouts``.
    loop_guard_factory: Callable[[], Any] | None = None
    #: optional, status only: required iff the rollout policy says ``status_only``.
    render_probe: Callable[[Path, str, int], RenderProbeRecord] | None = None

    _OPTIONAL = ("loop_guard_factory", "render_probe", "build_sft_batch")

    def require_complete(self, *, supervised_weight_micro: int, render_probe_mode: str) -> None:
        optional = set(self._OPTIONAL)
        for f in fields(self):
            if f.name in optional or getattr(self, f.name) is not None:
                continue
            raise RealRunPortMissing(f"required port {f.name!r} was not injected")
        if supervised_weight_micro > 0 and self.build_sft_batch is None:
            raise RealRunPortMissing(
                "port 'build_sft_batch' is required: supervised_weight_micro > 0"
            )
        if render_probe_mode == "status_only" and self.render_probe is None:
            raise RealRunPortMissing(
                "port 'render_probe' is required: the rollout policy says render_probe=status_only"
            )


@dataclass(frozen=True)
class LoopContext:
    """Run-level, immutable inputs of the per-page work."""

    opened: OpenedCandidate
    reference: nn.Module
    rollout_policy: Mapping[str, Any]
    reward_policy: Mapping[str, Any]
    heads: tuple[str, ...]
    run_seed: int
    provenance: RlLossProvenance
    supervised_weight_micro: int
    ports: RealRunPorts


@dataclass(frozen=True)
class PageRecord:
    """What the ledger and the telemetry need to know about one page's group."""

    lineage_root_id: str
    source: str
    cell: tuple[str, str, str, str]
    view_id: str
    rollout_count: int
    guard_intervention_counts: tuple[int | None, ...]
    rollout_seed_digest: str
    statuses: tuple[str, ...]
    scalar_micro: tuple[int | None, ...]
    #: the chain's page outcome per rollout (``terminated``, ``DecodeMaxStepsExceeded``, ...).
    outcomes: tuple[str, ...]
    render_probe_statuses: tuple[str, ...]
