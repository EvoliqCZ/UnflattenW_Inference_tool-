"""S6 dormant paired-arm trainer scaffolding (RFC005 §20, DEC-047).

DEC-047 authorizes the sparse Word Expert Adapter and this trainer as
DORMANT code ahead of S6/G2 ratification. Condition 5 ("no gradient runs")
means this module's real training path is refused unconditionally today:
no `docs/decisions/DEC-*.md` ratification record exists yet, so
:func:`require_s6_ratification` can only ever raise.

Arms are built by :func:`build_arms` (dense control and a `variant=sparse`
challenger from one seed). This module is on the short allowlist of stage
trainers that may select a variant (`test_candidate_model_architecture_variant
.py::TestNoProductionReferencesTheVariant`); production paths (service,
`s4_train`, `select_checkpoint`, decode CLIs) may not. Building arms is not
training. The one real CLI operation is `--admission-only` (exposure plans and
manifest, no model); `--dry-run` takes optimizer steps, which DEC-047 forbids
outside tests, so the CLI refuses it and the test suite calls
:func:`run_s6_dry_run` directly.

This module REUSES `s4_train`'s machinery by composition, never by
forking it (see the repo lesson "a copied stage module keeps old
identity"):

- dataset: `s1_train.ExactTextTrainingDataset` + `s4_typed_loss
  .S4TrainingDataset` (`stl`), unchanged;
- loss: `s4_typed_loss.s4_total_loss`, unchanged, called the same way
  `s4_train.evaluate_s4`/`run_s4_training` call it (the ~10-line
  encode_hidden -> composite_head -> s4_total_loss snippet is short
  enough that forking a shared helper out of the live, in-production
  `s4_train.py` for this dormant module would be a riskier change than
  the small duplication);
- checkpoint: `s1_train.save_checkpoint`/`load_checkpoint`, unchanged;
- ledger: `s2_exposure.ExposureLedger` + `start_attempt`/
  `finalize_attempt`, unchanged;
- eval: `s4_train.evaluate_s4`, unchanged, reused directly for BOTH arms
  (it is variant-agnostic: it only calls `model.encode_hidden`/
  `model.composite_head`, which both a dense- and sparse-variant
  `QwenDenseCandidateModel` expose identically).

Every S6 identity this module mints (`run_id` prefix, hash domains, the
manifest filename) is its OWN, never an S4 one -- `s4run_`/S4's domains
never appear in an S6 artefact (see `test_s6_train.py
::test_manifest_has_s6_identities_not_s4`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from unflatten.dataset.quality_gate.gate import require_data_quality_gate
from unflatten.model import composite_vocab as cv
from unflatten.model import onfly_visual as ov
from unflatten.model import s1_train
from unflatten.model import s2_exposure as ex
from unflatten.model import s4_train as s4t
from unflatten.model import s4_typed_loss as stl
from unflatten.model.candidate_architecture_variant import (
    CandidateArchitectureVariant,
    ExpertAdapterVariant,
    LevelRouterVariant,
)
from unflatten.model.candidate_model import (
    R1_EXPLICIT_ZERO_BIN_POINTERS,
    QwenDenseCandidateModel,
)
from unflatten.model.general_stage_gate_policy import general_stage_gate_policy_hash
from unflatten.model.hashing import domain_hash, make_id
from unflatten.model.level_router.policy import (
    LevelRouterPolicy,
    LevelRouterPolicyError,
    load_level_router_policy,
)
from unflatten.model.level_router.ratification import (
    LevelRouterRatificationRequired,
    require_level_router_ratification,
)
from unflatten.model.s6_paired_exposure import (
    ARM_DENSE_CONTROL,
    ARM_SPARSE_CHALLENGER,
    RouterTelemetry,
    build_g2_gate_policy_draft,
    build_paired_exposure_plans,
)
from unflatten.model.sparse_expert_adapter import telemetry as sparse_telemetry
from unflatten.model.sparse_expert_adapter.manifest import (
    ExpertRole,
    SparseExpertAdapterManifest,
)
from unflatten.model.stage_data_matrix import build_stage_data_matrix
from unflatten.model.typed_property_heads import IntegerHeadConfig

#: S6's own filename/domain/prefix contract -- distinct from every S1-S4 one
#: on purpose (design docstring above; `RUN_ID_PREFIX`s cannot collide).
S6_RUN_MANIFEST_FILENAME = "s6_run_manifest.json"
#: Bumped 0.1.0 -> 0.2.0: adds `mask_inactive_union_arms` (RFC005-ERRATA-008
#: E1, T1) and `explicit_zero_bin_pointers` (RFC004-ERRATA-006 E1, R1) --
#: both DEC-049 point 1/2 fields the run manifest now records, absent from
#: 0.1.0 payloads.
S6_RUN_MANIFEST_SCHEMA_VERSION = "0.2.0"
RUN_ID_PREFIX = "s6run"
RUN_ID_DOMAIN = "rfc005.s6-run-id/v1"
S6_RUN_MANIFEST_DOMAIN = "rfc005.s6-run-manifest/v1"
S6_STAGE_POLICY_DOMAIN = "rfc005.s6-stage-policy/v1"

#: The committed S6 stage-policy draft (router warm-up K + trainable-param
#: prefixes + lambda_router) -- every numeric value inside it is a named
#: PLACEHOLDER (DEC-047 condition 4: topology/thresholds are policy fields).
_S6_STAGE_POLICY_PATH = Path(__file__).parent / "policies" / "s6_stage_policy_draft.json"

#: S4's own committed stage policy (typed-head loss families/weights) is
#: reused UNCHANGED for both arms' composite loss -- S6 only adds a router
#: term on top for the sparse arm, it does not redefine the typed-head
#: loss surface S4 already owns.
_S4_STAGE_POLICY_PATH = s4t._POLICIES_DIR / "s4_stage_policy.json"

_ROUTER_TELEMETRY_FILENAME = "s6_router_telemetry.jsonl"


class S6TrainingRefused(RuntimeError):
    """A refusal specific to this module: missing ratification, an
    architecture-manifest-ref collision, or a malformed stage policy."""


# ---------------------------------------------------------------------------
# S6 stage policy (router warm-up K + trainable prefixes + lambda_router)
# ---------------------------------------------------------------------------


def load_s6_stage_policy(path: Path | str = _S6_STAGE_POLICY_PATH) -> dict[str, Any]:
    """Read the committed S6 stage-policy draft -- no schema validator of
    its own yet (mirrors `s6_paired_exposure.py`'s own lightweight,
    schema-free `general_stage_gate_policy` draft pattern): a hand-checked
    handful of fields, not a closed-envelope identity surface other
    stages already depend on."""
    payload = json.loads(Path(path).read_text())
    for key in (
        "s6_stage_policy_schema_version",
        "router_warmup_steps",
        "warmup_trainable_param_prefixes",
        "lambda_router_milli",
    ):
        if key not in payload:
            raise S6TrainingRefused(f"s6 stage policy is missing required key {key!r}")
    if not isinstance(payload["router_warmup_steps"], int) or payload["router_warmup_steps"] < 0:
        raise S6TrainingRefused("router_warmup_steps must be an integer >= 0")
    if not isinstance(payload["lambda_router_milli"], int) or payload["lambda_router_milli"] < 0:
        raise S6TrainingRefused("lambda_router_milli must be an integer >= 0")
    return payload


def s6_stage_policy_hash(policy: Mapping[str, Any]) -> str:
    return domain_hash(S6_STAGE_POLICY_DOMAIN, dict(policy))


def lambda_router(policy: Mapping[str, Any]) -> float:
    return policy["lambda_router_milli"] / 1000.0


# ---------------------------------------------------------------------------
# Two arms, one seed (RFC005 §20.2: same parent init for both) -- `build_arms`
# below builds them; the other functions operate on already-built models.
# ---------------------------------------------------------------------------


def default_sparse_manifest(*, d_model: int, dense_layers: int = 1) -> SparseExpertAdapterManifest:
    """A tiny, two-specialist reference topology for the dormant S6
    scaffold (DEC-047 condition 4: topology is a manifest field, never a
    module constant -- a real S6R specialization picks its own). Building
    a manifest mints no `nn.Module` and selects no variant by itself."""
    return SparseExpertAdapterManifest(
        d_model=d_model,
        hidden_multiplier=4,
        layers=dense_layers,
        dropout=0.0,
        top_k=1,
        alpha_shared=1.0,
        specialist_roles=(
            ExpertRole(expert_id="spec_structure", logical_role="structure_flow"),
            ExpertRole(expert_id="spec_text", logical_role="text_ocr"),
        ),
        insertion_point="node_level_bank",
    )


def resync_sparse_shared_expert(sparse_model: QwenDenseCandidateModel) -> None:
    """Re-copy `sparse_model.dense_adapter`'s CURRENT weights into
    `sparse_model.sparse_adapter.shared_expert` (the same copy
    `SparseExpertAdapter.from_dense` performs at construction).

    A same-seed construction already leaves this a no-op; it exists so a
    caller that later warm-starts BOTH arms from a real checkpoint (loading
    `dense_adapter`'s weights fresh into the sparse arm, which
    `sparse_adapter.shared_expert` was built from BEFORE that load) can
    re-establish step-0 dense/sparse equivalence with one explicit call,
    instead of silently drifting. Takes an ALREADY-BUILT sparse-variant
    model -- it never selects the variant itself.
    """
    assert sparse_model.sparse_adapter is not None
    sparse_model.sparse_adapter.shared_expert.blocks.load_state_dict(
        sparse_model.dense_adapter.layers.state_dict()
    )


# ---------------------------------------------------------------------------
# Router warm-up (RFC005 §20.2): freeze everything but the router for K steps
# ---------------------------------------------------------------------------


def apply_router_warmup(
    model: QwenDenseCandidateModel, *, trainable_param_prefixes: Sequence[str]
) -> dict[str, bool]:
    """Snapshot every named parameter's CURRENT `requires_grad`, then set
    it to `True` only for a name starting with one of
    `trainable_param_prefixes` (default policy: `sparse_adapter.router.`)
    and `False` for everything else. Returns the snapshot so
    :func:`restore_trainable_state` can undo this exactly once warm-up
    ends -- the "then the ratified trainable groups" half of RFC005
    §20.2, expressed as "whatever this model's trainable state was right
    before warm-up started", since S6 has no ratified group schedule of
    its own yet."""
    snapshot = {name: param.requires_grad for name, param in model.named_parameters()}
    for name, param in model.named_parameters():
        param.requires_grad_(any(name.startswith(prefix) for prefix in trainable_param_prefixes))
    return snapshot


def restore_trainable_state(model: QwenDenseCandidateModel, snapshot: Mapping[str, bool]) -> None:
    for name, param in model.named_parameters():
        param.requires_grad_(snapshot[name])


# ---------------------------------------------------------------------------
# Loss: the S4 composite loss (unchanged) + the sparse arm's router term
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ArmStepResult:
    """One arm's forward/loss result for one batch -- `router_loss` and
    `router_telemetry` are `None` on the dense arm (it has no router)."""

    total_loss: torch.Tensor
    composite_loss: torch.Tensor
    report: dict[str, Any]
    router_loss: torch.Tensor | None
    router_telemetry: RouterTelemetry | None


def _composite_forward(
    model: QwenDenseCandidateModel,
    *,
    batch: s1_train.S1TrainingBatch,
    typed_batch: stl.TypedBatch,
    region_batch: stl.RegionBatch,
    stage_policy: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
    """The same `encode_hidden -> composite_head -> s4_total_loss` call
    `s4_train.evaluate_s4`/`run_s4_training` make -- see the module
    docstring for why this ~10-line snippet is duplicated rather than
    imported from a private helper factored out of the live S4 trainer.
    Returns `(total, report, composite_hidden)`: the caller needs
    `composite_hidden` again for the sparse arm's own router telemetry.
    """
    hidden, prefix_len = model.encode_hidden(batch.model_batch)
    composite_hidden = hidden[:, prefix_len + batch.visual_steps :, :]
    logits = model.composite_head(composite_hidden)
    total, report = stl.s4_total_loss(
        logits=logits,
        target_ids=batch.target_ids,
        loss_mask=batch.loss_mask,
        loss_family_ids=batch.loss_family_ids,
        model=model,
        hidden=hidden,
        prefix_len=prefix_len,
        visual_steps=batch.visual_steps,
        typed_batch=typed_batch,
        stage_policy=stage_policy,
        region_batch=region_batch,
        image_grid_thw=batch.model_batch.image_grid_thw,
    )
    return total, report, composite_hidden


def dense_arm_step(
    model: QwenDenseCandidateModel,
    *,
    batch: s1_train.S1TrainingBatch,
    typed_batch: stl.TypedBatch,
    region_batch: stl.RegionBatch,
    stage_policy: Mapping[str, Any],
) -> ArmStepResult:
    total, report, _ = _composite_forward(
        model, batch=batch, typed_batch=typed_batch, region_batch=region_batch,
        stage_policy=stage_policy,
    )
    return ArmStepResult(
        total_loss=total, composite_loss=total, report=report,
        router_loss=None, router_telemetry=None,
    )


def sparse_arm_step(
    model: QwenDenseCandidateModel,
    *,
    batch: s1_train.S1TrainingBatch,
    typed_batch: stl.TypedBatch,
    region_batch: stl.RegionBatch,
    stage_policy: Mapping[str, Any],
    lambda_router_value: float,
) -> ArmStepResult:
    """The S4 composite loss PLUS `lambda_router_value * L_router`
    (RFC005 §27.8's `l_router`, `sparse_expert_adapter.telemetry`).

    `L_router`'s routing decision is taken over the POST-adapter composite
    hidden state (`composite_hidden`, what `_composite_forward` already
    computes) rather than the pre-adapter hidden `_apply_expert_adapter`
    actually routed on internally -- `candidate_model.py`'s pre-adapter
    hidden is not exposed publicly, and this module may not edit
    `candidate_model.py` (another agent owns its decode paths). This is a
    deliberate, documented simplification for a dormant scaffold that
    never takes a real gradient step: the ACTUAL specialist forward pass
    inside `model.encode_hidden` already used the correct pre-adapter
    hidden (so the composite loss/logits are exact), only this auxiliary
    router-telemetry/loss term is computed against a proxy tensor of the
    right shape and dtype.
    """
    assert model.sparse_adapter is not None
    total, report, composite_hidden = _composite_forward(
        model, batch=batch, typed_batch=typed_batch, region_batch=region_batch,
        stage_policy=stage_policy,
    )
    routing = model.sparse_adapter.route(composite_hidden)
    router_loss = sparse_telemetry.l_router(routing)
    counts = sparse_telemetry.expert_selection_counts(routing)
    telemetry = RouterTelemetry(
        dead_expert_count=len(sparse_telemetry.detect_dead_experts(counts)),
        router_entropy_micro=int(
            round(
                float(sparse_telemetry.routing_entropy(routing.weights).mean().detach())
                * 1_000_000
            )
        ),
        token_drop_count=0,
        total_tokens=int(routing.selected.shape[:-1].numel()),
    )
    combined = total + lambda_router_value * router_loss
    return ArmStepResult(
        total_loss=combined, composite_loss=total, report=report,
        router_loss=router_loss, router_telemetry=telemetry,
    )


# ---------------------------------------------------------------------------
# S6 run manifest -- S6's own identities, never S4's
# ---------------------------------------------------------------------------


def build_s6_run_manifest(
    *,
    run_seed: int,
    dense_arm: Mapping[str, Any],
    sparse_arm: Mapping[str, Any],
    s6_stage_policy_hash_value: str,
    g2_gate_policy_hash_value: str,
    mask_inactive_union_arms: bool = False,
    explicit_zero_bin_pointers: frozenset[tuple[str, str]] = frozenset(),
    structural_table_region: str = "abstain_not_visible",
    data_quality_gate_hash: str | None = None,
    level_router_policy_hash: str | None = None,
    visual_input: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The frozen S6 run manifest: `stage_id` is `"S6"`, `run_id` carries
    the `s6run_` prefix, and every hash inside it is either re-derived
    here (self-hash) or already the caller's own S6-domained hash
    (`dense_arm`/`sparse_arm` from `s6_paired_exposure`, which hashes
    under `rfc005.paired-exposure-arm/v1` -- never S4's plan-hash
    domain). Frozen BEFORE any gradient step, per DEC-047 condition 5.

    `mask_inactive_union_arms` (RFC005-ERRATA-008 E1, T1) and
    `explicit_zero_bin_pointers` (RFC004-ERRATA-006 E1, R1) are DEC-049
    points 1/2: both arms of one S6 run share exactly one value for each
    (`run_s6_dry_run`/`build_arms` never take an arm-specific value), so
    recording them here once is the run's own commitment, not per-arm
    data. Default `False`/`frozenset()` reproduces a 0.1.0 payload's
    fields exactly plus the two new keys -- never silently omitted.
    """
    if type(run_seed) is not int or run_seed < 0:
        raise S6TrainingRefused(f"run_seed must be an integer >= 0, got {run_seed!r}")
    base = {
        "s6_run_manifest_schema_version": S6_RUN_MANIFEST_SCHEMA_VERSION,
        "stage_id": "S6",
        "run_seed": run_seed,
        "dense_arm_id": dense_arm["arm_id"],
        "sparse_arm_id": sparse_arm["arm_id"],
        "dense_arm_hash": dense_arm["paired_exposure_arm_hash"],
        "sparse_arm_hash": sparse_arm["paired_exposure_arm_hash"],
        "s6_stage_policy_hash": s6_stage_policy_hash_value,
        "g2_gate_policy_hash": g2_gate_policy_hash_value,
        "mask_inactive_union_arms": mask_inactive_union_arms,
        "explicit_zero_bin_pointers": sorted(
            [list(entry) for entry in explicit_zero_bin_pointers]
        ),
    }
    # DEC-055 V6b: written only when selected -- absent == the v15b region
    # channel, so a run that does not use it keeps its exact 0.2.0 payload.
    if structural_table_region != "abstain_not_visible":
        base["structural_table_region"] = structural_table_region
    # DEC-057 point 6: a run on v16 caches carries the passing data-quality
    # gate result's hash in its identity (`quality_gate.gate.
    # require_data_quality_gate`); absent == a pre-v16 run, payload unchanged.
    if data_quality_gate_hash is not None:
        base["data_quality_gate_hash"] = data_quality_gate_hash
    # RFC007 variant B (PROPOSED): the same level router in BOTH arms, so the
    # dense/MoE comparison stays about the expert adapter; written only when
    # selected, absent == no router.
    if level_router_policy_hash is not None:
        base["level_router_policy_hash"] = level_router_policy_hash
    # DEC-056 addendum A: the on-the-fly ViT block (policy hash, DeepStack
    # tap, augmentation seed + DEC-040 policy hash) -- written only for an
    # on-the-fly run, so every other run keeps its exact 0.2.0 payload.
    if visual_input is not None:
        base["visual_input"] = dict(visual_input)
    run_id = make_id(RUN_ID_PREFIX, RUN_ID_DOMAIN, base)
    payload = {**base, "run_id": run_id}
    payload["s6_run_manifest_hash"] = domain_hash(S6_RUN_MANIFEST_DOMAIN, payload)
    return payload


def verify_s6_run_manifest(manifest: Mapping[str, Any]) -> None:
    payload = {k: v for k, v in manifest.items() if k != "s6_run_manifest_hash"}
    if domain_hash(S6_RUN_MANIFEST_DOMAIN, payload) != manifest.get("s6_run_manifest_hash"):
        raise S6TrainingRefused("s6_run_manifest_hash does not verify against the payload")


def load_s6_run_manifest(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    verify_s6_run_manifest(payload)
    return payload


# ---------------------------------------------------------------------------
# Ratification gate (DEC-047 condition 5) -- a real, tested refusal
# ---------------------------------------------------------------------------


def require_s6_ratification(record_path: Path | str | None) -> None:
    """No S6 gradient step may run without `--i-have-s6-ratification
    <path>` naming a FILE THAT EXISTS. No such ratification record exists
    in this repository today (`docs/decisions/` carries no S6/G2
    ratification), so this refuses unconditionally in production; the
    flag's only purpose today is to make that refusal a real, exercised
    code path rather than an unreachable `if` no test can prove."""
    if record_path is None:
        raise S6TrainingRefused(
            "S6 requires --i-have-s6-ratification <path to a DEC ratification "
            "record> naming a file that exists (DEC-047 condition 5: no "
            "gradient step without ratification) -- none was given"
        )
    if not Path(record_path).is_file():
        raise S6TrainingRefused(
            f"--i-have-s6-ratification {record_path!r} does not name an "
            "existing file -- no gradient step without ratification"
        )


def build_arms(
    *,
    candidate: Any,
    seed: int,
    d_model: int,
    visual_dim: int,
    sparse_manifest: Any,
    typed_integer_config: IntegerHeadConfig = IntegerHeadConfig(strategy="direct_regression"),
    explicit_zero_bin_pointers: frozenset[tuple[str, str]] = frozenset(),
    level_router_policy: LevelRouterPolicy | None = None,
    deepstack: bool = False,
) -> tuple[QwenDenseCandidateModel, QwenDenseCandidateModel]:
    """`(dense_control, sparse_challenger)` from the SAME seed (RFC005 §20.2):
    the sparse arm is `variant=sparse` over an identically initialised dense
    model, its shared expert re-synced to the dense adapter, so both arms give
    the same output at step 0. Building arms is not training: every gradient
    path in this module stays behind the S6 ratification gate.

    `typed_integer_config`/`explicit_zero_bin_pointers` (RFC004-ERRATA-006
    E1, R1 / DEC-049 point 2) are passed to `_build` for BOTH arms from the
    same closed-over values -- there is no per-arm override, which is what
    keeps the two arms' typed-head configuration identical by construction
    rather than by a separate equality check. `level_router_policy` (RFC007
    variant B, PROPOSED) follows the same rule: one value, both arms.

    `deepstack` (DEC-056, default off) builds both arms as DeepStack models
    (no parameter, no RNG draw) -- the same single value for both arms."""
    ranges = cv.packed_ranges(candidate.vocab_size)

    def _build(variant: CandidateArchitectureVariant | None) -> QwenDenseCandidateModel:
        torch.manual_seed(seed)
        return QwenDenseCandidateModel(
            decode=candidate.decode,
            inject_visual=candidate.inject_visual,
            visual_dim=visual_dim,
            d_model=d_model,
            ranges=ranges,
            qwen_embedding_weight=candidate.embed_tokens.weight,
            qwen_lm_head_weight=candidate.lm_head_weight,
            typed_integer_config=typed_integer_config,
            explicit_zero_bin_pointers=explicit_zero_bin_pointers,
            variant=variant,
            **(
                {"deepstack": True, "image_token_id": candidate.image_token_id}
                if deepstack
                else {}
            ),
        )

    router = (
        LevelRouterVariant(enabled=True, policy=level_router_policy)
        if level_router_policy is not None
        else LevelRouterVariant()
    )
    dense = _build(
        CandidateArchitectureVariant(level_router=router)
        if level_router_policy is not None
        else None
    )
    sparse = _build(
        CandidateArchitectureVariant(
            expert_adapter=ExpertAdapterVariant(mode="sparse", sparse_manifest=sparse_manifest),
            level_router=router,
        )
    )
    resync_sparse_shared_expert(sparse)
    # The sparse arm's specialists draw RNG before the router is built, so
    # the two routers would start from different weights; copy the dense
    # arm's so both arms start identical (RFC007 §7.2: B is not the S6 axis).
    if level_router_policy is not None:
        sparse.level_router.load_state_dict(dense.level_router.state_dict())
    return dense, sparse


# ---------------------------------------------------------------------------
# The dry-run loop: both arms, same batches, own ledgers
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ArmRunResult:
    ledger: ex.ExposureLedger
    router_telemetry_records: tuple[dict[str, Any], ...]


def _arm_run_dir(run_dir: Path, arm_id: str) -> Path:
    path = run_dir / arm_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_s6_dry_run(
    *,
    run_dir: Path,
    dense_model: QwenDenseCandidateModel,
    sparse_model: QwenDenseCandidateModel,
    dataset: stl.S4TrainingDataset,
    image_token_id: int,
    plan_order: Sequence[int],
    plan_lineages: Sequence[str],
    plan_cells: Sequence[tuple[str, str, str, str]],
    dense_run_id: str,
    sparse_run_id: str,
    steps: int,
    batch_size: int,
    stage_policy: Mapping[str, Any],
    allowed_head_ids: Collection[str],
    s6_stage_policy: Mapping[str, Any],
    optimizer_factory: Callable[[QwenDenseCandidateModel], torch.optim.Optimizer],
    device: torch.device | str = "cpu",
    grad_clip: float = 1.0,
    start_step: int = 0,
    dense_ledger: ex.ExposureLedger | None = None,
    sparse_ledger: ex.ExposureLedger | None = None,
    mask_inactive_union_arms: bool = False,
    structural_table_region: str = "abstain_not_visible",
    visual_encoder: ov.OnTheFlyVisualEncoder | None = None,
) -> tuple[ArmRunResult, ArmRunResult]:
    """Run BOTH arms over the SAME paired plan order for
    `steps - start_step` optimizer steps each (RFC005 §20.5: identical
    exposure/order). CPU-only, tiny, meant for the STUB candidate --
    never gated by :func:`require_s6_ratification` itself (a caller who
    reaches this function already decided to run it; the gate lives in
    `main()`, on the boundary between the CLI and this function, exactly
    once).

    Router warm-up (RFC005 §20.2) applies ONLY to the sparse arm, for the
    first `s6_stage_policy["router_warmup_steps"]` steps of THIS call
    (not the global step -- mirrors `s4_train.run_s4_training`'s own
    early-refusal window being "of this attempt", never the global step).

    `mask_inactive_union_arms` (RFC005-ERRATA-008 E1, T1 / DEC-049 point 1):
    default `False` is byte-identical to before this parameter existed
    (`stl.collate_typed`'s own default). Both arms see the SAME batches
    already (this is the whole point of a paired run), so passing one
    value here -- never a per-arm one -- is what keeps that guarantee for
    T1 too.

    `visual_encoder` (DEC-056 addendum A): `dataset` is then an
    `onfly_visual.OnTheFlyDrawDataset` (fetched by draw index) and each batch
    is materialized ONCE by the frozen ViT, so both arms see the same visual
    features and DeepStack levels. `None`: cached features, as before."""
    if visual_encoder is not None and not isinstance(dataset, ov.OnTheFlyDrawDataset):
        raise S6TrainingRefused("visual_encoder needs an onfly_visual.OnTheFlyDrawDataset")
    if start_step >= steps:
        raise S6TrainingRefused(
            f"start_step {start_step} has already reached the run's total "
            f"budget of {steps} step(s)"
        )
    warmup_steps = s6_stage_policy["router_warmup_steps"]
    warmup_prefixes = s6_stage_policy["warmup_trainable_param_prefixes"]
    lambda_router_value = lambda_router(s6_stage_policy)

    dense_optimizer = optimizer_factory(dense_model)
    sparse_optimizer = optimizer_factory(sparse_model)
    dense_ledger = dense_ledger if dense_ledger is not None else ex.ExposureLedger(dense_run_id)
    sparse_ledger = (
        sparse_ledger if sparse_ledger is not None else ex.ExposureLedger(sparse_run_id)
    )

    dense_dir = _arm_run_dir(run_dir, ARM_DENSE_CONTROL)
    sparse_dir = _arm_run_dir(run_dir, ARM_SPARSE_CHALLENGER)
    # Reuses `RUN_ID_DOMAIN` rather than minting a fourth S6 domain purely
    # for this integrity-record checksum -- the payload (not the domain)
    # is what makes the two hashes distinct.
    dense_integrity = {"dense_run_id": domain_hash(RUN_ID_DOMAIN, {"integrity_for": dense_run_id})}
    sparse_integrity = {
        "sparse_run_id": domain_hash(RUN_ID_DOMAIN, {"integrity_for": sparse_run_id})
    }
    dense_attempt = ex.start_attempt(
        run_id=dense_run_id, attempt_index=0,
        execution_profile={"device": str(device)},
        start_integrity=dense_integrity, ledger=dense_ledger,
    )
    sparse_attempt = ex.start_attempt(
        run_id=sparse_run_id, attempt_index=0,
        execution_profile={"device": str(device)},
        start_integrity=sparse_integrity, ledger=sparse_ledger,
    )

    router_telemetry_records: list[dict[str, Any]] = []
    warmup_snapshot: dict[str, bool] | None = None
    step = start_step
    for lo in range(start_step * batch_size, steps * batch_size, batch_size):
        step += 1
        steps_this_call = step - start_step
        if steps_this_call == 1 and warmup_steps > 0:
            warmup_snapshot = apply_router_warmup(
                sparse_model, trainable_param_prefixes=warmup_prefixes
            )
        if warmup_snapshot is not None and steps_this_call == warmup_steps + 1:
            restore_trainable_state(sparse_model, warmup_snapshot)
            warmup_snapshot = None

        hi = lo + batch_size
        indices = plan_order[lo:hi]
        if visual_encoder is None:
            samples = [dataset[i] for i in indices]
        else:
            samples = [dataset[lo + offset].sample for offset in range(len(indices))]
        batch = s1_train.collate_training_batch(
            [s.base for s in samples], ranges=dense_model.ranges,
            image_token_id=image_token_id, device=device,
        )
        if visual_encoder is not None:
            batch = visual_encoder.materialize(batch)
        typed_batch = stl.collate_typed(
            samples,
            allowed_head_ids=allowed_head_ids,
            mask_inactive_union_arms=mask_inactive_union_arms,
        )
        region_batch = stl.collate_regions(
            samples, **stl.region_kwargs(structural_table_region)
        )

        dense_optimizer.zero_grad(set_to_none=True)
        dense_result = dense_arm_step(
            dense_model, batch=batch, typed_batch=typed_batch, region_batch=region_batch,
            stage_policy=stage_policy,
        )
        dense_result.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in dense_model.parameters() if p.requires_grad], grad_clip
        )
        dense_optimizer.step()

        sparse_optimizer.zero_grad(set_to_none=True)
        sparse_result = sparse_arm_step(
            sparse_model, batch=batch, typed_batch=typed_batch, region_batch=region_batch,
            stage_policy=stage_policy, lambda_router_value=lambda_router_value,
        )
        sparse_result.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in sparse_model.parameters() if p.requires_grad], grad_clip
        )
        sparse_optimizer.step()

        for arm_ledger, attempt in (
            (dense_ledger, dense_attempt),
            (sparse_ledger, sparse_attempt),
        ):
            for offset, _idx in enumerate(indices):
                draw_index = lo + offset
                arm_ledger.append(
                    attempt_id=attempt["attempt_id"],
                    lineage_root_id=plan_lineages[draw_index],
                    cell=plan_cells[draw_index],
                    view_ids=[f"draw:{draw_index}"],
                    sample_count=1,
                    optimizer_step=step,
                    plan_draw_index=draw_index,
                )

        assert sparse_result.router_telemetry is not None
        telemetry_record = {
            "step": step,
            "router_loss": float(sparse_result.router_loss.detach()),
            "dead_expert_count": sparse_result.router_telemetry.dead_expert_count,
            "router_entropy_micro": sparse_result.router_telemetry.router_entropy_micro,
            "token_drop_count": sparse_result.router_telemetry.token_drop_count,
            "total_tokens": sparse_result.router_telemetry.total_tokens,
        }
        router_telemetry_records.append(telemetry_record)

    with open(sparse_dir / _ROUTER_TELEMETRY_FILENAME, "w") as f:
        for record in router_telemetry_records:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    dense_ledger.save_jsonl(dense_dir / "ledger.jsonl")
    sparse_ledger.save_jsonl(sparse_dir / "ledger.jsonl")
    ex.finalize_attempt(
        manifest=dense_attempt, terminal_state="completed", ledger=dense_ledger,
        end_integrity=dense_integrity,
    )
    ex.finalize_attempt(
        manifest=sparse_attempt, terminal_state="completed", ledger=sparse_ledger,
        end_integrity=sparse_integrity,
    )
    return (
        ArmRunResult(ledger=dense_ledger, router_telemetry_records=()),
        ArmRunResult(
            ledger=sparse_ledger, router_telemetry_records=tuple(router_telemetry_records)
        ),
    )


# ---------------------------------------------------------------------------
# CLI-only default matrix/lineage/policy fixtures for `build_paired_exposure_
# plans` (a single, tiny, ILLUSTRATIVE cell -- no owner-ratified mass,
# cohort or lineage; production code must never import from `tests/`, so
# these live here, not in a test module, even though the CLI's two
# operations can never take a real gradient step over them today).
# `stage_id="S2"`: `stage_data_matrix`/`s2_sampler`'s own schemas only
# enumerate `S2`/`S4` (they predate S6) -- this reuses S2's slot rather than
# widening a shared, hashed schema for a dormant scaffold's own fixtures.
# ---------------------------------------------------------------------------

_CLI_COHORT = "s6_dry_run_illustrative"


def _cli_default_matrix() -> dict[str, Any]:
    return build_stage_data_matrix(
        matrix_version="1.0.0",
        stage_id="S2",
        splits=["train"],
        pyramid_levels=["P2"],
        task_kinds=["structure_v0"],
        cohorts=[_CLI_COHORT],
        cells=[
            {
                "split": "train", "pyramid_level": "P2", "task_kind": "structure_v0",
                "cohort": _CLI_COHORT, "mass_micro": 1_000_000,
            }
        ],
        declared_marginals={},
    )


def _cli_default_lineages() -> list[dict[str, Any]]:
    cell = {"split": "train", "pyramid_level": "P2", "task_kind": "structure_v0"}
    return [
        {
            "lineage_root_id": f"lin_{i}",
            "eligible_cells": [{**cell, "cohort": _CLI_COHORT}],
            "feature_coverage_milli": 500,
            "hardness_score_milli": 200,
            "features": ["f1"],
        }
        for i in range(3)
    ]


def _cli_default_sampler_policy() -> dict[str, Any]:
    return {
        "s2_sampler_policy_schema_version": "1.0.0",
        "policy_version": "1.0.0",
        "stage_id": "S2",
        "feature_coverage_bounds": {"min_milli": 100, "max_milli": 4000},
        "hardness_bounds": {"min_milli": 0, "max_milli": 1000},
        "w_rare_cap_milli": 2000,
        "virtual_epoch_size": 20,
    }


def _cli_default_exposure_policy() -> dict[str, Any]:
    return {
        "s2_exposure_policy_schema_version": "1.0.0",
        "policy_version": "1.0.0",
        "stage_id": "S2",
        "max_views_per_lineage_per_virtual_epoch": 10_000,
        "max_fragment_overlap_exposure": 0,
        "max_augmentation_variants": 1,
        "p5_repeat_cap": 0,
        "synthetic_recipe_family_cap": 0,
    }


# ---------------------------------------------------------------------------
# CLI -- `--admission-only` and `--dry-run` only (see module docstring)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.s6_train", description=__doc__
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dense-architecture-ref", required=True)
    parser.add_argument("--sparse-architecture-ref", required=True)
    parser.add_argument("--stage-policy", type=Path, default=_S6_STAGE_POLICY_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--admission-only", action="store_true")
    parser.add_argument("--i-have-s6-ratification", type=Path, default=None)
    parser.add_argument(
        "--mask-inactive-union-arms", action="store_true",
        help="RFC005-ERRATA-008 E1 (T1) / DEC-049 point 1: both arms of this "
        "run share this single value, recorded in the S6 run manifest.",
    )
    parser.add_argument(
        "--explicit-zero-bin-r1", action="store_true",
        help="RFC004-ERRATA-006 E1 (R1) / DEC-049 point 2: turn on "
        "candidate_model.R1_EXPLICIT_ZERO_BIN_POINTERS for both arms of "
        "this run, recorded in the S6 run manifest.",
    )
    parser.add_argument(
        "--structural-table-region",
        choices=list(stl.STRUCTURAL_TABLE_REGION_POLICIES),
        default="abstain_not_visible",
        help="DEC-055 addendum B (V6b, PROPOSED): 'exclude' stops training "
        "structural table/row/cell records with no box as 'not visible' in the "
        "region channel (counted as structural_table). Both arms share it; "
        "recorded in the S6 run manifest only when not the default.",
    )
    parser.add_argument(
        "--cache-manifest", action="append", default=[], metavar="SPLIT=PATH",
        help="DEC-057: the exact-text cache manifest(s) this run trains/evaluates on. "
        "A cache built under a v16 target policy requires --data-quality-gate.",
    )
    parser.add_argument(
        "--data-quality-gate", type=Path, default=None,
        help="DEC-057: the passing data-quality-gate.json that binds the caches; "
        "its hash is recorded in the S6 run manifest (v16 caches only).",
    )
    parser.add_argument(
        "--level-router-policy", type=Path, default=None,
        help="RFC007 variant B (PROPOSED): the same LevelRouterPolicy JSON for both "
        "arms, recorded in the S6 run manifest. Refused without "
        "--level-router-ratification.",
    )
    parser.add_argument("--level-router-ratification", type=Path, default=None)
    parser.add_argument(
        "--visual-input-policy", type=Path, default=None,
        help="DEC-056 addendum A: the on-the-fly ViT policy JSON both arms train under "
        "(DeepStack on iff it names taps); recorded as the S6 manifest's visual_input.",
    )
    parser.add_argument(
        "--onfly-augmentation-seed", type=int, default=None,
        help="DEC-040 augmentation re-drawn per (seed, step, page) on the on-the-fly path",
    )
    args = parser.parse_args(argv)
    if args.onfly_augmentation_seed is not None and args.visual_input_policy is None:
        parser.error("--onfly-augmentation-seed needs --visual-input-policy")
    visual_input = None
    if args.visual_input_policy is not None:
        visual_input = ov.visual_input_manifest_block(
            ov.VisualInputPolicy.from_payload(json.loads(args.visual_input_policy.read_text())),
            None
            if args.onfly_augmentation_seed is None
            else ov.OnTheFlyAugmentation(seed=args.onfly_augmentation_seed),
        )
    r1_pointers = R1_EXPLICIT_ZERO_BIN_POINTERS if args.explicit_zero_bin_r1 else frozenset()
    cache_manifests = {}
    for spec in args.cache_manifest:
        split, _, path = spec.partition("=")
        if not path:
            raise S6TrainingRefused(f"--cache-manifest expects SPLIT=PATH, got {spec!r}")
        cache_manifests[split] = json.loads(Path(path).read_text())
    gate_hash = require_data_quality_gate(cache_manifests, args.data_quality_gate)
    level_router_policy = None
    if args.level_router_policy is not None:
        try:
            level_router_policy = load_level_router_policy(args.level_router_policy)
            require_level_router_ratification(
                stage="S6",
                policy=level_router_policy,
                ratification_record=args.level_router_ratification,
            )
        except (LevelRouterPolicyError, LevelRouterRatificationRequired) as exc:
            raise S6TrainingRefused(f"--level-router-policy: {exc}") from exc

    s6_stage_policy = load_s6_stage_policy(args.stage_policy)
    matrix = _cli_default_matrix()
    lineages = _cli_default_lineages()
    dense_arm, sparse_arm = build_paired_exposure_plans(
        matrix=matrix, lineages=lineages, sampler_policy=_cli_default_sampler_policy(),
        exposure_policy=_cli_default_exposure_policy(), run_seed=args.seed,
        dense_architecture_manifest_ref=args.dense_architecture_ref,
        sparse_architecture_manifest_ref=args.sparse_architecture_ref,
    )
    g2_policy = build_g2_gate_policy_draft()
    manifest = build_s6_run_manifest(
        run_seed=args.seed, dense_arm=dense_arm, sparse_arm=sparse_arm,
        s6_stage_policy_hash_value=s6_stage_policy_hash(s6_stage_policy),
        g2_gate_policy_hash_value=general_stage_gate_policy_hash(g2_policy),
        mask_inactive_union_arms=args.mask_inactive_union_arms,
        explicit_zero_bin_pointers=r1_pointers,
        structural_table_region=args.structural_table_region,
        data_quality_gate_hash=gate_hash,
        level_router_policy_hash=(
            level_router_policy.policy_hash() if level_router_policy is not None else None
        ),
        visual_input=visual_input,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.run_dir / S6_RUN_MANIFEST_FILENAME
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    if args.admission_only:
        print(json.dumps({"admission_only": True, "run_id": manifest["run_id"]}, sort_keys=True))
        return 0

    if args.dry_run:
        # A dry run takes optimizer steps; DEC-047 allows no gradient step
        # outside tests until S6 is ratified. Tests call `run_s6_dry_run`
        # directly with arms from `build_arms` on the stub candidate.
        raise S6TrainingRefused(
            "--dry-run takes optimizer steps, which DEC-047 allows only in tests "
            "until S6 is ratified; call run_s6_dry_run(...) with build_arms(...) "
            "arms, as tests/model/test_s6_train.py does"
        )

    require_s6_ratification(args.i_have_s6_ratification)
    # Unreachable today (see module docstring): no ratification record
    # exists, so `require_s6_ratification` above always raises first.
    raise S6TrainingRefused(  # pragma: no cover
        "a real S6 gradient path is not built yet -- only --dry-run (via the "
        "Python API) and --admission-only are implemented"
    )


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
