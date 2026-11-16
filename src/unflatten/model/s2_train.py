"""S2 = S1 admission + S2 layer (T5c, design §0/§5).

`s2_train.py` does not reimplement the S1 admission chain -- it REUSES
`s1_train`'s public functions (`ExactTextTrainingDataset`,
`collate_training_batch`, `train_step`, `save_checkpoint`,
`load_checkpoint`, `require_gradient_admissible`,
`unflatten.model.s1_runner.build_s1_optimizer`) and adds only what S1 has
no field for: an S2 run manifest binding a companion S1-format run
manifest, the committed matrix/sampler/exposure policies and ordered
exposure plan, the exposure ledger, and the stage-state check. Do not
modify `s1_train.py` behaviour (DEC-027 D1) -- if a change there ever
looks unavoidable, that is a stop, not a workaround here.

Admission runs in five refusal steps, all BEFORE the 4B model load (§5):

1. the S1 chain itself -- `s1_train.require_gradient_admissible` against
   the companion manifest, the allowlist, the checkpoint ancestry base and
   the pinned cohort expectation;
2. the S2 run manifest is bound to the SAME companion
   (`binding_s1_run_manifest_hash`) and declares `gradient_bearing`;
3. the committed matrix/sampler/exposure policies hash to what the S2
   manifest pins, and the ordered exposure plan verifies against them
   (`s2_run_manifest.verify_ordered_exposure_plan_against`) and against the
   manifest's own `ordered_exposure_plan_hash`;
4. every plan draw resolves to a page inside the pinned expectation and a
   position in the training dataset (`resolve_plan_pages` +
   `plan_dataset_order`) -- a draw naming a page outside the dataset is
   refused, never dropped;
5. the stage-state check (DEC-027 D4, RFC005-ERRATA-003): for every
   trainable-group-registry group, the state this MODEL actually exhibits
   (`trainable` iff it owns a `requires_grad` parameter under that group;
   otherwise the registry's own resting state for the group's kind) must
   be one `stage_state_policy` admits at S2, and a `promotion`-class run
   requires `node_visual_aligner` to be `trainable` (a frozen or
   not_applicable aligner is admitted only for S2 discovery).

The loop that follows is this module's own (~25 lines, §5): per step, the
next `batch_size` plan draws become dataset indices, collated and stepped
through `s1_train`'s own functions; one exposure-ledger record is appended
PER SAMPLE (never per batch), and a RunAttemptManifest brackets the whole
attempt (`completed` on a clean return, `failed` on any exception).
Checkpoint and ledger are written together, at the SAME cadence -- a
checkpoint whose companion ledger snapshot lags it cannot be resumed
consistently.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from unflatten.model import checkpoint_ancestry as ca
from unflatten.model import composite_vocab as cv
from unflatten.model import s1_train
from unflatten.model import s2_exposure as ex
from unflatten.model import s2_plan as sp
from unflatten.model import stage_state_policy as ssp
from unflatten.model.candidate_model import QwenDenseCandidateModel, parameter_group_ids
from unflatten.model.exact_text_cache import CacheExpectation, PageKey
from unflatten.model.s2_run_manifest import (
    S2RunManifestInvalid,
    load_s2_run_manifest,
    verify_ordered_exposure_plan_against,
)
from unflatten.model.s2_sampler import (
    load_s2_exposure_policy,
    load_s2_sampler_policy,
    s2_exposure_policy_hash,
    s2_sampler_policy_hash,
)
from unflatten.model.stage_data_matrix import load_stage_data_matrix, stage_data_matrix_hash
from unflatten.model.trainable_group_registry import trainable_group_registry

#: DEC-027 D3: the ledger's cell order for every S2 record -- the matrix
#: carries all its mass here.
CELL: tuple[str, str, str, str] = ("train", "P4", "structure_v0", "docxcorp")

_ATTEMPTS_DIRNAME = "attempts"
_LEDGER_FILENAME = "exposure_ledger.jsonl"

#: Design §12 S3: a worker that delivers nothing for this long raises
#: instead of hanging the run; the attempt then finalizes `failed`.
DATA_LOADER_TIMEOUT_S = 600
_PREFETCH_FACTOR = 2

#: The committed policy directory `main()` reads from by default
#: (overridable in tests via `monkeypatch.setattr(s2_train, "_POLICIES_DIR", ...)`,
#: the same pattern `s2_plan._POLICIES_DIR` uses).
_POLICIES_DIR = Path(__file__).parent / "policies"


class S2TrainingRefused(RuntimeError):
    """S2 admission (steps 1-5) or the resume consistency check refused
    the run. Always a refusal, never a downgrade -- there is no
    diagnostic mode for `s2_train`; a diagnostic S2 run uses `s1_train`
    directly (design §5)."""


# ---------------------------------------------------------------------------
# Step 2 -- the S2 manifest is bound to the presented companion
# ---------------------------------------------------------------------------


def require_step2_bound_to_companion(
    *, s2_manifest: Mapping[str, Any], companion: Mapping[str, Any]
) -> None:
    if s2_manifest["binding_s1_run_manifest_hash"] != companion["s1_run_manifest_hash"]:
        raise S2TrainingRefused(
            "s2_run_manifest.binding_s1_run_manifest_hash does not match the "
            "presented companion's s1_run_manifest_hash -- this S2 manifest "
            "is not bound to this companion"
        )
    if not s2_manifest["gradient_bearing"]:
        raise S2TrainingRefused(
            "s2_run_manifest declares gradient_bearing=False; it cannot "
            "authorize a gradient update"
        )


# ---------------------------------------------------------------------------
# Step 3 -- committed policies and plan verify against the S2 manifest
# ---------------------------------------------------------------------------


def require_step3_policies_bound(
    *,
    s2_manifest: Mapping[str, Any],
    matrix: Mapping[str, Any],
    sampler_policy: Mapping[str, Any],
    exposure_policy: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    checks = (
        ("stage_data_matrix_hash", stage_data_matrix_hash(matrix)),
        ("s2_sampler_policy_hash", s2_sampler_policy_hash(sampler_policy)),
        ("s2_exposure_policy_hash", s2_exposure_policy_hash(exposure_policy)),
    )
    for field, derived in checks:
        if s2_manifest[field] != derived:
            raise S2TrainingRefused(
                f"committed {field} ({derived}) does not match the pinned "
                f"s2_run_manifest value ({s2_manifest[field]})"
            )
    try:
        plan_hash = verify_ordered_exposure_plan_against(
            plan=plan,
            run_seed=s2_manifest["run_seed"],
            matrix=matrix,
            sampler_policy=sampler_policy,
            exposure_policy=exposure_policy,
        )
    except S2RunManifestInvalid as exc:
        raise S2TrainingRefused(f"ordered exposure plan refused: {exc}") from exc
    if plan_hash != s2_manifest["ordered_exposure_plan_hash"]:
        raise S2TrainingRefused(
            "the presented plan's hash does not match the pinned "
            "s2_run_manifest ordered_exposure_plan_hash"
        )


# ---------------------------------------------------------------------------
# Step 4 -- every plan draw resolves to a page inside the training dataset
# ---------------------------------------------------------------------------


def require_plan_covers_budget(*, plan: Mapping[str, Any], steps: int, batch_size: int) -> None:
    """Design §5 step 4: the plan order must cover EXACTLY `steps *
    batch_size` draws -- too few and the loop would run past the end of
    the plan (drawing an index that does not exist); too many and the
    frozen plan is not the one this run's budget claims."""
    expected = steps * batch_size
    actual = len(plan["draws"])
    if actual != expected:
        raise S2TrainingRefused(
            f"the plan carries {actual} draw(s), but steps ({steps}) x "
            f"batch_size ({batch_size}) = {expected} -- the plan order must "
            "cover exactly the run's budget"
        )


def resolve_plan_pages(
    *,
    plan: Mapping[str, Any],
    expectation: CacheExpectation,
    split_records: Sequence[Mapping[str, Any]],
) -> list[PageKey]:
    """The page :func:`s2_plan.select_page` draws for every plan draw, in
    draw order."""
    pages_by_lineage = sp.pages_by_lineage(expectation, split_records)
    resolved: list[PageKey] = []
    for draw in plan["draws"]:
        lineage = draw["lineage_root_id"]
        pages = pages_by_lineage.get(lineage)
        if not pages:
            raise S2TrainingRefused(
                f"plan draw {draw['draw_index']} names lineage "
                f"{lineage!r}, which has no pages under the pinned "
                "expectation"
            )
        resolved.append(
            sp.select_page(
                run_seed=plan["run_seed"], draw_index=draw["draw_index"], pages=pages
            )
        )
    return resolved


def plan_dataset_order(pages: Sequence[PageKey], expectation: CacheExpectation) -> list[int]:
    """Every drawn page's position in the training dataset (dataset index
    `i` is `expectation.target_keys()[i]`, per
    `ExactTextCacheReader._check_expectation`). A page the pinned
    expectation does not target is refused, never dropped."""
    index_of = {key: i for i, key in enumerate(expectation.target_keys())}
    order: list[int] = []
    for draw_index, page in enumerate(pages):
        position = index_of.get(page)
        if position is None:
            raise S2TrainingRefused(
                f"plan draw {draw_index} selected page {page!r}, which is "
                "not in the training dataset (outside the pinned expectation)"
            )
        order.append(position)
    return order


# ---------------------------------------------------------------------------
# Step 5 -- stage-state (DEC-027 D4, RFC005-ERRATA-003)
# ---------------------------------------------------------------------------


def _model_group_states(model: QwenDenseCandidateModel) -> dict[str, str]:
    """Every registry group's state under `model`: `trainable` iff it owns
    a `requires_grad` parameter here; otherwise the registry's own resting
    state for the group's kind (`frozen` for `frozen_base`,
    `not_applicable` for `not_applicable`/`lora`). The parameter -> group
    map is `candidate_model.parameter_group_ids` -- this function's own
    copy of it omitted the `node_visual_aligner` child, so an unfrozen
    aligner was reported `frozen` and S4 promotion refused (24.9.)."""
    group_id_of = parameter_group_ids(model)

    trainable_groups: set[str] = set()
    for name, param in model.named_parameters():
        group_id = group_id_of.get(name)
        if group_id is not None and param.requires_grad:
            trainable_groups.add(group_id)

    states: dict[str, str] = {}
    for group_id, group in trainable_group_registry().items():
        if group_id in trainable_groups:
            states[group_id] = "trainable"
        elif group.kind == "frozen_base":
            states[group_id] = "frozen"
        else:  # "not_applicable" or "lora"
            states[group_id] = "not_applicable"
    return states


def require_step5_stage_state(*, model: QwenDenseCandidateModel, run_class: str) -> None:
    states = _model_group_states(model)
    for group_id, state in states.items():
        try:
            ssp.validate_stage_state("s2", group_id, state)
        except ValueError as exc:
            raise S2TrainingRefused(f"stage-state check refused: {exc}") from exc
    if run_class == "promotion" and states.get("node_visual_aligner") != "trainable":
        raise S2TrainingRefused(
            "run_class='promotion' requires node_visual_aligner to be "
            "trainable (state is "
            f"{states.get('node_visual_aligner')!r}); RFC005-ERRATA-003 "
            "admits only S2 discovery while it is not trainable (DEC-027 D4)"
        )


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _attempts_dir(run_dir: Path) -> Path:
    directory = run_dir / _ATTEMPTS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _next_attempt_index(run_dir: Path) -> int:
    return len(list(_attempts_dir(run_dir).glob("attempt-*-start.json")))


def execution_profile_for(device: torch.device | str, num_workers: int) -> dict[str, str]:
    """The attempt's execution profile (RFC005 §43.2). `num_workers == 0`
    is today's serial path and keeps today's profile exactly -- `{"device":
    ...}`, hash unchanged (design §12 S2); only the prefetch path adds its
    loader parameters, incl. that worker seeds come from a private
    generator, never the global RNG (§43.1)."""
    profile = {"device": str(device)}
    if num_workers > 0:
        profile.update(
            {
                "data_loader": "ordered-prefetch-v1",
                "num_workers": str(num_workers),
                "prefetch_factor": str(_PREFETCH_FACTOR),
                "timeout": str(DATA_LOADER_TIMEOUT_S),
                "worker_seed": "private-generator",
            }
        )
    return profile


def write_execution_profile_sibling(
    attempts_dir: Path, attempt_index: int, profile: Mapping[str, str]
) -> None:
    """The attempt manifest carries only the profile's HASH and its schema
    is closed, so the plaintext goes to a sibling file (the S3
    typed-supervision precedent). `profile` must be the very dict passed to
    `ex.start_attempt`."""
    (attempts_dir / f"attempt-{attempt_index:04d}-execution-profile.json").write_text(
        json.dumps(dict(profile), sort_keys=True)
    )


def _clear_reader_lru(dataset: Dataset) -> None:
    """Drop the parsed shards the parent holds (e.g. from preflight) so
    `fork` does not copy them into every worker (design §12 S4). The
    `_verified` set is kept: workers re-hash at least as often as today."""
    reader = getattr(getattr(dataset, "base", dataset), "reader", None)
    if reader is not None:
        reader._loaded.clear()


def plan_batches(
    dataset: Dataset,
    plan_order: Sequence[int],
    *,
    start_step: int,
    steps: int,
    batch_size: int,
    num_workers: int,
) -> Iterator[list[Any]]:
    """Yield `steps - start_step` batches; batch `k` (1-based global step)
    is exactly `[dataset[i] for i in plan_order[(k-1)*B : k*B]]`, in plan
    order (design §3.1). Serves S2 and S3 alike.

    `num_workers == 0` is today's serial fetch. `num_workers > 0` prefetches
    the SAME indices in worker processes: `in_order=True` keeps the plan
    order whatever order workers finish in, and the private `generator`
    keeps DataLoader from drawing its worker base seed from the global RNG
    the checkpoint saves and restores. A worker exception (e.g. a shard
    whose sha256 changed) is re-raised here with its own type."""
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    indices = list(plan_order[start_step * batch_size : steps * batch_size])
    if len(indices) != (steps - start_step) * batch_size:
        raise ValueError(
            f"the plan order holds {len(plan_order)} draw(s), fewer than steps "
            f"({steps}) x batch_size ({batch_size}) -- a short plan must fail, "
            "not end the run early"
        )
    if num_workers == 0:
        for lo in range(0, len(indices), batch_size):
            yield [dataset[i] for i in indices[lo : lo + batch_size]]
        return
    _clear_reader_lru(dataset)
    generator = torch.Generator()
    generator.manual_seed(0)
    yield from DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=indices,
        collate_fn=list,
        num_workers=num_workers,
        prefetch_factor=_PREFETCH_FACTOR,
        in_order=True,
        generator=generator,
        timeout=DATA_LOADER_TIMEOUT_S,
        multiprocessing_context="fork",
    )


def require_resume_ledger_consistent(
    *, ledger: ex.ExposureLedger, step: int, batch_size: int
) -> None:
    """§5 resume rule: a checkpoint at step `k` must be paired with a
    ledger holding exactly `k * batch_size` records whose last
    `plan_draw_index` is `k * batch_size - 1` -- otherwise the ledger and
    the checkpoint disagree about where the run actually is."""
    expected = step * batch_size
    if ledger.record_count != expected:
        raise S2TrainingRefused(
            f"resumed checkpoint is at step {step} (batch_size {batch_size}, "
            f"expecting {expected} ledger records), but the ledger holds "
            f"{ledger.record_count}"
        )
    if expected and ledger.records[-1].plan_draw_index != expected - 1:
        raise S2TrainingRefused(
            f"resumed ledger's last record plan_draw_index is "
            f"{ledger.records[-1].plan_draw_index}, expected {expected - 1}"
        )


def run_training(
    *,
    run_dir: Path,
    model: QwenDenseCandidateModel,
    dataset: s1_train.ExactTextTrainingDataset,
    optimizer: torch.optim.Optimizer,
    image_token_id: int,
    plan_pages: Sequence[PageKey],
    plan_lineages: Sequence[str],
    plan_order: Sequence[int],
    companion_hash: str,
    s2_run_id: str,
    steps: int,
    batch_size: int,
    device: torch.device | str = "cpu",
    grad_clip: float = 1.0,
    checkpoint_every: int = 0,
    out: Path | None = None,
    start_step: int = 0,
    ledger: ex.ExposureLedger | None = None,
    execution_profile: Mapping[str, str] | None = None,
    ancestry_base: Mapping[str, Any] | None = None,
    integrity: Mapping[str, str],
    num_workers: int = 0,
) -> ex.ExposureLedger:
    """Run the S2 optimizer steps from `start_step` (0 for a fresh run, or
    the resumed checkpoint's step) up to `steps`, the run's TOTAL budget --
    the frozen `run_config.steps` the CLI must reproduce -- over the frozen
    plan order. A resume therefore runs only the remainder and can never
    walk past the plan's last draw (2026-09-14 review: `steps` used to mean
    "this many more"). `integrity` is the attempt's start/end integrity
    record (the S2 manifest and plan hashes). `num_workers` selects the
    fetch path (`plan_batches`); the batches are identical either way.
    Returns the ledger, appended in place."""
    if start_step >= steps:
        raise S2TrainingRefused(
            f"start_step {start_step} has already reached the run's total budget "
            f"of {steps} step(s) -- nothing left to run"
        )
    ledger = ledger if ledger is not None else ex.ExposureLedger(s2_run_id)
    execution_profile = (
        dict(execution_profile)
        if execution_profile
        else execution_profile_for(device, num_workers)
    )
    attempts_dir = _attempts_dir(run_dir)
    attempt_index = _next_attempt_index(run_dir)
    attempt = ex.start_attempt(
        run_id=s2_run_id,
        attempt_index=attempt_index,
        execution_profile=execution_profile,
        start_integrity=integrity,
        ledger=ledger,
    )
    (attempts_dir / f"attempt-{attempt_index:04d}-start.json").write_text(
        json.dumps(attempt, sort_keys=True)
    )
    if num_workers > 0:
        write_execution_profile_sibling(attempts_dir, attempt_index, execution_profile)

    def _finalize(terminal_state: str) -> None:
        finalized = ex.finalize_attempt(
            manifest=attempt,
            terminal_state=terminal_state,
            ledger=ledger,
            end_integrity=integrity,
        )
        (attempts_dir / f"attempt-{attempt_index:04d}-final.json").write_text(
            json.dumps(finalized, sort_keys=True)
        )

    step = start_step
    try:
        batches = plan_batches(
            dataset, plan_order, start_step=start_step, steps=steps,
            batch_size=batch_size, num_workers=num_workers,
        )
        # `data_wait_s`: blocked on the next batch. `step_s`: from there to
        # this step's log line (so it includes `data_wait_s`); checkpoint
        # writes fall in neither.
        waited_from = time.perf_counter()
        for step_samples in batches:
            data_wait_s = time.perf_counter() - waited_from
            step += 1
            lo, hi = (step - 1) * batch_size, step * batch_size
            indices = plan_order[lo:hi]
            batch = s1_train.collate_training_batch(
                step_samples,
                ranges=model.ranges,
                image_token_id=image_token_id,
                device=device,
            )
            evidence = s1_train.train_step(model, batch, optimizer, grad_clip=grad_clip)
            for offset, idx in enumerate(indices):
                draw_index = lo + offset
                bundle, page = plan_pages[draw_index]
                ledger.append(
                    attempt_id=attempt["attempt_id"],
                    lineage_root_id=plan_lineages[draw_index],
                    cell=CELL,
                    view_ids=[f"{bundle}:{page}"],
                    sample_count=1,
                    optimizer_step=step,
                    plan_draw_index=draw_index,
                )
            print(
                json.dumps(
                    {
                        "step": step,
                        "plan_draw_index": list(range(lo, hi)),
                        **evidence.report.payload(),
                        "grad_norm_total": evidence.grad_norm_total,
                        "grad_norm_groups": evidence.grad_norm_groups,
                        "data_wait_s": round(data_wait_s, 4),
                        "step_s": round(time.perf_counter() - waited_from, 4),
                    }
                ),
                flush=True,
            )
            if out is not None and checkpoint_every and step % checkpoint_every == 0:
                _checkpoint(
                    out, model=model, optimizer=optimizer, step=step,
                    dataset=dataset, companion_hash=companion_hash, s2_run_id=s2_run_id,
                    ancestry_base=ancestry_base,
                )
                ledger.save_jsonl(run_dir / _LEDGER_FILENAME)
            waited_from = time.perf_counter()
        if out is not None:
            _checkpoint(
                out, model=model, optimizer=optimizer, step=step,
                dataset=dataset, companion_hash=companion_hash, s2_run_id=s2_run_id,
                ancestry_base=ancestry_base,
            )
            ledger.save_jsonl(run_dir / _LEDGER_FILENAME)
    except Exception:
        # A failure in the final checkpoint/ledger write is exactly the
        # failure this finalises as: "completed" must mean the run's last
        # artefacts actually landed, not merely that the loop returned.
        _finalize("failed")
        raise
    _finalize("completed")
    return ledger


def _checkpoint(
    out: Path,
    *,
    model: QwenDenseCandidateModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    dataset: s1_train.ExactTextTrainingDataset,
    companion_hash: str,
    s2_run_id: str,
    ancestry_base: Mapping[str, Any] | None = None,
) -> None:
    s1_train.save_checkpoint(
        out,
        model=model,
        optimizer=optimizer,
        progress=s1_train.TrainingProgress(step=step),
        cache_identity=asdict(dataset.identity),
        s1_run_manifest_hash=companion_hash,
        feature_source=dataset.feature_source,
        ancestry_base=ancestry_base,
        run_id=s2_run_id,
    )
    print(f"[s2_train] checkpoint step {step} -> {out}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the S2 exact-text loop over a cache. Always gradient-bearing --
    a diagnostic S2 run uses `s1_train` directly (design §5).

    `--release-dir` is the one flag design §5's CLI list omits but step 4
    needs: mapping a plan draw's lineage to its candidate pages
    (`s2_plan.pages_by_lineage`) reads the release's split manifests
    (`s2_plan.load_split_records`, incl. `partial.jsonl`),
    and nothing else under `--run-dir` carries that information.
    """
    import argparse

    from unflatten.model.candidate_loader import load_qwen_candidate
    from unflatten.model.s1_runner import build_s1_optimizer

    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.s2_train", description=__doc__
    )
    parser.add_argument("cache", type=Path, help="exact-text cache directory")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--preflight-rows", type=int, default=4)
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="data prefetch worker processes; 0 = serial fetch in this process",
    )
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error(f"--num-workers must be >= 0, got {args.num_workers}")

    torch.manual_seed(args.seed)
    run_dir: Path = args.run_dir

    companion = json.loads((run_dir / "s1_run_manifest.json").read_text())
    allowlist = json.loads((run_dir / "qwen_text_allowlist.json").read_text())
    expectation = CacheExpectation.from_payload(
        json.loads((run_dir / "exact_text_cache_expectation.json").read_text())
    )
    ancestry = json.loads((run_dir / "checkpoint_ancestry_base.json").read_text())
    ca.verify_base(ancestry)
    # The same guard s1_train carries: a checkpoint under an S0 audited root
    # would be classified as unaudited legacy ancestry by the next S0 audit.
    if args.out:
        s1_train.refuse_out_under_audited_roots(args.out, ancestry)

    # --- step 1: the S1 admission chain, over the companion manifest -----
    dataset = s1_train.ExactTextTrainingDataset(
        args.cache, gradient_bearing=True, expectation=expectation
    )
    identity = asdict(dataset.identity)
    s1_train.require_gradient_admissible(
        cache_identity=identity,
        run_manifest=companion,
        allowlist=allowlist,
        ancestry=ancestry,
        expectation=expectation,
        cli_lr=args.lr,
        cli_algorithm=args.optimizer,
        cli_batch_size=args.batch_size,
        cli_seed=args.seed,
        cli_steps=args.steps,
        control_arm=False,
    )

    # --- step 2: the S2 manifest is bound to this companion --------------
    s2_manifest = load_s2_run_manifest(run_dir / "s2_run_manifest.json")
    require_step2_bound_to_companion(s2_manifest=s2_manifest, companion=companion)

    # --- step 3: committed policies and plan verify against the manifest -
    matrix = load_stage_data_matrix(_POLICIES_DIR / "s2_stage_data_matrix.json")
    sampler_policy = load_s2_sampler_policy(_POLICIES_DIR / "s2_sampler_policy.json")
    exposure_policy = load_s2_exposure_policy(_POLICIES_DIR / "s2_exposure_policy.json")
    plan = json.loads((run_dir / "ordered_exposure_plan.json").read_text())
    require_step3_policies_bound(
        s2_manifest=s2_manifest,
        matrix=matrix,
        sampler_policy=sampler_policy,
        exposure_policy=exposure_policy,
        plan=plan,
    )

    # --- step 4: every draw resolves inside the training dataset ---------
    require_plan_covers_budget(plan=plan, steps=args.steps, batch_size=args.batch_size)
    split_records = sp.load_split_records(args.release_dir)
    plan_pages = resolve_plan_pages(plan=plan, expectation=expectation, split_records=split_records)
    plan_order = plan_dataset_order(plan_pages, expectation)
    plan_lineages = [draw["lineage_root_id"] for draw in plan["draws"]]

    # --- step 5: stage-state, checked once the model exists --------------
    candidate = load_qwen_candidate(device=args.device)
    model = QwenDenseCandidateModel(
        decode=candidate.decode,
        inject_visual=candidate.inject_visual,
        visual_dim=candidate.visual_hidden_size,
        d_model=candidate.hidden_size,
        ranges=cv.packed_ranges(candidate.vocab_size),
        qwen_embedding_weight=candidate.embed_tokens.weight,
        qwen_lm_head_weight=candidate.lm_head_weight,
    )
    model.to(args.device)
    # The run class is the frozen manifest's, never a CLI default: a
    # promotion manifest must not pass step 5 as a discovery run.
    require_step5_stage_state(model=model, run_class=s2_manifest["run_class"])

    optimizer = build_s1_optimizer(model, lr=args.lr, algorithm=args.optimizer)

    start_step = 0
    ledger = ex.ExposureLedger(s2_manifest["run_id"])
    if args.resume:
        progress, _state = s1_train.load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            expect_cache_identity=identity,
            expect_feature_source=dataset.feature_source,
            expect_s1_run_manifest_hash=companion["s1_run_manifest_hash"],
            expect_base_ancestry_hash=ancestry["ancestry_hash"],
        )
        start_step = progress.step
        ledger = ex.ExposureLedger.load_jsonl(
            run_dir / _LEDGER_FILENAME, s2_manifest["run_id"]
        )
        require_resume_ledger_consistent(
            ledger=ledger, step=start_step, batch_size=args.batch_size
        )

    preflight_indices = s1_train.preflight_row_indices(len(dataset), args.preflight_rows)
    if preflight_indices:
        report = s1_train.preflight_native_signal(
            model,
            [dataset[i] for i in preflight_indices],
            image_token_id=candidate.image_token_id,
            device=args.device,
        )
        print(json.dumps({"preflight": report}), flush=True)
        s1_train.require_preflight_passes(report, gradient_bearing=True)

    run_training(
        run_dir=run_dir,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        image_token_id=candidate.image_token_id,
        plan_pages=plan_pages,
        plan_lineages=plan_lineages,
        plan_order=plan_order,
        companion_hash=companion["s1_run_manifest_hash"],
        s2_run_id=s2_manifest["run_id"],
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_every=args.checkpoint_every,
        out=args.out,
        ancestry_base=ancestry,
        start_step=start_step,
        ledger=ledger,
        integrity={
            "s2_run_manifest": s2_manifest["s2_run_manifest_hash"],
            "ordered_exposure_plan": plan["ordered_exposure_plan_hash"],
        },
        num_workers=args.num_workers,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
