"""RFC005 §12 / DEC-049 point 8: real `run_pilot` hooks for `s4_train`.

`run_pilot.PilotConfig` takes `worst_page_and_batch`/`forward_backward`/
`loss_reader` as opaque callables so it never has to import a trainer's
model code. This module is the S4-specific wiring the docstring there asks
for: `build_worst_page_and_batch`/`build_forward_backward`/
`build_loss_reader` close over the run's own model/optimizer/stage_policy
and reproduce `s4_train.run_s4_training`'s own per-step recipe (`s1_train.
collate_training_batch`, `s4_typed_loss.collate_typed`/`collate_regions`/
`s4_total_loss`) -- the whole point of a pilot is to exercise the SAME
forward/backward path a real run takes, on that run's own worst page,
before a promotion launch commits to it. None of this touches `s4_train.py`
itself (DEC-049 point 8's own design: a caller only ever supplies new hook
implementations).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from unflatten.model import onfly_visual as ov
from unflatten.model import run_pilot, s1_train
from unflatten.model import s4_typed_loss as stl
from unflatten.model.candidate_model import QwenDenseCandidateModel
from unflatten.model.exact_text_cache import CacheExpectation, PageKey

#: `s4_train.run_s4_training`'s own eval log filename (`_EVAL_LOG_FILENAME`)
#: -- kept as a literal here rather than imported, so this module never
#: needs to import `s4_train.py` (a private-module-attribute dependency
#: DEC-049 point 8 explicitly avoids: "a caller only ever supplies new hook
#: implementations, never a code change" to the trainer itself).
DEFAULT_EVAL_LOG_FILENAME = "s4_eval.jsonl"


def build_worst_page_and_batch(
    *,
    run_dir: Path,
    model: QwenDenseCandidateModel,
    image_token_id: int,
    allowed_head_ids: Collection[str],
    mask_inactive_union_arms: bool = False,
    device: str = "cpu",
    structural_table_region: str = "abstain_not_visible",
    visual_encoder: ov.OnTheFlyVisualEncoder | None = None,
    page_image_store: Path | None = None,
    processor: Any = None,
) -> Callable[[Path], tuple[Any, Any]]:
    """`PilotConfig.worst_page_and_batch`: resolves `cache_dir`'s largest
    page (`run_pilot.find_worst_page_index`) into a batch-of-one, collated
    exactly like one real `run_s4_training` step would (`s1_train.
    collate_training_batch` for the base/text channel, `s4_typed_loss.
    collate_typed`/`collate_regions` for the typed/region channels) --
    everything `build_forward_backward`'s hook needs, and nothing more.

    `run_dir` supplies the run's own pinned `CacheExpectation` (`exact_
    text_cache_expectation.json`), the SAME translation `s2_train.plan_
    dataset_order` uses to turn a `PageKey` into a dataset row index.

    `visual_encoder` (DEC-056 on-the-fly path, with `page_image_store` and
    `processor`): the worst page is a page-image row; its pixels are
    attached here and the ViT runs inside `build_forward_backward`'s hook,
    so the memory probe includes the ViT forward."""

    def _hook(cache_dir: Path) -> tuple[Any, Any]:
        expectation = CacheExpectation.from_payload(
            json.loads((run_dir / "exact_text_cache_expectation.json").read_text())
        )
        page_key, _visual_token_area = run_pilot.find_worst_page_index(cache_dir)
        index_of = {key: i for i, key in enumerate(expectation.target_keys())}
        if page_key not in index_of:
            raise ValueError(
                f"{page_key!r} (the cache's own worst page) is not among the pinned "
                "expectation's target_keys() -- cannot resolve a dataset row index"
            )
        base_dataset = s1_train.ExactTextTrainingDataset(
            cache_dir, gradient_bearing=True, expectation=expectation
        )
        dataset = stl.S4TrainingDataset(base_dataset)
        sample = dataset[index_of[page_key]]
        if visual_encoder is not None:
            sample = ov.attach_pixels(
                sample, store_dir=page_image_store, processor=processor,
                policy=visual_encoder.policy,
            )
        batch = s1_train.collate_training_batch(
            [sample.base], ranges=model.ranges, image_token_id=image_token_id, device=device
        )
        typed_batch = stl.collate_typed(
            [sample],
            allowed_head_ids=allowed_head_ids,
            mask_inactive_union_arms=mask_inactive_union_arms,
        )
        region_batch = stl.collate_regions(
            [sample], **stl.region_kwargs(structural_table_region)
        )
        return page_key, (batch, typed_batch, region_batch)

    return _hook


def build_forward_backward(
    *,
    model: QwenDenseCandidateModel,
    optimizer: torch.optim.Optimizer,
    stage_policy: Mapping[str, Any],
    consistency_lambda: float = 0.0,
    visual_encoder: ov.OnTheFlyVisualEncoder | None = None,
) -> Callable[[Any], None]:
    """`PilotConfig.forward_backward`: one real optimizer step -- forward,
    `s4_typed_loss.s4_total_loss`, `backward`, `optimizer.step()` -- over
    the `(batch, typed_batch, region_batch)` triple `build_worst_page_and_
    batch` produces. Mirrors `run_s4_training`'s own step body exactly
    (minus the ledger/checkpoint/eval bookkeeping a memory probe has no
    use for): a hook that skipped `backward()`/`optimizer.step()` would
    leave every trainable parameter bit-identical to before the call,
    which is exactly what `test_pilot_hooks_s4.py`'s mutant proves this
    real hook does NOT do."""

    def _hook(batch_triple: Any) -> None:
        batch, typed_batch, region_batch = batch_triple
        if visual_encoder is not None and batch.page_pixels is not None:
            batch = visual_encoder.materialize(batch)
        optimizer.zero_grad(set_to_none=True)
        hidden, prefix_len = model.encode_hidden(batch.model_batch)
        composite_hidden = hidden[:, prefix_len + batch.visual_steps :, :]
        logits = model.composite_head(composite_hidden)
        total, _report = stl.s4_total_loss(
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
            consistency_lambda=consistency_lambda,
        )
        total.backward()
        optimizer.step()

    return _hook


def build_loss_reader(
    *, eval_log_filename: str = DEFAULT_EVAL_LOG_FILENAME
) -> Callable[[Path, int], float]:
    """`PilotConfig.loss_reader`: `run_s4_training`'s own eval log
    (`s4_eval.jsonl`, one JSON record per `--eval-every` step) read back
    for the exact `step` requested -- `run_pilot._check_resume_determinism`
    only ever calls this at a step both the resumed and the uninterrupted
    run evaluated, so a caller wiring this hook must set `--eval-every` to
    land on that step. Raises `ValueError` (never returns a wrong-step
    value silently) when the log is missing or has no record for `step`."""

    def _hook(run_dir: Path, step: int) -> float:
        path = run_dir / eval_log_filename
        if not path.exists():
            raise ValueError(
                f"{path} does not exist -- was --eval-every configured to land on "
                f"step {step}?"
            )
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["step"] == step:
                return float(record["val"]["total_loss"])
        raise ValueError(f"no eval record for step {step} in {path}")

    return _hook


def build_trainer_entrypoint(
    *,
    model: QwenDenseCandidateModel,
    optimizer: torch.optim.Optimizer,
    dataset: stl.S4TrainingDataset,
    image_token_id: int,
    plan_pages: Sequence[PageKey],
    plan_lineages: Sequence[str],
    plan_cells: Sequence[tuple[str, str, str, str]],
    plan_order: Sequence[int],
    companion_hash: str,
    s4_run_id: str,
    stage_policy: Mapping[str, Any],
    allowed_head_ids: Collection[str],
    grad_clip: float,
    run_dir: Path,
    integrity: Mapping[str, str],
    batch_size: int = 1,
    structural_table_region: str = "abstain_not_visible",
    visual_encoder: ov.OnTheFlyVisualEncoder | None = None,
    device: str = "cpu",
    num_workers: int = 0,
    visual_prefetch: bool = False,
) -> Callable[[list[str]], int]:
    """`run_pilot.RunnerFn`'s own trainer entrypoint (`run_pilot.in_process_
    runner`'s `trainer_main`), calling `s4_train.run_s4_training` DIRECTLY
    instead of `s4_train.main`'s CLI.

    `s4_train.main`'s admission (`s1_train.require_gradient_admissible`,
    1.2.0 #21 C) refuses unless `--steps` exactly equals the S1 companion
    manifest's OWN frozen `run_config.steps` -- a real, deliberate
    production rule for the ONE promotion run that manifest belongs to.
    `run_pilot.py`'s generic pilot protocol is, by contrast, trainer-
    agnostic: it reruns the SAME `base_argv` first with `--steps
    pilot_steps` then with `--steps pilot_steps+resume_steps` (`_check_
    pilot_run`/`_check_resume`), which no single frozen `run_config.steps`
    value can satisfy for both calls, and RE-freezing a manifest between
    calls does not help either -- a resumed checkpoint also carries the
    `s1_run_manifest_hash` it was written under, and a resume onto a
    re-frozen (therefore different-hash) manifest is refused as "resuming
    across manifests, ... is a different run" (`s1_train.
    verify_checkpoint_state`). A real pilot is its own throwaway admission,
    separate from the promotion run's frozen manifest (DEC-049 point 8);
    this entrypoint stands in for that by calling the trainer's real step
    loop, typed loss, checkpointing and resume machinery (`run_s4_
    training` -- the exact function `s4_train.main` itself calls once ITS
    own admission has passed) directly, skipping only the outer CLI
    admission chain, which is a separate, already-tested concern
    (`test_s4_train.py`'s own `S4TrainingRefused` admission tests) that a
    pilot on the final cache/final code is not what re-derives.

    Understands the two CLI flags `run_pilot.py` actually overrides
    (`--steps`, `--checkpoint-every`) plus `--out`/`--resume`, matching
    `PilotConfig`'s own default flag names; every other S4 CLI flag (`--lr`,
    `--device`, admission-chain flags, ...) has no equivalent here because
    this entrypoint never reaches the admission chain those flags feed.

    `visual_encoder` (DEC-056): `dataset` is then an `onfly_visual.
    OnTheFlyDrawDataset` and every step runs the frozen ViT; a resume
    refuses a checkpoint whose `visual_feature_tap` is not this encoder's.

    `device` / `num_workers` (fresh S4 pilot): forwarded to the trainer, so a
    CUDA pilot collates onto the GPU and s/step includes the real prefetch
    workers (processor + augmentation); the defaults are the trainer's own.
    `visual_prefetch`: the overlap knob, forwarded only when on."""

    def _main(argv: list[str]) -> int:
        parser = argparse.ArgumentParser()
        parser.add_argument("--steps", type=int, required=True)
        parser.add_argument("--checkpoint-every", type=int, default=0)
        parser.add_argument("--out", type=Path, default=None)
        parser.add_argument("--resume", type=Path, default=None)
        args, _unused = parser.parse_known_args(argv)

        from unflatten.model import s2_exposure as ex
        from unflatten.model import s4_train

        start_step = 0
        ledger = ex.ExposureLedger(s4_run_id)
        # `run_pilot`'s `fresh_keys_logged` check (its own module docstring:
        # a warm start must log which keys it froze from scratch, never
        # silently assumed empty) is `s4_train.main`'s own `InitReport`
        # line -- skipped here along with the rest of the CLI's init-from
        # chain (this entrypoint never calls `load_init_weights`, so
        # nothing IS freshly initialised beyond `model`'s own construction,
        # which already happened before this entrypoint was built). Printed
        # truthfully empty rather than not printed at all.
        print(json.dumps({"fresh_keys": []}), flush=True)
        if args.resume is not None:
            state = torch.load(args.resume, weights_only=False)
            ov.require_checkpoint_visual_feature_tap(
                state,
                expected=(
                    visual_encoder.policy.visual_feature_tap()
                    if visual_encoder is not None
                    else ov.POOLER_OUTPUT_TAP
                ),
                what=f"pilot --resume {args.resume}",
            )
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            start_step = int(state["step"])
            ledger = ex.ExposureLedger.load_jsonl(
                run_dir / s4_train._LEDGER_FILENAME, s4_run_id
            )

        s4_train.run_s4_training(
            run_dir=run_dir,
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            image_token_id=image_token_id,
            plan_pages=plan_pages,
            plan_lineages=plan_lineages,
            plan_cells=plan_cells,
            plan_order=plan_order,
            companion_hash=companion_hash,
            s4_run_id=s4_run_id,
            steps=args.steps,
            batch_size=batch_size,
            stage_policy=stage_policy,
            allowed_head_ids=allowed_head_ids,
            grad_clip=grad_clip,
            checkpoint_every=args.checkpoint_every,
            out=args.out,
            start_step=start_step,
            ledger=ledger,
            integrity=integrity,
            **(
                {}
                if structural_table_region == "abstain_not_visible"
                else {"structural_table_region": structural_table_region}
            ),
            **({} if visual_encoder is None else {"visual_encoder": visual_encoder}),
            **({"visual_prefetch": True} if visual_prefetch else {}),
            device=device,
            num_workers=num_workers,
        )
        return 0

    return _main
