"""`python -m unflatten.model.s4_fresh_launch argv --run-dir RUN` -- the fresh
S4 run's (DEC-058) `s4_train` argument list, derived, never typed.

`scripts/s4_fresh_launch_training.sh` used to be where a v15b-style launcher
spelled every trainer flag by hand; each flag the frozen manifest did not
carry was an `s4_train` refusal -- after the services were stopped for the
night. Here the argv is read off the frozen run instead:

- `s1_run_manifest.json`'s `run_config` (steps, seed, batch, lr, grad clip,
  optimizer) and `s4_run_manifest.json` (reduced lr, T1, R3, V6b, the
  on-the-fly visual block and its augmentation seed, the gate, the router)
  decide every identity-bearing flag;
- `launch.json` (written by `scripts/s4_fresh_freeze_chain.sh`, execution
  only: paths, device, workers, eval/checkpoint cadence, the prefetch knob,
  the feature-cache directory, the router files) supplies the rest.

Before any of that the DEC-058 contract is checked: a `promotion` run from a
clean pretrained start (no `init_from`), a selection policy bound
(RFC005-ERRATA-005 E3), the DEC-057 gate hash, the on-the-fly visual path
with DeepStack (DEC-056 + A), V6b (`structural_table_region` `exclude`).

The one warm start the contract admits (RFC005-ERRATA-012, DEC-078) is the
run's own LR tail: `init_from` pins a `promotion` parent AND the manifest
freezes an `lr_schedule`; `launch.json` then names the parent file as
`init_from_checkpoint` (execution-only -- its identity is the manifest's pin)
and the argv carries `--init-from`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

LAUNCH_SCHEMA = "unflatten.s4-fresh-launch/1"
#: `launch.json` keys every fresh launch names (paths are absolute on prod).
REQUIRED_LAUNCH_KEYS = (
    "train_cache",
    "release_dir",
    "val_cache",
    "val_expectation",
    "page_image_store",
    "visual_input_policy",
    "data_quality_gate",
    "out",
    "device",
    "num_workers",
    "eval_every",
    "checkpoint_every",
    "keep_every",
    "eval_pages",
    "preflight_rows",
)
#: The augmentation policy that needs no `launch.json` path (the trainer default).
V1_POLICY_ID = "augmentation-v1"
_DEEPSTACK_TAP_PREFIX = "pooler_output+deepstack:"
#: Test sentinel: a manifest key to leave out of a fixture.
_DROP = object()


class LaunchRefused(ValueError):
    """The frozen run or `launch.json` does not meet the fresh-S4 contract."""


def _micro(value: int) -> str:
    return repr(value / 1_000_000)


def require_fresh_contract(manifest: Mapping[str, Any]) -> None:
    if manifest.get("run_class") != "promotion":
        raise LaunchRefused(
            f"the fresh S4 run is a promotion run (RFC005-ERRATA-005), the manifest says "
            f"{manifest.get('run_class')!r}"
        )
    init_from = manifest.get("init_from")
    if init_from is not None and (
        manifest.get("lr_schedule") is None
        or not isinstance(init_from, Mapping)
        or init_from.get("parent_run_class") != "promotion"
    ):
        raise LaunchRefused(
            "DEC-058: a clean pretrained start -- the manifest names an init_from parent "
            "that is not the RFC005-ERRATA-012 LR tail (a frozen lr_schedule AND a "
            "promotion parent)"
        )
    if not manifest.get("selection_policy_hash"):
        raise LaunchRefused(
            "RFC005-ERRATA-005 E3: no selection policy is bound -- it is frozen before "
            "the first gradient"
        )
    if not manifest.get("data_quality_gate_hash"):
        raise LaunchRefused("DEC-057: the manifest carries no data-quality gate hash")
    visual = manifest.get("visual_input")
    if not visual:
        raise LaunchRefused("DEC-056 A: the manifest has no on-the-fly visual_input block")
    if not str(visual.get("visual_feature_tap", "")).startswith(_DEEPSTACK_TAP_PREFIX):
        raise LaunchRefused(
            f"DEC-056: the fresh run trains with DeepStack, the manifest's tap is "
            f"{visual.get('visual_feature_tap')!r}"
        )
    if manifest.get("structural_table_region") != "exclude":
        raise LaunchRefused(
            "DEC-058 / DEC-055 V6b: structural_table_region must be frozen as 'exclude'"
        )


def train_argv(run_dir: Path, launch: Mapping[str, Any]) -> list[str]:
    """The full `s4_train` argv (without `--resume`/`--admission-only`,
    which the launcher adds) for the frozen run in `run_dir`."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "s4_run_manifest.json").read_text())
    companion = json.loads((run_dir / "s1_run_manifest.json").read_text())
    require_fresh_contract(manifest)
    missing = [key for key in REQUIRED_LAUNCH_KEYS if launch.get(key) is None]
    if missing:
        raise LaunchRefused(f"launch.json names no {missing}")
    init_from_checkpoint = launch.get("init_from_checkpoint")
    if (manifest.get("init_from") is not None) != (init_from_checkpoint is not None):
        raise LaunchRefused(
            "the manifest's init_from and launch.json init_from_checkpoint go together: "
            f"manifest {'pins' if manifest.get('init_from') else 'has no'} parent, launch.json "
            f"init_from_checkpoint is {init_from_checkpoint!r}"
        )

    run_config = companion["run_config"]
    optimizer = run_config["optimizer"]
    visual = manifest["visual_input"]
    argv = [
        str(launch["train_cache"]),
        "--run-dir", str(run_dir),
        "--release-dir", str(launch["release_dir"]),
        "--device", str(launch["device"]),
        "--seed", str(manifest["run_seed"]),
        "--steps", str(run_config["steps"]),
        "--batch-size", str(run_config["sampler"]["batch_size"]),
        "--lr", _micro(optimizer["lr_micro"]),
        "--lr-reduced", _micro(manifest["lr_reduced_micro"]),
        "--grad-clip", _micro(optimizer["grad_clip_micro"]),
        "--optimizer", str(optimizer["algorithm"]),
        "--val-cache", str(launch["val_cache"]),
        "--val-expectation", str(launch["val_expectation"]),
        "--eval-every", str(launch["eval_every"]),
        "--eval-pages", str(launch["eval_pages"]),
        "--out", str(launch["out"]),
        "--checkpoint-every", str(launch["checkpoint_every"]),
        "--keep-every", str(launch["keep_every"]),
        "--preflight-rows", str(launch["preflight_rows"]),
        "--num-workers", str(launch["num_workers"]),
        "--visual-input-policy", str(launch["visual_input_policy"]),
        "--page-image-store", str(launch["page_image_store"]),
        "--structural-table-region", str(manifest["structural_table_region"]),
        "--data-quality-gate", str(launch["data_quality_gate"]),
    ]  # fmt: skip
    if not launch["eval_every"]:
        drop = {"--val-cache", "--val-expectation", "--eval-pages"}
        argv = _without_flags(argv, drop)
    if visual.get("augmentation") is not None:
        argv += ["--onfly-augmentation-seed", str(visual["augmentation"]["seed"])]
        argv += _augmentation_policy_argv(visual["augmentation"], launch)
    if visual.get("batch_pages"):
        argv.append("--vit-batch-pages")
    cache_dir = launch.get("vit_feature_cache")
    if bool(visual.get("feature_cache")) != (cache_dir is not None):
        raise LaunchRefused(
            "the fp8 feature cache is frozen in the manifest's visual_input "
            f"({bool(visual.get('feature_cache'))}) but launch.json vit_feature_cache is "
            f"{cache_dir!r} -- both or neither"
        )
    if cache_dir is not None:
        argv += ["--vit-feature-cache", str(cache_dir)]
    if launch.get("vit_prefetch_overlap"):
        argv.append("--vit-prefetch-overlap")
    argv += _external_views_argv(visual, launch)
    if manifest.get("mask_inactive_union_arms"):
        argv.append("--mask-inactive-union-arms")
    lambda_milli = int(manifest.get("consistency_lambda_milli", 0))
    if lambda_milli:
        argv += ["--consistency-lambda", repr(lambda_milli / 1000)]
    argv += _v17_loss_argv(manifest)
    lr_schedule = manifest.get("lr_schedule")
    if lr_schedule is not None:
        argv += [
            "--lr-schedule", str(lr_schedule["kind"]),
            "--lr-schedule-start-step", str(lr_schedule["start_step"]),
            "--lr-schedule-floor-milli", str(lr_schedule["floor_milli"]),
        ]
    if init_from_checkpoint is not None:
        # The launcher drops this pair again on `--resume` (design §9: a resume
        # never carries --init-from; s4_train refuses the combination).
        argv += ["--init-from", str(init_from_checkpoint)]
    if manifest.get("level_router_policy_hash") is not None:
        policy = launch.get("level_router_policy")
        record = launch.get("level_router_ratification")
        if policy is None or record is None:
            raise LaunchRefused(
                "the manifest binds an RFC007 level_router policy: launch.json must name "
                "level_router_policy and the owner's level_router_ratification record"
            )
        argv += ["--level-router-policy", str(policy), "--level-router-ratification", str(record)]
    return argv


def _external_views_argv(visual: Mapping[str, Any], launch: Mapping[str, Any]) -> list[str]:
    """`--external-views PATH` (DEC-085): the manifest's `visual_input.external_views` block
    and `launch.json`'s `external_views` path go together -- both or neither. The path is
    execution-only; its identity is the block's table sha, which `s4_train` verifies."""
    path = launch.get("external_views")
    if ("external_views" in visual) != (path is not None):
        raise LaunchRefused(
            "the external-views table is frozen in the manifest's visual_input "
            f"({'external_views' in visual}) but launch.json external_views is {path!r} "
            "-- both or neither"
        )
    return [] if path is None else ["--external-views", str(path)]


def _augmentation_policy_argv(
    augmentation: Mapping[str, Any], launch: Mapping[str, Any]
) -> list[str]:
    """`--onfly-augmentation-policy PATH` (design §2). The v1 default needs no
    path (argv byte-identical to v16); any other bound policy must be named."""
    path = launch.get("augmentation_policy")
    policy_id = augmentation.get("policy_id")
    if path is None:
        if policy_id != V1_POLICY_ID:
            raise LaunchRefused(
                f"the manifest binds augmentation policy {policy_id!r}; launch.json names "
                "no augmentation_policy"
            )
        return []
    return ["--onfly-augmentation-policy", str(path)]


def _v17_loss_argv(manifest: Mapping[str, Any]) -> list[str]:
    """The 1.5.0 manifest keys as trainer flags -- only those present."""
    out: list[str] = []
    if manifest.get("collapsed_region_rule") is not None:
        out += ["--collapsed-region-rule", str(manifest["collapsed_region_rule"])]
    if manifest.get("story_decision_weight_milli") is not None:
        out += ["--story-decision-weight", repr(manifest["story_decision_weight_milli"] / 1000)]
    for node_type, milli in sorted((manifest.get("alignment_type_weights_milli") or {}).items()):
        out += ["--alignment-type-weight", f"{node_type}={milli / 1000!r}"]
    return out


def _without_flags(argv: Sequence[str], flags: set[str]) -> list[str]:
    out: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token in flags:
            skip = True
            continue
        out.append(token)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m unflatten.model.s4_fresh_launch")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("argv", help="print the s4_train argv, one argument per line")
    build.add_argument("--run-dir", type=Path, required=True)
    build.add_argument("--launch", type=Path, default=None, help="default: RUN/launch.json")
    args = parser.parse_args(argv)
    launch_path = args.launch or args.run_dir / "launch.json"
    launch = json.loads(launch_path.read_text())
    if launch.get("schema") != LAUNCH_SCHEMA:
        raise LaunchRefused(f"{launch_path}: schema {launch.get('schema')!r} != {LAUNCH_SCHEMA}")
    for token in train_argv(args.run_dir, launch):
        print(token)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
