"""CPU cost of augmentation-v2 and a visual gallery (design §1.8 CPU hook).

Timing::

    python -m unflatten.dataset.augment.bench --policy P --png PAGE --draws N

prints ``{"median_s", "p90_s", "per_op_median_s", ...}`` as JSON. A *view* is
one resolved recipe (``selection.resolve_recipe_v2``, families alternating
scan/photo, bundle ``bench``, page index = draw index) applied op by op with
``apply_recipe_v2``; ``median_s`` / ``p90_s`` are over the summed op time of a
view. The legibility guard is not part of this number.

Gallery::

    python -m unflatten.dataset.augment.bench --png PAGE --gallery OUT_DIR

writes ``<op>.png`` (every op, middle of the policy range, seed 1) and
``views/view-NN-<family>.png`` (resolved recipes) into OUT_DIR and nowhere else.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from unflatten.dataset.augment import ops_v2, policy, selection

_BUNDLE = "bench"
_FAMILIES = ("scan_simulation", "photo_simulation")
_GALLERY_VIEWS_PER_FAMILY = 6
_DEFAULT_POLICY = Path(__file__).parent / "policies" / "augmentation_policy_v2.json"


def draw_recipe(pol: dict[str, Any], index: int) -> tuple[list[dict[str, Any]], list[int]]:
    """The recipe the bench applies for draw `index` (families alternate)."""
    return selection.resolve_recipe_v2(
        _BUNDLE, index, _FAMILIES[index % len(_FAMILIES)], pol, policy.policy_hash(pol)
    )


def _timed_view(
    png: bytes, recipe: list[dict[str, Any]], seeds: list[int], ink_free: dict[str, Any]
) -> tuple[float, dict[str, float]]:
    """Apply `recipe` op by op; return (total seconds, seconds per op)."""
    per_op: dict[str, float] = {}
    current = png
    total = 0.0
    for step, seed in zip(recipe, seeds, strict=True):
        started = time.perf_counter()
        current, _matrix, _report = ops_v2.apply_recipe_v2(
            current, [step], [seed], ink_free=ink_free
        )
        spent = time.perf_counter() - started
        per_op[step["op"]] = spent
        total += spent
    return total, per_op


def run_bench(pol: dict[str, Any], png: bytes, *, draws: int) -> dict[str, Any]:
    """Time `draws` views of `png` under `pol`."""
    if draws < 1:
        raise ValueError(f"draws must be >= 1, got {draws}")
    policy.validate_policy(pol)
    totals: list[float] = []
    by_op: dict[str, list[float]] = {}
    for index in range(draws):
        recipe, seeds = draw_recipe(pol, index)
        total, per_op = _timed_view(png, recipe, seeds, pol["ink_free"])
        totals.append(total)
        for op, spent in per_op.items():
            by_op.setdefault(op, []).append(spent)
    return {
        "policy_hash": policy.policy_hash(pol),
        "draws": draws,
        "median_s": float(statistics.median(totals)),
        "p90_s": float(np.percentile(totals, 90)),
        "per_op_median_s": {
            op: float(statistics.median(values)) for op, values in sorted(by_op.items())
        },
    }


def _mid_params(pol: dict[str, Any], op: str) -> dict[str, Any]:
    for family in pol["families"].values():
        for entry in family["ops"]:
            if entry["op"] == op:
                return {
                    name: int(round((low + high) / 2))
                    if ops_v2.is_int_param(op, name)
                    else float((low + high) / 2)
                    for name, (low, high) in entry["params"].items()
                }
    raise ValueError(f"policy names no op {op!r}")


def write_gallery(pol: dict[str, Any], png: bytes, out_dir: Path) -> list[str]:
    """Write the gallery under `out_dir` (created); return the written relative names."""
    policy.validate_policy(pol)
    views = out_dir / "views"
    views.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for op in ops_v2.OPS_V2:
        out, _m, _r = ops_v2.apply_recipe_v2(
            png, [{"op": op, "params": _mid_params(pol, op)}], [1], ink_free=pol["ink_free"]
        )
        (out_dir / f"{op}.png").write_bytes(out)
        written.append(f"{op}.png")
    for index in range(_GALLERY_VIEWS_PER_FAMILY * len(_FAMILIES)):
        recipe, seeds = draw_recipe(pol, index)
        out, _m, _r = ops_v2.apply_recipe_v2(png, recipe, seeds, ink_free=pol["ink_free"])
        name = f"view-{index:02d}-{_FAMILIES[index % len(_FAMILIES)]}.png"
        (views / name).write_bytes(out)
        written.append(f"views/{name}")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m unflatten.dataset.augment.bench")
    parser.add_argument("--policy", type=Path, default=_DEFAULT_POLICY)
    parser.add_argument("--png", type=Path, required=True, help="clean page PNG")
    parser.add_argument("--draws", type=int, default=None, help="number of timed views")
    parser.add_argument("--gallery", type=Path, default=None, help="directory for the gallery")
    args = parser.parse_args(argv)
    if args.draws is None and args.gallery is None:
        parser.error("give --draws N (timing) and/or --gallery OUT_DIR")
    pol = policy.load_policy(args.policy)
    png = args.png.read_bytes()
    result: dict[str, Any] = {}
    if args.draws is not None:
        result.update(run_bench(pol, png, draws=args.draws))
    if args.gallery is not None:
        result["gallery"] = {
            "dir": str(args.gallery),
            "files": write_gallery(pol, png, args.gallery),
        }
    json.dump(result, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
