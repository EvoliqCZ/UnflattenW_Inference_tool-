"""`python -m unflatten.model.s4_visual_input_block` -- the fresh-S4 freeze
chain's `visual_input` manifest block, built from the visual input policy file
and an optional augmentation policy FILE (schema 1 or 2, RFC003-ERRATA-004).

This is the chain's former inline Python heredoc, lifted out so it is testable
without bash. Without `--policy` the block is the frozen v1 default (byte for
byte what the chain wrote before); with it the augmentation fields come from
the loaded and validated file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from unflatten.dataset.augment import policy as aug_policy
from unflatten.model import onfly_visual as ov


def build_block(
    vip_path: Path | str,
    seed: int,
    policy_path: Path | str | None = None,
    *,
    feature_cache: bool = False,
    batch_pages: bool = False,
) -> dict[str, Any]:
    visual = ov.VisualInputPolicy.from_payload(json.loads(Path(vip_path).read_text()))
    policy = aug_policy.load_policy(policy_path) if policy_path is not None else None
    return ov.visual_input_manifest_block(
        visual,
        ov.OnTheFlyAugmentation(policy, seed=int(seed)),
        feature_cache=feature_cache,
        batch_pages=batch_pages,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m unflatten.model.s4_visual_input_block")
    parser.add_argument("--visual-input-policy", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="on-the-fly augmentation seed")
    parser.add_argument("--policy", type=Path, default=None, help="augmentation policy file")
    parser.add_argument("--feature-cache", action="store_true")
    parser.add_argument("--batch-pages", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    block = build_block(
        args.visual_input_policy,
        args.seed,
        args.policy,
        feature_cache=args.feature_cache,
        batch_pages=args.batch_pages,
    )
    args.out.write_text(json.dumps(block, indent=1, sort_keys=True))
    print(json.dumps(block, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
