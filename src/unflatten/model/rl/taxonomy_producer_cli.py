"""CLI of the train taxonomy producer (DEC-047 addendum A v3 item 8).

``python -m unflatten.model.rl.taxonomy_producer_cli`` only turns arguments into a
:class:`ProducerConfig` and calls the library; all heavy work is lazy in
``adapters.taxonomy_producer_wiring``. It never trains (no ``torch.optim``), and
writes only below ``--out``, which must not exist. Executing it over a real
checkpoint needs the owner-approved window record for that checkpoint id.

Exit codes: 0 complete, 2 named refusal (one line on stderr), 3 stopped by the stop file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

EXIT_OK, EXIT_REFUSED, EXIT_STOPPED = 0, 2, 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="taxonomy_producer_cli",
        description="Train-cohort failure taxonomy over an approved checkpoint (inference only).",
    )
    for name in (
        "checkpoint",
        "sidecar",
        "release-dir",
        "store-dir",
        "cache-dir",
        "window-record",
        "out",
        "allowlist",
    ):
        p.add_argument(f"--{name}", type=Path, required=True)
    for name in ("base-ancestry-hash", "typed-head-config-hash"):
        p.add_argument(f"--expect-{name}", required=True)
    p.add_argument("--expect-s1-run-manifest-hash", default=None)
    p.add_argument("--visual-input-policy", type=Path, default=None)
    p.add_argument("--stop-file", type=Path, default=None)
    p.add_argument("--page-indices", type=int, nargs="+", default=[0])
    p.add_argument("--limit", type=int, default=None, help="max pages per page index")
    p.add_argument(
        "--offset", type=int, default=None,
        help="skip the first N sorted keys per page index (default: none)",
    )  # fmt: skip
    p.add_argument("--deepstack", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-steps", type=int, default=4608)
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--typed-decode", action="store_true")
    p.add_argument("--region-decode", action="store_true")
    p.add_argument("--loop-guard", choices=("off", "v3"), default="off")
    p.add_argument("--loop-guard-v3-arm", default="r1")
    p.add_argument(
        "--loop-guard-caps", type=Path, default=None,
        help="v2 caps file of the checkpoint's run (default: code tree's); needs --loop-guard v3",
    )  # fmt: skip
    return p


def _named_refusals() -> tuple[type[BaseException], ...]:
    from unflatten.model.checkpoint_ancestry import CheckpointAncestryInvalid
    from unflatten.model.exact_text_cache import ExactTextCacheIncompatible

    return (ValueError, OSError, CheckpointAncestryInvalid, ExactTextCacheIncompatible)


def main(argv: Sequence[str] | None = None, *, seams: Any = None) -> int:
    from unflatten.model.rl.adapters.taxonomy_producer_wiring import ProducerConfig, run_producer

    args = build_parser().parse_args(argv)
    config = ProducerConfig(
        checkpoint=args.checkpoint,
        sidecar=args.sidecar,
        release_dir=args.release_dir,
        store_dir=args.store_dir,
        cache_dir=args.cache_dir,
        window_record=args.window_record,
        out=args.out,
        allowlist=args.allowlist,
        expect_base_ancestry_hash=args.expect_base_ancestry_hash,
        expect_s1_run_manifest_hash=args.expect_s1_run_manifest_hash,
        expect_typed_head_config_hash=args.expect_typed_head_config_hash,
        page_indices=tuple(args.page_indices),
        limit=args.limit,
        offset=args.offset,
        stop_file=args.stop_file,
        visual_input_policy=args.visual_input_policy,
        deepstack=args.deepstack,
        device=args.device,
        max_steps=args.max_steps,
        budget=args.budget,
        typed_decode=args.typed_decode,
        region_decode=args.region_decode,
        loop_guard=args.loop_guard,
        loop_guard_v3_arm=args.loop_guard_v3_arm,
        loop_guard_caps=args.loop_guard_caps,
    )
    try:
        result = run_producer(config, seams=seams)
    except _named_refusals() as exc:
        line = f"{type(exc).__name__}: {exc}".replace("\n", " ")
        print(f"REFUSED {line}", file=sys.stderr)
        return EXIT_REFUSED
    print(json.dumps({"status": result.status, "out": str(result.out_dir)}, sort_keys=True))
    return EXIT_OK if result.status == "complete" else EXIT_STOPPED


if __name__ == "__main__":
    sys.exit(main())
