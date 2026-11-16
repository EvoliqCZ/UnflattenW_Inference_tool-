"""Write the models.json of the local launch scripts (dev-native.sh, dev-bundle.sh).

Reads the locally built bundles from stdin, one `<precision> <kv_cache>
<bundle_dir>` line each, and writes a validated models config listing them
as published variants plus the catalogue entries of
`deploy/docker/models.example.json` that are not published yet (owner
2026-09-29: BF16/FP8/INT8 `unpublished`, Unflatten W MoE `planned`). The
`--default` variant must be one of the local bundles. Nothing is written if
the result fails the service's models-config loader (which also refuses a
default that is not a published, i.e. local, variant).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from unflatten.service.models_config import (
    SCHEMA_VERSION,
    ModelsConfigInvalid,
    is_published,
    validate_models_config,
)

DEFAULT_CATALOGUE = Path(__file__).resolve().parents[1] / "docker" / "models.example.json"


def _local_variants(lines: list[str]) -> list[dict[str, str]]:
    variants = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        precision, kv_cache, bundle_dir = line.split(" ", 2)
        variants.append(
            {
                "variant_id": f"{precision}-{kv_cache}",
                "label": f"{precision.upper()} / KV {kv_cache.upper()}",
                "precision": precision,
                "kv_cache": kv_cache,
                "bundle": bundle_dir,
            }
        )
    return variants


def _catalogue_variants(catalogue_path: Path) -> list[dict]:
    catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))
    return [variant for variant in catalogue["variants"] if not is_published(variant)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default", required=True, help="variant_id of the default local bundle")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    args = parser.parse_args(argv)

    local = _local_variants(sys.stdin.readlines())
    try:
        config = validate_models_config(
            {
                "schema_version": SCHEMA_VERSION,
                "default_variant": args.default,
                "variants": local + _catalogue_variants(args.catalogue),
            }
        )
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        print(
            f"write_models_config: cannot read catalogue {args.catalogue}: {exc}", file=sys.stderr
        )
        return 1
    except ModelsConfigInvalid as exc:
        print(f"write_models_config: generated config is invalid: {exc}", file=sys.stderr)
        return 1
    args.out.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
