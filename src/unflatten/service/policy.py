from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema.protocols
import jsonschema.validators

from unflatten.ast.canonical_json import to_canonical_json

DEFAULT_RUNTIME_POLICY_PATH = Path(__file__).parent / "policies" / "runtime_policy.json"
_SCHEMA_PATH = Path(__file__).parent / "schemas" / "runtime_policy.schema.json"
#: DEC-062: policy_version 1.3.0 raises the PDF ceiling to 100 pages.
#: The closed schema remains 1.2.0; the complete policy has a new approved hash.
APPROVED_RUNTIME_POLICY_HASH = (
    "sha256:8ab57dc00aac96a67e8465c472a7fbf101b5cd5b91b9555107c7271981008418"
)


class RuntimePolicyInvalid(ValueError):
    """The required service runtime policy is absent, open, or internally unsafe."""


@cache
def _validator() -> jsonschema.protocols.Validator:
    with _SCHEMA_PATH.open(encoding="utf-8") as stream:
        schema = json.load(stream)
    validator_type = jsonschema.validators.validator_for(schema)
    validator_type.check_schema(schema)
    return validator_type(schema)


def validate_runtime_policy(policy: Mapping[str, Any]) -> None:
    """Validate the closed policy and relationships JSON Schema cannot express."""
    errors = sorted(_validator().iter_errors(policy), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        raise RuntimePolicyInvalid(
            f"runtime policy schema validation failed: {error.message} "
            f"at {list(error.absolute_path)}"
        )

    if policy["heartbeat_interval_seconds"] >= policy["lease_duration_seconds"]:
        raise RuntimePolicyInvalid("heartbeat interval must be strictly below lease duration")

    width_bound = policy["max_page_width_points"] * policy["raster_dpi"] // 72
    height_bound = policy["max_page_height_points"] * policy["raster_dpi"] // 72
    if policy["max_raster_width_pixels"] > width_bound:
        raise RuntimePolicyInvalid("raster width exceeds the page-point/DPI bound")
    if policy["max_raster_height_pixels"] > height_bound:
        raise RuntimePolicyInvalid("raster height exceeds the page-point/DPI bound")
    raster_area = policy["max_raster_width_pixels"] * policy["max_raster_height_pixels"]
    if policy["max_raster_pixels"] > raster_area:
        raise RuntimePolicyInvalid("raster pixel ceiling exceeds bounded raster dimensions")
    if policy["max_sandbox_output_bytes"] > policy["max_sandbox_tmpfs_bytes"]:
        raise RuntimePolicyInvalid("sandbox output ceiling exceeds sandbox tmpfs")
    if policy["max_png_bytes"] > policy["max_sandbox_output_bytes"]:
        raise RuntimePolicyInvalid("PNG ceiling exceeds sandbox output ceiling")
    if policy["scanner_cpu_time_seconds"] > policy["scanner_wall_time_seconds"]:
        raise RuntimePolicyInvalid("scanner CPU ceiling exceeds wall-time ceiling")
    if policy["active_jobs_per_principal"] > policy["global_active_jobs"]:
        raise RuntimePolicyInvalid("per-principal active jobs exceed global active jobs")


def load_runtime_policy(path: Path | str = DEFAULT_RUNTIME_POLICY_PATH) -> dict[str, Any]:
    """Load a required runtime policy, failing startup on read or validation errors."""
    policy_path = Path(path)
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimePolicyInvalid(f"cannot read runtime policy {policy_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimePolicyInvalid(
            f"runtime policy {policy_path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimePolicyInvalid("runtime policy root must be an object")
    validate_runtime_policy(payload)
    if (
        policy_path.resolve() == DEFAULT_RUNTIME_POLICY_PATH.resolve()
        and runtime_policy_hash(payload) != APPROVED_RUNTIME_POLICY_HASH
    ):
        raise RuntimePolicyInvalid("checked-in runtime policy does not match its approved hash")
    return payload


def runtime_policy_hash(policy: Mapping[str, Any]) -> str:
    """Return SHA-256 over RFC 8785 canonical bytes of the complete policy."""
    validate_runtime_policy(policy)
    return f"sha256:{hashlib.sha256(to_canonical_json(dict(policy))).hexdigest()}"
