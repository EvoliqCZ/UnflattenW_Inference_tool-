"""RFC005-ERRATA-007 / DEC-045 S7 gradient ratification gate.

DEC-045 ratifies the VC program's *design* only -- "no stage from S5 on is
unblocked" (project memory `second-model-programs-proposed`). A dormant S7
trainer (:mod:`s7_vc1_train`, :mod:`s7_vc2_train`) may exercise its stub
forward/backward path under ``--dry-run`` for tests, but a *real* gradient
step additionally requires an explicit, on-disk ratification record naming
the exact stage and child it authorizes. Today no such record exists
anywhere in the repo, so every non-dry-run call refuses -- this module's
whole purpose is to make that refusal real and tested, not aspirational.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

#: The only children this gate knows how to authorize (RFC005-ERRATA-007
#: E5): VC-3 (signal b) never reaches a gradient step at all -- it refuses
#: unconditionally via `PreEncoderSignalBUnavailable` -- so it is
#: deliberately absent here. VC-4 (combined PRE+POST,
#: :mod:`s7_vc4_combined`) is included even though it never trains anything
#: itself (it composes already-ratified VC-1/VC-2 evidence) -- it still runs
#: its own gated dry-run/manifest path and needs the same ratification
#: surface as any other S7 child. ``"dflash_speculator"`` (RFC004 §37.2,
#: :mod:`unflatten.model.speculative.s7_dflash_train`) is a second, unrelated
#: S7 program sharing only this gate -- the module_kind name, never the
#: forbidden bare "DFlash" (RFC004 ~:949-957).
#: RFC004 §37 / RFC005-ERRATA-007: `"MTP"` (`speculative/s7_mtp_train.py`)
#: trains only `MtpHeads` with the primary frozen, exactly like VC-1/VC-2
#: train only their own compressor -- same ratification surface, same gate.
KNOWN_S7_CHILDREN: Final[frozenset[str]] = frozenset(
    {"VC-1", "VC-2", "VC-4", "dflash_speculator", "MTP"}
)

_REQUIRED_STAGE = "S7"


class S7RatificationRequired(RuntimeError):
    """A gradient step was requested without a valid ratification record."""


@dataclass(frozen=True, slots=True)
class S7RatificationRecord:
    """One owner-signed authorization to run a real S7 gradient step."""

    decision_id: str
    stage: str
    child: str
    ratified: bool
    ratified_by: str
    ratified_at: str


def load_ratification_record(path: Path) -> S7RatificationRecord:
    """Read and validate a ratification record file.

    Raises :class:`S7RatificationRequired` for anything short of a
    well-formed, explicitly ``ratified: true`` record for stage ``"S7"`` --
    a malformed or missing record is exactly as authorizing as no record.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise S7RatificationRequired(
            f"ratification record {path} could not be read as JSON: {exc}"
        ) from exc
    required_fields = ("decision_id", "stage", "child", "ratified", "ratified_by", "ratified_at")
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise S7RatificationRequired(
            f"ratification record {path} is missing required field(s): {missing}"
        )
    if payload["stage"] != _REQUIRED_STAGE:
        raise S7RatificationRequired(
            f"ratification record {path} names stage {payload['stage']!r}, "
            f"not {_REQUIRED_STAGE!r}"
        )
    if payload["child"] not in KNOWN_S7_CHILDREN:
        raise S7RatificationRequired(
            f"ratification record {path} names child {payload['child']!r}, "
            f"not one of {sorted(KNOWN_S7_CHILDREN)}"
        )
    if payload["ratified"] is not True:
        raise S7RatificationRequired(
            f"ratification record {path} has ratified={payload['ratified']!r}, not true"
        )
    return S7RatificationRecord(
        decision_id=payload["decision_id"],
        stage=payload["stage"],
        child=payload["child"],
        ratified=payload["ratified"],
        ratified_by=payload["ratified_by"],
        ratified_at=payload["ratified_at"],
    )


def require_ratification(
    *, child: str, dry_run: bool, ratification_record: Path | None
) -> S7RatificationRecord | None:
    """Gate a real gradient step behind an explicit ratification record.

    ``dry_run=True`` never touches ``ratification_record`` and always
    returns ``None`` -- a dry run computes the stub forward/backward for
    tests but must never claim to be authorized. ``dry_run=False`` raises
    :class:`S7RatificationRequired` unless ``ratification_record`` points at
    a file that validates for exactly this ``child``.
    """
    if child not in KNOWN_S7_CHILDREN:
        raise ValueError(f"child must be one of {sorted(KNOWN_S7_CHILDREN)}, got {child!r}")
    if dry_run:
        return None
    if ratification_record is None:
        raise S7RatificationRequired(
            f"a real S7 {child} gradient step requires an explicit ratification "
            "record file (--ratification-record); pass --dry-run to exercise the "
            "stub path without one"
        )
    record = load_ratification_record(ratification_record)
    if record.child != child:
        raise S7RatificationRequired(
            f"ratification record {ratification_record} authorizes child "
            f"{record.child!r}, not the requested {child!r}"
        )
    return record
