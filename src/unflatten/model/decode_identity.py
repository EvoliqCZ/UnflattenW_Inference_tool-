"""Identity verdict for decode-speed options (inference-speed programme,
2026-10-11). Pure and CPU-only: no torch, no model.

An option may be called "identity-preserving" only when, on every page of
every compared set, ALL of these are equal to the baseline:

- the accepted token stream (and its length and termination),
- the hidden state hash of every `encode_step` (`step_hidden_sha256`),
- `prediction_ir_hash`, `ast_hash`,
- the diagnostic Word chain's reason codes and `docx_chain` (policy ids,
  outcomes and every edit count).

Anything else is labelled `changes output` and stays OFF. The verdict names
the first divergent page and step so a failing option can be debugged from
the report alone. An empty comparison is never `identical` (it proves
nothing).

`scripts/decode_identity_bench.py` produces the `PageIdentity` records on a
GPU; everything here is what the tests can prove without one.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

IDENTICAL = "identical"
CHANGES_OUTPUT = "changes output"

#: Comparison order inside one page; the first differing field is the one
#: reported first (tokens before hidden states before the downstream hashes).
_SCALAR_FIELDS = (
    "terminated",
    "issues",
    "prediction_ir_hash",
    "ast_hash",
    "docx_produced",
    "docx_reason_codes",
)


class SpeedComparisonInvalid(RuntimeError):
    """A timing comparison whose arms/runs did not decode the same thing:
    a speed number from an earlier termination is not a speed number."""


def tokens_sha256(tokens: Sequence[int]) -> str:
    """sha256 of the token ids as a canonical JSON list (type-independent:
    a tuple and a list of the same ints hash equal)."""
    payload = json.dumps([int(t) for t in tokens], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def hidden_sha256(raw: bytes, *, shape: Sequence[int], dtype: str) -> str:
    """sha256 of a hidden state's raw bytes plus its shape and dtype, so a
    reshape or a dtype change cannot hide behind equal bytes."""
    header = f"{dtype}|{','.join(str(int(d)) for d in shape)}|".encode()
    return hashlib.sha256(header + raw).hexdigest()


@dataclass(frozen=True, slots=True)
class PageIdentity:
    """Everything one page's decode + Word chain is compared on."""

    page_id: str
    tokens: tuple[int, ...]
    terminated: bool
    issues: tuple[str, ...]
    step_hidden_sha256: tuple[str, ...]
    prediction_ir_hash: str | None
    ast_hash: str | None
    docx_produced: bool
    docx_reason_codes: tuple[str, ...]
    docx_chain: Mapping[str, Any] | None

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    @property
    def tokens_sha256(self) -> str:
        return tokens_sha256(self.tokens)

    def to_json(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "tokens": list(self.tokens),
            "terminated": self.terminated,
            "issues": list(self.issues),
            "step_hidden_sha256": list(self.step_hidden_sha256),
            "prediction_ir_hash": self.prediction_ir_hash,
            "ast_hash": self.ast_hash,
            "docx_produced": self.docx_produced,
            "docx_reason_codes": list(self.docx_reason_codes),
            "docx_chain": json.loads(json.dumps(self.docx_chain)),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> PageIdentity:
        return cls(
            page_id=data["page_id"],
            tokens=tuple(int(t) for t in data["tokens"]),
            terminated=bool(data["terminated"]),
            issues=tuple(data["issues"]),
            step_hidden_sha256=tuple(data["step_hidden_sha256"]),
            prediction_ir_hash=data["prediction_ir_hash"],
            ast_hash=data["ast_hash"],
            docx_produced=bool(data["docx_produced"]),
            docx_reason_codes=tuple(data["docx_reason_codes"]),
            docx_chain=json.loads(json.dumps(data["docx_chain"])),
        )


@dataclass(frozen=True, slots=True)
class Divergence:
    """One field of one page that differs from the baseline. `step` is set
    for the per-step fields (`tokens`, `step_hidden_sha256`)."""

    page_id: str
    field: str
    step: int | None
    detail: str


@dataclass(frozen=True, slots=True)
class IdentityVerdict:
    pages_compared: int
    divergences: tuple[Divergence, ...]

    @property
    def identical(self) -> bool:
        return self.pages_compared > 0 and not self.divergences

    @property
    def label(self) -> str:
        return IDENTICAL if self.identical else CHANGES_OUTPUT

    @property
    def first_divergent_page(self) -> str | None:
        return self.divergences[0].page_id if self.divergences else None

    @property
    def first_divergent_step(self) -> int | None:
        return self.divergences[0].step if self.divergences else None

    def summary(self) -> str:
        if self.identical:
            return f"identical on {self.pages_compared} pages"
        if not self.divergences:
            return "changes output: nothing was compared (no pages)"
        first = self.divergences[0]
        where = f" step {first.step}" if first.step is not None else ""
        pages = len({d.page_id for d in self.divergences})
        return (
            f"changes output: first divergence page {first.page_id}{where} "
            f"field {first.field} ({first.detail}); {len(self.divergences)} differing "
            f"field(s) on {pages} of {self.pages_compared} page(s)"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "identical": self.identical,
            "pages_compared": self.pages_compared,
            "first_divergent_page": self.first_divergent_page,
            "first_divergent_step": self.first_divergent_step,
            "summary": self.summary(),
            "divergences": [
                {"page_id": d.page_id, "field": d.field, "step": d.step, "detail": d.detail}
                for d in self.divergences
            ],
        }


def _first_sequence_difference(a: Sequence[Any], b: Sequence[Any]) -> int | None:
    """First index where the sequences differ (a length mismatch counts at the
    shorter length), `None` when equal."""
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def diff_json_paths(a: Any, b: Any, _prefix: str = "") -> list[str]:
    """Slash-separated paths at which two JSON values differ, sorted. A list
    of different length is reported at the list itself."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        paths: list[str] = []
        for key in sorted(set(a) | set(b)):
            child = f"{_prefix}/{key}" if _prefix else str(key)
            if key not in a or key not in b:
                paths.append(child)
            else:
                paths.extend(diff_json_paths(a[key], b[key], child))
        return paths
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [_prefix or "/"]
        paths = []
        for index, (x, y) in enumerate(zip(a, b, strict=True)):
            child = f"{_prefix}/{index}" if _prefix else str(index)
            paths.extend(diff_json_paths(x, y, child))
        return paths
    return [] if a == b else [_prefix or "/"]


def _page_divergences(base: PageIdentity, cand: PageIdentity) -> list[Divergence]:
    out: list[Divergence] = []
    page = base.page_id
    index = _first_sequence_difference(base.tokens, cand.tokens)
    if index is not None:
        out.append(
            Divergence(
                page,
                "tokens",
                index,
                f"token {index}: {base.tokens[index] if index < len(base.tokens) else 'end'}"
                f" vs {cand.tokens[index] if index < len(cand.tokens) else 'end'}"
                f" (lengths {len(base.tokens)} vs {len(cand.tokens)})",
            )
        )
    index = _first_sequence_difference(base.step_hidden_sha256, cand.step_hidden_sha256)
    if index is None and len(base.tokens) > 1 and not base.step_hidden_sha256:
        # Both arms recorded no hidden states for a multi-token page: the
        # axis was not measured, so it must not pass vacuously.
        index = 0
    if index is not None:
        out.append(
            Divergence(
                page,
                "step_hidden_sha256",
                index,
                f"hidden state of step {index} differs "
                f"(steps {len(base.step_hidden_sha256)} vs {len(cand.step_hidden_sha256)})",
            )
        )
    for name in _SCALAR_FIELDS:
        left, right = getattr(base, name), getattr(cand, name)
        if left != right:
            out.append(Divergence(page, name, None, f"{left!r} vs {right!r}"))
    paths = diff_json_paths(base.docx_chain, cand.docx_chain)
    if paths:
        out.append(Divergence(page, "docx_chain", None, "differs at " + ", ".join(paths[:8])))
    return out


def identity_verdict(
    baseline: Sequence[PageIdentity], candidate: Sequence[PageIdentity]
) -> IdentityVerdict:
    """Compare two runs page by page (matched on `page_id`, reported in
    baseline order). A page present on one side only is a `page_set`
    divergence: the two runs did not decode the same cohort."""
    cand_by_id = {page.page_id: page for page in candidate}
    base_ids = [page.page_id for page in baseline]
    divergences: list[Divergence] = []
    for page in baseline:
        other = cand_by_id.get(page.page_id)
        if other is None:
            divergences.append(Divergence(page.page_id, "page_set", None, "missing in candidate"))
        else:
            divergences.extend(_page_divergences(page, other))
    for page in candidate:
        if page.page_id not in set(base_ids):
            divergences.append(Divergence(page.page_id, "page_set", None, "missing in baseline"))
    return IdentityVerdict(pages_compared=len(baseline), divergences=tuple(divergences))


# --- speed protocol -------------------------------------------------------


def arm_schedule(arms: Sequence[str], *, runs: int) -> list[tuple[int, str]]:
    """`(run, arm)` execution order: ABAB..., every arm once per run, arms in
    the given order each run. At least two runs, because run 0 is a warm-up
    that is discarded and one measured run is not a distribution."""
    if runs < 2:
        raise ValueError("runs must be >= 2: run 0 is discarded as warm-up")
    return [(run, arm) for run in range(runs) for arm in arms]


def summarize(values: Sequence[float]) -> dict[str, float]:
    """median, quartiles (inclusive method), iqr, min, max, n."""
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(float(v) for v in values)
    if len(ordered) >= 2:
        q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    else:
        q1 = q3 = ordered[0]
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": ordered[0],
        "max": ordered[-1],
    }


def assert_same_termination(rows: Sequence[Mapping[str, Any]]) -> None:
    """Every row (one decode of one page: `arm`, `run`, `tokens`,
    `terminated`) must have the same token count and termination. Raises
    `SpeedComparisonInvalid` otherwise -- timing from a decode that stopped
    earlier is not comparable."""
    if not rows:
        raise SpeedComparisonInvalid("no rows to compare")
    first = rows[0]
    for row in rows[1:]:
        for key in ("tokens", "terminated"):
            if row[key] != first[key]:
                raise SpeedComparisonInvalid(
                    f"{key} differs: arm {first['arm']} run {first['run']} = {first[key]!r}, "
                    f"arm {row['arm']} run {row['run']} = {row[key]!r}"
                )


def require_free_vram(*, free_mib: float, need_mib: float) -> None:
    """Refuse to start when the GPU has less free memory than the run needs."""
    if free_mib < need_mib:
        raise RuntimeError(
            f"free VRAM {free_mib:.0f} MiB < required {need_mib:.0f} MiB; "
            "refusing to start (something else holds the GPU)"
        )


def _first_run_by_page(rows: Sequence[Mapping[str, Any]], arm: str, run: int) -> list[PageIdentity]:
    seen: dict[str, PageIdentity] = {}
    for row in rows:
        if row["arm"] == arm and row["run"] == run and row["page_id"] not in seen:
            seen[row["page_id"]] = row["identity"]
    return list(seen.values())


def analyze_runs(
    rows: Sequence[Mapping[str, Any]], *, arms: Sequence[str], discard_runs: int = 1
) -> dict[str, Any]:
    """The whole measurement protocol over recorded executions.

    `rows`: one per (page, arm, run) with `page_id`, `arm`, `run` (0-based),
    `identity` (`PageIdentity`) and `timing` (`{metric: seconds}`). Runs
    `< discard_runs` are warm-up: they count for identity and termination but
    never for timing.

    - Termination: every execution of a page, in any arm or run, must carry
      the same token count and termination, else `speed_valid` is false and
      no timing is reported at all (speed from an earlier stop is no speed).
    - Self-determinism: each arm's later runs must equal its first run.
    - Arms: every arm after the first is compared to the first (`<arm>_vs_<first>`).
    - Timing: per page and arm over the measured runs, and per arm as the sum
      over pages for each measured run (`totals`), plus the paired per-run
      ratio to the first arm; each as median / quartiles / min / max.
    """
    arms = tuple(arms)
    runs = sorted({row["run"] for row in rows})
    measured = [run for run in runs if run >= discard_runs]
    if not measured:
        raise ValueError(f"no measured run: runs {runs} with {discard_runs} discarded")
    first_run = runs[0]

    pages = list(dict.fromkeys(row["page_id"] for row in rows))
    invalid: list[str] = []
    for page in pages:
        try:
            assert_same_termination(
                [
                    {
                        "arm": row["arm"],
                        "run": row["run"],
                        "tokens": row["identity"].token_count,
                        "terminated": row["identity"].terminated,
                    }
                    for row in rows
                    if row["page_id"] == page
                ]
            )
        except SpeedComparisonInvalid as exc:
            invalid.append(f"{page}: {exc}")

    self_identity: dict[str, Any] = {}
    for arm in arms:
        base = _first_run_by_page(rows, arm, first_run)
        later: IdentityVerdict | None = None
        for run in runs[1:]:
            verdict = identity_verdict(base, _first_run_by_page(rows, arm, run))
            if not verdict.identical:
                later = verdict
                break
        self_identity[arm] = {
            "identical": later is None and bool(base),
            "runs_compared": len(runs),
            "verdict": later.to_json() if later is not None else None,
        }
    identity: dict[str, Any] = {"self": self_identity}
    reference = _first_run_by_page(rows, arms[0], measured[0])
    for arm in arms[1:]:
        identity[f"{arm}_vs_{arms[0]}"] = identity_verdict(
            reference, _first_run_by_page(rows, arm, measured[0])
        ).to_json()

    report: dict[str, Any] = {
        "arms": list(arms),
        "pages": pages,
        "runs": runs,
        "discarded_runs": discard_runs,
        "identity": identity,
        "speed_valid": not invalid,
    }
    if invalid:
        report["speed_invalid_reason"] = "; ".join(invalid)
        return report

    metrics = sorted(
        set.intersection(
            *(
                {k for k, v in row["timing"].items() if isinstance(v, int | float)}
                for row in rows
            )
        )
    )
    value: dict[tuple[str, str, int, str], float] = {
        (row["page_id"], row["arm"], row["run"], metric): float(row["timing"][metric])
        for row in rows
        for metric in metrics
    }
    report["per_page"] = {
        page: {
            arm: {
                metric: summarize([value[(page, arm, run, metric)] for run in measured])
                for metric in metrics
            }
            for arm in arms
        }
        for page in pages
    }
    run_totals = {
        (arm, metric, run): sum(value[(page, arm, run, metric)] for page in pages)
        for arm in arms
        for metric in metrics
        for run in measured
    }
    report["totals"] = {
        arm: {
            metric: summarize([run_totals[(arm, metric, run)] for run in measured])
            for metric in metrics
        }
        for arm in arms
    }
    report["ratio_vs_first_arm"] = {
        arm: {
            metric: summarize(
                [
                    run_totals[(arm, metric, run)] / run_totals[(arms[0], metric, run)]
                    for run in measured
                ]
            )
            for metric in metrics
        }
        for arm in arms[1:]
    }
    return report
