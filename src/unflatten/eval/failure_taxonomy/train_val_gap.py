"""Train-vs-`val_iid` error-gap tool (RFC005-ERRATA-007 E3, RFC003-ERRATA-003
E3 OPEN-003-E3-3).

RFC005-ERRATA-007 E3 defers the choice of `producer_exposure`
(`in_sample` / `out_of_fold` / `earlier_checkpoint`) for
`reasoning_corrective_v0` candidate production to a measurement: "pred
výberom zmerať, ako sa líši miera a typ chýb rodiča na train a na
`val_iid`. Ak je rozdiel malý, stačí variant 2. Ak veľký, treba variant 1".
This module IS that measurement, over two already-built
`failure_taxonomy.report` reports (or two sealed cohort artefacts, built
into reports first via `report.build_report`) of the same checkpoint --
one over a train sample, one over `val_iid`.

Pure CPU: it only aggregates counts a report already carries and runs a
seeded percentile bootstrap over them, never touches a GPU or a cache
split itself.

Refuses a locked-test or P6 role on either side, exactly like
`report.check_role` (RFC005-ERRATA-007 E1: never locked test, never P6) --
a train *sample* used for this measurement must still be a development
cohort, not the withheld ones.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from unflatten.eval.failure_taxonomy import report as ft

REPORT_SCHEMA_VERSION = "0.1.0"
DEFAULT_SEED = 0
DEFAULT_RESAMPLES = 2000
CI_METHOD = (
    "seeded percentile bootstrap (95 %), each split resampled independently as "
    "Binomial(denominator, rate) draws; gap = val_iid rate - train rate"
)


def check_split_roles(train_role: str, val_iid_role: str) -> None:
    """Refuse a locked-test or P6 role on either side (RFC005-ERRATA-007 E1),
    exactly as `report.check_role` does for a single artefact."""
    ft.check_role(train_role)
    ft.check_role(val_iid_role)


def _rate(frequency: Mapping[str, Any]) -> float | None:
    if frequency["status"] != "counted":
        return None
    return frequency["count"] / frequency["denominator"]


def _bootstrap_gap_ci(
    count_train: int, denominator_train: int,
    count_val: int, denominator_val: int,
    *, seed: int, resamples: int,
) -> tuple[float, float] | None:
    """Seeded percentile 95 % CI of `rate_val - rate_train`, each split
    resampled independently as `Binomial(denominator, rate)` (equivalent in
    distribution to resampling-with-replacement the split's own 0/1 vector,
    without materializing it)."""
    if denominator_train <= 0 or denominator_val <= 0:
        return None
    rng = random.Random(seed)
    rate_train, rate_val = count_train / denominator_train, count_val / denominator_val
    deltas = sorted(
        rng.binomialvariate(denominator_val, rate_val) / denominator_val
        - rng.binomialvariate(denominator_train, rate_train) / denominator_train
        for _ in range(resamples)
    )
    lo_index = int(0.025 * resamples)
    hi_index = min(resamples - 1, int(0.975 * resamples))
    return (deltas[lo_index], deltas[hi_index])


def gap_report(
    train_report: Mapping[str, Any],
    val_iid_report: Mapping[str, Any],
    *, seed: int = DEFAULT_SEED, resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    """Per failure class: rate on `train_report` and on `val_iid_report`,
    their denominators, and a seeded bootstrap 95 % CI on the gap. Both
    reports must be over the **same checkpoint** (`checkpoint_sha256`);
    comparing different checkpoints would confound the model's own drift
    with the train/val_iid split difference this tool measures."""
    check_split_roles(train_report["role"], val_iid_report["role"])
    train_ckpt = train_report.get("checkpoint_sha256")
    val_ckpt = val_iid_report.get("checkpoint_sha256")
    if train_ckpt != val_ckpt:
        raise ValueError(
            f"train report checkpoint_sha256={train_ckpt!r} != "
            f"val_iid report checkpoint_sha256={val_ckpt!r}: not the same checkpoint"
        )
    by_train = {r["failure_class"]: r for r in train_report["classes"]}
    by_val = {r["failure_class"]: r for r in val_iid_report["classes"]}
    missing = (set(ft.CLASS_RULES) - by_train.keys()) | (set(ft.CLASS_RULES) - by_val.keys())
    if missing:
        raise KeyError(f"report missing failure classes: {sorted(missing)}")

    classes = []
    for name in ft.CLASS_RULES:
        f_train, f_val = by_train[name]["frequency"], by_val[name]["frequency"]
        rate_train, rate_val = _rate(f_train), _rate(f_val)
        ci = None
        if rate_train is not None and rate_val is not None:
            ci = _bootstrap_gap_ci(
                f_train["count"], f_train["denominator"],
                f_val["count"], f_val["denominator"],
                seed=seed, resamples=resamples,
            )
        classes.append({
            "failure_class": name,
            "train": {"count": f_train["count"], "denominator": f_train["denominator"],
                      "status": f_train["status"], "rate": rate_train},
            "val_iid": {"count": f_val["count"], "denominator": f_val["denominator"],
                       "status": f_val["status"], "rate": rate_val},
            "gap": {
                "value": (rate_val - rate_train) if ci is not None else None,
                "ci_95": list(ci) if ci is not None else None,
            },
        })
    return {
        "train_val_gap_schema_version": REPORT_SCHEMA_VERSION,
        "checkpoint_sha256": train_ckpt,
        "train_role": train_report["role"],
        "val_iid_role": val_iid_report["role"],
        "train_report_hash": train_report.get("report_hash"),
        "val_iid_report_hash": val_iid_report.get("report_hash"),
        "seed": seed,
        "resamples": resamples,
        "method": CI_METHOD,
        "classes": classes,
    }


def gap_report_from_artefacts(
    train_artefact: Any, train_gold_rows: Mapping[str, Any],
    val_iid_artefact: Any, val_iid_gold_rows: Mapping[str, Any],
    *, seed: int = DEFAULT_SEED, resamples: int = DEFAULT_RESAMPLES,
    train_build_kwargs: Mapping[str, Any] | None = None,
    val_iid_build_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: build both `failure_taxonomy_report`s (this is
    where the role refusal actually first triggers, in `report.build_report`
    itself) and then compare them. `train_build_kwargs`/`val_iid_build_kwargs`
    are forwarded to `report.build_report` (e.g. `pipeline`, `split_entries`)."""
    check_split_roles(train_artefact.role, val_iid_artefact.role)
    train_report = ft.build_report(
        train_artefact, train_gold_rows, **dict(train_build_kwargs or {})
    )
    val_iid_report = ft.build_report(
        val_iid_artefact, val_iid_gold_rows, **dict(val_iid_build_kwargs or {})
    )
    return gap_report(train_report, val_iid_report, seed=seed, resamples=resamples)


def _fmt_gap(record: Mapping[str, Any]) -> str:
    gap = record["gap"]
    if gap["value"] is None:
        return "insufficient_evidence"
    lo, hi = gap["ci_95"]
    return f"{100 * gap['value']:+.1f} pp [{100 * lo:+.1f}; {100 * hi:+.1f}]"


def render_markdown(gap: Mapping[str, Any]) -> str:
    lines = [
        "# Rozdiel chýb train vs val_iid (RFC005-ERRATA-007 E3)",
        "",
        f"- checkpoint: `{gap['checkpoint_sha256']}`",
        f"- train rola: `{gap['train_role']}`, val_iid rola: `{gap['val_iid_role']}`",
        f"- seed: `{gap['seed']}`, resamples: `{gap['resamples']}`",
        f"- metóda: {gap['method']}",
        "",
        "| trieda | train (n) | val_iid (n) | medzera [95 % CI] |",
        "|---|---|---|---|",
    ]
    for record in gap["classes"]:
        t, v = record["train"], record["val_iid"]
        t_cell = (
            f"{100 * t['rate']:.1f} % ({t['count']}/{t['denominator']})"
            if t["rate"] is not None else "insufficient_evidence"
        )
        v_cell = (
            f"{100 * v['rate']:.1f} % ({v['count']}/{v['denominator']})"
            if v["rate"] is not None else "insufficient_evidence"
        )
        lines.append(f"| `{record['failure_class']}` | {t_cell} | {v_cell} | {_fmt_gap(record)} |")
    lines.append("")
    return "\n".join(lines)
