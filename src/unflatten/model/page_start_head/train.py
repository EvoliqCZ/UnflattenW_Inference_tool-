"""Trainer of the DEC-061 page-start head (design §8.5, decision D-10).

    python -m unflatten.model.page_start_head.train --moments DIR --labels census.jsonl ... \\
        --sample sample.json --kind {logistic,mlp} --seeds 1,2,3 --epochs 30 \\
        --dec061-record docs/decisions/DEC-061-....md --out DIR

This is a discovery-class gradient of a separate module outside the model. It refuses before
the first gradient step unless `--dec061-record` names an existing file; the record's path and
sha256 go to `metrics.json` and to `provenance.json` next to the bundle (the bundle contract
schema is task 9's and stays untouched). Standardisation is fitted on `train_head` only, the
epoch is chosen on `dev_head` balanced accuracy, and the `diag` set is evaluated once, for the
selected head, after selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from unflatten.model.page_start_policy import load_page_start_policy

from .bundle import save_head_bundle
from .head import HEAD_KINDS
from .moments import MomentsContract

PageKey = tuple[str, int]
CURVE_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
SD_FLOOR = 1e-4


class TrainInputError(ValueError):
    """Moments, labels and sample disagree, or the run is not authorised."""


def require_dec061_record(path: Path | None) -> dict[str, str]:
    """Refuse the gradient without an existing DEC-061 record; returns its path and sha256."""
    if path is None or not Path(path).is_file():
        raise TrainInputError(
            f"page-start head gradient refused: --dec061-record {path!r} is not an existing file "
            "(DEC-061 must exist before the head is trained, design D-10)"
        )
    data = Path(path).read_bytes()
    return {
        "path": str(Path(path).resolve()),
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
    }


# -- data -----------------------------------------------------------------------------


def load_moments(directory: Path) -> tuple[MomentsContract, dict[PageKey, torch.Tensor]]:
    shards = sorted(Path(directory).glob("moments-*.pt"))
    if not shards:
        raise TrainInputError(f"no moments-*.pt shards in {directory}")
    contract: MomentsContract | None = None
    moments: dict[PageKey, torch.Tensor] = {}
    for shard in shards:
        blob = torch.load(shard, weights_only=True)
        this = MomentsContract.from_dict(blob["contract"])
        if contract is None:
            contract = this
        else:
            contract.refuse_mismatch(this)
        for bundle, page, row in zip(
            blob["bundles"], blob["page_indexes"].tolist(), blob["moments"], strict=True
        ):
            if (bundle, page) in moments:
                raise TrainInputError(f"{shard.name}: page {(bundle, page)} extracted twice")
            moments[(bundle, page)] = row
    assert contract is not None
    return contract, moments


def load_labels(paths: list[Path]) -> dict[PageKey, dict[str, Any]]:
    rows: dict[PageKey, dict[str, Any]] = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[(row["document_bundle_id"], int(row["page_index"]))] = row
    return rows


class Split:
    def __init__(self, keys: list[PageKey], moments: dict, labels: dict) -> None:
        missing = [k for k in keys if k not in moments or k not in labels]
        if missing:
            raise TrainInputError(
                f"{len(missing)} sampled pages lack moments or labels: {missing[:3]}"
            )
        self.keys = keys
        self.x = torch.stack([moments[k] for k in keys]).float()
        self.y = torch.tensor(
            [[float(labels[k]["has_header"]), float(labels[k]["has_footer"])] for k in keys]
        )
        self.empty = torch.tensor(
            [
                [labels[k].get("empty_header") is True, labels[k].get("empty_footer") is True]
                for k in keys
            ]
        )


# -- metrics --------------------------------------------------------------------------


def balanced_accuracy(pred: torch.Tensor, truth: torch.Tensor) -> float | None:
    """Mean of recall over the classes present; None when the set is empty."""
    recalls = []
    for cls in (True, False):
        mask = truth == cls
        if mask.any():
            recalls.append(float((pred[mask] == cls).float().mean()))
    return sum(recalls) / len(recalls) if recalls else None


def evaluate(head: nn.Module, split: Split, threshold: float) -> dict[str, Any]:
    head.eval()
    with torch.no_grad():
        p = torch.sigmoid(head(split.x))
    pred, truth = p >= 0.5, split.y >= 0.5
    conf = torch.maximum(p, 1 - p).min(dim=1).values
    ba_h, ba_f = (
        balanced_accuracy(pred[:, 0], truth[:, 0]),
        balanced_accuracy(pred[:, 1], truth[:, 1]),
    )
    out: dict[str, Any] = {
        "n": len(split.keys),
        "ba_has_header": ba_h,
        "ba_has_footer": ba_f,
        "ba_mean": None if ba_h is None or ba_f is None else (ba_h + ba_f) / 2,
    }
    for col, name in ((0, "empty_header"), (1, "empty_footer")):
        mask = split.empty[:, col]
        out[f"accuracy_{name}"] = (
            float((pred[mask, col] == truth[mask, col]).float().mean()) if mask.any() else None
        )
        out[f"n_{name}"] = int(mask.sum())
    out["coverage_at_threshold"] = float((conf >= threshold).float().mean())
    curve = []
    for t in CURVE_THRESHOLDS:
        keep = conf >= t
        if keep.any():
            parts = [balanced_accuracy(pred[keep, c], truth[keep, c]) for c in (0, 1)]
            ba = None if None in parts else sum(parts) / 2
        else:
            ba = None
        curve.append({"threshold": t, "coverage": float(keep.float().mean()), "ba_mean": ba})
    out["coverage_ba_curve"] = curve
    return out


def mean_sd(values: list[float | None]) -> dict[str, float | None]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "sd": None}
    return {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0}


# -- training -------------------------------------------------------------------------


def train_one_seed(
    kind: str,
    seed: int,
    train: Split,
    dev: Split,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    threshold: float,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    head = HEAD_KINDS[kind](train.x.shape[1])
    mu = train.x.mean(0)
    head.set_standardisation(mu, train.x.std(0, unbiased=False).clamp_min(SD_FLOOR))
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    best: tuple[float, int, dict[str, Any], dict[str, torch.Tensor]] | None = None
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(train.keys), generator=gen)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            opt.zero_grad()
            loss_fn(head(train.x[idx]), train.y[idx]).backward()
            opt.step()
        dev_metrics = evaluate(head, dev, threshold)
        score = dev_metrics["ba_mean"] if dev_metrics["ba_mean"] is not None else float("-inf")
        if best is None or score > best[0]:
            best = (score, epoch, dev_metrics, {k: v.clone() for k, v in head.state_dict().items()})
    assert best is not None
    head.load_state_dict(best[3])
    return head, {"seed": seed, "best_epoch": best[1], "dev": best[2]}


def run_training(
    *,
    moments_dir: Path,
    label_paths: list[Path],
    sample_path: Path,
    kind: str,
    seeds: list[int],
    epochs: int,
    dec061_record: Path | None,
    out: Path,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> dict[str, Any]:
    record = require_dec061_record(dec061_record)
    if (out / "bundle").exists():
        raise TrainInputError(
            f"{out / 'bundle'} already exists; refusing to overwrite a head bundle"
        )
    if kind not in HEAD_KINDS:
        raise TrainInputError(f"unknown head kind {kind!r}")
    contract, moments = load_moments(moments_dir)
    policy = load_page_start_policy()
    expected = policy.feature_identity_sha256(
        merge_size=contract.merge_size,
        visual_input_policy_hash=contract.visual_input_policy_hash,
        feature_tap=contract.feature_tap,
    )
    if (contract.band_fraction_top, contract.band_fraction_bottom) != (
        policy.band_fraction_top,
        policy.band_fraction_bottom,
    ):
        raise TrainInputError(
            f"moments bands {(contract.band_fraction_top, contract.band_fraction_bottom)} != "
            f"current policy bands {(policy.band_fraction_top, policy.band_fraction_bottom)}"
        )
    if contract.feature_identity_sha256 != expected:
        raise TrainInputError(
            f"moments feature identity {contract.feature_identity_sha256} != current policy "
            f"identity {expected}; the moments were extracted under other bands or semantics"
        )
    sample = json.loads(Path(sample_path).read_text())
    labels = load_labels(label_paths)
    for name in ("train_head", "dev_head", "diag"):
        if not sample["sets"][name]:
            raise TrainInputError(f"sample set {name!r} is empty")
    splits = {
        name: Split([(b, int(p)) for b, p in sample["sets"][name]], moments, labels)
        for name in ("train_head", "dev_head", "diag")
    }
    threshold = policy.confidence_threshold
    heads, per_seed = [], []
    for seed in seeds:
        head, record_seed = train_one_seed(
            kind,
            seed,
            splits["train_head"],
            splits["dev_head"],
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            threshold=threshold,
        )
        heads.append(head)
        per_seed.append(record_seed)
        print(
            f"[train] seed {seed}: dev ba_mean={record_seed['dev']['ba_mean']} "
            f"epoch {record_seed['best_epoch']}",
            flush=True,
        )
    def _dev_score(i: int) -> float:
        ba = per_seed[i]["dev"]["ba_mean"]
        return float("-inf") if ba is None else ba

    chosen = max(range(len(seeds)), key=_dev_score)
    diag = evaluate(heads[chosen], splits["diag"], threshold)
    dev_keys = [k for k in per_seed[0]["dev"] if k not in ("n", "coverage_ba_curve")]
    metrics = {
        "kind": kind,
        "seeds": seeds,
        "epochs": epochs,
        "confidence_threshold": threshold,
        "dec061_record": record,
        "sample_sha256": sample.get("sha256"),
        "per_seed": per_seed,
        "dev_mean_sd": {k: mean_sd([s["dev"][k] for s in per_seed]) for k in dev_keys},
        "selected_seed": seeds[chosen],
        "diag_selected_seed": diag,
    }
    out.mkdir(parents=True, exist_ok=True)
    sha = save_head_bundle(out / "bundle", contract=contract, head=heads[chosen])
    metrics["bundle_sha256"] = sha
    (out / "bundle" / "provenance.json").write_text(
        json.dumps(
            {"dec061_record": record, "selected_seed": seeds[chosen], "kind": kind}, indent=1
        )
        + "\n"
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--moments", type=Path, required=True)
    ap.add_argument("--labels", type=Path, nargs="+", required=True)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--kind", choices=sorted(HEAD_KINDS), required=True)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dec061-record", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        run_training(
            moments_dir=args.moments,
            label_paths=args.labels,
            sample_path=args.sample,
            kind=args.kind,
            seeds=[int(s) for s in args.seeds.split(",")],
            epochs=args.epochs,
            dec061_record=args.dec061_record,
            out=args.out,
            lr=args.lr,
            batch_size=args.batch_size,
        )
    except TrainInputError as exc:
        print(f"[train] refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
