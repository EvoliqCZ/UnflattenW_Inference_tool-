"""Linear and MLP probes per ViT level x signal (ViT-level probe).

For every representation and signal: a linear (softmax) probe and a
1-hidden-layer MLP probe, each over 3 seeds, plus a shuffled-label linear
control over 3 seeds. Scored by macro-F1 over the classes present in the
test split (val_iid pages, lineage-disjoint from train).

Nothing is tuned on test. Early stopping and the linear learning rate are
chosen on a dev split of TRAIN pages (whole pages, the same for every
seed), so a fixed optimiser setting cannot pose as a ceiling. The loss is
class-balanced, since macro-F1 weighs every class equally.

`level_gap` answers the RFC007 question directly: best non-pooler level
minus `pooler`, averaged over seeds, with its spread from a paired
bootstrap over TEST PAGES (tokens on one page are not independent).
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from unflatten.probe.vit_level_labels import N_CLASSES, SIGNALS

SEEDS = (0, 1, 2)
LINEAR_LRS = (1e-2, 1e-3)
MLP_LR = 1e-3
MLP_HIDDEN = 512
EPOCHS = 10
BATCH = 2048
TRAIN_CAP = 60_000
DEV_CAP = 20_000
DEV_PAGE_FRACTION = 0.15
BOOTSTRAP = 300
CONCAT = ("ds5", "ds11", "ds17", "pooler")
REPRESENTATIONS = (
    "ds5", "ds11", "ds17", "pooler", "raw5", "raw11", "raw17", "raw23", "concat",
)


def macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Mean F1 over the classes present in `truth`."""
    scores = []
    for c in np.unique(truth):
        tp = float(np.sum((predicted == c) & (truth == c)))
        fp = float(np.sum((predicted == c) & (truth != c)))
        fn = float(np.sum((predicted != c) & (truth == c)))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else float("nan")


def paired_page_bootstrap(
    truth: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    pages: np.ndarray,
    resamples: int = BOOTSTRAP,
    seed: int = 0,
) -> tuple[float, float]:
    """(macro-F1(a) - macro-F1(b), sd of that difference over page resamples)."""
    diff = macro_f1(truth, pred_a) - macro_f1(truth, pred_b)
    unique = np.unique(pages)
    rows_of = {p: np.flatnonzero(pages == p) for p in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        rows = np.concatenate([rows_of[p] for p in rng.choice(unique, len(unique))])
        draws.append(macro_f1(truth[rows], pred_a[rows]) - macro_f1(truth[rows], pred_b[rows]))
    return diff, float(np.std(draws, ddof=1))


# ------------------------------------------------------------------ data --


class Split:
    """Kept tokens of one split: labels, page ids, per-level features."""

    def __init__(self, out: Path, split: str) -> None:
        pages = json.loads((out / "pages.json").read_text())[split]
        from unflatten.probe.run_vit_levels import page_key

        self.keys = [
            page_key(p) for p in pages
            if (out / "features" / split / f"{page_key(p)}.npz").exists()
        ]
        self.out, self.split = out, split
        labels = {s: [] for s in SIGNALS}
        page_ids = []
        for number, key in enumerate(self.keys):
            z = np.load(out / "labels" / split / f"{key}.npz")
            for s in SIGNALS:
                labels[s].append(z[s].astype(np.int64))
            page_ids.append(np.full(len(z["kept"]), number))
        self.labels = {s: np.concatenate(v) for s, v in labels.items()}
        self.pages = np.concatenate(page_ids)

    def features(self, representation: str) -> np.ndarray:
        names = CONCAT if representation == "concat" else (representation,)
        parts = []
        for key in self.keys:
            z = np.load(self.out / "features" / self.split / f"{key}.npz")
            parts.append(np.hstack([z[n] for n in names]))
        return np.concatenate(parts)


# ---------------------------------------------------------------- probes --


def _batches(n: int, generator: torch.Generator | None) -> list[torch.Tensor]:
    order = torch.randperm(n, generator=generator) if generator is not None else torch.arange(n)
    return list(order.split(BATCH))


def _predict(model: torch.nn.Module, x: np.ndarray, mean, std, device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for rows in _batches(len(x), None):
            xb = (torch.from_numpy(x[rows.numpy()]).to(device).float() - mean) / std
            out.append(model(xb).argmax(1).cpu())
    return torch.cat(out).numpy()


def fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    n_classes: int,
    *,
    kind: str,
    lr: float,
    seed: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: str = "cpu",
) -> tuple[torch.nn.Module, float]:
    """Train with early stopping on dev macro-F1; returns (best model, dev F1)."""
    torch.manual_seed(seed)
    dim = x_train.shape[1]
    if kind == "linear":
        model = torch.nn.Linear(dim, n_classes)
    elif kind == "mlp":
        model = torch.nn.Sequential(
            torch.nn.Linear(dim, MLP_HIDDEN), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(MLP_HIDDEN, n_classes),
        )
    else:
        raise ValueError(f"unknown probe kind {kind!r}")
    model = model.to(device)
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / n_classes, 0.0)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(np.minimum(weights, 50.0), dtype=torch.float32, device=device)
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    y_t = torch.from_numpy(y_train)
    best_f1, best_state, best_epoch = -1.0, None, 0
    for epoch in range(EPOCHS):
        model.train()
        for rows in _batches(len(x_train), generator):
            xb = (torch.from_numpy(x_train[rows.numpy()]).to(device).float() - mean) / std
            loss = loss_fn(model(xb), y_t[rows].to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        f1 = macro_f1(y_dev, _predict(model, x_dev, mean, std, device))
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    # best at the last epoch means the probe may still be under-trained
    model.best_epoch = best_epoch
    return model, best_f1


def _subsample(rows: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(rows, cap, replace=False)) if len(rows) > cap else rows


def probe_signal(
    signal: str,
    x_train: np.ndarray,
    train: Split,
    x_test: np.ndarray,
    test: Split,
    dev_pages: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: str,
) -> dict[str, Any]:
    n_classes = N_CLASSES[signal]
    y_all, y_test_all = train.labels[signal], test.labels[signal]
    labelled = np.flatnonzero(y_all >= 0)
    in_dev = np.isin(train.pages[labelled], dev_pages)
    fit_rows_all, dev_rows_all = labelled[~in_dev], labelled[in_dev]
    test_rows = np.flatnonzero(y_test_all >= 0)
    y_test = y_test_all[test_rows]
    x_te = x_test[test_rows]
    result: dict[str, Any] = {
        "train_rows": int(len(fit_rows_all)), "test_rows": int(len(test_rows)),
        "test_pages": int(len(np.unique(test.pages[test_rows]))),
        "test_classes": np.bincount(y_test, minlength=n_classes).tolist(),
        "linear": [], "mlp": [], "control": [], "linear_lr": [], "best_epoch": [],
        "predictions": {},
    }
    majority = np.bincount(y_all[fit_rows_all], minlength=n_classes).argmax()
    result["majority"] = macro_f1(y_test, np.full_like(y_test, majority))
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        fit_rows = _subsample(fit_rows_all, TRAIN_CAP, rng)
        dev_rows = _subsample(dev_rows_all, DEV_CAP, rng)
        x_fit, y_fit = x_train[fit_rows], y_all[fit_rows]
        x_dev, y_dev = x_train[dev_rows], y_all[dev_rows]
        best = None
        for lr in LINEAR_LRS:
            model, dev_f1 = fit_probe(x_fit, y_fit, x_dev, y_dev, n_classes, kind="linear",
                                      lr=lr, seed=seed, mean=mean, std=std, device=device)
            if best is None or dev_f1 > best[1]:
                best = (model, dev_f1, lr)
        pred = _predict(best[0], x_te, mean, std, device)
        result["linear"].append(macro_f1(y_test, pred))
        result["linear_lr"].append(best[2])
        result["best_epoch"].append(best[0].best_epoch)
        result["predictions"][f"linear{seed}"] = pred.astype(np.int8)
        model, _ = fit_probe(x_fit, y_fit, x_dev, y_dev, n_classes, kind="mlp",
                             lr=MLP_LR, seed=seed, mean=mean, std=std, device=device)
        result["best_epoch"].append(model.best_epoch)
        pred = _predict(model, x_te, mean, std, device)
        result["mlp"].append(macro_f1(y_test, pred))
        result["predictions"][f"mlp{seed}"] = pred.astype(np.int8)
        model, _ = fit_probe(x_fit, rng.permutation(y_fit), x_dev, rng.permutation(y_dev),
                             n_classes, kind="linear", lr=LINEAR_LRS[0], seed=seed,
                             mean=mean, std=std, device=device)
        result["control"].append(macro_f1(y_test, _predict(model, x_te, mean, std, device)))
    return result


def _column_stats(
    x: np.ndarray, rows: np.ndarray | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Column mean and sd over `rows` of `x` (all rows by default)."""
    rows = np.arange(len(x)) if rows is None else rows
    total = np.zeros(x.shape[1])
    square = np.zeros(x.shape[1])
    for start in range(0, len(rows), 20_000):
        chunk = x[rows[start : start + 20_000]].astype(np.float64)
        total += chunk.sum(0)
        square += (chunk**2).sum(0)
    mean = total / len(rows)
    std = np.sqrt(np.maximum(square / len(rows) - mean**2, 0.0))
    std = np.where(std > 1e-6, std, 1.0)
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


#: (candidates, reference): each comparison holds width and merger fixed.
#: DeepStack features vs pooler are all 2560-d merger outputs; raw block
#: outputs vs raw23 (the last block, pooler's own input) are all 4096-d.
COMPARISONS = {
    "deepstack_vs_pooler": (("ds5", "ds11", "ds17"), "pooler"),
    "raw_vs_raw23": (("raw5", "raw11", "raw17"), "raw23"),
}


def level_gap(
    results: dict[str, Any],
    test: Split,
    signal: str,
    kind: str,
    candidates: tuple[str, ...],
    reference: str,
) -> dict[str, Any]:
    """Best candidate level vs the reference: mean diff over seeds, the seed
    sd of the per-seed diff, and the paired page-bootstrap sd."""
    means = {rep: float(np.mean(results[rep][signal][kind])) for rep in candidates}
    best = max(means, key=means.get)
    truth = test.labels[signal]
    rows = np.flatnonzero(truth >= 0)
    diffs, sds = [], []
    for seed in SEEDS:
        a = results[best][signal]["predictions"][f"{kind}{seed}"]
        b = results[reference][signal]["predictions"][f"{kind}{seed}"]
        diff, sd = paired_page_bootstrap(truth[rows], a, b, test.pages[rows], seed=seed)
        diffs.append(diff)
        sds.append(sd)
    per_seed = np.array(results[best][signal][kind]) - np.array(results[reference][signal][kind])
    return {
        "best_level": best, "reference": reference, "gap": float(np.mean(diffs)),
        "seed_sd": float(np.std(per_seed, ddof=1)), "page_bootstrap_sd": float(np.mean(sds)),
    }


def run_probes(out: Path, threads: int = 16, device: str = "cpu") -> dict[str, Any]:
    torch.set_num_threads(threads)
    train, test = Split(out, "train"), Split(out, "test")
    rng = np.random.default_rng(12345)
    n_pages = len(train.keys)
    dev_pages = rng.choice(n_pages, max(1, round(n_pages * DEV_PAGE_FRACTION)), replace=False)
    results_path = out / "probe-results.json"
    results: dict[str, Any] = {}
    (out / "probe").mkdir(exist_ok=True)
    for rep in REPRESENTATIONS:
        done_path = out / "probe" / f"{rep}.pkl"
        if done_path.exists():  # resumable per representation
            results[rep] = pickle.loads(done_path.read_bytes())
            continue
        started = time.time()
        x_train, x_test = train.features(rep), test.features(rep)
        mean, std = _column_stats(x_train, np.flatnonzero(~np.isin(train.pages, dev_pages)))
        mean, std = mean.to(device), std.to(device)
        results[rep] = {}
        for signal in SIGNALS:
            results[rep][signal] = probe_signal(
                signal, x_train, train, x_test, test, dev_pages, mean, std, device
            )
            r = results[rep][signal]
            print(f"[probe] {rep:7s} {signal:14s} linear {np.mean(r['linear']):.3f}"
                  f"±{np.std(r['linear'], ddof=1):.3f} mlp {np.mean(r['mlp']):.3f}"
                  f"±{np.std(r['mlp'], ddof=1):.3f} ctrl {np.mean(r['control']):.3f}"
                  f" maj {r['majority']:.3f}", flush=True)
        del x_train, x_test
        done_path.write_bytes(pickle.dumps(results[rep]))
        print(f"[probe] {rep} done in {time.time() - started:.0f}s", flush=True)
    summary: dict[str, Any] = {"representations": {}, "gaps": {}}
    for rep, per_signal in results.items():
        summary["representations"][rep] = {
            s: {k: v for k, v in r.items() if k != "predictions"} for s, r in per_signal.items()
        }
    for signal in SIGNALS:
        summary["gaps"][signal] = {
            f"{name}/{kind}": level_gap(results, test, signal, kind, candidates, reference)
            for name, (candidates, reference) in COMPARISONS.items()
            for kind in ("linear", "mlp")
        }
    summary["train_pages"], summary["test_pages"] = len(train.keys), len(test.keys)
    summary["dev_pages"] = int(len(dev_pages))
    results_path.write_text(json.dumps(summary, indent=1))
    return summary


# ---------------------------------------------------------------- report --


def _cell(values: list[float]) -> str:
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"


def report_tables(summary: dict[str, Any]) -> str:
    """Markdown tables: per probe kind, signal x representation macro-F1."""
    reps = [r for r in REPRESENTATIONS if r in summary["representations"]]
    lines: list[str] = []
    for kind in ("linear", "mlp"):
        lines += [f"### {kind} probe, test macro-F1 (mean ± sd over {len(SEEDS)} seeds)", ""]
        lines.append("| signal | " + " | ".join(reps) + " | control | majority |")
        lines.append("|---|" + "---:|" * (len(reps) + 2))
        for signal in SIGNALS:
            row = [_cell(summary["representations"][r][signal][kind]) for r in reps]
            ref = summary["representations"]["pooler"][signal]
            lines.append(
                f"| {signal} | " + " | ".join(row)
                + f" | {_cell(ref['control'])} | {ref['majority']:.3f} |"
            )
        lines.append("")
    lines += ["### Best level minus reference (same width, same merger)", ""]
    lines.append("| signal | comparison | probe | best | gap | seed sd | page-bootstrap sd "
                 "| gap / page sd |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for signal in SIGNALS:
        for key, gap in summary["gaps"][signal].items():
            name, kind = key.split("/")
            sd = gap["page_bootstrap_sd"]
            ratio = gap["gap"] / sd if sd else float("nan")
            lines.append(
                f"| {signal} | {name} | {kind} | {gap['best_level']} vs {gap['reference']} "
                f"| {gap['gap']:+.3f} | {gap['seed_sd']:.3f} | {gap['page_bootstrap_sd']:.3f} "
                f"| {ratio:+.1f} |"
            )
    lines.append("")
    lines += ["### Rows per signal (pooler run)", ""]
    lines.append("| signal | train rows | test rows | test pages | test class counts |")
    lines.append("|---|---:|---:|---:|---|")
    for signal in SIGNALS:
        r = summary["representations"]["pooler"][signal]
        lines.append(f"| {signal} | {r['train_rows']} | {r['test_rows']} | {r['test_pages']} "
                     f"| {r['test_classes']} |")
    return "\n".join(lines) + "\n"
