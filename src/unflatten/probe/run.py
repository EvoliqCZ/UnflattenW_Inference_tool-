"""Stage-0 backbone probe.

Fits linear heads on frozen backbone features and asks, per signal,
whether they beat a constant baseline. This is the gate that decides
whether a backbone is worth building on at all: if paragraph structure,
table presence, run style and paragraph spacing are not linearly
readable from the features, no adapter or decoder stacked on top will
conjure them, and the response is to move the feature tap, unfreeze
partially, or swap the backbone.

Feature extraction dominates the cost, so features are cached to disk.
Refitting heads is then seconds, which is what makes it practical to
iterate on the label set. The cache carries an identity — release,
backbone revision, processor, pixel budget, label derivation — and a
read that does not match it is refused: the same flags over a new
release, or after the label semantics changed, would otherwise reuse the
old numbers and make the change invisible.

The verdict is deliberately coarse — PASS when a majority of scored
signals beat their baseline — because the probe's job is a go/no-go, not
a ranking. Ranking backbones against each other is RFC005's ablation,
scored on product metrics. Coarse is not the same as thin, though: a
verdict over one scored signal and seven skipped ones is no evidence
about the backbone, so fewer than `min_scored` scored signals is
INCONCLUSIVE, never PASS.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from unflatten.probe.heads import (
    ALPHAS,
    accuracy,
    cross_validate_alpha,
    fold_plan,
    kernel_ridge_predict,
    majority_accuracy,
    permutation_null,
    r2_score,
    roc_auc,
    roc_auc_noise_sd,
    spearman,
    spearman_noise_sd,
    standardize,
    take_labelled,
)

#: Labels scored as binary classification.
BINARY_LABELS: tuple[str, ...] = ("has_table", "has_list", "has_image")

#: Labels scored as regression.
REGRESSION_LABELS: tuple[str, ...] = (
    "paragraph_count",
    "run_count",
    "bold_run_ratio",
    "modal_font_size_mpt",
    "modal_space_after_mpt",
)

#: A label needs at least this many labelled rows on each side to score.
MIN_EXAMPLES = 24

#: Heavy-tailed count labels: a linear head on raw counts extrapolates
#: badly on the long tail and R² collapses while the ordering stays
#: informative. Fitted on log1p, scored on both R² and rank correlation.
COUNT_LABELS: frozenset[str] = frozenset({"paragraph_count", "run_count"})

#: How many standard deviations a metric must clear its null value to
#: count as signal. A fixed margin is the wrong instrument: 0.02 above
#: chance AUROC is 0.6 sigma at n=1000 with 9 % positives — noise — while
#: the same margin at n=100000 would be overwhelming. Thresholds are
#: therefore derived from the evaluated sample size. Three sigma over
#: eight labels leaves an expected false-positive count near 0.01.
SIGNAL_SIGMA = 3.0

#: Minimum R² to call a regression head calibrated, independent of rank.
MIN_R2 = 0.02

#: How many signals must actually be scored for a verdict to mean
#: anything. Half of the eight declared labels: a go/no-go decided by a
#: single surviving head is a coin flip with a citation.
MIN_SCORED_LABELS = 4

#: AUROC of a coin flip. Binary labels here are heavily imbalanced (91 %
#: of pages have no table), which makes accuracy-against-the-majority a
#: bad test: a head can rank almost perfectly and still lose on accuracy
#: because the 0.5 threshold is uncalibrated. AUROC is threshold-free.
CHANCE_AUROC = 0.5

#: The penalty is CHOSEN PER LABEL by cross-validation on the train
#: split (`heads.ALPHAS`, `heads.cross_validate_alpha`), not fixed.
#: A single fixed value measures the value, not the backbone: at 10 the
#: spacing head (then `mean_space_after_mpt`) scored R² -1.122 and rho -0.008 — read as
#: "no signal" — while the SAME features under a cross-validated penalty
#: score +0.225 and +0.528. `run_count` moved 0.11 -> 0.60 the same way.
#: Those numbers were about to be quoted into RFC004 as evidence of what
#: the frozen backbone carries.
#:
#: Selecting per label is itself a search, so the baseline is re-measured
#: under the same search on shuffled labels (`heads.permutation_null`)
#: rather than assumed to be unmoved: a head only counts as signal if it
#: beats what the selection can extract from noise.


@dataclass(frozen=True)
class ProbeOutcome:
    """One signal's probe result."""

    label: str
    kind: str
    n_train: int
    n_eval: int
    #: R² for regression, accuracy for binary. None when skipped.
    score: float | None
    #: What a constant predictor achieves: 0.0 (R²) or the majority rate.
    baseline: float | None
    auroc: float | None
    #: Rank correlation between target and prediction.
    spearman: float | None
    beats_baseline: bool | None
    skipped_reason: str | None
    #: Penalty cross-validated on train for THIS label. A per-label
    #: hyperparameter is a result, not an implementation detail.
    alpha: float | None = None
    #: Best the same procedure reaches on shuffled labels: the floor this
    #: label's score had to clear on top of the constant baseline.
    null_score: float | None = None
    null_spearman: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_label(
    label: str,
    train_features: torch.Tensor,
    train_labels: Sequence[Any],
    eval_features: torch.Tensor,
    eval_labels: Sequence[Any],
    *,
    alphas: Sequence[float] = ALPHAS,
    min_examples: int = MIN_EXAMPLES,
) -> ProbeOutcome:
    """Fit a linear head for `label` and score it against a constant.

    The penalty is cross-validated on the train split alone; eval is
    never consulted for the choice. The constant baseline is joined by a
    permutation floor, so a label cannot pass on an improvement the
    selection would have produced from noise.
    """
    kind = "binary" if label in BINARY_LABELS else "regression"
    x_train, y_train = take_labelled(train_features, train_labels)
    x_eval, y_eval = take_labelled(eval_features, eval_labels)
    if label in COUNT_LABELS:
        y_train, y_eval = torch.log1p(y_train), torch.log1p(y_eval)
    if min(x_train.shape[0], x_eval.shape[0]) < min_examples:
        reason = (
            "no_labelled_examples"
            if min(x_train.shape[0], x_eval.shape[0]) == 0
            else "too_few_labelled_examples"
        )
        return ProbeOutcome(label, kind, x_train.shape[0], x_eval.shape[0],
                            None, None, None, None, None, reason)

    if kind == "binary" and (
        int(y_train.sum()) in (0, y_train.numel())
        or int(y_eval.sum()) in (0, y_eval.numel())
    ):
        # One class on either side is not a hard signal, it is no
        # measurement: AUROC is undefined, and accuracy against a
        # constant target is 1.0 for a head that learned nothing. Scoring
        # it would let a split with no tables cast a FAIL vote and count
        # toward `MIN_SCORED_LABELS`.
        return ProbeOutcome(label, kind, x_train.shape[0], x_eval.shape[0],
                            None, None, None, None, None, "single_class_target")

    # Statistics come from train only; using eval rows would leak.
    z_train, mean, std = standardize(x_train)
    z_eval, _, _ = standardize(x_eval, mean=mean, std=std)
    # Centred columns, so the kernel form is the same estimator
    # `fit_ridge` is (`kernel_ridge_predict`). It is used because one
    # Gram matrix then serves every penalty, every fold and every
    # permutation — the search below is otherwise unaffordable at 8704
    # columns.
    gram = z_train @ z_train.T
    cross = z_eval @ z_train.T
    # one set of fold factorisations for the selection AND the null: they
    # cross-validate the same folds over the same penalties, and factoring
    # them twice is the whole cost of the step
    plan = fold_plan(gram, alphas=alphas)
    alpha, _ = cross_validate_alpha(gram, y_train, kind, alphas=alphas, plan=plan)
    scores = kernel_ridge_predict(gram, y_train, cross, alpha=alpha)
    null = permutation_null(
        gram, y_train, cross, y_eval, kind, alphas=alphas, plan=plan
    )

    rho = spearman(y_eval, scores)
    rank = None if rho != rho else rho
    if kind == "binary":
        # Reported score stays accuracy (readable), but the VERDICT is
        # AUROC: threshold-free, and these classes are heavily imbalanced.
        value = accuracy(y_eval, scores)
        baseline = majority_accuracy(y_eval)
        area = roc_auc(y_eval, scores)
        auroc = None if area != area else area  # NaN -> single class
        positives = int(y_eval.sum())
        threshold = CHANCE_AUROC + SIGNAL_SIGMA * roc_auc_noise_sd(
            positives, int(y_eval.numel()) - positives
        )
        if null["auroc"] is not None:
            threshold = max(threshold, null["auroc"])
        beats = auroc is not None and auroc > threshold
    else:
        value = r2_score(y_eval, scores)
        baseline = 0.0
        auroc = None
        # A head can rank a heavy-tailed target well and still miss its
        # scale; either is evidence the information is present.
        rank_threshold = SIGNAL_SIGMA * spearman_noise_sd(int(y_eval.numel()))
        r2_threshold = MIN_R2
        if null["r2"] is not None:
            r2_threshold = max(r2_threshold, null["r2"])
        if null["spearman"] is not None:
            rank_threshold = max(rank_threshold, null["spearman"])
        # POSITIVE rank only: a head that orders the target backwards has
        # not read the property, whatever |rho| says.
        beats = value > r2_threshold or (
            rank is not None and rank > rank_threshold
        )
    if value != value:  # constant target -> R² undefined
        return ProbeOutcome(label, kind, x_train.shape[0], x_eval.shape[0],
                            None, None, None, None, None, "constant_target")
    return ProbeOutcome(label, kind, x_train.shape[0], x_eval.shape[0],
                        value, baseline, auroc, rank, beats, None,
                        alpha, null["auroc"] if kind == "binary" else null["r2"],
                        null["spearman"])


def summarize(
    outcomes: Sequence[ProbeOutcome], *, min_scored: int = MIN_SCORED_LABELS
) -> dict[str, Any]:
    """Aggregate per-signal outcomes into a go/no-go verdict.

    Fewer than `min_scored` scored signals is INCONCLUSIVE: the probe
    reports that it could not decide, rather than passing a backbone on
    the strength of whichever head happened to have labels.
    """
    scored = [o for o in outcomes if o.skipped_reason is None]
    beat = [o for o in scored if o.beats_baseline]
    if len(scored) < min_scored:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS" if len(beat) * 2 > len(scored) else "FAIL"
    return {
        "verdict": verdict,
        "scored": len(scored),
        "skipped": len(outcomes) - len(scored),
        "beat_baseline": len(beat),
        "labels": [o.as_dict() for o in outcomes],
    }


class ProbeCacheIdentityError(RuntimeError):
    """A cached feature file cannot prove it is what the run asked for."""


@dataclass(frozen=True)
class ProbeCacheIdentity:
    """What a probe feature cache's contents are bound to.

    The cache path keys only the flags a human typed. Everything that
    actually determines the payload — which release was read, which
    weights the tower resolved to, which label semantics ran — is
    invisible in a filename, so re-running the same command after a new
    release or a change in label derivation silently reused the old
    numbers. That is how a supervision fix becomes invisible.

    `labels_version` is the field `model.feature_cache.CacheIdentity`
    has no analogue for: there the defect to catch is different features,
    here it is the same features carrying differently derived labels.

    `backbone_revision` is best-effort, exactly as in that module: with
    no loader passing `revision=`, `revision_source == "unavailable"`
    means the cache is pinned to a NAME and not to bytes.
    """

    backbone: str | None = None
    backbone_revision: str | None = None
    revision_source: str = "unavailable"
    #: Processor config digest, not a name: the pixel budget is applied
    #: to the processor, so its config is part of what produced the rows.
    processor: str | None = None
    #: Which feature tap the rows came from, for a backbone that exposes
    #: more than one. Two taps of one checkpoint are two different feature
    #: sets under the same model id, so a cache built at one must not
    #: satisfy a run that asked for the other.
    feature_tap: str | None = None
    max_pixels: int | None = None
    task_kind: str | None = None
    split: str | None = None
    release_root_hash: str | None = None
    #: `labels.LABELS_VERSION` — the derivation, not the label names.
    labels_version: str | None = None
    #: Admission boundary the extraction ran under: a permissive cache
    #: must not satisfy a run that asked for the strict one.
    require_privacy_review: bool = False
    #: `--train-pages` / `--eval-pages` for THIS split's cache. A cache
    #: built under a smaller budget holds fewer rows than the run asked
    #: for, and nothing in the numbers says so: the sample size would be
    #: reported as the one requested. Per split, because a change to the
    #: train budget is no reason to re-extract the eval split.
    page_limit: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def differences(self, recorded: dict[str, Any]) -> list[str]:
        """Fields where `recorded` disagrees with this identity."""
        return [
            f"{name}: cache has {recorded.get(name)!r}, run wants {value!r}"
            for name, value in self.as_dict().items()
            if recorded.get(name) != value
        ]


def probe_cache_identity(
    backbone: Any,
    release_dir: Path | str,
    *,
    split: str,
    task_kind: str,
    max_pixels: int | None,
    page_limit: int | None = None,
    require_privacy_review: bool = False,
) -> ProbeCacheIdentity:
    """Build the identity a probe run's cache must match."""
    import hashlib

    from unflatten.model.build_cache import resolve_backbone_revision
    from unflatten.probe.labels import LABELS_VERSION

    try:
        manifest = json.loads(
            (Path(release_dir) / "release-manifest.json").read_text()
        )
        release_root_hash = manifest.get("release_root_hash")
    except (OSError, json.JSONDecodeError):
        # An unreadable manifest is recorded as absent rather than
        # guessed: a None here compares unequal to a real hash, so it
        # refuses a cache built against a named release.
        release_root_hash = None

    processor = getattr(backbone, "processor", None)
    digest = None
    if processor is not None and hasattr(processor, "to_json_string"):
        digest = hashlib.sha256(
            processor.to_json_string().encode("utf-8")
        ).hexdigest()[:16]

    model_id = getattr(backbone, "model_id", None)
    revision, source = (
        resolve_backbone_revision(model_id) if model_id else (None, "unavailable")
    )
    return ProbeCacheIdentity(
        backbone=model_id,
        backbone_revision=revision,
        revision_source=source,
        processor=digest,
        feature_tap=getattr(backbone, "tap", None),
        max_pixels=max_pixels,
        task_kind=task_kind,
        split=split,
        release_root_hash=release_root_hash,
        labels_version=LABELS_VERSION,
        require_privacy_review=require_privacy_review,
        page_limit=page_limit,
    )


def _require_an_identifiable_release(
    identity: ProbeCacheIdentity, cache_path: Path
) -> None:
    """Refuse to read or write a cache under an unidentifiable release.

    `None` compares equal to `None`, so an identity whose
    `release_root_hash` could not be read would match any other run whose
    manifest was equally unreadable — two absences taken for agreement,
    which is the shape of every defect this module has had. An identity
    that identifies nothing must not authorise reuse.
    """
    if identity.release_root_hash is None:
        raise ProbeCacheIdentityError(
            f"{cache_path}: this run cannot name its release "
            "(release-manifest.json is missing or unreadable), so it can "
            "neither prove a cache belongs to it nor write one that could; "
            "run without --cache-dir, or point it at a readable release"
        )


def _check_cache_identity(
    blob: dict[str, Any], identity: ProbeCacheIdentity, cache_path: Path
) -> None:
    """Refuse a cache that cannot prove it matches `identity`."""
    recorded = blob.get("identity")
    if not isinstance(recorded, dict):
        raise ProbeCacheIdentityError(
            f"{cache_path}: no recorded identity, so it cannot say which "
            "release, backbone or label derivation produced it; delete it "
            "and re-extract"
        )
    differences = identity.differences(recorded)
    if differences:
        joined = "; ".join(differences)
        raise ProbeCacheIdentityError(
            f"{cache_path}: cached features were produced under a different "
            f"identity ({joined})"
        )


# ------------------------------------------------------------------ harness --


def extract_split_features(
    backbone: Any,
    release_dir: Path | str,
    store_dir: Path | str,
    split: str,
    task_kind: str,
    *,
    limit: int,
    cache_path: Path | None = None,
    identity: ProbeCacheIdentity | None = None,
    progress_every: int = 100,
    require_privacy_review: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Pooled features + probe labels for up to `limit` pages of a split.

    Extraction is the expensive step; `cache_path` stores the result so
    heads can be refitted without touching the GPU again. `identity` is
    what that cache is bound to: given one, a stored file must carry a
    matching identity or the read is refused, naming what differs. Omit
    it only where the caller makes no claim about lineage.

    `require_privacy_review` is the RFC005 §9.2 training admission
    boundary. It defaults to False because this probe is a diagnostic
    and fits closed-form heads rather than gradients, but a caller that
    wants the strict boundary must be able to demand it — the gap this
    closes is that it previously could not ask at all.
    """
    import io

    from PIL import Image

    from unflatten.dataset.items import build_item
    from unflatten.dataset.samples import iter_page_samples
    from unflatten.probe.labels import extract_labels

    if cache_path is not None and identity is not None:
        _require_an_identifiable_release(identity, Path(cache_path))
    if cache_path is not None and Path(cache_path).exists():
        # tensors + plain-python label dicts: nothing here needs the
        # arbitrary-unpickling escape hatch, and --cache-dir is caller-supplied
        blob = torch.load(cache_path, weights_only=True)
        if identity is not None:
            _check_cache_identity(blob, identity, Path(cache_path))
        return blob["features"], blob["labels"]

    store_dir = Path(store_dir)
    rows: list[torch.Tensor] = []
    labels: list[dict[str, Any]] = []
    skipped_unlabelled = 0
    pack_cache: dict[str, Any] = {}
    for index, sample in enumerate(
        iter_page_samples(
            release_dir,
            store_dir,
            split,
            task_kind,
            require_privacy_review=require_privacy_review,
        )
    ):
        if len(rows) >= limit:
            break
        item = build_item(store_dir, sample)
        if not item.fragment:
            # no fragment: the page has no labels at all. Labelling it
            # from an empty dict would report "no table, no paragraphs".
            skipped_unlabelled += 1
            continue
        image = Image.open(io.BytesIO(item.image_bytes))
        image.load()
        features = backbone.extract(image)
        rows.append(features.pooled().cpu())
        labels.append(
            extract_labels(
                item.fragment, _supervised_targets(item, store_dir, pack_cache)
            ).as_dict()
        )
        if progress_every and (index + 1) % progress_every == 0:
            print(f"[probe] {len(rows)}/{limit} strán", flush=True)

    if skipped_unlabelled:
        print(
            f"[probe] {skipped_unlabelled} strán bez fragmentu preskočených "
            "(bez labelov)",
            flush=True,
        )
    stacked = torch.stack(rows) if rows else torch.empty(0)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"features": stacked, "labels": labels}
        if identity is not None:
            payload["identity"] = identity.as_dict()
        torch.save(payload, cache_path)
    return stacked, labels


def _render_font_pack_hash(store_dir: Path, sample: Any, cache: dict) -> str | None:
    """Font pack the page's PIXELS were rendered under, from the sidecar.

    Cached per document: pages of one document share one sidecar, and
    the sidecar is large, so it is read once per bundle rather than once
    per page.
    """
    key = sample.document_bundle_id
    if key not in cache:
        import json as _json

        from unflatten.dataset.gates.store import read_object

        ref = sample.observation_sidecar
        sidecar = _json.loads(read_object(store_dir, ref)) if ref is not None else None
        cache[key] = ((sidecar or {}).get("renderer") or {}).get("font_pack_hash")
    return cache[key]


def _supervised_targets(
    item: Any, store_dir: Path, pack_cache: dict
) -> dict[str, Any]:
    """Targets with both §29 withdrawals applied, in the load-bearing
    order: the font projection first (it re-derives `supervision_mask`
    for the pointer it assesses), the placement withdrawal last. Both in
    one `apply_masks` pass — one deep copy and one `target_bundle_hash`
    instead of two, and the hash is what a page-sized bundle pays for.

    The projection is only meaningful against the render it was derived
    for, so the sidecar's recorded `font_pack_hash` — the pixels this
    probe is actually holding — is handed to `apply_masks`, whose font
    projection step refuses a projection from any other pack (and refuses
    a render it cannot name). The refusal lives there, not here, so no
    future call site can forget it.
    """
    from unflatten.dataset.targets.masks import apply_masks

    targets = item.targets or {}
    if not targets:
        return {}
    masked, _flipped = apply_masks(
        targets,
        item.mask_projection,
        [item.fragment],
        render_font_pack_hash=_render_font_pack_hash(store_dir, item.sample, pack_cache)
        if item.mask_projection is not None
        else None,
    )
    return masked


def run_probe(
    train_features: torch.Tensor,
    train_labels: Sequence[dict[str, Any]],
    eval_features: torch.Tensor,
    eval_labels: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Score every declared signal and return the summary."""
    outcomes = [
        score_label(
            name,
            train_features,
            [row.get(name) for row in train_labels],
            eval_features,
            [row.get(name) for row in eval_labels],
        )
        for name in (*BINARY_LABELS, *REGRESSION_LABELS)
    ]
    return summarize(outcomes)


def report(summary: dict[str, Any]) -> str:
    """Human-readable probe report."""
    lines = [
        f"{'signal':22}{'kind':12}{'n_train':>8}{'n_eval':>7}"
        f"{'score':>9}{'baseline':>10}{'null':>8}{'auroc':>8}{'rank':>8}"
        f"{'alpha':>9}  verdict",
        "-" * 109,
    ]
    for row in summary["labels"]:
        if row["skipped_reason"]:
            lines.append(
                f"{row['label']:22}{row['kind']:12}{row['n_train']:>8}"
                f"{row['n_eval']:>7}{'-':>9}{'-':>10}{'-':>8}{'-':>8}{'-':>8}"
                f"{'-':>9}  skipped ({row['skipped_reason']})"
            )
            continue
        auroc = "-" if row["auroc"] is None else f"{row['auroc']:.3f}"
        rank = "-" if row["spearman"] is None else f"{row['spearman']:.3f}"
        null = "-" if row["null_score"] is None else f"{row['null_score']:+.3f}"
        alpha = "-" if row["alpha"] is None else f"{row['alpha']:.0e}"
        lines.append(
            f"{row['label']:22}{row['kind']:12}{row['n_train']:>8}"
            f"{row['n_eval']:>7}{row['score']:>9.3f}{row['baseline']:>10.3f}"
            f"{null:>8}{auroc:>8}{rank:>8}{alpha:>9}"
            f"  {'signal' if row['beats_baseline'] else 'none'}"
        )
    lines.append("-" * 109)
    lines.append(
        f"{summary['beat_baseline']}/{summary['scored']} signálov nad baseline"
        f"  ({summary['skipped']} preskočených)  ->  {summary['verdict']}"
    )
    return "\n".join(lines)


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m unflatten.probe.run", description=__doc__
    )
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("store_dir", type=Path)
    parser.add_argument("--task-kind", default="structure_v0")
    parser.add_argument("--train-pages", type=int, default=1500)
    parser.add_argument("--eval-pages", type=int, default=500)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--backbone",
        default=None,
        help="model id; omit for the default",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tap",
        default=None,
        help="feature tap, for a backbone that exposes several "
             "(Granite Vision: 'siglip4' four raw SigLIP layers, "
             "'projected8' all eight projected streams); omit for a "
             "backbone with only one",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="cap pixels per page (one visual token per 32x32 px); "
             "downscales stored rasters, never upscales",
    )
    parser.add_argument(
        "--require-privacy-review",
        action="store_true",
        help="demand the RFC005 §9.2 training admission boundary: refuse a "
             "document whose privacy decision is not complete (off by "
             "default — this probe is a diagnostic, not a training run)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from unflatten.probe.features import DEFAULT_BACKBONE, FrozenVisionBackbone

    model_id = args.backbone or DEFAULT_BACKBONE
    backbone = FrozenVisionBackbone(
        model_id, device=args.device, max_pixels=args.max_pixels, tap=args.tap
    )
    tag = f" tap {backbone.tap}" if backbone.tap else ""
    print(
        f"[probe] backbone {model_id}{tag}, feature dim {backbone.feature_dim}",
        flush=True,
    )
    # the pixel budget decides the resolution every number below was
    # measured at, so the run says what it actually applied
    print(f"[probe] pixel budget: {backbone.pixel_budget_applied or 'native'}"
          f"  measured: {backbone.pixel_budget_measured or 'not capped'}",
          flush=True)

    def cache(name: str) -> Path | None:
        if args.cache_dir is None:
            return None
        tag = model_id.replace("/", "_")
        # the pixel budget changes the features, so it changes the cache key
        budget = args.max_pixels or "native"
        # the tap changes the features, so it changes the cache key too
        tap = f"-{backbone.tap}" if backbone.tap else ""
        return Path(args.cache_dir) / (
            f"{tag}{tap}-{args.task_kind}-{budget}-{name}.pt"
        )

    def identity(split: str, page_limit: int) -> ProbeCacheIdentity:
        return probe_cache_identity(
            backbone, args.release_dir, split=split, task_kind=args.task_kind,
            max_pixels=args.max_pixels, page_limit=page_limit,
            require_privacy_review=args.require_privacy_review,
        )

    train_x, train_y = extract_split_features(
        backbone, args.release_dir, args.store_dir, "train", args.task_kind,
        limit=args.train_pages, cache_path=cache("train"),
        identity=identity("train", args.train_pages),
        require_privacy_review=args.require_privacy_review,
    )
    eval_x, eval_y = extract_split_features(
        backbone, args.release_dir, args.store_dir, "val_iid", args.task_kind,
        limit=args.eval_pages, cache_path=cache("val"),
        identity=identity("val_iid", args.eval_pages),
        require_privacy_review=args.require_privacy_review,
    )
    summary = run_probe(train_x, train_y, eval_x, eval_y)
    summary["backbone"] = model_id
    summary["feature_tap"] = backbone.tap
    summary["task_kind"] = args.task_kind
    print(report(summary))
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
