"""ViT-level probe runner (discovery, 2026-09-29, evidence for RFC007).

Which Qwen3-VL ViT levels carry fine document signals that the final
`pooler_output` lacks? Frozen tower, 300 DPI renders, small probes only;
no model is trained.

Steps, each resumable and each writing under `--out`:

    select   train / test page list (train: release train split minus every
             bundle held out in ANY local release and the v15b selection
             cohort; test: val_iid minus that cohort and minus every lineage
             the train pages use)
    labels   per-page token labels + the token subset kept (CPU)
    extract  per-page features of the kept tokens for every level (GPU,
             under the shared gpu.lock, only when no foreign process holds
             more than 1 GB)
    probe    linear + 1-hidden-layer MLP probes, 3 seeds, shuffled-label
             control, per signal x level (CPU)

Pages come from the local store only; nothing under
`~/unflatten-runs/s5-control-sample` is read.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from unflatten.probe.vit_level_labels import (
    SIGNALS,
    grid_from_thw,
    index_ast,
    page_labels,
    parse_text_lines,
)

DATA_ROOT = Path.home() / "unflatten-data"
DEFAULT_RELEASE = DATA_ROOT / "release-v12"
DEFAULT_STORE = DATA_ROOT / "m2a-v7"
DEFAULT_OUT = Path.home() / "unflatten-runs" / "vit-level-probe-2026-09-29"
GPU_LOCK = Path.home() / "unflatten-runs" / "gpu.lock"
REPO_ROOT = Path(__file__).resolve().parents[3]
SELECTION_POLICY = REPO_ROOT / "src/unflatten/model/policies/s4_selection_policy.v15b.json"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
RAW_LAYERS = (5, 11, 17, 23)
HELD_OUT = ("val_iid.jsonl", "test.jsonl", "val_template_ood.jsonl")

#: Tokens kept per page, by stratum (random within a stratum, page-seeded).
#: Rare signals are over-represented on purpose; every level sees the same
#: tokens, so the comparison between levels is unaffected.
STRATA_CAPS = {
    "spacing": 300,
    "grid_line": 200,
    "table_no_line": 100,
    "bold_or_italic": 150,
    "text": 300,
    "blank": 100,
}


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


# ---------------------------------------------------------------- select --


def excluded_bundles(data_root: Path, policy_path: Path) -> set[str]:
    out: set[str] = set()
    for splits in sorted(data_root.glob("release-v*/splits")):
        for name in HELD_OUT:
            if (splits / name).exists():
                out.update(row["authority_id"] for row in _jsonl(splits / name))
        if (splits / "partial.jsonl").exists():
            for row in _jsonl(splits / "partial.jsonl"):
                if row.get("split_name") != "train":
                    out.add(row["authority_id"])
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    out.update(bundle for bundle, _page in policy["cohort"]["pinned_pages"])
    return out


def select_pages(
    release: Path, store: Path, n_train: int, n_test: int, seed: int
) -> dict[str, Any]:
    policy = json.loads(SELECTION_POLICY.read_text(encoding="utf-8"))
    cohort = {bundle for bundle, _page in policy["cohort"]["pinned_pages"]}
    excluded = excluded_bundles(DATA_ROOT, SELECTION_POLICY)
    splits = release / "splits"
    train_rows = [
        r for r in _jsonl(splits / "train.jsonl")
        if r["authority_id"] not in excluded and r.get("eligibility") == "train_ready"
    ]
    test_rows = [
        r for r in _jsonl(splits / "val_iid.jsonl")
        if r["authority_id"] not in cohort and r.get("eligibility") == "train_ready"
    ]
    wanted = {r["authority_id"] for r in train_rows + test_rows}
    renders: dict[str, dict[str, Any]] = {}
    for row in _jsonl(store / "renders.jsonl"):
        bundle = row.get("document_bundle_id")
        if bundle not in wanted or row.get("render_status") != "rendered":
            continue
        art = row.get("artifacts") or {}
        if (row.get("render_profile") or {}).get("render_dpi") != 300:
            continue
        if not (art.get("pages") and art.get("observation_sidecar") and art.get("pdf")):
            continue
        renders[bundle] = row

    index: dict[str, str] = {}
    for row in _jsonl(store / "index.jsonl"):
        env = row.get("envelope") or {}
        if env.get("document_bundle_id") in renders:
            index[env["document_bundle_id"]] = env["ast"]["ref"]

    rng = random.Random(seed)

    def pick(rows: list[dict[str, Any]], n: int, banned_lineages: set[str]) -> list[dict]:
        rows = sorted(
            (r for r in rows if r["authority_id"] in renders and r["authority_id"] in index
             and r["lineage_root_id"] not in banned_lineages),
            key=lambda r: r["authority_id"],
        )
        rng.shuffle(rows)
        chosen, lineages = [], set()
        for row in rows:
            if len(chosen) >= n:
                break
            if row["lineage_root_id"] in lineages:  # one page per lineage
                continue
            render = renders[row["authority_id"]]
            pages = render["artifacts"]["pages"]
            page_index = rng.randrange(len(pages))
            lineages.add(row["lineage_root_id"])
            chosen.append({
                "bundle": row["authority_id"],
                "lineage": row["lineage_root_id"],
                "page_index": page_index,
                "png": pages[page_index]["ref"],
                "sidecar": render["artifacts"]["observation_sidecar"]["ref"],
                "pdf": render["artifacts"]["pdf"]["ref"],
                "ast": index[row["authority_id"]],
            })
        return chosen

    train = pick(train_rows, n_train, set())
    test = pick(test_rows, n_test, {p["lineage"] for p in train})
    return {
        "release": str(release), "store": str(store), "seed": seed,
        "train": train, "test": test,
    }


# ---------------------------------------------------------------- labels --


def page_key(page: dict[str, Any]) -> str:
    return f"{page['bundle'][7:23]}_p{page['page_index']}"


def keep_tokens(labels: dict[str, np.ndarray], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    strata = {
        "spacing": (labels["space_before"] >= 0) | (labels["space_after"] >= 0),
        "grid_line": labels["grid_line"] == 1,
        "table_no_line": labels["grid_line"] == 0,
        "bold_or_italic": (labels["bold"] == 1) | (labels["italic"] == 1),
        "text": labels["text_blank"] == 1,
        "blank": labels["text_blank"] == 0,
    }
    kept: set[int] = set()
    for name, mask in strata.items():
        pool = np.flatnonzero(mask)
        cap = STRATA_CAPS[name]
        if len(pool) > cap:
            pool = rng.choice(pool, cap, replace=False)
        kept.update(int(i) for i in pool)
    return np.array(sorted(kept), dtype=np.int64)


def _processor():
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


def label_page(page: dict[str, Any], store: Path, processor: Any) -> dict[str, Any]:
    from PIL import Image

    image = Image.open(store / page["png"])
    image.load()
    rgb = image.convert("RGB")
    gray = np.asarray(image.convert("L"))
    grid_thw = processor(images=rgb, return_tensors="pt")["image_grid_thw"][0].tolist()
    grid = grid_from_thw(grid_thw, 2, rgb.width, rgb.height)
    sidecar = json.loads((store / page["sidecar"]).read_text(encoding="utf-8"))
    sc_page = next(
        (p for p in sidecar.get("pages") or [] if p.get("page_index") == page["page_index"]), None
    )
    observations = (sc_page or {}).get("observations") or []
    ast = index_ast(json.loads((store / page["ast"]).read_text(encoding="utf-8")))
    number = str(page["page_index"] + 1)
    proc = subprocess.run(
        ["pdftotext", "-bbox-layout", "-f", number, "-l", number, str(store / page["pdf"]), "-"],
        capture_output=True, timeout=120, check=True,
    )
    lines_per_page, heights = parse_text_lines(proc.stdout)
    if len(lines_per_page) != 1:
        raise ValueError(f"{page_key(page)}: pdftotext returned {len(lines_per_page)} pages")
    result = page_labels(grid, observations, ast, lines_per_page[0], heights[0], gray)
    return {
        "grid_thw": grid_thw,
        "grid": [grid.rows, grid.cols, grid.width_px, grid.height_px],
        "labels": result.labels,
        "table_checks": result.table_checks,
        "observations": len(observations),
    }


def run_labels(out: Path, store: Path, split: str, workers: int) -> None:
    from multiprocessing import Pool

    pages = json.loads((out / "pages.json").read_text())[split]
    todo = [p for p in pages if not (out / "labels" / split / f"{page_key(p)}.npz").exists()]
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    with Pool(workers, initializer=_init_worker, initargs=(str(store), str(out), split)) as pool:
        for done, key in enumerate(pool.imap_unordered(_label_worker, todo), 1):
            if done % 20 == 0:
                print(f"[labels {split}] {done}/{len(todo)} {key}", flush=True)


_WORKER: dict[str, Any] = {}


def _init_worker(store: str, out: str, split: str) -> None:
    _WORKER.update(store=Path(store), out=Path(out), split=split, processor=_processor())


def _label_worker(page: dict[str, Any]) -> str:
    key = page_key(page)
    try:
        info = label_page(page, _WORKER["store"], _WORKER["processor"])
    except (ValueError, subprocess.SubprocessError, OSError) as exc:
        (_WORKER["out"] / "labels" / _WORKER["split"] / f"{key}.error").write_text(str(exc))
        return f"{key} ERROR {exc}"
    kept = keep_tokens(info["labels"], seed=int(page["bundle"][7:15], 16))
    np.savez_compressed(
        _WORKER["out"] / "labels" / _WORKER["split"] / f"{key}.npz",
        kept=kept,
        grid_thw=np.array(info["grid_thw"]),
        grid=np.array(info["grid"]),
        **{name: info["labels"][name][kept].astype(np.int8) for name in SIGNALS},
        **{f"full_{name}": info["labels"][name].astype(np.int8) for name in SIGNALS},
    )
    (_WORKER["out"] / "labels" / _WORKER["split"] / f"{key}.json").write_text(
        json.dumps({"table_checks": info["table_checks"], "observations": info["observations"]})
    )
    return key


# --------------------------------------------------------------- extract --


def foreign_gpu_mib() -> dict[int, int]:
    """GPU memory per process other than this one, in MiB."""
    proc = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    usage: dict[int, int] = {}
    for line in proc.stdout.strip().splitlines():
        pid, mib = (int(v.strip()) for v in line.split(","))
        if pid != os.getpid():
            usage[pid] = mib
    return usage


class GpuSlot:
    """Hold gpu.lock only while no foreign process holds more than 1 GB."""

    def __init__(self, lock: Path, poll_s: int = 60) -> None:
        self.lock, self.poll_s = lock, poll_s

    def __enter__(self) -> GpuSlot:
        self.handle = self.lock.open("a")
        while True:
            fcntl.flock(self.handle, fcntl.LOCK_EX)
            busy = {p: m for p, m in foreign_gpu_mib().items() if m > 1024}
            if not busy:
                print(f"[gpu] lock held, foreign usage {foreign_gpu_mib()}", flush=True)
                return self
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            print(f"[gpu] busy {busy}, retry in {self.poll_s}s", flush=True)
            time.sleep(self.poll_s)

    def __exit__(self, *exc: object) -> None:
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()


def load_tower(device: str):
    import torch
    from transformers import AutoModelForImageTextToText

    from unflatten.probe.backbones import _drop_language_half, optimize_patch_embed

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    visual = model.model.visual
    _drop_language_half(model)
    visual = visual.to(device).eval()
    optimize_patch_embed(visual)
    return visual


def run_extract(out: Path, store: Path, splits: list[str], device: str) -> None:
    import torch
    from PIL import Image

    from unflatten.probe.vit_level_taps import tap_levels

    processor = _processor()
    todo: list[tuple[str, dict[str, Any]]] = []
    for split in splits:
        for page in json.loads((out / "pages.json").read_text())[split]:
            key = page_key(page)
            if (out / "labels" / split / f"{key}.npz").exists() and not (
                out / "features" / split / f"{key}.npz"
            ).exists():
                todo.append((split, page))
    print(f"[extract] {len(todo)} pages to do", flush=True)
    if not todo:
        return
    with GpuSlot(GPU_LOCK):
        visual = load_tower(device)
        started = time.time()
        for done, (split, page) in enumerate(todo, 1):
            key = page_key(page)
            meta = np.load(out / "labels" / split / f"{key}.npz")
            image = Image.open(store / page["png"]).convert("RGB")
            batch = processor(images=image, return_tensors="pt")
            grid = batch["image_grid_thw"]
            if grid[0].tolist() != meta["grid_thw"].tolist():
                raise RuntimeError(f"{key}: grid {grid[0].tolist()} != labels' grid")
            taps = tap_levels(
                visual,
                batch["pixel_values"].to(device, torch.bfloat16),
                grid.to(device),
                RAW_LAYERS,
            )
            kept = torch.from_numpy(meta["kept"]).to(device)
            (out / "features" / split).mkdir(parents=True, exist_ok=True)
            np.savez(
                out / "features" / split / f"{key}.npz",
                **{name: t.index_select(0, kept).to(torch.float16).cpu().numpy()
                   for name, t in taps.items()},
            )
            del taps
            if done % 10 == 0:
                rate = (time.time() - started) / done
                print(f"[extract] {done}/{len(todo)} {rate:.1f}s/page "
                      f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)
        del visual
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ main --


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["select", "labels", "extract", "probe", "report"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.step == "select":
        pages = select_pages(args.release, args.store, args.n_train, args.n_test, args.seed)
        (args.out / "pages.json").write_text(json.dumps(pages, indent=1))
        print(f"train {len(pages['train'])} test {len(pages['test'])}")
    elif args.step == "labels":
        for split in ("train", "test"):
            run_labels(args.out, args.store, split, args.workers)
    elif args.step == "extract":
        run_extract(args.out, args.store, ["train", "test"], args.device)
    elif args.step == "probe":
        if args.device == "cpu":
            # torch's Adam step probes the accelerator even for CPU tensors,
            # which opens a CUDA context on a GPU other agents are queued for
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from unflatten.probe.vit_level_probes import run_probes

        run_probes(args.out, threads=args.threads, device=args.device)
    elif args.step == "report":
        from unflatten.probe.vit_level_probes import report_tables

        summary = json.loads((args.out / "probe-results.json").read_text())
        (args.out / "tables.md").write_text(report_tables(summary))
        print(report_tables(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
