"""Independent cached-feature head training; never enables a model stage.

Inputs contain verified visible-property questions, not AST targets. Checkpoints
are local weights-only-loadable Torch states; exported inference uses safetensors.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor

from .bundle import FeatureContract
from .config import DecisionHeadConfig
from .head import DecisionHead


def _hash(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _identity(value: dict) -> str:
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _artifact(base: Path, ref: dict) -> Path:
    path = (base / ref["ref"]).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError(f"decision training cache: invalid artifact path {ref['ref']}")
    if _hash(path) != ref["sha256"]:
        raise ValueError(f"decision training cache: artifact hash mismatch {path}")
    return path


@dataclass
class CachedDataset:
    manifest: dict
    identity: str
    moments: Tensor
    candidates: dict[str, Tensor]
    view_index: dict[str, int]
    rows: dict[str, list[dict]]

    @property
    def config(self) -> DecisionHeadConfig:
        return DecisionHeadConfig(**self.manifest["head_config"])

    @property
    def contract(self) -> FeatureContract:
        return FeatureContract(**self.manifest["feature_contract"])

    @classmethod
    def load(
        cls, manifest_path: Path, *, roles: tuple[str, ...] = ("train", "dev")
    ) -> CachedDataset:
        path = Path(manifest_path)
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != "head-training-cache.v1":
            raise ValueError("decision training cache: unsupported schema")
        if manifest.get("data_gate", {}).get("status") != "pass":
            raise ValueError(
                "decision training cache: independent target/pixel data gate not passed"
            )
        config = DecisionHeadConfig(**manifest["head_config"])
        FeatureContract(**manifest["feature_contract"])
        if config.kind != "bilinear_moments":
            raise ValueError("decision training cache: cached moments require bilinear_moments")
        views: dict[str, int] = {}
        chunks = []
        offset = 0
        for shard in manifest["feature_shards"]:
            state = load_file(str(_artifact(path.parent, shard)))
            if set(state) != {"moments"}:
                raise ValueError("decision training cache: unexpected feature tensors")
            values = state["moments"]
            ids = shard["view_ids"]
            if values.shape != (len(ids), config.visual_dim * 2) or not values.is_floating_point():
                raise ValueError("decision training cache: moments shape/dtype mismatch")
            if not torch.isfinite(values).all() or (values[:, config.visual_dim :] < 0).any():
                raise ValueError("decision training cache: invalid visual moments")
            for i, name in enumerate(ids):
                if not isinstance(name, str) or not name or name in views:
                    raise ValueError("decision training cache: duplicate/invalid view id")
                views[name] = offset + i
            chunks.append(values.float())
            offset += len(ids)
        if not chunks:
            raise ValueError("decision training cache: no feature shards")
        candidates = load_file(str(_artifact(path.parent, manifest["candidate_file"])))
        registry = manifest["registry"]
        if not registry or set(candidates) != set(registry):
            raise ValueError("decision training cache: candidate registry differs")
        for task, tensor in candidates.items():
            options = registry[task]["options"]
            if len(options) < 2 or len(set(options)) != len(options):
                raise ValueError("decision training cache: invalid registered options")
            if tensor.shape != (len(options), config.text_dim) or not tensor.is_floating_point():
                raise ValueError("decision training cache: candidate dimension mismatch")
            if not torch.isfinite(tensor).all():
                raise ValueError("decision training cache: nonfinite candidate features")
        rows = {}
        lineage_roles: dict[str, str] = {}
        document_roles: dict[str, str] = {}
        pixel_roles: dict[str, str] = {}
        for role in roles:
            if role not in manifest["rows"]:
                raise ValueError(f"decision training cache: missing requested role {role}")
            source = _artifact(path.parent, manifest["rows"][role])
            rr = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
            if not rr:
                raise ValueError(f"decision training cache: empty role {role}")
            seen = set()
            for row in rr:
                task, label = row["task"], row["label"]
                if row["role"] != role or task not in registry or row["view_id"] not in views:
                    raise ValueError("decision training cache: invalid role/task/view in question")
                if type(label) is not int or not 0 <= label < len(registry[task]["options"]):
                    raise ValueError("decision training cache: invalid answer index")
                key = (row["view_id"], task)
                if key in seen:
                    raise ValueError("decision training cache: duplicated role/view/task question")
                seen.add(key)
                for field, assigned in [
                    ("lineage", lineage_roles),
                    ("document", document_roles),
                    ("pixel_sha256", pixel_roles),
                ]:
                    value = row[field]
                    if not isinstance(value, str) or not value:
                        raise ValueError(f"decision training cache: invalid {field}")
                    if value in assigned and assigned[value] != role:
                        raise ValueError(f"decision training cache: {field} crosses roles")
                    assigned[value] = role
            rows[role] = rr
        return cls(manifest, _hash(path), torch.cat(chunks), candidates, views, rows)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 256
    seed: int = 42
    learning_rate: float = 0.001
    minimum_learning_rate: float = 0.0001
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"decision training config: {name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("decision training config: invalid seed")
        for name in ("learning_rate", "minimum_learning_rate", "weight_decay", "gradient_clip"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"decision training config: invalid {name}")
        if (
            not 0 < self.minimum_learning_rate <= self.learning_rate
            or self.weight_decay < 0
            or self.gradient_clip <= 0
        ):
            raise ValueError("decision training config: invalid optimizer bounds")


class TrainRunner:
    """One full-epoch run with exact local resume; train and dev targets only."""

    def __init__(self, cache: CachedDataset, config: TrainingConfig, *, device: str) -> None:
        if set(cache.rows) != {"train", "dev"}:
            raise ValueError("decision training: only train/dev roles may enter TrainRunner")
        self.cache, self.config = cache, config
        self.device = torch.device(device)
        source_dir = Path(__file__).parent
        code = {
            n: _hash(source_dir / n) for n in ("training.py", "head.py", "config.py", "bundle.py")
        }
        self.identity = _identity(
            {
                "dataset": cache.identity,
                "config": asdict(config),
                "code": code,
                "torch": str(torch.__version__),
                "device": str(self.device),
            }
        )
        torch.manual_seed(config.seed)
        self.generator = torch.Generator().manual_seed(config.seed + 1000)
        self.head = DecisionHead(cache.config).to(self.device)
        self.moments = cache.moments.to(self.device)
        tasks = sorted(cache.candidates)
        self.tasks = tasks
        self.task_index = {task: i for i, task in enumerate(tasks)}
        largest = max(c.shape[0] for c in cache.candidates.values())
        self.candidates = torch.zeros(
            len(tasks), largest, cache.config.text_dim, device=self.device
        )
        self.masks = torch.zeros(len(tasks), largest, dtype=torch.bool, device=self.device)
        for i, task in enumerate(tasks):
            c = cache.candidates[task].to(self.device)
            self.candidates[i, : len(c)] = c
            self.masks[i, : len(c)] = True
        self.train = self._indices(cache.rows["train"])
        unique = torch.tensor(
            sorted({cache.view_index[r["view_id"]] for r in cache.rows["train"]}),
            dtype=torch.long,
            device=self.device,
        )
        self.head.set_train_moment_stats(
            self.moments[unique],
            visual_present=torch.ones(len(unique), dtype=torch.bool, device=self.device),
        )
        by = Counter((r["task"], r["lineage"]) for r in cache.rows["train"])
        lineages = {
            task: len({r["lineage"] for r in cache.rows["train"] if r["task"] == task})
            for task in tasks
        }
        if not all(lineages.values()):
            raise ValueError("decision training: every registered task needs train supervision")
        n = len(cache.rows["train"])
        self.weights = torch.tensor(
            [
                n / (len(tasks) * lineages[r["task"]] * by[r["task"], r["lineage"]])
                for r in cache.rows["train"]
            ],
            device=self.device,
        )
        self.optimizer = torch.optim.AdamW(
            self.head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.order = torch.randperm(n, generator=self.generator)
        self.epoch, self.position, self.step = 0, 0, 0
        self.history: list[dict] = []
        self.best_metric = -1.0
        self.best_state: dict[str, Tensor] | None = None

    def _indices(self, rows: list[dict]) -> tuple[Tensor, Tensor, Tensor]:
        return tuple(
            torch.tensor(values, dtype=torch.long, device=self.device)
            for values in (
                [self.cache.view_index[r["view_id"]] for r in rows],
                [self.task_index[r["task"]] for r in rows],
                [r["label"] for r in rows],
            )
        )

    @property
    def finished(self) -> bool:
        return self.epoch == self.config.epochs - 1 and self.position == len(self.order)

    def train_step(self) -> float | None:
        if self.finished:
            return None
        if self.position == len(self.order):
            self.epoch += 1
            self.position = 0
            self.order = torch.randperm(len(self.order), generator=self.generator)
        nsteps = self.config.epochs * math.ceil(len(self.order) / self.config.batch_size)
        progress = self.step / max(nsteps - 1, 1)
        lr = (
            self.config.minimum_learning_rate
            + (self.config.learning_rate - self.config.minimum_learning_rate)
            * (1 + math.cos(math.pi * progress))
            / 2
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        end = min(self.position + self.config.batch_size, len(self.order))
        ix = self.order[self.position : end].to(self.device)
        views, tasks, targets = (x[ix] for x in self.train)
        flags = torch.ones(len(ix), dtype=torch.bool, device=self.device)
        self.head.train()
        self.optimizer.zero_grad(set_to_none=True)
        result = self.head.forward_moments(
            self.moments[views],
            self.candidates[tasks],
            visual_present=flags,
            candidate_mask=self.masks[tasks],
            eligible=flags,
        )
        loss = (
            torch.nn.functional.cross_entropy(result.logits, targets, reduction="none")
            * self.weights[ix]
        ).mean()
        if not torch.isfinite(loss):
            raise ValueError(f"decision training: nonfinite loss at step {self.step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.head.parameters(), self.config.gradient_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.position, self.step = end, self.step + 1
        return float(loss.detach())

    def run_steps(self, count: int) -> None:
        for _ in range(count):
            if self.train_step() is None:
                break

    @torch.no_grad()
    def evaluate_dev(self) -> dict:
        self.head.eval()
        rows = self.cache.rows["dev"]
        indices = self._indices(rows)
        predictions = []
        for start in range(0, len(rows), self.config.batch_size):
            views, tasks, _ = (x[start : start + self.config.batch_size] for x in indices)
            flags = torch.ones(len(views), dtype=torch.bool, device=self.device)
            result = self.head.forward_moments(
                self.moments[views],
                self.candidates[tasks],
                visual_present=flags,
                candidate_mask=self.masks[tasks],
                eligible=flags,
            )
            predictions.extend(result.choice_index.cpu().tolist())
        report = metrics(rows, predictions, self.cache.manifest["registry"])
        score = report["mean_balanced_accuracy"]
        if score > self.best_metric:
            self.best_metric = score
            self.best_state = {
                k: v.detach().cpu().clone() for k, v in self.head.state_dict().items()
            }
        entry = {
            "epoch": self.epoch + 1,
            "step": self.step,
            "best_metric": self.best_metric,
            **report,
        }
        self.history.append(entry)
        return entry

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "schema": "decision-training-state.v1",
            "identity": self.identity,
            "head": self.head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "position": self.position,
            "step": self.step,
            "order": self.order,
            "generator": self.generator.get_state(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            "history": self.history,
            "best_metric": self.best_metric,
            "best_state": self.best_state,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)

    def restore_checkpoint(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (
            state.get("schema") != "decision-training-state.v1"
            or state.get("identity") != self.identity
        ):
            raise ValueError("decision training resume: checkpoint identity mismatch")
        order, epoch, position, step = (state[k] for k in ("order", "epoch", "position", "step"))
        if (
            order.dtype != torch.long
            or order.shape != self.order.shape
            or not torch.equal(order.sort().values, torch.arange(len(order)))
        ):
            raise ValueError("decision training resume: invalid sample order")
        if (
            type(epoch) is not int
            or not 0 <= epoch < self.config.epochs
            or type(position) is not int
            or not 0 <= position <= len(order)
            or type(step) is not int
            or step < 0
        ):
            raise ValueError("decision training resume: invalid progress state")
        self.head.load_state_dict(state["head"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.epoch, self.position, self.step = epoch, position, step
        self.order = order
        self.generator.set_state(state["generator"])
        torch.set_rng_state(state["cpu_rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        self.history, self.best_metric, self.best_state = (
            state["history"],
            state["best_metric"],
            state["best_state"],
        )


def metrics(rows: list[dict], predictions: list[int], registry: dict) -> dict:
    """Equal lineage mass per task; supported-class BA, plus explicit support."""
    if len(rows) != len(predictions):
        raise ValueError("decision metrics: prediction count differs")
    reports = {}
    for task, definition in registry.items():
        selected = [(r, p) for r, p in zip(rows, predictions) if r["task"] == task]
        if not selected:
            raise ValueError(f"decision metrics: no questions for task {task}")
        size = len(definition["options"])
        counts = Counter(r["lineage"] for r, _ in selected)
        matrix = torch.zeros(size, size, dtype=torch.float64)
        support = defaultdict(set)
        for row, prediction in selected:
            if not 0 <= prediction < size:
                raise ValueError("decision metrics: invalid predicted answer")
            matrix[row["label"], prediction] += 1 / counts[row["lineage"]]
            support[row["label"]].add(row["lineage"])
        total = matrix.sum(1)
        present = total > 0
        recall = matrix.diag() / total.clamp_min(1e-30)
        precision = matrix.diag() / matrix.sum(0).clamp_min(1e-30)
        f1 = 2 * recall * precision / (recall + precision).clamp_min(1e-30)
        reports[task] = {
            "accuracy": float(matrix.diag().sum() / matrix.sum()),
            "balanced_accuracy": float(recall[present].mean()),
            "macro_f1": float(f1[present].mean()),
            "questions": len(selected),
            "lineages": len(counts),
            "confusion_lineage_mass": matrix.tolist(),
            "class_lineages": {
                option: len(support[i]) for i, option in enumerate(definition["options"])
            },
            "class_recall": {
                option: float(recall[i]) if present[i] else None
                for i, option in enumerate(definition["options"])
            },
        }
    return {
        "tasks": reports,
        "mean_balanced_accuracy": sum(r["balanced_accuracy"] for r in reports.values())
        / len(reports),
    }
