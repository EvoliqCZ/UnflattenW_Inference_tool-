"""Design §7 Guard A / task 11 — refusal at the boundary, plus the
diagnostic's own training step.

`TypedPropertyHeadDiagnosticRun` is this diagnostic's only public entry
point. Its `run_class` argument is mandatory and keyword-only (no
default -- mirroring `probe/run.py`'s pattern of a training-shaped flag
defaulting to the SAFE value, never the dangerous one) and is refused
unless it is exactly `"diagnostic_canary"`. This is the boundary doc's
Guard A: a caller cannot construct this run under any run class that
would make it look like S1/S2/a promoted run.

`training_step` wires the pieces already proven in isolation by earlier
tasks: a list of `dataset.PropertyExample` (each already carrying
`family`, sourced from `pointer_map.DIAGNOSTIC_POINTER_TO_FAMILY`) is
grouped by `(node_type, pointer)`, dispatched to one of the three
`heads.py` families, reduced through `masking.masked_mean` inside each
head's own `loss()`, and optimized with a plain `torch.optim.Adam` this
run owns itself -- it never claims a group in the S1/S2 trainable-group
registry, never builds any S1 run-manifest object or its promotion-gate
hash field, and never touches the S1 loss-unit/locator/loss-family
machinery. See
`tests/diagnostics/typed_property_head/test_run_training_step.py`, which
also statically scans this file's source for those exact identifiers.

Two scope notes this diagnostic does not pretend away (design §8):

- `d_model` is inferred from the first anchor a given `(node_type,
  pointer)` head is built for, never hardcoded -- this stays honest
  about being a small feasibility canary over whatever backbone width
  happens to be plugged in.
- an enum head's vocabulary is the pointer's FULL domain from the AST
  schema (`enum_domain.enum_domain`, sorted by `str` so it is
  deterministic), never observed data: it is fixed before the first
  example arrives and does not depend on which values a given batch
  happens to contain or the order batches arrive in. A later
  `training_step` call that sees a value outside that schema domain
  still raises `UnknownEnumValueError` -- that is a real registry-drift
  bug -- but a schema-legal value merely unseen so far never does.
- a KNOWN example can still carry `value=None` (RFC003 §29.1: an
  authoritative null, e.g. a body paragraph's `/semantic/heading_level`
  is genuinely absent, not unknown). None of the three head families has
  a null class, so `training_step` folds `value is None` into the mask
  passed to `masked_mean` -- such an example is dropped from the loss
  exactly like `known_mask=False`, never coerced into a fabricated `0`
  or vocabulary entry that a downstream reader could mistake for a real
  observed value.
- the integer family's `direct_regression` vs `bins_plus_residual` choice
  (RFC004 §52.1 / RFC005 S3) is selectable via `integer_strategy` at
  construction time, defaulting to `direct_regression`; `bins_plus_residual`
  requires caller-supplied `integer_bin_edges` per `(node_type, pointer)`
  and raises `MissingIntegerBinEdgesError` otherwise -- this diagnostic
  does not invent bin boundaries (design §4).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

import torch

from unflatten.diagnostics.typed_property_head.dataset import PropertyExample
from unflatten.diagnostics.typed_property_head.enum_domain import enum_domain
from unflatten.diagnostics.typed_property_head.heads import (
    BooleanPropertyHead,
    EnumPropertyHead,
    IntegerHeadConfig,
    IntegerPropertyHead,
    TypedPropertyHead,
)

DIAGNOSTIC_CANARY_RUN_CLASS = "diagnostic_canary"

_PropertyKey = tuple[str, str]


class UnknownEnumValueError(ValueError):
    """An enum example's value is not in the pointer's AST-schema domain
    (design §8 scope note above / finding 1) -- a registry-drift bug, not
    a value this diagnostic ever grows its classifier to accommodate."""


class MissingIntegerBinEdgesError(ValueError):
    """`integer_strategy="bins_plus_residual"` was selected but no
    `bin_edges` were supplied for this `(node_type, pointer)` key -- the
    concrete bin boundaries are an OPEN measurement question (design §4),
    never a value this diagnostic may invent."""


class TypedPropertyHeadDiagnosticRun:
    """Design §7 Guard A boundary, extended by task 11 with the training
    step. `run_class` refusal is unchanged by the extension."""

    def __init__(
        self,
        *,
        run_class: str,
        learning_rate: float = 1e-3,
        integer_strategy: Literal["direct_regression", "bins_plus_residual"] = "direct_regression",
        integer_bin_edges: dict[_PropertyKey, tuple[int, ...]] | None = None,
    ) -> None:
        if run_class != DIAGNOSTIC_CANARY_RUN_CLASS:
            raise ValueError(
                f"TypedPropertyHeadDiagnosticRun refuses run_class={run_class!r}; "
                f"only {DIAGNOSTIC_CANARY_RUN_CLASS!r} is accepted (design §7 Guard A)"
            )
        self.run_class = run_class
        self._learning_rate = learning_rate
        self._integer_strategy = integer_strategy
        self._integer_bin_edges = integer_bin_edges
        self._heads: dict[_PropertyKey, TypedPropertyHead] = {}
        self._enum_vocab: dict[_PropertyKey, dict[Any, int]] = {}
        self._optimizer: torch.optim.Optimizer | None = None
        self._optimizer_param_ids: frozenset[int] = frozenset()

    def _head_for(
        self, key: _PropertyKey, family: str, examples: list[PropertyExample]
    ) -> TypedPropertyHead:
        if key in self._heads:
            return self._heads[key]

        d_model = examples[0].anchor.shape[-1]
        if family == "boolean":
            head: TypedPropertyHead = BooleanPropertyHead(d_model=d_model)
        elif family == "enum":
            node_type, pointer = key
            vocab = {
                value: index
                for index, value in enumerate(sorted(enum_domain(node_type, pointer), key=str))
            }
            self._enum_vocab[key] = vocab
            head = EnumPropertyHead(d_model=d_model, num_classes=len(vocab))
        elif family == "integer":
            # Strategy is selected by the caller at construction time
            # (RFC004 §52.1 / RFC005 S3 ablation this run only feeds, not
            # answers). `bins_plus_residual` needs a per-pointer bin_edges
            # profile (design §4) that this diagnostic does not invent --
            # it must be supplied by the caller or construction refuses.
            if self._integer_strategy == "bins_plus_residual":
                if self._integer_bin_edges is None or key not in self._integer_bin_edges:
                    raise MissingIntegerBinEdgesError(
                        f"{key!r}: integer_strategy='bins_plus_residual' requires "
                        "caller-supplied bin_edges for this (node_type, pointer); "
                        "none were given"
                    )
                config = IntegerHeadConfig(
                    strategy="bins_plus_residual",
                    bin_edges=self._integer_bin_edges[key],
                )
            else:
                config = IntegerHeadConfig(strategy="direct_regression")
            head = IntegerPropertyHead(d_model=d_model, config=config)
        else:
            raise ValueError(f"{key!r}: unknown family {family!r}")

        self._heads[key] = head
        return head

    def _target_tensor(
        self, key: _PropertyKey, family: str, examples: list[PropertyExample]
    ) -> torch.Tensor:
        """Builds a numeric target for every example, including ones whose
        `value is None` (an authoritative, KNOWN null -- RFC003 §29.1 --
        e.g. a body paragraph's `/semantic/heading_level`). None of the
        three head families has a "null" class, so a null example's
        target is a placeholder (index/value `0`) that is never read: the
        caller combines `known_mask` with `value is not None` before
        calling `masked_mean`, which zeroes exactly those examples'
        contribution regardless of what placeholder they carry."""
        if family == "boolean":
            return torch.tensor([float(e.value) if e.value is not None else 0.0 for e in examples])
        if family == "integer":
            return torch.tensor(
                [int(e.value) if e.value is not None else 0 for e in examples], dtype=torch.long
            )

        vocab = self._enum_vocab[key]
        indices = []
        for example in examples:
            if example.value is None:
                indices.append(0)
                continue
            if example.value not in vocab:
                raise UnknownEnumValueError(
                    f"{key!r}: value {example.value!r} is not in this pointer's "
                    f"AST-schema domain ({sorted(vocab, key=str)!r})"
                )
            indices.append(vocab[example.value])
        return torch.tensor(indices, dtype=torch.long)

    def _ensure_optimizer(self) -> None:
        """Finding 2: a newly appeared head's parameters are added to the
        EXISTING `Adam` via `add_param_group`, never by rebuilding it --
        rebuilding discards every already-trained head's moment estimates
        (`exp_avg`/`exp_avg_sq`), which happens up to once per newly
        appearing pointer/enum-value early in training."""
        params = [p for head in self._heads.values() for p in head.parameters()]
        param_ids = frozenset(id(p) for p in params)
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(params, lr=self._learning_rate)
            self._optimizer_param_ids = param_ids
        elif param_ids != self._optimizer_param_ids:
            new_params = [p for p in params if id(p) not in self._optimizer_param_ids]
            if new_params:
                self._optimizer.add_param_group({"params": new_params})
            self._optimizer_param_ids = param_ids

    def training_step(self, examples: list[PropertyExample]) -> torch.Tensor:
        """One masked-loss, one optimizer step, over this run's OWN heads
        and OWN `torch.optim.Adam` -- no trainable-group-registry
        involvement and no manifest of any kind (module docstring).

        Raises `ValueError` on an empty batch (there is nothing to step
        on) and `UnknownEnumValueError` if an enum example's value falls
        outside its pointer's AST-schema domain.
        """
        if not examples:
            raise ValueError("training_step requires at least one example")

        grouped: dict[_PropertyKey, list[PropertyExample]] = defaultdict(list)
        family_of: dict[_PropertyKey, str] = {}
        for example in examples:
            key = (example.node_type, example.pointer)
            grouped[key].append(example)
            family_of[key] = example.family

        losses = []
        for key, group in grouped.items():
            family = family_of[key]
            head = self._head_for(key, family, group)
            anchors = torch.stack([e.anchor for e in group])
            # A known-but-null example (design note on _target_tensor) has
            # no target any head family can represent -- fold it into the
            # mask so masked_mean drops it exactly like known_mask=False.
            known_mask = torch.tensor(
                [e.known_mask and e.value is not None for e in group], dtype=torch.bool
            )
            target = self._target_tensor(key, family, group)
            prediction = head(anchors)
            losses.append(head.loss(prediction, target, known_mask))

        total_loss = torch.stack(losses).sum()

        self._ensure_optimizer()
        assert self._optimizer is not None
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        return total_loss.detach()
