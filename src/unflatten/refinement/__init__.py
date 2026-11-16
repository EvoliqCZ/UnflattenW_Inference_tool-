"""RFC006 W3-12 — deterministic, model-free refinement core (dormant).

DEC-047 (2026-09-26) authorizes this package as DORMANT code ahead of
RFC006 ratification: default off, no change to any existing identity or
production path, no model training. RFC006 itself remains PROPOSED with
product enablement BLOCKED (CLAUDE.md). Nothing outside this package may
import it; nothing in this package trains anything or has commit
authority over a production document.

Scope (RFC006 §57.1 "approved to implement", model-free only):

- `target_catalog` — RFC006 §19 Authority Target Catalog + model view,
  derived by *reading* the existing typed pointer map
  (`unflatten.model.typed_property_pointer_map`) and the RFC006 §21
  mutable-property allowlist. Never mutates either.
- `patch_language` — RFC006 §20-22, the closed `set_typed_property`
  operation and `ASTPatchSet`, plus inverse-patch generation.
- `patch_applier` — RFC006 §24 deterministic, transactional PatchApplier
  and Conservation Report over `src/unflatten/ast/`.
- `render_engine` / `verifier` — RFC006 §25-26 Relative Fidelity
  Verifier, deterministic decision logic behind an injectable
  `RenderEngine` so unit tests never need LibreOffice.
- `journal` — RFC006 §27/§33.5 patch journal and replay.
- `corruption` — RFC006 §38 offline corruption episode factory (R0).

Explicitly NOT implemented here (out of scope for W3-12, and most of it
requires RL/GRPO/PPO in the refiner (REJECTED V0, RFC006-ERRATA-001 §42.8(a)) or cross-RFC ratification DEC-047 does not cover):
the Pixel Evidence Ledger (§11-16), Active Foveated Evidence Retriever
(§17), the learned Source-Render Residual Critic (§18) and Typed Patch
Decoder (§20) themselves, the Refinement Controller state machine
(§28), confidence/calibration (§30), and any commit authority over a
real production document.
"""
