"""RFC003-ERRATA-003 second-model reasoning data frameworks (S6R groundwork).

Three FRAMEWORKS only — the content registers they plug into
(OPEN-003-E3-1 defect types, OPEN-003-E3-2 aux tag families) stay OPEN
until the owner selects a specialization (RFC005-ERRATA-007 E2):

- :mod:`unflatten.dataset.reasoning.derivation` — D4, §30.11
  `reasoning-derivation-policy`.
- :mod:`unflatten.dataset.reasoning.diff` — D5, §30.12
  `candidate-diff-policy`.
- :mod:`unflatten.dataset.reasoning.corruption` — D7, controlled-corruption
  generator for corrective examples.

Nothing here writes to a release, a cache, or a training loop; every
concrete policy is a test-only or future plug-in.
"""
