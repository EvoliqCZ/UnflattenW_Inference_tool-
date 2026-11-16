"""Diagnostic child runs — never part of any promotable S1/S2 pipeline.

Per `docs/plans/typed-property-head-module-boundary.md` Q1: this package is
a sibling to, and never a descendant of, `src/unflatten/model/`, so no
diagnostic module can trip `tests/model/test_m1_design_invariants.py`'s
file-inventory guard or M1 acceptance criterion 15. Nothing under this
package may be imported from `run_binding.py`, `trainable_group_registry.py`
or any promotion-path module (Guard B).
"""
