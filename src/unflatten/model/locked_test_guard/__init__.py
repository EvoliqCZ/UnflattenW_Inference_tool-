"""RFC005 §23.2-23.3/§53 #37/#46 locked-test access guard.

Dormant per `docs/decisions/DEC-047-spiaci-kod-v-predstihu.md`: this
guards a not-yet-built S9 locked test set. No caller in this repo opens a
locked test today; the guard exists so a future S9 program has an atomic
`sealed_unopened -> access_reserved -> consumed` state machine ready
before the first real access.
"""

from unflatten.model.locked_test_guard.guard import (
    LockedTestGuard,
    LockedTestState,
    SecondAccessRefusedError,
)
from unflatten.model.locked_test_guard.product_audit_plan import ProductAuditPlan

__all__ = [
    "LockedTestGuard",
    "LockedTestState",
    "SecondAccessRefusedError",
    "ProductAuditPlan",
]
