"""RFC005 §9/§13 S0 admission report (dataset half).

S0 performs **no gradient update** (§13.1). It freezes the evidence base:
it verifies the release identity, measures the §9.2 blocking conditions,
recomputes the §43 coverage minima, writes the §9.4 migration report that
turns an inventory count into a training count, and marks the model half
`blocked` while the RFC004 artifacts §9.1 requires do not exist.

Nothing here trains, loads a model or touches a GPU. The runner's run
class is `discovery` (§12.1) and the report says so.
"""

from unflatten.dataset.admission.evidence import AdmissionError
from unflatten.dataset.admission.report import build_admission_report

__all__ = ["AdmissionError", "build_admission_report"]
