"""RFC005 §25 LoRA delta module — dormant code over PROPOSED S4/S6 LoRA
child runs, per `docs/decisions/DEC-047-spiaci-kod-v-predstihu.md`.

This package is a **standalone** LoRA wrapper for `nn.Linear` layers of a
frozen decoder. It is never wired into `trainable_group_registry`'s four
closed `lora.*` groups (which stay `not_applicable`) or into
`candidate_track.ADAPTATION_KIND` (which stays
`"adapter_plus_new_heads_no_lora"`) — DEC-047 condition 3 forbids moving
those hashed identities, and the S4 v15b promoted run and the DEC-044
diagnostic service never import this package. Nothing here runs a gradient.
"""

from unflatten.model.lora.config import LoRAConfig
from unflatten.model.lora.linear import LoRALinear

__all__ = ["LoRAConfig", "LoRALinear"]
