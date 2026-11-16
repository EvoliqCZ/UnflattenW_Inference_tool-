"""DEC-050: export the model as one complete Hugging Face package.

`configuration_unflatten.py` and `modeling_unflatten.py` are copied
byte-for-byte into every exported repository as its `trust_remote_code`
custom code (`scripts/export_complete_model.py`) -- this package is their
source of truth, not something a caller imports directly at inference time.
"""
