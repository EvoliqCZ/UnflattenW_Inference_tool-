"""RFC004 §37 optional speculative decoding: protocol (`drafter.py`), MTP
heads (`mtp.py`, `mtp_drafter.py`), and the opt-in wire-up into the real
grammar-constrained decode (`decode_integration.py`).

DEC-047: dormant and default off. `decode_integration.
speculative_decode_free_running`'s `speculator=None` default is exactly
`free_running_decode.decode_free_running`, unmodified -- production
decode is never called through this package unless a caller explicitly
opts in. See `drafter.py`'s module docstring for what closed the KV-cache
rollback gap this package used to name (`IncrementalDecodeState.snapshot`/
`.crop`/`.rollback_to`, `candidate_model.py`).
"""
