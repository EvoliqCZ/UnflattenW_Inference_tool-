"""Deterministic page assembler (assembler design `spec-s4-readiness`, DEC-033 §2).

Must not import `unflatten.dataset`, `unflatten.eval` or any training/cache
module: it runs on model output with no store (design §3, enforced by
`tests/assembly/test_hypothesis.py`).
"""
