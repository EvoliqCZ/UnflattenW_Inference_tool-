"""RFC003 §32 M2 — migration of the pre-existing real-DOCX corpus.

M2a bulk import: registry-driven import of the raw blob store into
content-addressed DocumentBundles with lineage + rights records
(rights basis: docs/decisions/DEC-001, option A). Every source file —
including failures — leaves an auditable record; nothing is silently
skipped (RFC003 §45).
"""
