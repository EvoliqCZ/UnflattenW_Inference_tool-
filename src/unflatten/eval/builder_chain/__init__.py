"""The T8 end-to-end AST Builder chain report (design §2.6, plan P7).

Gold arm: corpus fixture -> gold page hypothesis -> `assemble` ->
`build_strict` -> `promote` -> `build_review_candidate` -> one report row.
Model arm: a sealed S3 decode artefact -> `sealed_adapter.page_hypothesis`
-> the same chain. `report.py` aggregates rows into one
`builder_chain_report.json` under domain `rfc004.builder-chain-report/v0`.
"""
