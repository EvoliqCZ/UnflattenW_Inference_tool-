"""Typed property head diagnostic child run (`run_class="diagnostic_canary"`).

Slice 1: `expert_adapter.dense` restricted to `paragraph`, 35 pointers — see
`docs/plans/typed-property-head-module-design.md` §1.1. This package never
claims a `trainable_group_registry` group and is never imported by any
promotion-path module (boundary doc Q2/Q3).
"""
