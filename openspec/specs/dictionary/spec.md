# dictionary Specification

## Purpose

The core data model: `Feature` (a single feature with cluster
assignment, β, α, γ, φ knobs) and `Dictionary` (a collection of
features under a chosen encoding). Defines the `MPSRung1` encoding
marker and the analytic-Gram surface (`Dictionary.gram()`) that
downstream experiments and the analysis layer build on.

## Requirements
### Requirement: MPSRung1 encoding marker

Polygram SHALL expose `MPSRung1(bond_dim=2, phase_knobs=True)` as a
config marker for the rung-1 MPS encoding. v0 supports `bond_dim=2`
only.

#### Scenario: bond_dim != 2 rejected

- **WHEN** `MPSRung1(bond_dim=3)` is constructed
- **THEN** `__post_init__` raises `ValueError` mentioning that v0
  supports rung-1 (χ=2) only

