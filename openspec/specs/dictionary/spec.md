# dictionary Specification

## Purpose
TBD - created by archiving change core-dictionary-mpsrung1. Update Purpose after archive.
## Requirements
### Requirement: MPSRung1 encoding marker

Polygram SHALL expose `MPSRung1(bond_dim=2, phase_knobs=True)` as a
config marker for the rung-1 MPS encoding. v0 supports `bond_dim=2`
only.

#### Scenario: bond_dim != 2 rejected

- **WHEN** `MPSRung1(bond_dim=3)` is constructed
- **THEN** `__post_init__` raises `ValueError` mentioning that v0
  supports rung-1 (χ=2) only

