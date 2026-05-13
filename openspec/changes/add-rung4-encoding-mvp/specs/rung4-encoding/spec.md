## ADDED Requirements

### Requirement: Rung4 is a 5-qubit encoding parallel to Rung3 with a product amp branch

`polygram.encoding.Rung4` SHALL be a frozen dataclass parallel to
`Rung3`, exposing a single field `bond_dim: int = 2` with the same
`__post_init__` validation. The class SHALL declare
`max_features = 32` (consumed by `per-encoding-feature-cap`).

The amplitude branch SHALL be parameterised as the product of two
independent single-qubit amplitudes:

```
|amp(θ_a, ψ_a, θ_b, ψ_b)⟩ = |u(θ_a, ψ_a)⟩_{q3} ⊗ |v(θ_b, ψ_b)⟩_{q4}
```

where `|u(θ, ψ)⟩ = cos(θ)|0⟩ + e^(iψ) sin(θ)|1⟩` is a single-qubit
state.

#### Scenario: Rung4 is constructible with default bond_dim

- **WHEN** `Rung4()` is constructed
- **THEN** the result has `bond_dim == 2` and `max_features == 32`

#### Scenario: bond_dim != 2 rejected

- **WHEN** `Rung4(bond_dim=3)` is constructed
- **THEN** a `ValueError` is raised matching the Rung3 message shape

### Requirement: Rung4 amp overlap factors through single-qubit overlaps

`polygram.encoding.rung4_amp_overlap` SHALL return the product of two single-qubit overlap calls — one for the q3 amp pair, one for the q4 amp pair.

For inputs `(θ_a3, ψ_a3, θ_a4, ψ_a4, θ_b3, ψ_b3, θ_b4, ψ_b4)` the returned complex value SHALL equal:

```
⟨amp_a | amp_b⟩ = _single_qubit_overlap(θ_a3, ψ_a3, θ_b3, ψ_b3)
                * _single_qubit_overlap(θ_a4, ψ_a4, θ_b4, ψ_b4)
```

where `_single_qubit_overlap` is a helper exposing the existing
single-qubit overlap math (the body of the current
`rung3_amp_overlap` function).

`polygram.encoding.rung4_amp_overlap_squared(...)` SHALL return
`abs(rung4_amp_overlap(...)) ** 2` and equivalently equal the product
of the two single-qubit squared overlaps.

#### Scenario: product factorisation holds

- **WHEN** `rung4_amp_overlap(θ_a3, ψ_a3, θ_a4, ψ_a4, θ_b3, ψ_b3, θ_b4, ψ_b4)`
  is called with two independent (θ, ψ) pairs per feature
- **THEN** the result equals the product of two `_single_qubit_overlap`
  calls

### Requirement: Rung4 default knobs reduce to MPSRung1-equivalent grams

A Rung4 dictionary with every feature holding `theta_amp == psi_aux == theta_amp_b == psi_amp_b == 0` SHALL produce a gram equal (within float64 tolerance, 1e-12 absolute) to the MPSRung1 gram evaluated on the same (α, β, γ, φ).

Under those default knobs the amp overlap factor MUST be identically 1 for every pair, so the elementwise-product factorisation collapses to the MPS gram.

#### Scenario: default-knob equivalence

- **WHEN** a 4-feature Rung4 dictionary is built with all
  `theta_amp = theta_amp_b = psi_aux = psi_amp_b = 0`
- **THEN** `dictionary.gram()` equals the gram of the same dictionary
  with `encoding=MPSRung1()` to 1e-12 absolute tolerance

### Requirement: Rung4 reaches 32 linearly-independent features empirically

A diverse-parameter Rung4 fixture at N=32 SHALL produce a gram of empirical rank 32, where rank is measured as the count of singular values above 1e-12 relative to σ_max.

At N=40 the gram SHALL saturate at rank 32 (every additional singular value MUST fall below 1e-12 relative to σ_max). Parameters are uniformly sampled over the full parameter ranges for all six per-feature knobs.

The `examples/rung4_rank_verification.py` artifact SHALL be the
reproducible source of this empirical bound.

#### Scenario: Rung4 saturates at rank 32

- **WHEN** `examples/rung4_rank_verification.py` is run with N ∈ {32, 40}
- **THEN** rank at relative tolerance 1e-12 equals 32 at both sizes

### Requirement: Rung4 surface is torch-free and analytic

The Rung4 encoding surface SHALL NOT import torch or transformers, and all math SHALL be implementable in numpy + math only (the polygram baseline stack).

The surface in scope includes `polygram.encoding.Rung4`, `rung4_amp_overlap`, `rung4_amp_overlap_squared`, and `Rung4State`.

#### Scenario: import without torch installed

- **WHEN** `from polygram.encoding import Rung4` is executed in a
  Python environment without torch
- **THEN** the import succeeds
