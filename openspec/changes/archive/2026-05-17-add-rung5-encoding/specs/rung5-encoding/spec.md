## ADDED Requirements

### Requirement: Rung5 is a configurable-amp-width encoding parallel to Rung4

`polygram.encoding.Rung5` SHALL be a frozen dataclass exposing two
fields: `bond_dim: int = 2` (validated like every prior rung —
`bond_dim != 2` raises `ValueError`) and `n_amp_qubits: int` (no
default, must be supplied at construction). `n_amp_qubits` MUST be
a positive integer; `n_amp_qubits < 1` SHALL raise `ValueError`.

`Rung5.max_features` SHALL be exposed as an instance property
returning `8 * 2 ** n_amp_qubits`.

The amplitude branch SHALL be parameterised as the k-fold product
of independent single-qubit amplitudes, one per amp qubit on
q3..q3+k−1 where `k = n_amp_qubits`:

```
|amp(θ_0, ψ_0, …, θ_{k−1}, ψ_{k−1})⟩
    = ⊗_{i=0}^{k−1} |u(θ_i, ψ_i)⟩_{q(3+i)}
```

where `|u(θ, ψ)⟩ = cos(θ)|0⟩ + e^(iψ) sin(θ)|1⟩` is a single-qubit
state. No entangling gates SHALL be applied between amp qubits.

#### Scenario: Rung5 is constructible with explicit n_amp_qubits

- **WHEN** `Rung5(n_amp_qubits=3)` is constructed
- **THEN** the result has `bond_dim == 2`, `n_amp_qubits == 3`, and
  `max_features == 64`

#### Scenario: bond_dim != 2 rejected

- **WHEN** `Rung5(bond_dim=3, n_amp_qubits=2)` is constructed
- **THEN** a `ValueError` is raised matching the Rung3/Rung4 message
  shape

#### Scenario: n_amp_qubits < 1 rejected

- **WHEN** `Rung5(n_amp_qubits=0)` is constructed
- **THEN** a `ValueError` is raised mentioning that `n_amp_qubits`
  must be ≥ 1

#### Scenario: max_features scales as 8 · 2^k

- **WHEN** `Rung5(n_amp_qubits=k).max_features` is read for
  k ∈ {1, 2, 3, 4, 5}
- **THEN** the returned values are `{16, 32, 64, 128, 256}`
  respectively

### Requirement: Rung5 enforces an upper bound on n_amp_qubits

`Rung5(n_amp_qubits=k)` SHALL raise `ValueError` when
`k > RUNG5_MAX_N_AMP_QUBITS` (a module-level constant in
`polygram.encoding`). The default cap SHALL be 16, giving a
maximum per-feature Hilbert dim of `8 · 2^16 = 524288`.

`RUNG5_MAX_N_AMP_QUBITS` SHALL be importable from
`polygram.encoding` so sae-forge and other consumers can read the
cap without hard-coding the value.

#### Scenario: n_amp_qubits above cap rejected

- **WHEN** `Rung5(n_amp_qubits=17)` is constructed (with the
  default cap of 16)
- **THEN** a `ValueError` is raised mentioning `n_amp_qubits`, the
  attempted value, and the cap

#### Scenario: n_amp_qubits at cap accepted

- **WHEN** `Rung5(n_amp_qubits=16)` is constructed
- **THEN** the result is valid and `max_features == 524288`

### Requirement: Rung5 amp overlap factors through single-qubit overlaps

`polygram.encoding.rung5_amp_overlap(amp_a, amp_b)` SHALL accept
two equal-length tuples of `(θ, ψ)` pairs and SHALL return the
product of `_single_qubit_overlap` calls — one per amp qubit:

```
⟨amp_a | amp_b⟩ = ∏_{i=0}^{k−1} _single_qubit_overlap(
    amp_a[i][0], amp_a[i][1], amp_b[i][0], amp_b[i][1]
)
```

`rung5_amp_overlap_squared(amp_a, amp_b)` SHALL return
`abs(rung5_amp_overlap(amp_a, amp_b)) ** 2` and equivalently equal
the product of the k single-qubit squared overlaps.

Both functions SHALL raise `ValueError` when the two input tuples
have different lengths.

#### Scenario: k-fold product factorisation holds

- **WHEN** `rung5_amp_overlap(amp_a, amp_b)` is called with two
  length-3 tuples of `(θ, ψ)` pairs
- **THEN** the result equals the product of three
  `_single_qubit_overlap` calls — one per amp-qubit index

#### Scenario: mismatched amp tuple lengths rejected

- **WHEN** `rung5_amp_overlap(amp_a, amp_b)` is called with
  `len(amp_a) != len(amp_b)`
- **THEN** a `ValueError` is raised mentioning both lengths

### Requirement: Rung5 default knobs reduce to MPSRung1-equivalent grams

A Rung5 dictionary with every feature holding `amp_knobs ==
((0.0, 0.0),) * encoding.n_amp_qubits` SHALL produce a gram equal
(within float64 tolerance, 1e-12 absolute) to the MPSRung1 gram
evaluated on the same (α, β, γ, φ).

Under those default knobs every single-qubit overlap factor MUST
equal 1, so the k-fold product collapses to 1 and the gram reduces
to the MPSRung1 gram via the elementwise-product factorisation.

#### Scenario: default-knob equivalence at k=3

- **WHEN** a 4-feature `Rung5(n_amp_qubits=3)` dictionary is built
  with every feature's `amp_knobs = ((0, 0), (0, 0), (0, 0))`
- **THEN** `dictionary.gram()` equals the gram of the same dictionary
  with `encoding=MPSRung1()` to 1e-12 absolute tolerance

#### Scenario: default-knob equivalence at k=5

- **WHEN** a 4-feature `Rung5(n_amp_qubits=5)` dictionary is built
  with every feature's `amp_knobs = ((0, 0),) * 5`
- **THEN** `dictionary.gram()` equals the gram of the same dictionary
  with `encoding=MPSRung1()` to 1e-12 absolute tolerance

### Requirement: Rung5 saturates at 8 · 2^k linearly-independent features empirically

A diverse-parameter Rung5 fixture at `N = 8 · 2^k` SHALL produce a
gram of empirical rank `8 · 2^k`, where rank is measured as the
count of singular values above 1e-12 relative to σ_max.

At `N > 8 · 2^k` the gram SHALL saturate at rank `8 · 2^k` (every
additional singular value MUST fall below 1e-12 relative to σ_max).
Parameters MUST be uniformly sampled over the full parameter ranges
for all `4 + 2k` per-feature knobs.

The `examples/rung5_rank_verification.py` artifact SHALL be the
reproducible source of this empirical bound at multiple k values.

#### Scenario: Rung5 saturates at rank 8·2^k

- **WHEN** `examples/rung5_rank_verification.py` is run for
  k ∈ {2, 3, 4}, each at `N = 8 · 2^k` and `N = 2 · 8 · 2^k`
- **THEN** rank at relative tolerance 1e-12 equals `8 · 2^k` at both
  sizes for every k tested

### Requirement: Rung5 surface is torch-free and analytic

The Rung5 encoding surface SHALL NOT import torch or transformers,
and all math SHALL be implementable in numpy + math only (the
polygram baseline stack).

The surface in scope includes `polygram.encoding.Rung5`,
`rung5_amp_overlap`, `rung5_amp_overlap_squared`, and `Rung5State`.

#### Scenario: import without torch installed

- **WHEN** `from polygram.encoding import Rung5` is executed in a
  Python environment without torch
- **THEN** the import succeeds

### Requirement: Q-OrCA emission supports Rung5 with k-parameterised actions

`polygram.emit.write_qorca` (and the underlying `_qorca_emit`
module) SHALL emit Rung5 dictionaries with an amp-branch action
accepting `2k` amp knobs per feature (where `k = encoding.n_amp_qubits`).
The emitted circuit SHALL apply `k` independent single-qubit
preparations on qubits q3..q3+k−1, with no entangling gates between
amp qubits.

The action signature in the emitted machine SHALL list the `2k` amp
knobs in interleaved `(θ_i, ψ_i)` order for `i ∈ [0, k)`.

#### Scenario: Rung5 emitter writes k single-qubit amp preparations

- **WHEN** `write_qorca` is called on a `Rung5(n_amp_qubits=3)`
  dictionary
- **THEN** the emitted machine contains an amp-branch action with 6
  amp knobs and three single-qubit preparations on q3, q4, q5

#### Scenario: no entangling gates between amp qubits

- **WHEN** the Rung5 amp-branch circuit is inspected for any value
  of k
- **THEN** no two-qubit gates appear between any pair of amp qubits
  q3..q3+k−1
