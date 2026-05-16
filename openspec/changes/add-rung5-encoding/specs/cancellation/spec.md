## ADDED Requirements

### Requirement: Cancellation accepts encoding="rung5"

`polygram.cancellation.Cancellation` SHALL accept `encoding="rung5"`
alongside the existing `"mps"`, `"hea"`, `"rung3"`, and `"rung4"`
values. `SUPPORTED_ENCODINGS` SHALL list `"rung5"`.

When `encoding="rung5"`, `Cancellation.__post_init__` SHALL read
`k = dictionary.encoding.n_amp_qubits` from the supplied dictionary
and assemble the default-knob list as:

```
[
  "a.phi",
  "b.phi",
  "b.amp_knobs[0].theta",
  "b.amp_knobs[0].psi",
  "b.amp_knobs[1].theta",
  "b.amp_knobs[1].psi",
  ...
  "b.amp_knobs[k-1].theta",
  "b.amp_knobs[k-1].psi",
]
```

producing a `2 + 2k`-knob default optimisation space.

#### Scenario: encoding="rung5" accepted

- **WHEN** `Cancellation(encoding="rung5", dictionary=d)` is
  constructed where `d.encoding == Rung5(n_amp_qubits=3)`
- **THEN** construction succeeds and the default knob list has 8
  entries (2 φ knobs + 2·3 amp knobs)

#### Scenario: knob list scales with n_amp_qubits

- **WHEN** `Cancellation(encoding="rung5", dictionary=d)` is
  constructed for `d.encoding == Rung5(n_amp_qubits=k)` for
  k ∈ {2, 3, 5}
- **THEN** the default knob list has `2 + 2k` entries (6, 8, 12
  respectively)

### Requirement: Rung5 joint solves route to the scipy backend

When the resolved default-knob list for a `"rung5"` cancellation
exceeds `GRID_KNOB_LIMIT` (today 4), the grid backend SHALL raise
`ValueError` with a clear message recommending the scipy backend.
This applies for every `k ≥ 1` since the default knob count is
already `2 + 2k ≥ 4`, exceeding the grid limit at k ≥ 2.

Single-knob solves (e.g. only `a.phi`) on a `"rung5"` dictionary
SHALL continue to be available on the grid backend, matching the
existing behaviour for Rung3 and Rung4.

#### Scenario: default Rung5 joint solve on grid backend errors

- **WHEN** `Cancellation(encoding="rung5", method="grid",
  dictionary=d)` is run with the default knob list at
  `d.encoding == Rung5(n_amp_qubits=2)`
- **THEN** a `ValueError` is raised mentioning the grid knob limit
  and recommending the scipy backend

#### Scenario: φ-only Rung5 solve on grid backend succeeds

- **WHEN** `Cancellation(encoding="rung5", method="grid",
  dictionary=d, knobs=["a.phi", "b.phi"])` is run
- **THEN** the grid backend executes normally and returns a result

### Requirement: Rung5 min_amp_overlap applies to the k-fold product overlap

For `encoding="rung5"`, the `min_amp_overlap` constraint SHALL be
evaluated against `rung5_amp_overlap_squared(amp_a, amp_b)` — the
k-fold product of single-qubit squared overlaps — at the candidate
knob values produced by the optimiser.

#### Scenario: min_amp_overlap enforced on product overlap

- **WHEN** `Cancellation(encoding="rung5", min_amp_overlap=0.5)` is
  run on a 2-feature `Rung5(n_amp_qubits=3)` dictionary
- **THEN** the returned solution satisfies
  `rung5_amp_overlap_squared(...) ≥ 0.5` at the chosen knobs

### Requirement: Rung5 structural_floor reduces to the MPS-phase-only floor

For `encoding="rung5"`, `Cancellation.structural_floor` SHALL be the
same expression as for Rung3 and Rung4: the MPS-phase-only floor of
the gram entry evaluated at the target pair's (α, β, γ).

The floor SHALL NOT depend on `n_amp_qubits` — it is a property of
the MPSRung1 core only.

#### Scenario: structural_floor matches MPSRung1 for any k

- **WHEN** `Cancellation(encoding="rung5", dictionary=d).structural_floor()`
  is computed for `d.encoding == Rung5(n_amp_qubits=k)` for any k
- **THEN** the result equals the MPSRung1 floor on the same (α, β, γ)
  to 1e-12 absolute tolerance

### Requirement: Cancellation reads k from the dictionary, not as a separate parameter

`Cancellation` SHALL NOT accept a separate `n_amp_qubits` or `k`
parameter. The value SHALL be derived exclusively from
`dictionary.encoding.n_amp_qubits` to prevent drift between the
optimiser shape and the dictionary's actual amp width.

#### Scenario: explicit k parameter rejected

- **WHEN** `Cancellation(encoding="rung5", n_amp_qubits=3, ...)` is
  attempted (passing k as a kwarg)
- **THEN** a `TypeError` is raised (no such kwarg in the signature)
