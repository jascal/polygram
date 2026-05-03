## ADDED Requirements

### Requirement: HEA_Rung2 encoding marker

Polygram SHALL expose `HEA_Rung2(depth, entangler="ring",
rotations=("Ry", "Rz"), tier_separation_bound=0.025, n_qubits=3)`
as a config marker for the rung-2 hardware-efficient ansatz
encoding. `depth` is required; the remaining fields have defaults
matching the q-orca-lang `examples/larql-hea-minimal.q.orca.md`
spike. Construction SHALL validate that `depth >= 1`, `entangler ∈
{"ring", "chain"}`, every rotation ∈ `{"Rx", "Ry", "Rz"}`, and (when
not `None`) `0.0 <= tier_separation_bound <= 1.0`.

#### Scenario: defaults match the q-orca-lang spike

- **WHEN** `HEA_Rung2(depth=3)` is constructed
- **THEN** `entangler == "ring"`, `rotations == ("Ry", "Rz")`,
  `tier_separation_bound == 0.025`, `n_qubits == 3`

#### Scenario: invalid depth rejected

- **WHEN** `HEA_Rung2(depth=0)` is constructed
- **THEN** `__post_init__` raises `ValueError` mentioning the
  `depth >= 1` constraint

#### Scenario: unknown rotation rejected

- **WHEN** `HEA_Rung2(depth=2, rotations=("Ry", "Rq"))` is
  constructed
- **THEN** `__post_init__` raises `ValueError` naming `"Rq"` as
  outside `{"Rx", "Ry", "Rz"}`

#### Scenario: explicit None suppresses invariant emission

- **WHEN** `HEA_Rung2(depth=2, tier_separation_bound=None)` is
  constructed
- **THEN** the field is permitted (validation skipped); downstream
  emitters SHALL omit the `concept_gram_tier_separation` invariant
  for dictionaries using this encoding

### Requirement: Feature carries an optional HEA θ tensor

`Feature` SHALL accept an optional `theta: np.ndarray | None = None`
field. When the surrounding `Dictionary.encoding` is an instance of
`HEA_Rung2`, an explicitly-passed `theta` SHALL have shape
`(|encoding.rotations|, encoding.depth, encoding.n_qubits)`; a
shape mismatch SHALL raise `ValueError` naming the offending feature.

When `theta is None` and the encoding is `HEA_Rung2`, the emitter
SHALL synthesize a default tensor from the existing
`(α, β, γ, φ)` knobs.

#### Scenario: default theta is None

- **WHEN** a `Feature(name="poodle", cluster="dogs", beta=0.1)` is
  constructed
- **THEN** `feature.theta is None`

#### Scenario: explicit theta with wrong shape rejected

- **GIVEN** a `Dictionary` with `encoding=HEA_Rung2(depth=3,
  rotations=("Ry", "Rz"))` (so the expected θ shape is
  `(2, 3, 3)`)
- **WHEN** one of its features is constructed with a `(2, 2, 3)`
  numpy array
- **THEN** `Dictionary.__post_init__` raises `ValueError` naming
  the offending feature and the expected vs actual shape

### Requirement: Dictionary.gram() and tier_separation() dispatch on encoding

`Dictionary.gram()` SHALL dispatch on the type of
`self.encoding`: an `MPSRung1` encoding SHALL invoke
`q_orca.compiler.concept_gram_mps.compute_concept_gram_mps`
(unchanged from `core-dictionary-mpsrung1`); an `HEA_Rung2`
encoding SHALL invoke
`q_orca.compiler.concept_gram_hea.compute_concept_gram_hea`.

A new `Dictionary.tier_separation() -> float | None` method SHALL
return
`q_orca.compiler.concept_gram_hea.compute_tier_separation(
self.gram(), [f.cluster for f in self.features])`. The method
SHALL return `None` when every cluster is a singleton (matching the
helper's contract).

#### Scenario: HEA dictionary returns a complex Gram via compute_concept_gram_hea

- **GIVEN** a `Dictionary(encoding=HEA_Rung2(depth=2))` with two
  features in cluster `s1` and one in cluster `s2`, all with
  default `theta=None`
- **WHEN** `dictionary.gram()` is called
- **THEN** the returned value is a `(3, 3)` complex numpy array
  with `gram[i, i] ≈ 1.0` for every `i`

#### Scenario: tier_separation returns a positive float for tiered fixtures

- **GIVEN** an `HEA_Rung2` dictionary whose intra-cluster pairs
  all have squared overlap ≥ 0.95 and whose cross-cluster pair
  has squared overlap ≤ 0.40
- **WHEN** `dictionary.tier_separation()` is called
- **THEN** the returned value is a positive float ≥ 0.5
