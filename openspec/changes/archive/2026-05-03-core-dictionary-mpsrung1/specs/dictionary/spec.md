## ADDED Requirements

### Requirement: Feature and Dictionary declarations

Polygram SHALL expose a `Feature` dataclass with fields `name: str`,
`cluster: str`, `alpha: float`, `beta: float`, `gamma: float`, `phi: float`,
and a `Dictionary` dataclass with fields `name: str`,
`features: list[Feature]`, `hierarchy: dict[str, list[str]]`. Each feature
MUST belong to exactly one cluster declared in `hierarchy`.

#### Scenario: feature declared outside hierarchy raises

- **WHEN** a `Dictionary` is constructed with a `Feature` whose `cluster`
  does not appear as a key in `hierarchy`
- **THEN** `__post_init__` raises `ValueError` naming the offending
  feature and cluster

#### Scenario: feature listed in two clusters raises

- **WHEN** the `hierarchy` value lists the same feature name under two
  cluster keys
- **THEN** `__post_init__` raises `ValueError` naming the duplicate

#### Scenario: well-formed dictionary exposes feature_index

- **WHEN** a valid 4-feature, 2-cluster `Dictionary` is constructed
- **THEN** `Dictionary.feature_index("Dog_Beagle")` returns the integer
  index of `Dog_Beagle` in the `features` list, suitable for indexing
  into the Gram matrix

### Requirement: Default-angle assignment helper

Polygram SHALL expose an API that, given a hierarchy, assigns default
`β` values evenly spread in `[-0.5, 0.5]` per cluster, with `α = γ = φ = 0`
on every feature.

#### Scenario: two clusters split β symmetrically

- **WHEN** `Dictionary.with_default_angles({"dogs": [...], "birds": [...]})`
  is called
- **THEN** every feature in `dogs` has `β = -0.5` and every feature in
  `birds` has `β = +0.5`

### Requirement: Analytic Gram via q-orca

Polygram SHALL expose `Dictionary.gram() -> np.ndarray` returning the
analytic Gram matrix produced by
`q_orca.compiler.concept_gram_mps.compute_concept_gram_mps` using
`form="preparation"` (avoiding the inverse-form `Rz` symmetry break
documented in q-orca-lang's `larql-animals-interference.q.orca.md`).

#### Scenario: 4-feature dictionary reproduces published Gram tiers

- **WHEN** `Dictionary.gram()` is called on a 4-feature dictionary with
  α=γ=0, β ∈ {-0.5, +0.5}, φ ∈ {0, π/2}
- **THEN** the off-diagonal magnitudes |⟨c_i|c_j⟩|² fall into the three
  tiers 0.8851 / 0.6816 / 0.5931 within 1e-4, matching
  `larql-animals-interference.q.orca.md`

## ADDED Requirements

### Requirement: MPSRung1 encoding marker

Polygram SHALL expose `MPSRung1(bond_dim=2, phase_knobs=True)` as a
config marker for the rung-1 MPS encoding. v0 supports `bond_dim=2`
only.

#### Scenario: bond_dim != 2 rejected

- **WHEN** `MPSRung1(bond_dim=3)` is constructed
- **THEN** `__post_init__` raises `ValueError` mentioning that v0
  supports rung-1 (χ=2) only
