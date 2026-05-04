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

### Requirement: Dictionary.with_knob applies a single named slot

`Dictionary` SHALL expose a `with_knob(path: str, value: float) ->
Dictionary` method that returns a new `Dictionary` with the named
parameter slot mutated. The grammar of `path` is:

- `<feature_name>.phi` — sets `Feature.phi` on the named feature.
  Accepted on both `MPSRung1` and `HEA_Rung2` encodings.
- `<feature_name>.theta[r,d,q]` — sets the `(r, d, q)` slot of the
  named feature's θ tensor. Accepted only on `HEA_Rung2`. When the
  feature's `theta` is `None`, the helper SHALL first materialize
  the default tensor via `_default_hea_theta(feature, encoding)`,
  copy it, then set the named slot.
- `<cluster_name>.phi` — *cluster-shared* φ. Sets `Feature.phi` to
  the same value on every feature listed in
  `dictionary.hierarchy[cluster_name]`. Accepted on both encodings.
- `<cluster_name>.theta[r,d,q]` — *cluster-shared* θ slot. Sets the
  `(r, d, q)` slot of every member feature's θ tensor to the same
  value. Members whose `theta is None` are first materialized via
  `_default_hea_theta`. Accepted only on `HEA_Rung2`.

The leading identifier SHALL resolve as a feature name first; if
no feature matches, the identifier resolves as a cluster name.
Construction-time uniqueness (see "Feature and cluster names SHALL
NOT collide") guarantees the resolution is unambiguous.

`with_knob` SHALL raise `ValueError` for:

- a leading identifier that matches neither a feature name nor a
  cluster name; the message SHALL name both candidate spaces,
- a `.theta[...]` path on `MPSRung1`, whether per-feature or
  cluster-shared,
- an `(r, d, q)` triple outside `encoding.theta_shape`,
- any other malformed path that does not match the grammar above.

The cluster-shared application's Gram-preservation guarantee is
**conditional on the encoding and the knob path**:

- **MPSRung1 `<cluster>.phi`** — bit-for-bit preserved (to numeric
  round-off): when every sibling in the cluster shares the same
  pre-mutation `phi`, the cluster-shared write produces the same
  outer `Rz(qs[1], v)` on each branch, and the
  `<U_C a | U_C b> = <a|U_C†U_C|b> = <a|b>` cancellation holds.
- **HEA_Rung2 `<cluster>.theta[r,d,q]` or `<cluster>.phi`** — NOT a
  bit-for-bit invariant in general. Sharing a single rotation slot
  across siblings does not produce identical sibling unitaries
  unless the sibling baselines (every other θ slot) already agree.
  In the degenerate case of fully identical siblings (overlap 1.0),
  the invariant trivially holds; for diverse siblings it does not.
  The HEA cluster-shared paths are a search-space dimensionality
  reduction (one axis per cluster, bounding optimizer leverage),
  not an algebraic preservation guarantee.

#### Scenario: .phi works on both encodings

- **GIVEN** a `Dictionary` with a feature `a`
- **WHEN** `dictionary.with_knob("a.phi", 0.7)` is called on either
  an `MPSRung1`- or `HEA_Rung2`-encoded dictionary
- **THEN** the returned dictionary's `feature("a").phi == 0.7` and
  every other feature is unchanged

#### Scenario: .theta[r,d,q] rejected on MPS rung-1

- **GIVEN** a `Dictionary(encoding=MPSRung1())` with a feature `a`
- **WHEN** `dictionary.with_knob("a.theta[0,0,1]", 0.3)` is called
- **THEN** a `ValueError` is raised naming the encoding and the
  unsupported `.theta` path

#### Scenario: .theta[r,d,q] writes a single slot on HEA

- **GIVEN** a `Dictionary(encoding=HEA_Rung2(depth=2))` with a
  feature `a` whose `theta is None`
- **WHEN** `dictionary.with_knob("a.theta[1,0,1]", 0.5)` is called
- **THEN** the returned dictionary's `feature("a").theta` has shape
  `(2, 2, 3)`, the `(1, 0, 1)` slot equals `0.5`, and every other
  slot equals the corresponding entry of
  `_default_hea_theta(original_feature, encoding)`

#### Scenario: out-of-range slot rejected

- **GIVEN** a `Dictionary(encoding=HEA_Rung2(depth=2))` (so
  `theta_shape == (2, 2, 3)`)
- **WHEN** `dictionary.with_knob("a.theta[2,0,0]", 0.0)` is called
- **THEN** a `ValueError` is raised naming the offending triple
  `(2, 0, 0)` and the encoding's `theta_shape`

#### Scenario: malformed path rejected

- **WHEN** `dictionary.with_knob("a.theta", 0.0)` or
  `dictionary.with_knob("a", 0.0)` is called
- **THEN** a `ValueError` is raised describing the expected grammar
  (`<feature>.phi`, `<feature>.theta[r,d,q]`,
  `<cluster>.phi`, or `<cluster>.theta[r,d,q]`)

#### Scenario: cluster-shared .phi fans out across siblings

- **GIVEN** a `Dictionary` with `hierarchy = {"dogs":
  ["dog_poodle", "dog_beagle"], "birds": ["bird_hawk"]}`
- **WHEN** `dictionary.with_knob("dogs.phi", 0.7)` is called
- **THEN** the returned dictionary's `feature("dog_poodle").phi ==
  0.7`, `feature("dog_beagle").phi == 0.7`, and
  `feature("bird_hawk").phi` is unchanged

#### Scenario: cluster-shared .theta[r,d,q] fans out on HEA

- **GIVEN** a `Dictionary(encoding=HEA_Rung2(depth=2))` with two
  features in cluster `"dogs"` (both with `theta is None`) and one
  feature in cluster `"birds"`
- **WHEN** `dictionary.with_knob("dogs.theta[0,0,0]", 1.5)` is
  called
- **THEN** every feature in `hierarchy["dogs"]` has a materialized
  θ tensor whose `(0, 0, 0)` slot equals `1.5` and every other slot
  equals the corresponding entry of
  `_default_hea_theta(original_feature, encoding)`; the bird
  feature is unchanged

#### Scenario: cluster-shared MPS phi preserves within-cluster Gram

- **GIVEN** a `Dictionary(encoding=MPSRung1())` with at least one
  cluster of size ≥ 2 in which every sibling shares the same
  pre-mutation `phi` value
- **WHEN** the dictionary is mutated by `with_knob("<cluster>.phi",
  v)` for any value `v` in `(0, 2π)`
- **THEN** for every pair of features `a, b` both in the named
  cluster, `mutated.gram()[i_a, i_b]` equals
  `original.gram()[i_a, i_b]` to within numeric round-off
  (`abs(after − before) < 1e-9`)

#### Scenario: HEA cluster-shared path may shift within-cluster Gram

- **GIVEN** an HEA-encoded `Dictionary` with at least one cluster of
  size ≥ 2 whose sibling features have any per-feature parameter
  variation (e.g. distinct `alpha` or `gamma`)
- **WHEN** the dictionary is mutated by `with_knob("<cluster>.<slot>",
  v)` for a value `v` in the per-knob bounds
- **THEN** the call SHALL succeed (cluster-shared paths are accepted
  on HEA), but the within-cluster Gram entries MAY differ from the
  pre-mutation values. The implementation makes no bit-for-bit
  preservation guarantee in this regime; cluster-shared on HEA is a
  search-space dimensionality reduction, not an algebraic invariant.

#### Scenario: unknown identifier rejected

- **GIVEN** a `Dictionary` whose features are `["dog_poodle"]` and
  whose hierarchy keys are `["dogs"]`
- **WHEN** `dictionary.with_knob("cats.phi", 0.0)` is called
- **THEN** a `ValueError` is raised naming the identifier `"cats"`
  and listing both the available feature names and cluster names

### Requirement: Feature and cluster names SHALL NOT collide

`Dictionary.__post_init__` SHALL raise `ValueError` when any cluster
key in `hierarchy` matches any feature name in `features`. The error
message SHALL name the colliding identifier so callers can rename
either the feature or the cluster.

This invariant is required for unambiguous parsing of cluster-shared
knob paths (`<cluster>.phi`, `<cluster>.theta[r,d,q]`); without it,
`with_knob` cannot decide whether `"dogs.phi"` means "set the feature
named `dogs`" or "set every feature in the cluster `dogs`".

#### Scenario: feature/cluster name collision rejected

- **WHEN** a `Dictionary` is constructed with a `features` list that
  contains a feature named `"dogs"` and a `hierarchy` that also has a
  cluster key `"dogs"`
- **THEN** `__post_init__` raises `ValueError` naming the
  identifier `"dogs"`

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

