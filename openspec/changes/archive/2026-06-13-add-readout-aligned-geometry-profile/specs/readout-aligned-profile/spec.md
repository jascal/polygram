# readout-aligned-profile Specification (delta)

## ADDED Requirements

### Requirement: Readout-aligned geometric profile

Polygram SHALL provide a `readout-aligned` geometric profile (registered alongside `clustered` /
`uniform_sphere`) that builds the dictionary geometry — clustering, cluster centroids, residual-variance
fidelity, and γ assignment — on feature directions **projected onto the model's readout subspace**, not on the
raw decoder vectors.

The readout subspace SHALL be the top-`r` right singular directions of `gain ⊙ U` (`U`: the host unembed,
`(vocab, d_model)`; `gain`: the final-norm gain, `(d_model,)`, default ones), and the geometry SHALL be built
on `projections @ Rᵀ`. The projection SHALL occur **inside** the profile's knob-assignment strategy, leaving
the `from_sae_lens` call site and the `GeometricProfile`/registry plumbing structurally unchanged.

`get_profile("readout-aligned")` SHALL resolve to this profile. The `clustered` profile SHALL remain the
default; a run that does not select `readout-aligned` SHALL be byte-identical to current behaviour.

#### Scenario: readout-aligned geometry differs from raw-decoder geometry

- **GIVEN** a feature set and a host unembed `U` whose readout subspace re-orders the raw decoder directions
- **WHEN** a dictionary is built with `profile="readout-aligned"` and again with `profile="clustered"`
- **THEN** the two SHALL produce different cluster assignments / β/γ knobs (the geometry is built on a
  different basis), while the `clustered` result is unchanged from the pre-change behaviour

#### Scenario: deterministic given the readout inputs

- **GIVEN** identical `(projections, U, gain, seed)`
- **WHEN** the readout-aligned profile assigns knobs
- **THEN** the assignment SHALL be identical across runs

### Requirement: `from_sae_lens` accepts the readout geometry

`from_sae_lens` SHALL accept optional `u_matrix: np.ndarray | None` (`(vocab, d_model)`) and
`gain: np.ndarray | float | None` (default ones). When the resolved profile is `readout-aligned`, `u_matrix`
SHALL be **required** and 2-D with its second axis equal to the feature dimension; absent or malformed,
construction SHALL raise a clear `ValueError`. Polygram SHALL NOT load a host model itself to obtain `U`/`gain`
— the caller supplies them (keeping the core dependency-light).

#### Scenario: readout-aligned profile without a host unembed is rejected

- **GIVEN** `profile="readout-aligned"` and `u_matrix=None`
- **WHEN** `from_sae_lens(...)` is called
- **THEN** it SHALL raise `ValueError` naming the missing `u_matrix`

#### Scenario: default profile needs no readout inputs

- **GIVEN** no `profile` (or `profile="clustered"`) and no `u_matrix`
- **WHEN** `from_sae_lens(...)` is called
- **THEN** it SHALL succeed and produce a result byte-identical to the pre-change behaviour
