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
Dictionary` method that returns a new `Dictionary` with one named
parameter slot mutated. The grammar of `path` is:

- `<feature_name>.phi` — sets `Feature.phi` on the named feature.
  Accepted on both `MPSRung1` and `HEA_Rung2` encodings.
- `<feature_name>.theta[r,d,q]` — sets the `(r, d, q)` slot of the
  named feature's θ tensor. Accepted only on `HEA_Rung2`. When the
  feature's `theta` is `None`, the helper SHALL first materialize
  the default tensor via `_default_hea_theta(feature, encoding)`,
  copy it, then set the named slot.

`with_knob` SHALL raise `ValueError` for:

- a feature name not present in `dictionary.features`,
- a `.theta[...]` path on `MPSRung1`,
- an `(r, d, q)` triple outside `encoding.theta_shape`,
- any other malformed path that does not match the grammar above.

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
  (`<feature>.phi` or `<feature>.theta[r,d,q]`)

