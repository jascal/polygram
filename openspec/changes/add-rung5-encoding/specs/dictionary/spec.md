## ADDED Requirements

### Requirement: Feature carries an optional amp_knobs tuple for product-amp encodings

`polygram.dictionary.Feature` SHALL accept an optional
`amp_knobs: tuple[tuple[float, float], ...] = ()` field. Each entry
is a `(theta, psi)` pair of floats, one per amp qubit.

The empty-tuple default SHALL apply to every non-Rung5 encoding
(MPSRung1, HEA_Rung2, Rung3, Rung4). Those encodings SHALL NOT read
from `amp_knobs` and SHALL be unaffected by its value. Rung3 and
Rung4 continue to read their existing `theta_amp`, `psi_aux`,
`theta_amp_b`, `psi_amp_b` fields.

`amp_knobs` SHALL be persisted in `Dictionary` serialisation when
non-empty; the empty-tuple default SHALL serialise as omitted /
absent to preserve round-trip identity for existing Rung3/Rung4
dictionaries.

#### Scenario: Feature constructible without amp_knobs

- **WHEN** `Feature(name="f0", cluster="c0", beta=0.1)` is
  constructed
- **THEN** the result has `amp_knobs == ()`

#### Scenario: Feature constructible with amp_knobs

- **WHEN** `Feature(name="f0", cluster="c0", beta=0.1,
  amp_knobs=((0.1, 0.2), (0.3, 0.4)))` is constructed
- **THEN** the result has `amp_knobs == ((0.1, 0.2), (0.3, 0.4))`

#### Scenario: existing Rung3/Rung4 dictionaries unaffected

- **WHEN** an existing Rung3 or Rung4 dictionary fixture is loaded
  whose features omit `amp_knobs`
- **THEN** every feature's `amp_knobs == ()` and
  `dictionary.gram()` returns the same values it did before this
  change

### Requirement: Dictionary validates amp_knobs length for Rung5 encodings

`Dictionary.__post_init__` SHALL, when
`isinstance(self.encoding, Rung5)`, validate that every feature's
`amp_knobs` length equals `self.encoding.n_amp_qubits`. The
validation SHALL also confirm each entry is a 2-tuple of `(float,
float)`.

On mismatch, `__post_init__` SHALL raise `ValueError` mentioning
the feature's name, the expected length (`n_amp_qubits`), and the
actual length.

This validation SHALL NOT run for non-Rung5 encodings; `amp_knobs`
on a Rung3/Rung4/MPSRung1/HEA dictionary SHALL be ignored regardless
of its length.

#### Scenario: Rung5 dictionary with correctly-shaped amp_knobs

- **WHEN** a `Dictionary(encoding=Rung5(n_amp_qubits=2),
  features=[Feature(..., amp_knobs=((0, 0), (0, 0))), ...])` is
  constructed
- **THEN** construction succeeds

#### Scenario: Rung5 dictionary with mis-sized amp_knobs

- **WHEN** a `Dictionary(encoding=Rung5(n_amp_qubits=3),
  features=[Feature(name="f0", ..., amp_knobs=((0, 0), (0, 0))),
  ...])` is constructed (length 2, expected 3)
- **THEN** a `ValueError` is raised mentioning `"f0"`, expected
  length 3, and actual length 2

#### Scenario: non-Rung5 dictionary with non-empty amp_knobs is accepted

- **WHEN** a `Dictionary(encoding=MPSRung1(), features=[Feature(...,
  amp_knobs=((0.1, 0.2),)), ...])` is constructed
- **THEN** construction succeeds and `dictionary.gram()` returns the
  same value it would with `amp_knobs=()` (MPSRung1 ignores
  `amp_knobs`)

### Requirement: Feature.with_default_amp_knobs helper pads to encoding width

`polygram.dictionary.Feature.with_default_amp_knobs(encoding) ->
Feature` SHALL return a copy of the feature with `amp_knobs`
expanded to `((0.0, 0.0),) * encoding.n_amp_qubits` when
`isinstance(encoding, Rung5)`, and SHALL return the feature
unchanged otherwise.

The helper SHALL preserve every other field of the feature
unchanged.

#### Scenario: helper pads amp_knobs to k for Rung5

- **WHEN** `feature.with_default_amp_knobs(Rung5(n_amp_qubits=4))`
  is called on a feature with `amp_knobs == ()`
- **THEN** the returned feature has
  `amp_knobs == ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))`
  and every other field equal to the input feature

#### Scenario: helper is a no-op for non-Rung5 encodings

- **WHEN** `feature.with_default_amp_knobs(MPSRung1())` is called
- **THEN** the returned feature equals the input feature
  (identity-equal field-by-field)

### Requirement: Dictionary.with_knob accepts amp_knobs paths for Rung5

`Dictionary.with_knob` SHALL extend its path grammar with:

- `<feature_name>.amp_knobs[i].theta` — sets the theta entry of the
  i-th amp-knob pair on the named feature.
- `<feature_name>.amp_knobs[i].psi` — sets the psi entry of the
  i-th amp-knob pair on the named feature.
- `<cluster_name>.amp_knobs[i].theta` — cluster-shared theta entry.
- `<cluster_name>.amp_knobs[i].psi` — cluster-shared psi entry.

These paths SHALL be accepted only when the dictionary's encoding
is `Rung5`. On other encodings, `with_knob` SHALL raise `ValueError`
mentioning that `amp_knobs` paths require Rung5.

When the target feature's `amp_knobs` is `()` (the default),
`with_knob` SHALL first materialise the default-padded tuple
(`((0.0, 0.0),) * encoding.n_amp_qubits`) before setting the named
slot.

`i` SHALL be a non-negative integer in `[0, n_amp_qubits)`. Values
outside this range SHALL raise `ValueError` mentioning the
encoding's k and the attempted index.

#### Scenario: with_knob sets amp_knobs slot on Rung5 dictionary

- **WHEN** `dictionary.with_knob("f0.amp_knobs[1].theta", 0.7)` is
  called on a `Rung5(n_amp_qubits=3)` dictionary where feature `f0`
  has `amp_knobs == ()`
- **THEN** the returned dictionary's `f0.amp_knobs` equals
  `((0.0, 0.0), (0.7, 0.0), (0.0, 0.0))`

#### Scenario: with_knob amp_knobs index out of range rejected

- **WHEN** `dictionary.with_knob("f0.amp_knobs[5].theta", 0.7)` is
  called on a `Rung5(n_amp_qubits=3)` dictionary
- **THEN** a `ValueError` is raised mentioning k=3 and the attempted
  index 5

#### Scenario: with_knob amp_knobs path on non-Rung5 dictionary rejected

- **WHEN** `dictionary.with_knob("f0.amp_knobs[0].theta", 0.7)` is
  called on a `Rung4` (or any non-Rung5) dictionary
- **THEN** a `ValueError` is raised mentioning that `amp_knobs`
  paths require Rung5

### Requirement: Dictionary.gram() dispatches to Rung5

`Dictionary.gram()` SHALL, when
`isinstance(self.encoding, Rung5)`, compute the elementwise product
of the MPSRung1-on-(α, β, γ, φ) overlap and the Rung5 amp overlap
(via `rung5_amp_overlap`) for every feature pair.

The output SHALL be a numpy float64 array of shape
`(n_features, n_features)` with real-valued entries equal to
`|⟨feature_a | feature_b⟩|²` under the Rung5 encoding.

#### Scenario: Rung5 gram is symmetric PSD with unit diagonal

- **WHEN** `Dictionary(encoding=Rung5(n_amp_qubits=k), features=[...])
  .gram()` is computed for any valid Rung5 dictionary
- **THEN** the returned matrix is symmetric, has all-ones on the
  diagonal (within 1e-12), and is positive semi-definite
