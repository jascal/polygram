## MODIFIED Requirements

### Requirement: render_report emits a markdown triage report

`polygram.analysis.render_report(prediction) -> str` SHALL return a
markdown document that contains:

- A title and one-line summary stating the suitability score
- A "Per-pair predictions" table with columns
  `(feature_a, feature_b, current_overlap, structural_floor,
  cancellation_gap, is_cross_cluster)`, sorted by descending
  `cancellation_gap`
- A "Per-feature sensitivity" section with one row per feature
- A "Choosing knobs" section that quotes the module-level
  `KNOB_SELECTION_GUIDANCE` constant
- A "Suitability formula" footer that quotes
  `SUITABILITY_FORMULA`

The report SHALL be deterministic given the input (no timestamps,
no random ordering).

`KNOB_SELECTION_GUIDANCE` SHALL be a module-level string constant in
`polygram.analysis` whose content names, at minimum, the following
empirical findings already captured in archived OpenSpec changes and
research notes:

- The default `<feature>.phi` knob is the "last-layer Rz" axis on
  both encodings.
- Multi-feature binding is preferred via the cluster-shared
  grammar (`<cluster>.phi`, `<cluster>.theta[r,d,q]`) rather than a
  list of per-feature paths; bit-for-bit Gram preservation is
  scoped to `MPSRung1 <cluster>.phi`, while HEA cluster-shared
  paths ship as a search-space dimensionality reduction.
- Per-feature θ on diverse-sibling HEA fixtures is the
  cluster-shatterer (cite: 4-θ Ry experiment that drove the
  `(dog_poodle, bird_hawk)` overlap to ≈0 while inverting
  `(dog_poodle, dog_beagle)` from `0.9999 → 0.5735`).
- HEA Rz at depth 0 has zero leverage on `|0⟩` initial states.
- Pure-phase search has a structural floor at `M − |V|`; piercing
  it requires β/α/γ adjustment or a richer encoding.
- Per-feature sensitivity ranking is two lines in user code; no
  callable helper is provided.

`KNOB_SELECTION_GUIDANCE` SHALL be re-exported from
`polygram.analysis` so callers can quote it without reaching into
the renderer.

#### Scenario: report contains all required sections

- **WHEN** `render_report(predict_cancellation_depth(toy_sae, [0, 1,
  4, 5]))` is called
- **THEN** the returned string contains the headings "Per-pair
  predictions", "Per-feature sensitivity", "Choosing knobs", and
  "Suitability formula", and includes every selected feature's name
  at least once

#### Scenario: report quotes the knob-selection guidance constant

- **WHEN** `render_report(predict_cancellation_depth(toy_sae, [0, 1,
  4, 5]))` is called
- **THEN** the returned string contains the substring
  `KNOB_SELECTION_GUIDANCE`'s text (or a stable identifying
  substring from it, e.g. `"cluster-shatterer"`)
