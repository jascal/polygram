# analysis Specification

## Purpose

Pre-encoding triage of SAE feature subsets using only the rung-1
closed-form Gram. Lets researchers score and compare candidate
subsets before committing to encoding or simulation: per-pair
`(M, V, structural_floor, cancellation_gap)`, per-feature
sensitivity, and an aggregate `encoding_suitability_score`. No
quantum simulation — `O(n_features)` Gram evaluations only.
## Requirements
### Requirement: predict_cancellation_depth returns per-pair triage predictions

The function `polygram.analysis.predict_cancellation_depth` SHALL
build a rung-1 Polygram `Dictionary` via `polygram.from_sae_lens` and
return a `TriagePrediction` whose `pairs` list contains one
`PairPrediction` per ordered pair `(i, j)` with `i < j`. The function
accepts the SAE record dict, the selected feature ids, and any
keyword arguments forwarded to `from_sae_lens`.

Each `PairPrediction` SHALL expose:

- `feature_a`, `feature_b` — feature names
- `cluster_a`, `cluster_b` — cluster names
- `current_overlap` — `|<i|j>|²` at the dictionary's current φs
  (equivalent to `m_zero` when all φs are zero)
- `m_pi` — `|<i|j>|²` evaluated with `φ_i = π` (and `φ_j` left
  unchanged); by symmetry this equals the value with `φ_j = π`,
  `φ_i = 0`, since the squared overlap depends only on `δ = φ_i − φ_j`
- `M` — `(current_overlap + m_pi) / 2`
- `V` — `(current_overlap − m_pi) / 2` (signed)
- `structural_floor` — `min(current_overlap, m_pi)`, equivalent to
  `M − |V|`
- `cancellation_gap` — `current_overlap − structural_floor`
- `is_cross_cluster` — derived property, `cluster_a != cluster_b`

The implementation SHALL evaluate exactly `n_features + 1` Gram
configurations: one all-zero baseline plus one with `φ_i = π` per
feature. It SHALL NOT call any q-orca simulator or build any quantum
state.

#### Scenario: per-pair fields agree with the closed-form Gram

- **WHEN** `predict_cancellation_depth(toy_sae, [0, 1, 4, 5])` is
  called on the test fixture
- **THEN** every `PairPrediction.current_overlap` equals
  `|dictionary.gram()[i, j]|²` to within 1e-9
- **AND** every `PairPrediction.structural_floor` equals
  `min(current_overlap, m_pi)` exactly
- **AND** every `PairPrediction.cancellation_gap` is non-negative

### Requirement: feature_sensitivity reports mean swing magnitude per feature

`polygram.analysis.feature_sensitivity(...)` SHALL return a
`dict[str, float]` mapping each selected feature's name to the mean
`|V_ij|` across pairs containing that feature. The dict SHALL contain
exactly one entry per selected feature.

#### Scenario: sensitivity dict is keyed by feature name

- **WHEN** `feature_sensitivity(toy_sae, [0, 1, 4, 5])` is called
- **THEN** the returned dict has exactly four keys, each matching a
  built feature's `name`, and each value is non-negative

### Requirement: encoding_suitability_score lives in [0, 1]

`polygram.analysis.encoding_suitability_score(...)` SHALL return a
single `float` in `[0.0, 1.0]` computed as
`mean_cancellation_gap × min_pairwise_separation`, where
`min_pairwise_separation = 1 − max_pair_current_overlap`. Both
factors live in `[0, 1]` so the product does too. Higher is better.

The full formula SHALL be exposed as a module-level constant
`SUITABILITY_FORMULA: str` and re-emitted on the `TriagePrediction`
via the `suitability_formula` field, so consumers (CLI report,
notebooks) can quote the definition without duplicating it.

#### Scenario: score is a number in the unit interval

- **WHEN** `encoding_suitability_score(toy_sae, [0, 1, 4, 5])` is
  called
- **THEN** the returned value is a `float` with
  `0.0 <= value <= 1.0`

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

