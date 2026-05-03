## ADDED Requirements

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
- A "Suitability formula" footer that quotes
  `SUITABILITY_FORMULA`

The report SHALL be deterministic given the input (no timestamps,
no random ordering).

#### Scenario: report contains all required sections

- **WHEN** `render_report(predict_cancellation_depth(toy_sae, [0, 1,
  4, 5]))` is called
- **THEN** the returned string contains the headings "Per-pair
  predictions", "Per-feature sensitivity", and "Suitability
  formula", and includes every selected feature's name at least once
