## ADDED Requirements

### Requirement: `EpochReport` surfaces `redundancy_ratio` as a first-class diagnostic

`polygram.compression.epoch_report.EpochReport` SHALL expose a `redundancy_ratio: float` field whose value equals `n_features_zeroed_total / n_features_input` (with the divisor exposed as a sibling `n_features_input: int` field so the ratio is computable without external context).

The ratio is the load-bearing diagnostic for whether a compression run helped or hurt downstream behavioural metrics — observed in the 60–74% range across recent Gemma-2 SAE runs, correlating strongly with KL outcomes (helpful on sparse SAEs, catastrophic on dense ones). It SHALL surface in both the JSON serialisation and the human-readable CLI report (`polygram analyze`).

The serialisation schema bumps `schema_version`. `EpochReport.from_dict` / `from_json` SHALL accept older payloads — either by computing the ratio from any available divisor or by setting `redundancy_ratio` to `None` when no recovery path exists — so existing serialised reports continue to load.

#### Scenario: redundancy_ratio matches the computed value on a fresh report

- **WHEN** `EpochCompressor.run()` finishes on the toy compression fixture
- **THEN** `report.redundancy_ratio == report.n_features_zeroed_total / report.n_features_input` within `1e-12`

#### Scenario: legacy payload loads without the new fields

- **WHEN** an `EpochReport` JSON written before this change is loaded via `EpochReport.from_json`
- **THEN** the load succeeds without raising, and `redundancy_ratio` is either computed from any available divisor (if recoverable) or is `None` (when the divisor is not present in the payload)

#### Scenario: CLI surfaces the ratio

- **WHEN** `polygram analyze ...` is invoked against a fixture that exercises compression
- **THEN** the human-readable output includes a `redundancy_ratio` line alongside the existing `n_features_zeroed_total` line
