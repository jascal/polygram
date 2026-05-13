## MODIFIED Requirements

### Requirement: EpochCompressor's panel decomposition consumes ClusteredDictionary

`polygram.compression.epoch.EpochCompressor`'s internal panel decomposition (`_select_panels`, `_validate_panels`, `_synthesize_validation_report`) SHALL be re-expressed as consumers of `polygram.clustered.ClusteredDictionary`. The external behaviour of `EpochCompressor.run()` SHALL be byte-identical to the pre-refactor implementation on every shipped fixture and every seed.

The external surface (`EpochCompressor` dataclass fields, the `EpochResult` shape, the `EpochReport` shape) SHALL be unchanged. This is a refactor, not a behaviour change.

A differential regression test SHALL pin the byte-identical invariant by comparing the post-refactor `EpochResult` against a frozen reference produced by the pre-refactor path on the bundled GPT-2-small SAE fixture.

#### Scenario: existing compression test suite passes unchanged

- **WHEN** the full `tests/test_compression*.py` test suite is run against the post-refactor implementation
- **THEN** every test passes without modification

#### Scenario: differential test pins byte-identity

- **WHEN** the differential regression test compares the post-refactor `EpochResult` against the frozen pre-refactor reference on the bundled fixture with seed 0
- **THEN** the two results are byte-identical (every field equals to bit precision for numerics, set-equality for collections)

#### Scenario: panel selection delegates to ClusteredDictionary

- **WHEN** `EpochCompressor._select_panels` is invoked during a compression iteration
- **THEN** the panel construction internally goes through `ClusteredDictionary.from_sae_state(... , block_formation=BlockFormation(strategy="cosine", ...))` rather than the pre-refactor inline implementation
