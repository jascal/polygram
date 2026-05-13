## ADDED Requirements

### Requirement: `EpochCompressor` accepts a configurable encoding

`polygram.compression.epoch.EpochCompressor` SHALL accept an `encoding` constructor parameter of type `MPSRung1 | HEA_Rung2 | Rung3 | Rung4 | None`, defaulting to `None`.

When `None`, `__post_init__` SHALL resolve the parameter to `MPSRung1()` so the default behaviour is byte-identical to the pre-change pipeline.

When non-`None`, the supplied encoding SHALL be threaded through to the per-iteration `ClusteredDictionary.from_compression_panels` call inside `EpochCompressor.run`.

The supplied encoding's `max_features` attribute SHALL be a positive integer; values less than 2 SHALL be rejected with `ValueError` at construction time.

#### Scenario: default `encoding=None` resolves to `MPSRung1()`

- **WHEN** `EpochCompressor(sae_checkpoint=..., prompts=..., layer=...)` is constructed with no explicit `encoding` argument
- **THEN** `self.encoding` is set to `MPSRung1()` after `__post_init__` completes
- **AND** the resulting `EpochResult` is byte-identical to the pre-change pipeline (locked by the existing differential regression test against the frozen reference)

#### Scenario: explicit `encoding=MPSRung1()` matches default

- **WHEN** `EpochCompressor` is constructed with `encoding=MPSRung1()` explicitly
- **THEN** the resulting `EpochResult` is byte-identical to the same constructor with `encoding=None`

#### Scenario: explicit `encoding=Rung3()` enables larger panels

- **WHEN** `EpochCompressor` is constructed with `encoding=Rung3()` and run on a fixture engineered with redundancy clusters of >8 features
- **THEN** at least one selected panel has more than 8 feature IDs
- **AND** every selected panel has at most 16 feature IDs (`Rung3.max_features`)

#### Scenario: encoding with degenerate `max_features` rejected

- **WHEN** an encoding object whose `max_features` is less than 2 is passed
- **THEN** `__post_init__` raises `ValueError` with a message naming the offending value

## MODIFIED Requirements

### Requirement: Panel selection respects encoding feature cap

`polygram.compression.epoch._select_panels` SHALL accept a `max_panel_size: int` keyword argument. The neighbour-count cap inside the greedy seeded-coverage loop SHALL be `max_panel_size - 1` (so panels reach at most `max_panel_size` features including the anchor).

`EpochCompressor.run` SHALL pass `max_panel_size=self.encoding.max_features` to `_select_panels`.

The hardcoded `len(neighbours) >= 7` (the implicit 8-feature MPSRung1 cap minus the anchor) SHALL be replaced.

#### Scenario: panel size scales with encoding

- **WHEN** `_select_panels` is called with `max_panel_size=16`
- **AND** the eligible pool has more than 16 features mutually cosine-similar above the threshold
- **THEN** the returned panels each contain at most 16 feature IDs

#### Scenario: panel size at MPSRung1 default unchanged

- **WHEN** `_select_panels` is called with `max_panel_size=8` (the `MPSRung1.max_features` value)
- **THEN** the returned panels are byte-identical to those produced by the pre-change hardcoded-`7`-neighbour-cap implementation on the same inputs

### Requirement: Internal `ClusteredDictionary` view uses configured encoding

`EpochCompressor.run` SHALL pass `encoding=self.encoding` to `ClusteredDictionary.from_compression_panels`. The previous hardcoded `encoding=MPSRung1()` literal at that call site SHALL be removed.

The per-iteration `ClusteredDictionary` SHALL carry the configured encoding in every block's `Dictionary`. Downstream `_validate_panels` and `_synthesize_validation_report` consume the `ClusteredDictionary` and SHALL continue to function unchanged — they don't query the encoding directly.

#### Scenario: per-iteration ClusteredDictionary carries the configured encoding

- **WHEN** `EpochCompressor` is constructed with `encoding=Rung3()` and `run()` is called
- **AND** the internal `ClusteredDictionary` view is captured (e.g., via test instrumentation on `_validate_panels`)
- **THEN** every block in the captured `ClusteredDictionary` reports `encoding=Rung3()`
