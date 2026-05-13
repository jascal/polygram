## MODIFIED Requirements

### Requirement: `_validate_panels` consumes `ClusteredDictionary`

`polygram.compression.epoch._validate_panels` SHALL accept a `ClusteredDictionary` as its first argument instead of a `list[Panel]`. Its body SHALL iterate `clustered.blocks` and run per-block validation via the existing `BehaviouralValidator.validate` path.

The per-block validation logic SHALL be byte-identical to the pre-refactor per-panel validation: same per-block forward pass, same per-pair confirmation thresholds, same numeric output.

The return type SHALL remain `list[ValidationReport]` with one entry per block, in the same order as `clustered.blocks`.

#### Scenario: per-block validation results match pre-refactor per-panel results

- **WHEN** `_validate_panels(clustered, ...)` is called on a `ClusteredDictionary` constructed via `ClusteredDictionary.from_compression_panels(panels, ...)`
- **THEN** the returned `list[ValidationReport]` is element-equal to what the pre-refactor `_validate_panels(panels, ...)` would have returned on the same `panels`

#### Scenario: block order matches panel order

- **WHEN** a `ClusteredDictionary` is constructed via `from_compression_panels` from an ordered list of panels
- **THEN** `clustered.blocks[k]` corresponds to `panels[k]` and `_validate_panels`'s returned reports preserve that ordering

### Requirement: `_synthesize_validation_report` consumes `ClusteredDictionary` + per-block reports

`polygram.compression.epoch._synthesize_validation_report` SHALL accept `(clustered: ClusteredDictionary, block_reports: list[ValidationReport], sae_checkpoint: Path)` instead of `(panels: list[Panel], per_panel_reports: list[ValidationReport], sae_checkpoint: Path)`.

The synthesis logic SHALL produce a cross-block `ValidationReport` byte-identical to the pre-refactor cross-panel `ValidationReport` on the same inputs (same confirmed-pair set, same candidate-pair set, same fields).

#### Scenario: cross-block synthesis matches cross-panel synthesis

- **WHEN** `_synthesize_validation_report(clustered, block_reports, sae_checkpoint)` is called on a `ClusteredDictionary` derived from a panel list and the corresponding `block_reports`
- **THEN** the returned cross-block `ValidationReport` is field-equal to what the pre-refactor `_synthesize_validation_report(panels, per_panel_reports, sae_checkpoint)` would have returned

### Requirement: `EpochCompressor.run` constructs `ClusteredDictionary` per iteration

`polygram.compression.epoch.EpochCompressor.run` SHALL construct a `ClusteredDictionary` per iteration by calling `ClusteredDictionary.from_compression_panels(panels=..., state_dict=current_state, encoding=MPSRung1(), name=f"<stem>_iter<i>")` immediately after `_select_panels` returns.

The constructed `ClusteredDictionary` SHALL be passed to `_validate_panels` and `_synthesize_validation_report` in place of the raw panel list.

`EpochCompressor.run`'s external surface — its constructor fields, its `EpochResult` return type, its `EpochReport` shape, the convergence-state determination, the per-iteration disk artifacts — SHALL remain byte-identical to the pre-refactor implementation on every shipped fixture and every seed.

#### Scenario: byte-identical EpochResult on bundled fixture

- **WHEN** `EpochCompressor.run` is invoked on the bundled `tests/fixtures/toy_sae.json` fixture with seed 0 and the shipped defaults, post-refactor
- **THEN** the resulting `EpochResult` is bit-equal to the frozen reference captured at `tests/compression/data/epoch_result_reference.json` (numeric fields compared without tolerance; collections compared by element equality)

#### Scenario: convergence-state determination unchanged

- **WHEN** the iteration loop reaches a `_REASON_STABLE_CLUSTERS` / `_REASON_QUALITY_BREACHED` / `_REASON_MAX_ITERATIONS` / `_REASON_NO_PRIORITY_CANDIDATES` state on the bundled fixture
- **THEN** the convergence reason is identical to the pre-refactor reason at the same iteration index

#### Scenario: existing compression test suite passes unchanged

- **WHEN** `tests/test_compression*.py` and `tests/compression/` are run against the post-refactor implementation
- **THEN** every test passes without modification (zero behaviour drift)

### Requirement: `_select_panels` is unchanged

`polygram.compression.epoch._select_panels` SHALL be untouched by this change. Its signature, body, return type, and algorithm (priority-driven seeded coverage with visit cap, coverage target, anchor-only fallback, ≤7 neighbour cap) all stay exactly as shipped pre-refactor.

This is the load-bearing risk-mitigation: the priority-driven algorithm is preserved verbatim so behaviour-divergence at the partition-selection level is impossible by construction.

#### Scenario: _select_panels output is bit-identical

- **WHEN** `_select_panels(...)` is called with identical arguments pre- and post-refactor
- **THEN** the returned `(list[Panel], coverage)` tuple is bit-identical
