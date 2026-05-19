## ADDED Requirements

### Requirement: `ClusteredDictionary` materialises per-block `Dictionary` instances lazily

`polygram.clustered_dictionary.ClusteredDictionary` SHALL store per-block data as lightweight `BlockView` value types internally (a frozen dataclass holding `indices: tuple[int, ...]`, `decoder_slice: np.ndarray`, `encoding`, `feature_names: tuple[str, ...]`). The public `blocks` accessor SHALL remain a sequence of `Dictionary` instances — materialised on first access via a lazy `@property` so existing callers see no API shape change.

This replaces today's eager-materialisation pass in `_materialise_blocks`, where every per-block `Feature` is constructed via a `dataclasses.replace` round-trip at build time. At full N=24,576 that pass alone allocates ~35–41 k Python objects and ~435 MB RSS (measured in `docs/research/data/clustered_amortised_benchmark_full_*.json`); the lazy contract pushes that cost to the consumers who actually need the `Dictionary` shape.

#### Scenario: lazy access returns a Dictionary on demand

- **WHEN** a caller indexes `cd.blocks[i]` for the first time on a `ClusteredDictionary` built via `build_clustered_dictionary(...)`
- **THEN** the returned object is a `Dictionary` whose feature names, decoder slice, and encoding match the underlying `BlockView` at position `i`

#### Scenario: lazy access is idempotent

- **WHEN** a caller indexes `cd.blocks[i]` twice in succession
- **THEN** both calls return the same `Dictionary` instance (cached, not re-materialised)

### Requirement: `materialise_dictionaries=True` preserves eager-build behaviour

`build_clustered_dictionary(..., materialise_dictionaries=True)` SHALL build all per-block `Dictionary` instances eagerly at construction time, byte-identical to the pre-change build path. This preserves a deterministic path for callers — notably the compression `from_compression_panels` adapter — that already have real `Feature` carriers in hand and want them retained without lazy re-synthesis.

#### Scenario: eager build matches pre-change byte equality

- **WHEN** `build_clustered_dictionary(..., materialise_dictionaries=True)` is called on the existing toy compression fixture
- **THEN** the resulting `ClusteredDictionary` is byte-identical to the pre-change build path (per the existing `tests/compression/test_epoch_clustered_consume.py` frozen-reference suite)

### Requirement: lazy-materialised features satisfy `Dictionary.__post_init__` validation

The lazy-materialisation path SHALL synthesise per-block `Feature` carriers whose knob values satisfy the validation `Dictionary.__post_init__` already enforces — including Rung5's amp-knob length check via `Feature.with_default_amp_knobs`.

#### Scenario: Rung5 lazy blocks pass amp-knob validation

- **WHEN** a `ClusteredDictionary` is built with `Rung5(n_amp_qubits=k)` and the lazy materialisation path runs
- **THEN** `cd.blocks[i]` returns a valid `Dictionary` whose features carry `amp_knobs` of length `k`, matching `Feature.with_default_amp_knobs`'s contract
