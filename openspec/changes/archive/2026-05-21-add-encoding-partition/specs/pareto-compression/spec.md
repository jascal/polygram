# pareto-compression Specification (delta)

## ADDED Requirements

### Requirement: BlockSpec dataclass

`BlockSpec` SHALL be a frozen dataclass with five fields:

| Field | Type | Semantics |
|-------|------|-----------|
| `block_id` | `str` | Non-empty human-readable id; surfaces in `BlockReport.block_id` |
| `encoding_class` | `str` | One of `{"MPSRung1", "Rung3", "Rung4", "Rung5", "HEA_Rung2"}` |
| `encoding_kwargs` | `dict[str, Any]` | Per-family kwargs per the validator registry |
| `learn_axis_assignment` | `bool` | Per-block axis-assignment policy |
| `feature_ids` | `tuple[int, ...]` | Non-empty, non-negative, no duplicates within block |

`__post_init__` SHALL validate via a module-level
`_BLOCK_SPEC_KWARG_VALIDATORS: dict[str, Callable[[dict], None]]`
registry, not an inline branch chain. Future encoding families add a
new validator + one registry entry.

`__hash__` SHALL hash a `frozenset` of `encoding_kwargs.items()`
(dict is itself unhashable). This makes `BlockSpec` hashable as
required by downstream consumers' cache-key contracts (e.g.
sae-forge's `compute_cache_key`).

#### Scenario: Valid BlockSpec with Rung5 + n_amp_qubits

- **WHEN** `BlockSpec(block_id="heavy", encoding_class="Rung5", encoding_kwargs={"n_amp_qubits": 4}, feature_ids=(0, 1, 2, 3))` is constructed
- **THEN** validation passes and the instance is hashable

#### Scenario: Rung5 without n_amp_qubits is rejected

- **WHEN** `BlockSpec(encoding_class="Rung5", encoding_kwargs={}, ...)` is constructed
- **THEN** `ValueError` is raised naming the missing `n_amp_qubits` kwarg

#### Scenario: MPSRung1 with non-empty kwargs is rejected

- **WHEN** `BlockSpec(encoding_class="MPSRung1", encoding_kwargs={"foo": 1}, ...)` is constructed
- **THEN** `ValueError` is raised noting the encoding accepts no kwargs

#### Scenario: feature_ids with duplicates rejected

- **WHEN** `BlockSpec(feature_ids=(1, 2, 1), ...)` is constructed
- **THEN** `ValueError` names the duplicate id

#### Scenario: feature_ids with negative ids rejected

- **WHEN** `BlockSpec(feature_ids=(0, -1), ...)` is constructed
- **THEN** `ValueError` notes feature_ids must be non-negative

### Requirement: PartitionCoverageError + validate_partition_coverage

`PartitionCoverageError` SHALL subclass `ValueError`.
`validate_partition_coverage(partition, *, n_features_input)` SHALL
check:

- **Disjointness**: no feature id in more than one block. Violation
  raises `PartitionCoverageError` naming up to 10 duplicate ids with
  their containing block_ids.
- **Completeness**: the union of all blocks' `feature_ids` equals
  `set(range(n_features_input))`. Missing ids raise an `incomplete`
  error; extras raise an `extra` error; both name up to 10 ids.

Called by `Compressor.apply` immediately after loading the input
SAE; NOT called from `BlockSpec.__post_init__` (because
`n_features_input` is unknown at BlockSpec construction).

#### Scenario: Disjoint complete partition passes

- **WHEN** partition `(BlockSpec(feature_ids=(0,1,2,3)), BlockSpec(feature_ids=(4,5,6,7)))` is validated against `n_features_input=8`
- **THEN** no exception is raised

#### Scenario: Overlapping partition raises naming duplicates

- **WHEN** two blocks share feature_id 3
- **THEN** `PartitionCoverageError` is raised mentioning `3` and both blocks' `block_id`s

#### Scenario: Incomplete partition raises naming missing ids

- **WHEN** blocks cover `{0,1,2}` against `n_features_input=4` (missing 3)
- **THEN** `PartitionCoverageError` is raised naming the missing id

### Requirement: CompressionConfig.encoding_partition

`CompressionConfig` SHALL accept an optional
`encoding_partition: tuple[BlockSpec, ...] | None = None` field.

When `None` (default), `Compressor.apply` runs the historical
single-encoding path byte-equivalently — no behavioural drift for
existing callers.

When set, `__post_init__` SHALL validate:
- Type is `tuple` (NOT list — required for `CompressionConfig`'s
  frozen-hashable contract).
- Every element is a `BlockSpec` instance.
- Tuple is non-empty.

Coverage validation against `n_features_input` is deferred to
`Compressor.apply` (where `n_features_input` becomes known).

#### Scenario: Default encoding_partition is None

- **WHEN** `CompressionConfig()` is constructed without arguments
- **THEN** `cfg.encoding_partition is None`

#### Scenario: Partition as list is rejected

- **WHEN** `CompressionConfig(encoding_partition=[BlockSpec(...)])` (list, not tuple) is constructed
- **THEN** `TypeError` is raised noting "tuple of BlockSpec"

#### Scenario: Non-BlockSpec member rejected

- **WHEN** `CompressionConfig(encoding_partition=("not-a-blockspec",))` is constructed
- **THEN** `TypeError` is raised naming `BlockSpec`

#### Scenario: Empty partition tuple rejected

- **WHEN** `CompressionConfig(encoding_partition=())` is constructed
- **THEN** `ValueError` is raised noting non-empty requirement

### Requirement: CompressionReport.blocks + schema v2 → v3

`CompressionReport` SHALL gain a
`blocks: tuple[BlockReport, ...] | None = None` field. `BlockReport`
mirrors the top-level diagnostic fields per block (plus the
encoding/feature-id metadata):

| Field | Type | Default |
|-------|------|---------|
| `block_id` | `str` | required |
| `encoding_class` | `str` | required |
| `encoding_kwargs` | `dict` | required |
| `learn_axis_assignment` | `bool` | required |
| `feature_ids` | `tuple[int, ...]` | required |
| `n_features_kept` | `int` | required |
| `n_features_zeroed` | `int` | required |
| `n_clusters` | `int` | required |
| `cluster_assignments` | `tuple[int, ...] \| None` | `None` |
| `scale_compression_ratio` | `float` | `1.0` |
| `rank_ratio` | `float \| None` | `None` |
| `post_A` | `float \| None` | `None` |
| `forge_mse` | `float \| None` | `None` |
| `informative_metric` | `Literal[...] \| None` | `None` |

`CompressionReport.SCHEMA_VERSION` SHALL be bumped from `2` to `3`.
`from_json` SHALL accept v2 payloads (no `blocks` key) and default
`blocks=None`. Equality + hash SHALL include `blocks`.

The module SHALL export `MAX_CLUSTERS_PER_BLOCK = 10_000` as the
constant for the global cluster-id namespace.

#### Scenario: SCHEMA_VERSION bumped to 3

- **WHEN** `polygram.compression.report.SCHEMA_VERSION` is read
- **THEN** the value is `3`

#### Scenario: v3 report with blocks round-trips through JSON

- **WHEN** a `CompressionReport` with two populated `BlockReport`s is serialised via `to_json` and reconstructed via `from_json`
- **THEN** the round-tripped instance equals the original (NaN-aware on float fields, exact-equal on the rest)

#### Scenario: v2 payload loads with blocks=None

- **GIVEN** a JSON payload with `schema_version=2` and no `blocks` key
- **WHEN** `CompressionReport.from_json(payload)` is called
- **THEN** the call succeeds; `r.schema_version == 2`, `r.blocks is None`

#### Scenario: v3 report with blocks=None round-trips

- **WHEN** a `CompressionReport` with `schema_version=3` and `blocks=None` (the unpartitioned-compress case) round-trips via `to_json` + `from_json`
- **THEN** the result has `blocks is None` and equals the original

### Requirement: Compressor.apply refuses encoding_partition (Phase 1)

`Compressor.apply` SHALL detect `self.config.encoding_partition is
not None` immediately upon entry and raise `NotImplementedError`
naming `add-encoding-partition` Phase 2 as the load-bearing follow-up.

This refusal is the Phase 1 contract: the API surface is locked
(`BlockSpec` + `CompressionConfig.encoding_partition` +
`CompressionReport.blocks`) but the actual per-block dispatch is
deferred to Phase 2. Without the refusal, downstream consumers
would silently get a single-encoding compression instead of the
partitioned one they requested.

#### Scenario: Compressor.apply refuses partitioned config

- **GIVEN** a `Compressor` constructed with a `CompressionConfig` whose `encoding_partition` is a non-empty tuple
- **WHEN** `compressor.apply(output_checkpoint=...)` is called
- **THEN** `NotImplementedError` is raised mentioning "Phase 2 follow-up"
