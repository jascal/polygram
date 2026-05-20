# pareto-compression Specification (delta)

## MODIFIED Requirements

### Requirement: EpochReport schema includes cluster-structure fields

`EpochReport` SHALL include the following fields in addition to the
v3 schema (`add-cancellation-and-compression-diagnostics`):

| Field | Type | Default | Semantics |
|-------|------|---------|-----------|
| `n_clusters` | `int` | `0` | Count of distinct non-negative ids in `cluster_assignments`. Counts both multi-feature clusters AND surviving singletons. |
| `cluster_assignments` | `tuple[int, ...] \| None` | `None` | Per-feature cluster id (length `n_features_input`). Elements are integers in `[0, n_clusters)` for features that belong to a cluster (multi-feature OR singleton) in the FINAL iteration's compression, OR `-1` for features fully zeroed across all iterations. |

`SCHEMA_VERSION` is bumped from `3` to `4`. `from_json` SHALL accept
v3 payloads (no `n_clusters` / `cluster_assignments` keys) without
error, defaulting both fields to the values above so the new
fields' absence in legacy reports is observable but not fatal.

`__post_init__` validation: `n_clusters >= 0`. When
`cluster_assignments is not None`, every element is `>= -1`.

#### Scenario: v4 EpochReport round-trips through JSON

- **WHEN** an `EpochReport` with `n_clusters=3`,
  `cluster_assignments=(0, 0, 1, 1, 2, -1, ...)` is serialised via
  `to_json()` and reconstructed via `from_json(...)`
- **THEN** the reconstructed instance equals the original
  (`schema_version=4`, `n_clusters=3`, same `cluster_assignments`)

#### Scenario: v3 payload loads with default fields

- **GIVEN** a JSON payload with `schema_version=3` and no
  `n_clusters` / `cluster_assignments` keys
- **WHEN** `EpochReport.from_json(payload)` is called
- **THEN** the call succeeds; the returned instance has
  `n_clusters == 0` and `cluster_assignments is None`

#### Scenario: negative n_clusters rejected

- **WHEN** `EpochReport(..., n_clusters=-1)` is constructed
- **THEN** `__post_init__` raises `ValueError` naming the field

#### Scenario: cluster_assignments rejects values below -1

- **WHEN** `EpochReport(..., cluster_assignments=(0, -2, 1, ...))`
  is constructed
- **THEN** `__post_init__` raises `ValueError` naming the field
  (the only valid sub-zero sentinel is `-1`)

### Requirement: EpochCompressor.run writes a sidecar compression report

`EpochCompressor.run(out_path)` SHALL write a sidecar JSON file
adjacent to the safetensors output, with filename
`<out_path>_compression_report.json` (using `out_path.stem` plus
the `_compression_report.json` suffix). The sidecar contains the
serialised v4 `EpochReport` produced by the run, including the
populated `n_clusters` and `cluster_assignments` fields.

The sidecar write SHALL be atomic via the same tempfile + os.replace
pattern used for the safetensors write. If the sidecar write fails,
its temp file SHALL be cleaned up (no orphan `.tmp` files); the
safetensors file may already have been replaced successfully, but
the failure SHALL propagate to the caller.

The naming convention matches `Compressor.apply()`'s existing
sidecar pattern, so downstream consumers
(e.g. sae-forge's `FeatureBasis.from_polygram_checkpoint`, which
already inspects `_compression_report.json` suffixes) discover the
new metadata automatically — no downstream code changes required.

#### Scenario: sidecar written alongside the safetensors

- **WHEN** `EpochCompressor(...).run(out_path)` completes successfully
- **THEN** both `out_path` (safetensors) AND
  `out_path.with_name(out_path.stem + "_compression_report.json")`
  (JSON sidecar) exist

#### Scenario: sidecar parses as a v4 EpochReport

- **GIVEN** a sidecar JSON written by `EpochCompressor.run`
- **WHEN** the file is parsed as JSON
- **THEN** the result has `schema_version == 4`, contains the keys
  `n_clusters` and `cluster_assignments`, and round-trips cleanly
  via `EpochReport.from_json`

#### Scenario: cluster_assignments length matches n_features_input

- **WHEN** an EpochReport is written by `EpochCompressor.run`
- **THEN** `len(report.cluster_assignments) == report.n_features_input`

#### Scenario: n_clusters matches distinct non-negative ids

- **WHEN** an EpochReport is written by `EpochCompressor.run`
- **THEN** `report.n_clusters == len({cid for cid in
  report.cluster_assignments if cid >= 0})`

#### Scenario: atomic write — no orphan sidecar temp on failure

- **GIVEN** the JSON `os.replace` to the sidecar path is forced to
  raise `OSError`
- **WHEN** `EpochCompressor.run` is invoked
- **THEN** the error propagates AND no `.tmp` file matching the
  sidecar's name pattern remains in the output directory
