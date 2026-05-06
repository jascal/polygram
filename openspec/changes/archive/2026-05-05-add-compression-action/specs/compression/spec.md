## ADDED Requirements

### Requirement: Compressor consumes a ValidationReport and rewrites the SAE checkpoint

`polygram.compression.Compressor` SHALL be a dataclass that consumes a `ValidationReport` (or its `confirmed` pair list) and rewrites a source SAE checkpoint into a new `.safetensors` file with redundant features zeroed per the `zero` strategy.

The dataclass exposes the following fields:

- `validation_report: ValidationReport` — required. The upstream
  validator output. The `confirmed` field drives cluster
  construction; `pairs` is consulted for representative
  selection.
- `sae_checkpoint: Path` — required. The source `.safetensors`
  file. MUST exist on disk. The compressor reads it; it never
  writes to it.
- `strategy: str` — required. Must be the literal string
  `"zero"`. Other values raise `ValueError`. The strategy field
  is required (not defaulted) so the call site is explicit about
  which transformation is applied; future strategies (e.g.,
  `"merge"`) will land as separate spec changes.
- `representatives: dict[int, int] | None = None` — optional
  per-cluster override mapping cluster id (assigned in `plan()`
  by ascending min-fid order) to a chosen feature id. The chosen
  fid MUST be a member of the cluster.

`__post_init__` SHALL validate every field constraint named above and SHALL raise `ValueError` (with field name and offending value) on any violation.

#### Scenario: construction succeeds with a valid ValidationReport and existing checkpoint

- **GIVEN** a synthetic SAE checkpoint at `tmp/sae.safetensors`
- **AND** a `ValidationReport` with non-empty `confirmed` list
- **WHEN** `Compressor(validation_report=report, sae_checkpoint=path, strategy='zero')` is constructed
- **THEN** `__post_init__` SHALL succeed without raising

#### Scenario: missing checkpoint raises ValueError

- **WHEN** `Compressor(validation_report=report, sae_checkpoint=Path('/tmp/does-not-exist.safetensors'), strategy='zero')` is constructed
- **THEN** `__post_init__` SHALL raise `ValueError` whose message names the missing path

#### Scenario: unsupported strategy raises ValueError

- **WHEN** `Compressor(validation_report=report, sae_checkpoint=path, strategy='merge')` is constructed
- **THEN** `__post_init__` SHALL raise `ValueError` whose message names the rejected strategy `'merge'` and lists the supported set

#### Scenario: representative override referencing a non-member raises

- **GIVEN** a `ValidationReport` whose plan would produce cluster 0 = `{0, 1}`
- **WHEN** `Compressor(..., representatives={0: 7})` is constructed (fid 7 is not in cluster 0)
- **THEN** `__post_init__` SHALL raise `ValueError` whose message names the offending fid `7` and the cluster's actual members

### Requirement: plan() builds connected components and selects representatives

`Compressor.plan() -> CompressionPlan` SHALL:

1. Build connected components from
   `self.validation_report.confirmed` via a Union-Find data
   structure.
2. Discard singleton features (features that appear in
   `feature_ids` but are not in any confirmed pair) — they are
   not part of any cluster and are not affected by compression.
3. For each cluster, pick a representative:
   - If `self.representatives` is not None and contains the
     cluster id: use the override fid. The override fid MUST be
     a member of the cluster else raise `ValueError`.
   - Otherwise: pick the cluster member with the highest summed
     `n_fires` across the cluster's pairs in
     `self.validation_report.pairs`. Tiebreak: lowest feature
     id.
4. Return a `CompressionPlan` with one entry per cluster (sorted
   by ascending min-fid), each entry carrying:
   - `cluster_id: int` — assigned by ascending min-fid order
     (cluster 0 has the lowest min-fid).
   - `members: tuple[int, ...]` — sorted ascending.
   - `representative: int` — selected per the rule above.
   - `zeroed: tuple[int, ...]` — `members` minus
     `representative`, sorted ascending.

`plan()` SHALL NOT read from `self.sae_checkpoint`. It SHALL NOT import torch or transformers. It SHALL be deterministic across runs given the same input.

#### Scenario: clique collapses to a single connected component

- **GIVEN** a `ValidationReport` with `confirmed` containing all 6 pairs from the 4-clique on `{8371, 2287, 68, 13737}`
- **WHEN** `Compressor(...).plan()` runs
- **THEN** the returned `CompressionPlan.clusters` SHALL contain exactly one cluster whose `members` equals the sorted tuple `(68, 2287, 8371, 13737)`

#### Scenario: representative selection picks the highest-summed-n_fires member

- **GIVEN** a 3-feature cluster `{3, 4, 5}` with hand-set pair `n_fires` such that fid 3 sums to 100, fid 4 to 50, fid 5 to 1
- **WHEN** `plan()` runs
- **THEN** the cluster's `representative` SHALL equal `3`
- **AND** `zeroed` SHALL equal `(4, 5)`

#### Scenario: lowest-fid tiebreak on equal n_fires

- **GIVEN** a cluster `{2, 3}` with equal n_fires sums on both members
- **WHEN** `plan()` runs
- **THEN** the cluster's `representative` SHALL equal `2`

#### Scenario: representative override is honored

- **GIVEN** a `ValidationReport` whose default plan would pick fid 3 as cluster 0's rep
- **WHEN** `Compressor(..., representatives={0: 5}).plan()` runs (fid 5 is in cluster 0)
- **THEN** the returned cluster 0's `representative` SHALL equal `5`

#### Scenario: clusters are ordered by ascending min-fid

- **GIVEN** confirmed pairs producing clusters `{8, 9}` and `{2, 3}`
- **WHEN** `plan()` runs
- **THEN** `clusters[0].cluster_id` SHALL equal `0` AND `clusters[0].members[0]` SHALL equal `2`
- **AND** `clusters[1].cluster_id` SHALL equal `1` AND `clusters[1].members[0]` SHALL equal `8`

### Requirement: apply() rewrites the SAE checkpoint and returns a CompressionResult

`Compressor.apply(plan: CompressionPlan | None = None, output_checkpoint: Path) -> CompressionResult` SHALL:

1. When `plan is None`, default to `self.plan()`.
2. Reject `output_checkpoint` paths that resolve (`Path.resolve()`)
   to the same path as `self.sae_checkpoint`. Raise `ValueError`
   on collision.
3. Read all four tensors `W_enc, b_enc, W_dec, b_dec` from
   `self.sae_checkpoint` via
   `safetensors.numpy.load_file`.
4. For every feature id in any cluster's `zeroed` list, set
   `W_enc[:, fid] = 0`, `b_enc[fid] = 0`, `W_dec[fid, :] = 0`.
   `b_dec` is unchanged.
5. Write the rewritten tensors to `output_checkpoint` atomically:
   write to a sibling temp path, then `os.replace()` to
   `output_checkpoint`. The source checkpoint is never modified.
6. Compute SHA256 hashes of both the source checkpoint (as read)
   and the output checkpoint (as written).
7. Build a `CompressionReport` carrying provenance, the applied
   plan, both checkpoint hashes, and aggregate counters
   (`n_features_zeroed`, `n_features_kept`, `n_clusters`).
8. Return a `CompressionResult` with three fields: `plan`
   (the applied plan), `report` (the `CompressionReport`), and
   `output_checkpoint` (the resolved final path).

`apply()` SHALL NOT import torch or transformers.

#### Scenario: apply() writes the rewritten checkpoint and returns a CompressionResult

- **GIVEN** a `Compressor` whose plan zeroes 3 of 8 features
- **WHEN** `apply(output_checkpoint=tmp/'out.safetensors')` is called
- **THEN** the file `tmp/'out.safetensors'` SHALL exist
- **AND** `result.output_checkpoint` SHALL resolve to that path
- **AND** `result.report.n_features_zeroed` SHALL equal `3`

#### Scenario: source-equals-output collision raises before any I/O

- **GIVEN** a `Compressor` with `sae_checkpoint=path`
- **WHEN** `apply(plan, output_checkpoint=path)` is called
- **THEN** `ValueError` SHALL be raised before any rewrite happens
- **AND** the message SHALL name both paths

#### Scenario: source bytes are unchanged after a successful run

- **GIVEN** `before = path.read_bytes()`
- **WHEN** `Compressor(...).run(output_checkpoint=other_path)` completes successfully
- **THEN** `path.read_bytes() == before`

### Requirement: run() is the convenience wrapper for plan() + apply()

`Compressor.run(output_checkpoint: Path) -> CompressionResult` SHALL be exactly equivalent to `self.apply(self.plan(), output_checkpoint=output_checkpoint)`.

#### Scenario: run() result matches a separate plan+apply invocation

- **GIVEN** a `Compressor` instance `c`
- **WHEN** `r1 = c.run(output_a)` and a fresh `c2` runs `r2 = c2.apply(c2.plan(), output_checkpoint=output_b)` with identical inputs
- **THEN** `r1.report.source_checkpoint_sha256 == r2.report.source_checkpoint_sha256`
- **AND** `r1.plan == r2.plan`

### Requirement: CompressionPlan describes the per-cluster transformation

`CompressionPlan` SHALL be a frozen dataclass with the following fields:

- `clusters: tuple[ClusterPlan, ...]` — one entry per
  multi-feature cluster, ordered by ascending min-fid.
- `feature_ids: tuple[int, ...]` — the validator's input
  `feature_ids` list, preserved verbatim for provenance.

Each `ClusterPlan` is a frozen dataclass with:

- `cluster_id: int` — ascending from 0 by min-fid order.
- `members: tuple[int, ...]` — sorted ascending.
- `representative: int` — must be in `members`.
- `zeroed: tuple[int, ...]` — `members` minus
  `representative`, sorted ascending.

`CompressionPlan` SHALL be JSON-serializable via the same `to_json` / `from_json` round-trip pattern as `ValidationReport`.

#### Scenario: cluster fields are populated as specified

- **GIVEN** a plan whose cluster 0 has members `(0, 1, 2)` and rep `2`
- **THEN** `cluster.cluster_id == 0`
- **AND** `cluster.members == (0, 1, 2)`
- **AND** `cluster.representative == 2`
- **AND** `cluster.zeroed == (0, 1)`

#### Scenario: feature_ids preserved verbatim from the source ValidationReport

- **GIVEN** a `ValidationReport` with `feature_ids = (12999, 19398, 4192, 23625, 8371, 2287, 68, 13737)`
- **WHEN** `Compressor(...).plan()` runs
- **THEN** the returned `CompressionPlan.feature_ids` SHALL equal the source `feature_ids` exactly

### Requirement: CompressionReport carries source-validation provenance

`CompressionReport` SHALL be a frozen dataclass with the following fields:

- `schema_version: int` — currently `1`.
- `source_checkpoint: Path` — the source path as supplied.
- `source_checkpoint_sha256: str` — hex SHA256 of the source
  bytes as read.
- `output_checkpoint: Path` — the output path as written.
- `output_checkpoint_sha256: str` — hex SHA256 of the output
  bytes as written.
- `validation_report_dictionary_name: str` — copied from the
  source `ValidationReport`.
- `validation_report_schema_version: int` — copied from the
  source `ValidationReport`.
- `strategy: str` — the applied strategy (initial release:
  `"zero"`).
- `plan: CompressionPlan` — the plan that was applied.
- `n_features_zeroed: int` — total count across all clusters.
- `n_features_kept: int` — total count across all clusters
  (one per cluster, the representative).
- `n_clusters: int` — number of multi-feature clusters.

`CompressionReport.to_json(path)` SHALL write a deterministic JSON representation matching the schema in `design.md` Decision 6. Floats SHALL be formatted via `format(v, ".6g")`.

`CompressionReport.from_json(path) -> CompressionReport` SHALL be the inverse of `to_json`. The round-trip property `from_json(to_json(r)) == r` SHALL hold for any `r` reachable from `Compressor.run()`.

#### Scenario: sha256 fields are populated and differ between source and output

- **GIVEN** a successful `Compressor.run()` invocation
- **THEN** `result.report.source_checkpoint_sha256` SHALL be a 64-character hex string
- **AND** `result.report.output_checkpoint_sha256` SHALL be a 64-character hex string
- **AND** the two sha256 values SHALL NOT be equal

#### Scenario: validation_report fields are propagated through to the report

- **GIVEN** a `ValidationReport` with `dictionary_name='ScaleupBlocks10'` and `schema_version=1`
- **WHEN** `Compressor(...).run(...)` produces a `CompressionResult`
- **THEN** `result.report.validation_report_dictionary_name` SHALL equal `'ScaleupBlocks10'`
- **AND** `result.report.validation_report_schema_version` SHALL equal `1`

#### Scenario: JSON round-trip preserves equality

- **GIVEN** a `CompressionReport` instance `r` returned from `Compressor.run()`
- **WHEN** `r2 = CompressionReport.from_json(r.to_json())`
- **THEN** `r2 == r`

### Requirement: zero strategy zeroes encoder and decoder rows for non-representatives

When `strategy == "zero"`, for every feature id in any cluster's `zeroed` list, `apply()` SHALL set:

- `W_enc[:, fid] = 0` — the encoder column for that feature.
- `b_enc[fid] = 0` — the encoder bias entry.
- `W_dec[fid, :] = 0` — the decoder row for that feature.

`b_dec` (the decoder bias) SHALL be unchanged — it is global, not feature-specific.

The representative feature's `W_enc / b_enc / W_dec` rows SHALL be unchanged.

Singleton features (those not in any cluster) SHALL be unchanged.

#### Scenario: zeroed feature has all three tensors silenced

- **GIVEN** a plan with cluster 0 = `{0, 1}` rep = `1` zeroed = `(0,)`
- **WHEN** `apply()` writes the rewritten checkpoint
- **AND** the rewritten state-dict is loaded via `safetensors.numpy.load_file`
- **THEN** every entry of `state['W_enc'][:, 0]` SHALL equal `0.0`
- **AND** `state['b_enc'][0]` SHALL equal `0.0`
- **AND** every entry of `state['W_dec'][0, :]` SHALL equal `0.0`

#### Scenario: representative tensors are byte-equal to source

- **GIVEN** the same plan as above (rep = `1`)
- **WHEN** the rewritten checkpoint is loaded
- **THEN** `state['W_enc'][:, 1]` SHALL be element-wise byte-equal to the source's `W_enc[:, 1]`
- **AND** `state['W_dec'][1, :]` SHALL be element-wise byte-equal to the source's `W_dec[1, :]`

#### Scenario: b_dec is global and unchanged

- **GIVEN** any `Compressor.run()` invocation
- **WHEN** the rewritten checkpoint is loaded
- **THEN** `state['b_dec']` SHALL be element-wise byte-equal to the source's `b_dec`

#### Scenario: singleton features are untouched

- **GIVEN** an SAE with 8 features whose plan compresses only cluster `{0, 1}`
- **WHEN** the rewritten checkpoint is loaded
- **THEN** for every fid in `(2, 3, 4, 5, 6, 7)`, both `W_enc[:, fid]` and `W_dec[fid, :]` SHALL be byte-equal to the source's

### Requirement: apply() produces a Dictionary that round-trips through from_sae_lens

The `CompressionResult` returned by `apply()` SHALL include a `dictionary: Dictionary` field built by re-running `polygram.from_sae_lens` on the rewritten checkpoint, using the same `feature_ids` list as the source `ValidationReport`. Cluster (β / γ) assignments on the new Dictionary are derived afresh from the rewritten geometry; they are not transferred from the source dictionary.

#### Scenario: rebuilt Dictionary has the expected feature count and name

- **GIVEN** a `Compressor` whose source `ValidationReport.feature_ids` has length 8 and `dictionary_name='Test'`
- **WHEN** `result = compressor.run(output)` completes
- **THEN** `len(result.dictionary.features)` SHALL equal `8`
- **AND** `result.dictionary.name` SHALL equal `'Test'`

### Requirement: Compressor is torch-free

`polygram.compression` SHALL NOT import `torch` or `transformers` at module load time or during `plan()` / `apply()` / `run()`. The action operates on `numpy.ndarray` tensors loaded via `safetensors.numpy.load_file`. No `[behavioural]` or other optional extra is required.

#### Scenario: importing the module does not import torch

- **GIVEN** a fresh Python process
- **WHEN** `import polygram.compression` is executed
- **THEN** `'torch' not in sys.modules`
- **AND** `'transformers' not in sys.modules`

#### Scenario: full Compressor.run() does not import torch

- **GIVEN** a fresh Python process that has executed `from polygram import Compressor`
- **WHEN** `Compressor(...).run(...)` completes successfully on a synthetic SAE
- **THEN** `'torch' not in sys.modules`
- **AND** `'transformers' not in sys.modules`
