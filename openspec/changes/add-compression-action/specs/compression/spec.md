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

### Requirement: run() is the convenience wrapper for plan() + apply()

`Compressor.run(output_checkpoint: Path) -> CompressionResult` SHALL be exactly equivalent to `self.apply(self.plan(), output_checkpoint=output_checkpoint)`.

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

### Requirement: zero strategy zeroes encoder and decoder rows for non-representatives

When `strategy == "zero"`, for every feature id in any cluster's `zeroed` list, `apply()` SHALL set:

- `W_enc[:, fid] = 0` — the encoder column for that feature.
- `b_enc[fid] = 0` — the encoder bias entry.
- `W_dec[fid, :] = 0` — the decoder row for that feature.

`b_dec` (the decoder bias) SHALL be unchanged — it is global, not feature-specific.

The representative feature's `W_enc / b_enc / W_dec` rows SHALL be unchanged.

Singleton features (those not in any cluster) SHALL be unchanged.

### Requirement: apply() produces a Dictionary that round-trips through from_sae_lens

The `CompressionResult` returned by `apply()` SHALL include a `dictionary: Dictionary` field built by re-running `polygram.from_sae_lens` on the rewritten checkpoint, using the same `feature_ids` list as the source `ValidationReport`. Cluster (β / γ) assignments on the new Dictionary are derived afresh from the rewritten geometry; they are not transferred from the source dictionary.

### Requirement: Compressor is torch-free

`polygram.compression` SHALL NOT import `torch` or `transformers` at module load time or during `plan()` / `apply()` / `run()`. The action operates on `numpy.ndarray` tensors loaded via `safetensors.numpy.load_file`. No `[behavioural]` or other optional extra is required.
