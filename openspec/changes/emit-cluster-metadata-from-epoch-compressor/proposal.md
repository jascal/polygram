# emit-cluster-metadata-from-epoch-compressor

## Why

Wave B's `add-concept-anchored-finetune` in sae-forge (PR #69, merged
2026-05-20) added a polygram-clusters label-source backend that reads
`n_clusters` and `cluster_assignments` from the compressed-SAE basis
to drive a supervised concept-anchoring loss. The §8.4 falsifiable
smoke run on a real GPT-2 + `jbloom/GPT2-Small-SAEs-Reformatted`
compressed via `EpochCompressor` surfaced a load-bearing gap:

**`EpochCompressor.run(out_path)` writes ONLY the safetensors. It does
not write a sidecar `_compression_report.json`.** This is in contrast
to `Compressor.apply()` (the single-shot path) which DOES write a
sidecar report. Downstream consumers reading the compressed checkpoint
via `FeatureBasis.from_polygram_checkpoint()` therefore get
`n_clusters: 0` (a defensive default) and no cluster_assignments,
even though `EpochCompressor` internally tracked cluster structure
across iterations via `cluster_fingerprints`.

Concrete impact, measured: the §8.4 smoke's
`PolygramClusterLabelSource.prepare()` correctly refuses the basis
("n_clusters <= 1; trivially clustered") and the smoke had to
inject synthetic cluster_assignments to proceed. Any production
sae-forge user invoking concept-anchored fine-tune against an
EpochCompressor-built basis hits the same wall. The sae-forge side
falls back to a round-robin partition (with a clear UserWarning),
but the supervised signal under that fallback is only coarsely aligned
with the true cluster structure the user paid compute for.

P1 — blocks the production utility of Wave B's add-concept-anchored-
finetune; without this, the recipe works at smoke scale but not on
real EpochCompressor outputs (which is the typical production path).

## What Changes

Three coordinated additions:

### `EpochReport` gains two fields

- `n_clusters: int` — count of multi-feature clusters in the final
  iteration's compression. Already computed internally by
  `EpochCompressor.run()` (the `cluster_fingerprints` tracking).
- `cluster_assignments: tuple[int, ...] | None` — per-feature cluster
  id of length `n_features_input`. Maps each input feature to its
  cluster id (0..n_clusters-1) OR `-1` for features that ended up
  fully zeroed across all iterations.

Schema bump `EpochReport.SCHEMA_VERSION 3 -> 4`. Back-compat loader
defaults missing fields to `None` / `0` and continues to load v3
payloads without error.

### `EpochCompressor.run()` writes a sidecar JSON report

After the safetensors write, emit `<out_path>_compression_report.json`
containing the `EpochReport`'s JSON form. Matches the convention used
by `Compressor.apply()` (which already writes the sidecar with that
exact suffix). Existing `FeatureBasis.from_polygram_checkpoint`
infrastructure in sae-forge already looks for the same suffixes
(`_compression_report.json`, `.compression_report.json`,
`_report.json`); no downstream changes required for discovery.

### `FeatureBasis`-layer metadata population

Already-shipped sae-forge code reads the sidecar JSON when present
and populates `basis.metadata` with `n_clusters`, `n_features_kept`,
`n_features_zeroed`. This change ALSO surfaces `cluster_assignments`
in the same metadata dict so downstream consumers
(`add-concept-anchored-finetune`'s polygram-clusters backend) can
read it without re-loading the sidecar.

## Capabilities

### Modified Capabilities

- `pareto-compression`: `EpochReport` gains `n_clusters` and
  `cluster_assignments` fields; `EpochCompressor.run()` writes a
  sidecar `<out>_compression_report.json` mirroring
  `Compressor.apply()`'s pattern. Schema v3 -> v4 with back-compat
  loader.

## Impact

- `polygram/compression/epoch_report.py` — two new fields with
  validation + JSON round-trip; schema version bump.
- `polygram/compression/epoch.py` — sidecar write at end of `run()`;
  cluster-fingerprint -> per-feature-id materialisation for the
  `cluster_assignments` field.
- `tests/compression/test_epoch_report_roundtrip.py` — round-trip
  the new fields.
- `tests/compression/test_epoch_apply.py` (or equivalent) — assert
  the sidecar JSON is written alongside the safetensors and contains
  the expected fields.
- `tests/compression/test_epoch_clustered_consume.py` — update the
  frozen-reference JSON fixtures to include the new fields.
- CHANGELOG entry under [Unreleased].

No breaking changes: schema bump has a back-compat loader; the
sidecar-write is additive (consumers that don't look for it are
unaffected). Existing tests stay green.

## Out of scope

- **Compressor.apply()-side cluster_assignments.** Compressor.apply
  already writes a sidecar with `n_clusters`. Its `CompressionPlan`
  carries `clusters: tuple[ClusterPlan, ...]` where each
  `ClusterPlan.members` is the per-cluster feature list — the
  cluster_assignments are recoverable from this. A follow-up
  could mirror EpochReport's new field on CompressionReport for
  symmetry, but the sae-forge side can already derive it from
  `plan.clusters[*].members`; not strictly required for this change.
- **Per-iteration cluster history.** Only the final-iteration
  assignments are surfaced. Per-iteration debugging info stays
  inside `EpochIteration`.
- **safetensors-header metadata.** Polygram could ALSO write
  `n_clusters` into the safetensors header so a single
  .safetensors file is fully self-describing without a sidecar.
  Out of scope for this change; the sidecar is sufficient.

## Acceptance

This change ships when:

1. `EpochCompressor.run(out_path)` writes both `<out>.safetensors`
   AND `<out>_compression_report.json`.
2. The JSON sidecar contains the v4-schema `EpochReport` JSON with
   `n_clusters` and `cluster_assignments` populated correctly.
3. `FeatureBasis.from_polygram_checkpoint(<out>.safetensors)` on
   the sae-forge side surfaces `n_clusters` and `cluster_assignments`
   in `basis.metadata`.
4. The sae-forge §8.4 smoke that surfaced this gap can be re-run
   WITHOUT the synthetic cluster injection workaround and the
   concept-anchored path runs cleanly against the real polygram
   metadata.
