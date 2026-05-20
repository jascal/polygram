# tasks — emit-cluster-metadata-from-epoch-compressor

## 1. `EpochReport` schema extension

- [ ] 1.1 Add two fields to `polygram/compression/epoch_report.py::EpochReport`:
  - `n_clusters: int = 0`
  - `cluster_assignments: tuple[int, ...] | None = None`
- [ ] 1.2 Bump `SCHEMA_VERSION` from `3` to `4`.
- [ ] 1.3 `from_json` SHALL accept v3 payloads (no `n_clusters` /
      `cluster_assignments` keys); default both to the sentinel
      values above. Add a unit test confirming a v3 payload still
      loads without error.
- [ ] 1.4 `to_json` SHALL serialise both fields. `cluster_assignments`
      serialises as a JSON list when populated; as `null` otherwise.
      `n_clusters` is a plain int.
- [ ] 1.5 `__eq__` and `__hash__` include the new fields. Equality
      compares tuples element-wise; None equals None; tuple equals
      tuple iff same length + same per-element values.
- [ ] 1.6 Validation in `__post_init__` (extend existing block):
      `n_clusters >= 0`; when `cluster_assignments is not None`,
      every element is `>= -1` (the sentinel for fully-zeroed
      features).

## 2. `EpochCompressor.run()` sidecar write

- [ ] 2.1 In `polygram/compression/epoch.py::EpochCompressor.run`,
      after the safetensors write completes, materialise the final
      `cluster_assignments` from `cluster_fingerprints` (Decision 4)
      AND set `n_clusters` to the count of distinct non-negative
      ids in the assignments.
- [ ] 2.2 Construct the `EpochReport` with the new fields populated.
- [ ] 2.3 Write `<out_path>_compression_report.json` with
      `report.to_json()`. Use the same atomic-write pattern as the
      safetensors (temp file in same dir + `os.replace`).
- [ ] 2.4 On either write failing, the cleanup logic SHALL remove
      BOTH temp files (leave no half-written artefacts).
- [ ] 2.5 Helper `_final_cluster_assignments(...)` per Decision 4.
      Pure-function, easy to unit-test.

## 3. Cluster-assignments materialisation

- [ ] 3.1 `_final_cluster_assignments(cluster_fingerprints,
      n_features_input, final_state_dict) -> tuple[int, ...]`:
      - Each frozenset in `cluster_fingerprints[-1]` becomes a
        cluster; assign ids in sorted-by-min-fid order.
      - Features in those frozensets get their cluster's id.
      - Surviving singletons (features with non-zero W_dec norm in
        final_state_dict that aren't in any cluster) get fresh
        cluster ids appended after the multi-feature ones.
      - Fully-zeroed features get `-1`.
- [ ] 3.2 Unit test: hand-construct a cluster_fingerprints + state
      dict; assert the materialised assignments match the expected
      tuple element-wise.
- [ ] 3.3 Unit test: empty cluster_fingerprints (max_iterations=0 or
      first-iteration-already-converged) → assignments are
      `(-1,) * n_features_input` when state is all-zero, OR per-
      feature singleton ids when state has surviving features.

## 4. Tests

### 4.1 Round-trip + back-compat

- [ ] 4.1.1 `tests/compression/test_epoch_report_roundtrip.py::test_epoch_report_roundtrips_cluster_fields`:
      populate `n_clusters=3`, `cluster_assignments=(0, 0, 1, 1, 2, -1)`,
      round-trip via `to_json` + `from_json`, assert equality.
- [ ] 4.1.2 `test_epoch_report_loads_v3_payload`: hand-craft a JSON
      payload with `schema_version=3` and no `n_clusters` /
      `cluster_assignments` keys; assert `from_json` succeeds and
      defaults the new fields to `0` / `None`.
- [ ] 4.1.3 `test_epoch_report_validation_rejects_negative_n_clusters`:
      `n_clusters=-1` → `ValueError`.
- [ ] 4.1.4 `test_epoch_report_validation_rejects_below_minus_one_in_assignments`:
      `cluster_assignments=(0, -2, 1)` → `ValueError`.

### 4.2 EpochCompressor sidecar

- [ ] 4.2.1 `tests/compression/test_compressor_epoch.py::test_run_writes_sidecar_compression_report`:
      construct an `EpochCompressor`, run on a synth SAE; assert
      `<out>_compression_report.json` exists alongside the
      safetensors; assert its parsed JSON has `schema_version=4`,
      a populated `n_clusters`, and a populated
      `cluster_assignments` of length `n_features_input`.
- [ ] 4.2.2 `test_sidecar_assignments_consistent_with_zeroed_features`:
      cross-check that every feature id at `cluster_assignments[i] == -1`
      has the corresponding `W_dec` row in the compressed state at
      zero (or near zero per polygram's zero strategy).
- [ ] 4.2.3 `test_sidecar_n_clusters_matches_distinct_ids`:
      `n_clusters == len(set(cid for cid in cluster_assignments if cid >= 0))`.
- [ ] 4.2.4 `test_atomic_sidecar_write_no_orphans_on_failure`:
      monkeypatch the JSON write to fail; assert the safetensors
      temp file is also cleaned up and no partial artefacts remain.

### 4.3 Frozen fixtures

- [ ] 4.3.1 Regenerate `tests/compression/data/epoch_result_reference.json`
      and `epoch_result_reference_multi_iter.json` with the new
      schema_version=4 + populated `n_clusters` / `cluster_assignments`.
      Same regeneration approach used by the v0.11.0 release for
      the rank_ratio / post_A / forge_mse / informative_metric
      additions.
- [ ] 4.3.2 `test_byte_identical_epoch_result_against_frozen_reference`
      continues to pass after the regeneration.

## 5. Spec

- [ ] 5.1 Author `specs/pareto-compression/spec.md` delta:
      MODIFIED `EpochReport` requirement to include the two new fields
      and the schema bump; MODIFIED `EpochCompressor.run` requirement
      to also write the sidecar.

## 6. Docs

- [ ] 6.1 `CHANGELOG.md` entry under `[Unreleased]` → `### Added
      (emit-cluster-metadata-from-epoch-compressor)`.

## 7. Validation

- [ ] 7.1 `openspec validate emit-cluster-metadata-from-epoch-compressor --strict` is green.
- [ ] 7.2 Full `pytest tests/` is green (existing test count + new
      cases, no regressions).
- [ ] 7.3 `ruff check polygram/ tests/` clean on touched files.
- [ ] 7.4 Live re-run of sae-forge's §8.4 paired smoke (from PR #69's
      comment) WITHOUT the synthetic cluster injection workaround;
      assert `basis.metadata['n_clusters'] >= 2` and the
      concept-anchored fine-tune runs end-to-end against real
      polygram metadata.
- [ ] 7.5 `openspec archive emit-cluster-metadata-from-epoch-compressor`
      after merge.

## 8. Coordination with sae-forge

- [ ] 8.1 No sae-forge code changes required — `FeatureBasis.from_polygram_checkpoint`
      already looks for the sidecar suffix `_compression_report.json`
      and surfaces its contents into `basis.metadata`. The
      `cluster_assignments` field will appear in `basis.metadata`
      once the polygram side ships.
- [ ] 8.2 After this change ships and lands in a polygram tag, the
      sae-forge concept-anchored test should be re-run against a
      real EpochCompressor-built basis; the round-robin warning
      stops firing and the supervised signal aligns with the real
      cluster structure.

## 9. What this change explicitly defers

- [ ] 9.1 Mirroring `cluster_assignments` onto `CompressionReport`
      (the single-shot `Compressor.apply` path). It's already
      derivable from `plan.clusters[*].members`; sae-forge can
      compute it client-side. File a separate proposal if a real
      consumer asks for the field directly.
- [ ] 9.2 safetensors-header metadata. Embedding `n_clusters` /
      `cluster_assignments` into the safetensors header so the
      .safetensors is fully self-describing without a sidecar.
      Out of scope for v1.
- [ ] 9.3 Per-iteration cluster history. Only the final-iteration
      assignments are surfaced; per-iteration debugging info stays
      inside `EpochIteration`.
