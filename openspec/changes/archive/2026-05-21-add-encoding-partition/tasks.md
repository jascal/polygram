# tasks — add-encoding-partition

## 1. `BlockSpec` dataclass

- [ ] 1.1 Add `BlockSpec` to `polygram/compression/config.py` (or a
      new module if `config.py` is getting crowded). Frozen dataclass
      with fields: `block_id: str`, `encoding_class: str`,
      `encoding_kwargs: dict[str, Any]`, `learn_axis_assignment: bool`,
      `feature_ids: tuple[int, ...]`.
- [ ] 1.2 `__post_init__` validation — uses the per-family validator
      registry (see Decision 2b in design.md):
      - `encoding_class` ∈ `{"MPSRung1", "Rung3", "Rung4", "Rung5", "HEA_Rung2"}`.
      - `encoding_kwargs` shape dispatched through a module-level
        `_BLOCK_SPEC_KWARG_VALIDATORS: dict[str, Callable[[dict], None]]`
        registry rather than an inline match/elif chain. Adding a
        future family is one new validator function + one registry
        entry.
      - `feature_ids` is a non-empty tuple of non-negative ints, no
        duplicates.
- [ ] 1.3 `BlockSpec.__eq__` / `__hash__` cover all fields
      (frozen=True gives this automatically; just confirm the dict
      and tuple fields hash correctly — tuples do; dicts don't, so
      convert `encoding_kwargs` to a frozenset of items for hash).
- [ ] 1.4 `make_default_block(...)` convenience constructor (see
      Decision 2c in design.md): produces a BlockSpec covering all
      `range(n_features_input)` minus `excluded_feature_ids`. Useful
      for "default + heavy override" partition patterns. Returns a
      regular BlockSpec; no special-case path in downstream consumers.

## 2. `CompressionConfig.encoding_partition`

- [ ] 2.1 Add `encoding_partition: tuple[BlockSpec, ...] | None = None`
      to `CompressionConfig`.
- [ ] 2.2 `CompressionConfig.__post_init__` extension:
      - When `encoding_partition is not None`, verify it's a tuple of
        `BlockSpec` instances. Other iterables raise `TypeError`.
      - Coverage validation requires `n_features_input` which isn't
        known at config-construction time → defer to `Compressor.apply`.
      - Document the "partition + top-level default" pattern.

## 3. Partition-coverage validation

- [ ] 3.1 New helper
      `polygram/compression/partition.py::validate_partition_coverage(
        partition: tuple[BlockSpec, ...],
        n_features_input: int,
      ) -> None`.
      - Builds the union of all blocks' `feature_ids` as a Python set.
      - Disjointness: track first 10 duplicate ids across blocks; on
        violation raise `PartitionCoverageError` naming the
        duplicates.
      - Completeness: compute `expected = set(range(n_features_input))`;
        on `expected - union`, raise naming missing ids (first 10);
        on `union - expected`, raise naming extras (first 10).
- [ ] 3.2 `PartitionCoverageError` exception class (new) extends
      `ValueError`.

## 4. `Compressor.apply()` per-block dispatch

- [ ] 4.1 In `polygram/compression/compressor.py::Compressor.apply`,
      detect `config.encoding_partition is not None` and branch into
      a `_apply_partitioned(...)` private method.
- [ ] 4.2 `_apply_partitioned(sae_state, validation_report, config)`:
      1. Call `validate_partition_coverage(config.encoding_partition,
         sae_state["W_dec"].shape[0])`.
      2. For each block: slice the relevant rows/columns of
         W_dec / W_enc / b_enc out into a per-block state-dict.
      3. Build a per-block sub-`CompressionConfig`: encoding_class
         + kwargs + learn_axis_assignment from the block; everything
         else from the original config.
      4. Recursively call `Compressor.apply` (or factor out
         `_apply_single_encoding`) on each sub-state.
      5. Per-block returns yield `(compressed_sae, sub_report)`.
      6. Stitch: produce a single output W_dec / W_enc / b_enc / b_dec
         by writing each block's compressed rows back into the
         original feature_ids positions. b_dec is invariant under
         feature-axis slicing — verify the per-block b_dec values
         agree (they should; the strategy doesn't touch b_dec by
         construction) and pick block 0's.
      7. Aggregate the sub-reports into a CompressionReport with
         the new `blocks` field populated.

## 5. `CompressionReport.blocks` field

- [ ] 5.1 New `BlockReport` dataclass in
      `polygram/compression/report.py`:
      ```python
      @dataclass(frozen=True)
      class BlockReport:
          block_id: str
          encoding_class: str
          encoding_kwargs: dict[str, Any]
          learn_axis_assignment: bool
          feature_ids: tuple[int, ...]
          n_features_kept: int
          n_features_zeroed: int
          n_clusters: int
          cluster_assignments: tuple[int, ...] | None
          scale_compression_ratio: float
          rank_ratio: float | None
          post_A: float | None
          forge_mse: float | None
          informative_metric: Literal["post_A", "both", "forge_mse"] | None
      ```
- [ ] 5.2 Add `blocks: tuple[BlockReport, ...] | None = None` to
      `CompressionReport`.
- [ ] 5.3 `CompressionReport.SCHEMA_VERSION 5 → 6`. Back-compat loader
      defaults `blocks=None` when key is absent.
- [ ] 5.4 Equality / hash include `blocks` (when populated; tuple of
      frozen BlockReports hashes naturally).
- [ ] 5.5 JSON serialisation of `blocks` is a list-of-dicts (each
      BlockReport is dict-ified via standard fields).

## 6. Top-level aggregation in CompressionReport (when partitioned)

- [ ] 6.1 `n_features_kept` aggregates as sum of per-block.
- [ ] 6.2 `n_features_zeroed` aggregates as sum of per-block.
- [ ] 6.3 `n_clusters` aggregates as sum of per-block.
- [ ] 6.4 `scale_compression_ratio`: feature-count-weighted geometric
      mean of per-block ratios.
- [ ] 6.5 `cluster_assignments`: global ids per Decision 2 (block_idx
      × MAX_CLUSTERS_PER_BLOCK + local_id). `MAX_CLUSTERS_PER_BLOCK = 10000`
      module-level constant in `report.py`.
- [ ] 6.6 `rank_ratio` / `post_A` / `forge_mse` aggregate as
      feature-count-weighted means (when defined per-block).
- [ ] 6.7 `informative_metric`: set to `None` at top-level when
      partitioned (the per-block reports have their own).

## 7. `from_sae_lens(..., encoding_partition=...)`

- [ ] 7.1 In `polygram/sae_import.py`, extend `from_sae_lens`
      signature with `encoding_partition: tuple[BlockSpec, ...] | None = None`.
- [ ] 7.2 Thread it into the constructed `CompressionConfig`.
- [ ] 7.3 Same-call mutual exclusion: NOT enforced (per Decision 6 —
      they're complementary).

## 8. `EpochCompressor.run` refusal

- [ ] 8.1 In `polygram/compression/epoch.py::EpochCompressor.run`,
      add a defensive check at the top: if
      `self.config.encoding_partition is not None`, raise
      `NotImplementedError` with a clear message naming the v2
      follow-up change (`epoch-compressor-encoding-partition`).

## 9. Tests

### 9.1 `BlockSpec` validation

- [ ] 9.1.1 `test_block_spec_rejects_unknown_encoding_class` —
      `BlockSpec(..., encoding_class="MadeUp", ...)` raises ValueError.
- [ ] 9.1.2 `test_block_spec_rejects_missing_n_amp_qubits_for_rung5` —
      `BlockSpec(encoding_class="Rung5", encoding_kwargs={}, ...)`
      raises ValueError.
- [ ] 9.1.3 `test_block_spec_rejects_extra_kwargs_for_mps_rung1` —
      `BlockSpec(encoding_class="MPSRung1", encoding_kwargs={"foo": 1}, ...)`
      raises ValueError.
- [ ] 9.1.4 `test_block_spec_rejects_empty_feature_ids` — empty tuple
      raises ValueError.
- [ ] 9.1.5 `test_block_spec_rejects_duplicate_feature_ids_within_block`
      — `feature_ids=(1, 2, 1)` raises ValueError.
- [ ] 9.1.6 `test_block_spec_rejects_negative_feature_id` — `(0, -1)`
      raises ValueError.

### 9.2 Partition coverage

- [ ] 9.2.1 `test_partition_coverage_disjoint_complete_passes` —
      two blocks splitting `range(8)` with no overlap pass.
- [ ] 9.2.2 `test_partition_coverage_overlap_raises_naming_dupes` —
      blocks sharing id 3 raise `PartitionCoverageError` mentioning 3.
- [ ] 9.2.3 `test_partition_coverage_missing_raises_naming_holes` —
      blocks covering only `{0,1,2,4,5,6,7}` (missing 3) raise.
- [ ] 9.2.4 `test_partition_coverage_extras_raises_naming_extras` —
      blocks covering `{0..7, 99}` (extra 99) raise.

### 9.3 `Compressor.apply` partitioned dispatch

- [ ] 9.3.1 `test_apply_partitioned_round_trip_w_dec` — feed an SAE
      with 16 features into a 2-block partition (heavy:{0,1,2,3} with
      Rung4, tail:{4..15} with MPSRung1). Assert output W_dec has the
      heavy rows compressed under Rung4-style cluster geometry, tail
      rows under MPSRung1. Hash the per-block subarrays separately.
- [ ] 9.3.2 `test_apply_partitioned_matches_single_block_dispatch` —
      a partition with ONE block covering all features SHALL produce
      output bit-identical to single-encoding apply with that block's
      encoding. Round-trip via JSON-report comparison.
- [ ] 9.3.3 `test_apply_partitioned_independent_strategies` — verify
      block A's `strategy="zero"` and block B's `strategy="merge"`
      are applied independently (peak at `n_features_zeroed` per
      block in `BlockReport`).
- [ ] 9.3.4 `test_apply_partitioned_b_dec_invariant` — `b_dec` is the
      same in all per-block sub-states (since strategies don't touch
      it); top-level output's `b_dec` matches the input's.

### 9.4 `CompressionReport.blocks` round-trip

- [ ] 9.4.1 `test_compression_report_v6_blocks_round_trip` — populate
      a CompressionReport with 2 BlockReports, round-trip via
      to_json/from_json, assert equality.
- [ ] 9.4.2 `test_compression_report_v5_loads_with_blocks_none` —
      hand-craft v5 JSON without `blocks` key, load via from_json,
      assert blocks is None. Use a v5 payload pulled from the
      pre-change frozen fixtures (or a synthesised one matching
      the v5 schema's set of keys) to guarantee the back-compat
      contract against a REAL pre-change payload, not just a stub.
- [ ] 9.4.3 `test_compression_report_blocks_round_trip_with_nan_aware_eq`
      — populate with `rank_ratio=float('nan')` in a block; round-trip;
      equality holds (NaN-aware `__eq__`).
- [ ] 9.4.4 `test_compression_report_v6_blocks_explicitly_none_round_trip`
      — populate with `blocks=None` and schema_version=6; round-trip;
      `blocks is None`. Confirms the v6 schema accepts the absent
      case (so an unpartitioned compress produces a v6 report with
      `blocks=None`, indistinguishable from the v5 contract).

### 9.5 `from_sae_lens` kwarg

- [ ] 9.5.1 `test_from_sae_lens_accepts_encoding_partition` — call
      with a 2-block partition; assert the constructed
      `CompressionConfig.encoding_partition` equals the input.
- [ ] 9.5.2 `test_from_sae_lens_partition_overrides_top_level` —
      pass both `encoding_class="MPSRung1"` AND a partition with one
      Rung4 block. Assert per-block encoding wins.

### 9.6 EpochCompressor refusal

- [ ] 9.6.1 `test_epoch_compressor_refuses_encoding_partition` —
      construct EpochCompressor with a partitioned CompressionConfig
      (via the helper that builds the config); calling `.run()`
      raises `NotImplementedError` mentioning `epoch-compressor-encoding-partition`.

## 10. Documentation

- [ ] 10.1 `CHANGELOG.md` entry under `[Unreleased]`:
      `### Added (add-encoding-partition)`.
- [ ] 10.2 Inline docstrings on `BlockSpec`, `validate_partition_coverage`,
      `CompressionReport.blocks`, and the per-block branch of
      `Compressor.apply`.
- [ ] 10.3 `openspec/changes/add-encoding-partition/specs/pareto-compression/spec.md`
      delta: NEW Requirement "Compressor.apply supports
      encoding_partition" with 4 scenarios from §9 above.

## 11. Validation

- [ ] 11.1 `openspec validate add-encoding-partition --strict` is green.
- [ ] 11.2 Full pytest suite is green (1020 → ~1040 cases).
- [ ] 11.3 Ruff clean on touched files.
- [ ] 11.4 No regression on existing `Compressor.apply` byte-identity
      tests (the new code path is only exercised when
      `encoding_partition is not None`).

## 12. Release coordination

- [ ] 12.1 Tag `polygram v0.13.0` after merge.
- [ ] 12.2 PyPI auto-publish via the existing release workflow.
- [ ] 12.3 Update sae-forge's `polygram>=0.13.0` pin in
      `add-block-structured-sae`'s Phase 0.1 task — that unblocks
      the downstream impl.

## 13. What this change explicitly DEFERS

- [ ] 13.1 `EpochCompressor.run(encoding_partition=...)` support
      — `epoch-compressor-encoding-partition` v2 follow-up.
- [ ] 13.2 Auto-partitioning heuristics ("partition by firing rate").
- [ ] 13.3 safetensors-header per-block metadata.
- [ ] 13.4 Per-feature (not per-block) heterogeneity. v3+ optimisation.
