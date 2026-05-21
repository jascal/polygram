# add-encoding-partition

## Why

Today every feature in a single polygram-compressed SAE dictionary is
allocated the same encoding family and the same axis-assignment policy.
The `CompressionConfig.encoding_class` choice picks `MPSRung1` (cap=8),
`Rung3` (cap=16), `Rung4` (cap=32), `Rung5(k)` (cap=8·2^k), or
`HEA_Rung2(n_qubits=N)` (cap=2^N) for the *whole* dictionary. `learn_axis_assignment`
is similarly all-or-nothing.

Real SAE dictionaries are not uniform across features. Heavy-hitting
features with broad decoder support and high firing rates plausibly need
more substrate than a feature that fires once per ten thousand tokens on
a single neighbourhood. The current single-encoding regime forces the
analyst to pick the *worst-case* budget for the *best-case* feature:
choosing `Rung5(k=4)` over `MPSRung1` because a handful of features
need the capacity inflates the parameter cost for the entire
dictionary, including the long tail where it isn't needed. Choosing
`MPSRung1` puts the heavy features in a bucket too small to reconstruct
cleanly. Neither choice is right for both subsets simultaneously.

**This change ships the polygram-side capability to support per-block
heterogeneous encoding within a single SAE dictionary.** It does not
itself ship a partitioning recipe (how to decide which features go in
which block) — that's a downstream analyst-driven decision encoded in a
manifest. The sae-forge `add-block-structured-sae` change (which has
been on file pending this prerequisite) consumes the API surface this
change locks in.

P1 — this is the gating prerequisite for sae-forge's per-block
heterogeneous-encoding feature. Without it, the Wave C feature roadmap
stays blocked.

## What Changes

### `BlockSpec` dataclass (new)

A single block's encoding + axis-assignment + feature-id list:

```python
@dataclass(frozen=True)
class BlockSpec:
    block_id: str
    encoding_class: Literal["MPSRung1", "Rung3", "Rung4", "Rung5", "HEA_Rung2"]
    encoding_kwargs: dict[str, Any]  # e.g. {"n_amp_qubits": 4} for Rung5
    learn_axis_assignment: bool
    feature_ids: tuple[int, ...]
```

Validation in `__post_init__`:

- `encoding_class` ∈ the supported set above. Mismatch → `ValueError`.
- `encoding_kwargs` shape per family:
  - `Rung5` requires `n_amp_qubits: int`.
  - `HEA_Rung2` requires `n_qubits: int`.
  - `MPSRung1` / `Rung3` / `Rung4` require an empty dict.
- `feature_ids` is non-empty, contains only non-negative ints, and is
  internally unique (no duplicate ids within the block).

### `CompressionConfig.encoding_partition` (new optional field)

```python
@dataclass
class CompressionConfig:
    encoding_class: Literal[...] | None = None
    encoding_kwargs: dict[str, Any] = field(default_factory=dict)
    learn_axis_assignment: bool = False
    encoding_partition: tuple[BlockSpec, ...] | None = None  # NEW
    ...
```

When `encoding_partition` is set:

- `encoding_class`, `encoding_kwargs`, `learn_axis_assignment` SHALL be
  treated as defaults (or left at their None / empty / False zero
  values); per-block values from the partition take precedence.
- Coverage validation: the union of all blocks' `feature_ids` SHALL
  cover every input feature exactly once. Overlap or omission → raise
  `PartitionCoverageError` at config-validation time (not compress
  time), naming the offending ids capped at first 10.

Mutual-exclusion: setting BOTH a per-block partition AND a top-level
`encoding_class` simultaneously is **not** an error — the top-level
serves as a "default" and the partition overrides per block. This
permits ergonomic patterns like "the heavy features get Rung5, the
rest defaults to MPSRung1."

### `from_sae_lens(..., encoding_partition=...)` (new kwarg)

The high-level entry point gains an `encoding_partition` kwarg that is
threaded into the `CompressionConfig` it builds. When supplied, the
single-encoding kwargs (`encoding_class`, `encoding_kwargs`,
`learn_axis_assignment`) on `from_sae_lens` SHALL be either omitted or
used as the default tier. Mutually exclusive with passing them
positionally; a clear `TypeError` fires when both forms are present
for the same encoding axis.

### `Compressor.apply()` per-block dispatch

`Compressor.apply()` SHALL accept a config carrying
`encoding_partition`. The per-block dispatch:

1. For each block: slice the input SAE checkpoint's `W_dec`/`W_enc`/etc.
   down to the block's `feature_ids`.
2. Build a per-block sub-`CompressionConfig` from the BlockSpec's
   encoding + kwargs + axis-assignment policy.
3. Run the existing compress-strategy pipeline on the sub-config.
4. Stitch the per-block results back into a single output SAE:
   - Features outside any block (which there cannot be, given coverage
     validation, but defensive) → pass through untouched.
   - Per-block compressed rows → assigned back to their original
     feature_ids.
5. The output safetensors has the SAME shape as the input
   (single-tensor per key: W_dec / W_enc / b_enc / b_dec). The per-block
   structure is recorded in the compression report.

The on-disk safetensors layout is unchanged — no per-block sub-tables.
This preserves the round-trip contract with `FeatureBasis.from_polygram_checkpoint`
and the existing `_compression_report.json` sidecar convention.

### `CompressionReport` per-block sections (extension)

`CompressionReport` gains:

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
    # Per-block diagnostic mirrors of the top-level fields
    rank_ratio: float | None
    post_A: float | None
    forge_mse: float | None
    informative_metric: Literal["post_A", "both", "forge_mse"] | None


@dataclass(frozen=True)
class CompressionReport:
    ...
    blocks: tuple[BlockReport, ...] | None = None  # NEW
```

Schema bump `CompressionReport.SCHEMA_VERSION 5 → 6` with a back-compat
loader for v5 payloads (defaults `blocks=None`).

When `encoding_partition` is set, `blocks` is populated; the top-level
fields aggregate across blocks (sum for counts; weighted average for
ratios where meaningful; `None` for fields that don't aggregate
sensibly like `informative_metric`).

When `encoding_partition` is None, `blocks` is `None` and behaviour is
unchanged from v5.

### `EpochCompressor.run()` — NOT YET supported

`EpochCompressor.run()` SHALL refuse a `CompressionConfig` carrying
`encoding_partition` with a clear `NotImplementedError`:

> "EpochCompressor does not yet support encoding_partition. Use
> Compressor.apply directly for partitioned compression in v1. Filing
> `epoch-compressor-encoding-partition` is the documented v2
> follow-up."

The epoch iterator's panel-selection + per-iteration cluster
fingerprinting is its own complex machinery; mixing per-block
heterogeneous encoding into that flow is out of scope for v1.

## Capabilities

### Modified Capabilities

- `pareto-compression`:
  - `CompressionConfig` gains optional `encoding_partition` field.
  - New `BlockSpec` dataclass with coverage + kwargs validation.
  - `Compressor.apply()` per-block dispatch when partition is set.
  - `CompressionReport.blocks` field; schema bump v5 → v6.
  - `from_sae_lens` accepts `encoding_partition` kwarg.
  - `EpochCompressor.run` refuses partition (NotImplementedError; v2 follow-up).

## Impact

- `polygram/compression/config.py` — `CompressionConfig.encoding_partition` field + `BlockSpec` dataclass + validation helpers (~100 lines).
- `polygram/compression/compressor.py` — `Compressor.apply` per-block dispatch + stitching logic (~150 lines).
- `polygram/compression/report.py` — `BlockReport` dataclass + `CompressionReport.blocks` + schema bump 5→6 + back-compat loader (~80 lines).
- `polygram/compression/epoch.py` — `EpochCompressor.run` refusal branch (~10 lines).
- `polygram/sae_import.py` — `from_sae_lens` accepts `encoding_partition` (~20 lines).
- New tests under `tests/compression/`:
  - `test_block_spec.py` — BlockSpec validation (kwargs per family, feature_ids non-empty, no duplicates).
  - `test_encoding_partition_coverage.py` — partition coverage validation (overlap, omission).
  - `test_compressor_apply_partitioned.py` — per-block dispatch produces correct sliced W_dec rows; stitching restores original feature_ids ordering.
  - `test_compression_report_blocks_roundtrip.py` — v5 back-compat + v6 round-trip + per-block fields populated correctly.
  - `test_epoch_compressor_partition_refusal.py` — clean error message when partition supplied to EpochCompressor.

No breaking changes; schema bump has back-compat loader; new fields are additive.

## Risks / Trade-offs

- **Test fixture proliferation**: each compression strategy × each encoding family × partition vs single-encoding = many combinations. Mitigation: shared parametrised fixtures + a "smoke matrix" that covers the cross-product at one tiny SAE size.
- **Cluster-id namespace** across blocks: a feature in block A's cluster 3 is unrelated to a feature in block B's cluster 3. The aggregated `cluster_assignments` in the top-level report uses GLOBAL cluster ids (block_index * MAX_CLUSTERS_PER_BLOCK + block_local_id), with per-block report carrying the local ids. Documented in the design; alternative would be only emitting per-block IDs but breaks the existing `add-concept-anchored-finetune` consumer's expectation of a flat per-feature cluster_id.
- **Per-block convergence diagnostics**: `rank_ratio` / `post_A` are computed per-block; the top-level aggregate uses feature-count-weighted means. Documented in the design.
- **`scale_compression_ratio` semantics**: per-block ratio is well-defined (norm preservation within the block); top-level aggregate is a weighted product. Documented in the design.

## Out of scope (deferred follow-ups)

- **EpochCompressor partition support** — file `epoch-compressor-encoding-partition` after this lands. The iterative panel selection + cluster fingerprinting needs per-block extension.
- **Auto-partitioning heuristics** (e.g., "auto-cluster by firing rate"). Out of scope; the partition is analyst-supplied via the manifest.
- **safetensors-header per-block metadata**. Sidecar is sufficient.
- **`scale_compression_ratio` mixed-block monotonicity guarantees**. Per-block monotonicity holds within each block; cross-block ordering is left to the analyst.

## Acceptance

This change ships when:

1. `BlockSpec` + `CompressionConfig.encoding_partition` validation is in place.
2. `Compressor.apply()` correctly dispatches per-block and stitches output (round-trip test passes against a hand-crafted 2-block partition).
3. `CompressionReport.blocks` round-trips through JSON; schema-v5 payloads still load.
4. `from_sae_lens(encoding_partition=...)` is accepted and threaded through.
5. `EpochCompressor.run` refuses with a clear NotImplementedError when partition is set.
6. Full polygram test suite passes (~1020 tests pre-change; ~1040 expected post-change with the new cases).
7. The sae-forge `add-block-structured-sae` change can begin its Phase 2 implementation against the locked API surface.

## Coordination with sae-forge

sae-forge's `add-block-structured-sae/proposal.md` references this proposal under "Phase 0 (blocking)". Once this lands and tags as polygram v0.13.0 (or whichever version ships the contract), sae-forge can bump its pin and start Phase 2.

Per-block API surface locked in this proposal:

- `CompressionConfig.encoding_partition: tuple[BlockSpec, ...] | None`
- `BlockSpec(block_id, encoding_class, encoding_kwargs, learn_axis_assignment, feature_ids)`
- `from_sae_lens(..., encoding_partition=...)`

Drift in any of these field names blocks the downstream sae-forge implementation. The two changes' API contracts are co-locked.

### Note on `tuple` vs `list` (corrects sae-forge proposal's shorthand)

The sae-forge `add-block-structured-sae/tasks.md` task 0.2 cites the
locked surface as `CompressionConfig.encoding_partition: list[BlockSpec] | None`.
**This proposal corrects that to `tuple[BlockSpec, ...]`** — frozen
tuples make `CompressionConfig` (and `BlockSpec`) hashable, which is
load-bearing for the auto-materialise cache-key contract in sae-forge
(`compute_cache_key` records the partition's SHA-256, derived from a
deterministic serialisation that requires hashable members).

The sae-forge change's tasks.md will be updated to match
`tuple[BlockSpec, ...]` when its impl PR lands; flagging the
correction here so downstream implementers don't carry the `list`
typing into the wired code.
