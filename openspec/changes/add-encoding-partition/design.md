# design — add-encoding-partition

## Context

The decision tree the analyst faces under the current single-encoding
regime is binary at the dictionary level: either pay the parameter
budget of the most-demanding feature uniformly across all features, or
choose a smaller budget and accept poor reconstruction on the heavy
tail. The sae-forge `quality_tier="degenerate"` rows from
`add-forge-quality-diagnostics` already confound two distinct
failure modes in this regime:

1. K (kept-feature count) is too small for the host's residual rank.
2. The chosen encoding capacity is too small for the kept features.

Per-block encoding lets the analyst distinguish these — heavy features
get the big bucket, tail features get the small one, both at the same
K. This is the "concrete frontier-row knob" sae-forge's
`add-block-structured-sae` proposes; polygram has to ship the
underlying capability first.

## Goals / Non-Goals

**Goals:**

- Lock a stable per-block API surface that sae-forge's downstream
  change can wire to without coordinating subsequent renames.
- Support arbitrary partitions of the feature space (analyst-supplied
  via a manifest; polygram doesn't itself choose the partitioning).
- Preserve the on-disk safetensors layout (single tensor per key, same
  shape as input). Per-block structure lives in the sidecar report.
- Schema-versioned, back-compat-loadable `CompressionReport`.
- Same atomic-write + sidecar-emission convention used by
  `Compressor.apply()` and `EpochCompressor.run()` today.

**Non-Goals:**

- Picking the partition automatically. Analyst-supplied.
- Heterogeneous encoding in `EpochCompressor.run()`. v2 follow-up.
- Per-feature (rather than per-block) heterogeneity. Per-block is the
  v1 granularity; per-feature is a v3+ optimisation.
- Cross-block constraints (e.g., "the per-block parameter budgets must
  sum to a fixed total"). Out of scope; analyst chooses the budgets.

## Decisions

### Decision 1 — On-disk layout: single tensors with per-block report metadata

Output safetensors is a single `{W_dec, W_enc, b_enc, b_dec}` set
matching the input shape. The per-block structure is recorded in the
`_compression_report.json` sidecar.

**Rationale**: keeping the safetensors layout unchanged preserves the
round-trip contract with `FeatureBasis.from_polygram_checkpoint` (no
per-block reader logic needed downstream) and with any other consumer
that just wants the W_dec geometry. The per-block info is for
analysts inspecting the report; the runtime path doesn't need it.

**Alternative considered**: per-block sub-tables in the safetensors
(e.g., `block_heavy.W_dec`, `block_tail.W_dec`). Rejected — every
downstream consumer would need a per-block reader, and the safetensors
format's grouping primitives are awkward.

### Decision 2 — Global cluster-id namespace in top-level cluster_assignments

When `encoding_partition` is set, the top-level
`CompressionReport.cluster_assignments` field uses GLOBAL cluster ids
computed as:

```
global_id(block_index, local_cluster_id) = block_index * MAX_CLUSTERS_PER_BLOCK + local_cluster_id
```

where `MAX_CLUSTERS_PER_BLOCK = 10000` (defensive cap; reality is
~hundreds per block). The per-block `BlockReport.cluster_assignments`
field carries LOCAL ids in `[0, n_clusters_in_block)`.

**Rationale**: the `add-concept-anchored-finetune` consumer (sae-forge
PR #69) reads `cluster_assignments` as a flat per-feature integer
array; preserving that contract means cluster ids must be unique
globally. Global encoding via `block_index * MAX + local` is
deterministic and reversible.

**Alternative considered**: have the consumer concatenate per-block
arrays itself. Rejected — increases coupling between polygram's report
schema and downstream consumers' code paths.

### Decision 2b — Extensibility: per-family validator registry

`BlockSpec.__post_init__` validates `encoding_kwargs` shape per family
(Rung5 needs `n_amp_qubits`, HEA_Rung2 needs `n_qubits`, MPSRung1/3/4
need empty kwargs). This is currently a `match`-style branch over the
5 supported encoding classes. As polygram adds new families (Rung6,
larger HEA variants, etc.), that branch becomes a maintenance trap.

The implementation SHALL use a module-level validator registry:

```python
_BLOCK_SPEC_KWARG_VALIDATORS: dict[str, Callable[[dict], None]] = {
    "MPSRung1": _validate_empty_kwargs,
    "Rung3":    _validate_empty_kwargs,
    "Rung4":    _validate_empty_kwargs,
    "Rung5":    _validate_rung5_kwargs,        # checks n_amp_qubits: int >= 1
    "HEA_Rung2": _validate_hea_rung2_kwargs,   # checks n_qubits: int >= 1
}
```

Adding a future family is one new validator function + one registry
entry, not a new `elif` branch.

The registry SHALL be private to `polygram/compression/partition.py`
(or wherever `BlockSpec` lives) — not a public extension point.
Encoding families are a polygram-controlled vocabulary; opening it
for third-party extension would weaken the "supported encoding set"
contract.

### Decision 2c — Ergonomic helpers for common partition patterns

The "default + heavy override" pattern is common enough to deserve a
helper. The implementation SHALL ship one convenience constructor:

```python
def make_default_block(
    *,
    block_id: str = "default",
    encoding_class: Literal[...],
    encoding_kwargs: dict[str, Any] = None,
    learn_axis_assignment: bool = False,
    n_features_input: int,
    excluded_feature_ids: set[int] = frozenset(),
) -> BlockSpec:
    """Build a BlockSpec covering all feature_ids NOT in
    `excluded_feature_ids`. Useful for 'tail bucket' partitions where
    a small set of heavy features get a custom block and the rest
    falls through to a default.

    Example:
        heavy = BlockSpec(block_id="heavy", encoding_class="Rung5",
                          encoding_kwargs={"n_amp_qubits": 4},
                          feature_ids=(0, 1, 2, 3))
        tail  = make_default_block(
            encoding_class="MPSRung1",
            n_features_input=n_features,
            excluded_feature_ids={0, 1, 2, 3},
        )
        partition = (heavy, tail)
    """
```

This is sugar over `BlockSpec(feature_ids=tuple(sorted(
range(n_features_input)) - excluded_feature_ids))`. The naming and
the explicit `n_features_input` argument make the "I want everything
that's not in the heavy block" intent visually clear.

### Decision 3 — Mutual-exclusion semantics: partition overrides top-level

When `encoding_partition` is set, top-level `encoding_class` /
`encoding_kwargs` / `learn_axis_assignment` SHALL be silently treated
as defaults that the per-block specs override. This permits ergonomic
"default + override" patterns:

```python
CompressionConfig(
    encoding_class="MPSRung1",         # the tail's encoding
    encoding_partition=(
        BlockSpec(block_id="heavy", encoding_class="Rung5",
                  encoding_kwargs={"n_amp_qubits": 4}, ...),
        # any features NOT in 'heavy' fall through to MPSRung1
    ),
)
```

But: **partition coverage validation still requires every feature to
appear in some block**. The "fall through" semantics above don't
implicitly create a "default block" — that would be the analyst's
responsibility to add as an explicit block in the partition.

**Rationale**: implicit default blocks make the partition harder to
audit; explicit is better.

### Decision 4 — Schema bump CompressionReport v5 → v6

`CompressionReport.SCHEMA_VERSION` bumps from 5 to 6. The new field is
`blocks: tuple[BlockReport, ...] | None = None`. `from_json` accepts
v5 payloads (no `blocks` key) and defaults to `None`. Equality / hash
include the new field.

`BlockReport`'s own schema is locked to whatever fields it has at v6
ship; future per-block field additions get their own schema bump.

### Decision 5 — `EpochCompressor` refusal, not silent acceptance

`EpochCompressor.run()` MUST raise `NotImplementedError` when called
with `encoding_partition` set, naming the v2 follow-up change
(`epoch-compressor-encoding-partition`).

**Rationale**: silently dropping the partition (treating
`encoding_partition` as if it were `None`) would surprise the user.
The error is loud and points at the right next step.

### Decision 6 — `from_sae_lens` API: same-call mutual exclusion

`from_sae_lens(...)` accepts `encoding_partition` as a new kwarg. When
present:

- The per-encoding kwargs (`encoding_class`, `encoding_kwargs`,
  `learn_axis_assignment`) MAY be present as defaults; partition
  overrides them per block.

- Passing both `encoding_class` and `encoding_partition` is NOT an
  error (they're complementary).

- Passing `encoding_partition` with overlapping feature_ids across
  blocks raises `PartitionCoverageError` (caught in `BlockSpec`
  validation).

## Risks / Trade-offs

- **Test combinatorics**: each strategy (zero, merge) × each encoding
  family (5 of them) × partition vs single-encoding = 20 base
  combinations. Mitigation: a parametrised "smoke matrix" at one tiny
  SAE size (n_features=16, d_model=8) covers the cross-product in
  ~5s of test runtime.

- **Cluster-id namespace collisions if MAX_CLUSTERS_PER_BLOCK is too
  small**: in production we expect ~100-1000 clusters per block, so
  10000 is a 10x-100x safety margin. If a future analyst pushes
  beyond that, the cap can be bumped (it's a per-schema constant).

- **`scale_compression_ratio` aggregation**: top-level value is the
  feature-count-weighted geometric mean of per-block ratios. The
  weighting matches the "average bytes-per-feature" semantics.

- **`rank_ratio` aggregation**: per-block rank_ratios are weighted
  averages of block-kept-feature-counts. Documented but not
  load-bearing.

## Migration

- **No migration required for existing callers.** `encoding_partition`
  defaults to `None`; all existing code paths stay byte-equivalent.

- **Existing `_compression_report.json` files** stay valid; their
  schema version is `5` and the v5 loader returns `blocks=None`.

- **`EpochCompressor` callers** are unaffected unless they pass a
  partitioned config (which they wouldn't have done before this
  change; the option didn't exist).

- **Downstream consumers** (sae-forge's `FeatureBasis`) read
  `cluster_assignments` the same way; the global-id encoding is
  transparent to them.
