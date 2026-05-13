## Why

`clustered-dictionary-analysis` (PR #43 / #44) shipped the
`ClusteredDictionary` primitive as the v1 abstraction for SAE-scale
block-decomposed analyses. §7 of that change *promised* to refactor
`EpochCompressor`'s panel-decomposition internals
(`_select_panels`, `_validate_panels`, `_synthesize_validation_report`)
into wrappers over `ClusteredDictionary` methods so compression
would become **one consumer** of the shared primitive.

What actually shipped is a **one-way adapter**: `ClusteredDictionary.from_compression_panels`
takes the *output* of `_select_panels` and wraps it as a
`ClusteredDictionary` view. Compression's internals remained
completely untouched. The deferral was deliberate: the priority-driven
seeded-coverage algorithm in `_select_panels` has visit-tracking,
coverage-target early termination, anchor-only fallback with warning,
neighbour cap (≤7), and Panel-specific output. A byte-identical
extraction into a `BlockFormation` strategy is ~500 LOC of careful
algorithm-mirroring with real behaviour-divergence risk against the
production compression pipeline.

That deferral document at
`openspec/changes/clustered-dictionary-analysis/tasks.md:47-53` left
the architectural debt explicit. This change addresses it — with a
deliberate scope *pivot* away from the original §7 framing.

### The pivot

**Short version:** the two partition algorithms solve different
problems (geometric clustering vs priority-budgeted panel selection);
unifying them is the wrong shape. The right shape is to share the
*data type* (`ClusteredDictionary`) without sharing the algorithm.

The original §7 framing (unify the partition algorithms by extracting
`_select_panels` into a `BlockFormation` strategy) was the wrong shape
of integration:

| Algorithm | Goal | Coverage | State |
|---|---|---|---|
| `_form_blocks_cosine` | Partition ALL features for SAE-scale analysis | Every feature ends up in some block | Stateless; pure geometric |
| `_select_panels` | SELECT ≤K-feature panels for behavioural validation | Not every feature ends up in a panel | Priority-driven, visit-budgeted, coverage-targeted, anchor-only fallback |

These aren't competing implementations of the same algorithm — they
solve different problems. Trying to express them as one
`BlockFormation` strategy with a flag would couple two intentionally-
distinct algorithms together and pollute `BlockFormation`'s simple
config dataclass with `priority`, `n_visits_per_feature`,
`coverage_target`, `n_panels_max`, and `zeroed` state.

The *right* integration is to keep the algorithms distinct and have
`EpochCompressor` **consume `ClusteredDictionary` as its internal
data type** for `_validate_panels` and `_synthesize_validation_report`.
The conversion from `_select_panels`'s output to
`ClusteredDictionary` is already shipped via
`ClusteredDictionary.from_compression_panels` — this change wires it
through the rest of the compression pipeline.

## What Changes

- **`_validate_panels` consumes `ClusteredDictionary`.** Currently
  takes `panels: list[Panel]` and iterates feature_ids. Reframed to
  take `clustered: ClusteredDictionary` and iterate
  `clustered.blocks`. Per-block validation output stays identical;
  the differential test pins this.
- **`_synthesize_validation_report` consumes `ClusteredDictionary` +
  per-block `ValidationReport`s.** Currently takes `panels: list[Panel]`
  alongside the per-panel reports; reframed to take
  `clustered: ClusteredDictionary` + `block_reports:
  list[ValidationReport]`. Cross-panel evidence aggregation
  unchanged.
- **`EpochCompressor.run()` builds `ClusteredDictionary` internally
  per iteration.** The output of `_select_panels` (panels +
  coverage) gets wrapped via
  `ClusteredDictionary.from_compression_panels` and that
  `ClusteredDictionary` becomes the data type passed through to
  validation + synthesis. The compression-pipeline output
  (`EpochResult`, `EpochReport`) is bit-for-bit unchanged.
- **`_select_panels` stays untouched.** No algorithm extraction. No
  `BlockFormation` strategy addition. The priority-driven seeded-
  coverage logic stays in `polygram/compression/epoch.py` where it
  belongs.
- **Differential regression test.** Capture `EpochResult` from
  current main on the bundled GPT-2-small SAE fixture at seed 0;
  freeze as a JSON reference under `tests/compression/data/`. The
  post-refactor pipeline must produce a bit-identical `EpochResult`
  on the same inputs. Test runs on every CI build.

## Capabilities

### Modified Capabilities

- `compression`: `_validate_panels` and `_synthesize_validation_report`
  signatures accept `ClusteredDictionary` instead of `list[Panel]`.
  `EpochCompressor.run` constructs `ClusteredDictionary` per
  iteration from `_select_panels`'s output via
  `from_compression_panels`. External surface (`EpochCompressor`
  dataclass fields, `EpochResult`, `EpochReport`) unchanged. Existing
  `tests/test_compression*.py` tests pass without modification.
- `clustered-dictionary`: documents that `ClusteredDictionary.from_compression_panels`
  is the production conversion point used by `EpochCompressor`
  internally. No new API surface — `from_compression_panels` already
  exists from PR #44.

## Impact

- `polygram/compression/epoch.py` — `_validate_panels` reframed to
  iterate `clustered.blocks`; `_synthesize_validation_report` reframed
  to take `clustered` + `block_reports`; `EpochCompressor.run` builds
  `ClusteredDictionary` from `_select_panels` output and threads it
  through.
- `tests/compression/data/epoch_result_reference.json` (new) — frozen
  reference `EpochResult` on the bundled fixture at seed 0 for the
  differential regression test.
- `tests/compression/test_epoch_clustered_consume.py` (new) — the
  differential regression test plus a small set of structural
  invariants on the per-iteration `ClusteredDictionary` shape.

**Sequencing dependencies:**

- Depends on `clustered-dictionary-analysis` (PR #44) being merged
  first. This change consumes `ClusteredDictionary.from_compression_panels`,
  which §7 of that PR shipped.

**No breaking changes.** Every external surface
(`EpochCompressor.run` signature, `EpochResult` / `EpochReport`
shape, the `polygram analyze` CLI, sae-forge's compression
consumer) preserves byte-identical behaviour by construction. The
differential regression test is the load-bearing guarantee.

## What This Change Explicitly Does NOT Do

- **Does NOT extract `_select_panels` into a `BlockFormation` strategy.**
  See the "pivot" section above. The algorithms have different goals
  and unifying them is the wrong abstraction. The deep-refactor
  vision from `clustered-dictionary-analysis` §7 is reframed as
  "EpochCompressor consumes `ClusteredDictionary` as data type",
  not "algorithms unified".
- **Does NOT change the panel-selection algorithm.** `_select_panels`
  keeps every line of its current behaviour: priority-driven
  iteration, visit-counting, coverage-target early stop, anchor-only
  fallback, ≤7 neighbour cap.
- **Does NOT extend `BlockFormation` with new strategies.** The
  `co_firing` strategy remains reserved (NotImplementedError); the
  three existing strategies (`cosine`, `co_firing`, `user_declared`)
  are unchanged.
- **Does NOT change `EpochCompressor.run`'s external signature or
  return type.** Internal data type changes; external API frozen.
