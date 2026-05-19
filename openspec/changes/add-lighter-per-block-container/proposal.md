## Why

PR #93 diagnosed and PR #94 confirmed: the load-bearing remaining
bottleneck in clustered analyses at SAE scale is **per-block
`Dictionary` materialisation**, dominated by the
`dataclasses.replace` round-trip in
`polygram/clustered_dictionary.py:_materialise_blocks` (line ~1346).
At full N=24,576 the cost is:

- **Block formation:** ~57 s wall-clock (MPS K=8, dominated by
  per-Feature object allocation in `_materialise_blocks`).
- **RSS delta:** ~435 MB across the whole `ClusteredDictionary`
  build (measured directly in
  `docs/research/data/clustered_amortised_benchmark_full_mps.json`).
- **Object delta:** ~35–41 k Python objects allocated during the
  same build (measured via `gc.get_count()` deltas in the same JSON).
- **Net effect on the bet:** the amortised-benchmark's `gram`
  cross-over criterion (≤ 8 ops) **fails** at every shipped K
  (MPS=50, Rung3=372, Rung4=`none`) — not because per-block gram
  is asymptotically wrong, but because the one-time block-formation
  cost dwarfs the per-op savings.

PR #94's cache fix already dropped the *per-op* `cross_block_overlap`
cost 287× (51.6 ms → 0.18 ms). The remaining ceiling is the
one-time build cost itself. Closing it is the next step the
[interpretability bet's "smaller + cheaper at inference"
sub-claim](../../../../.claude/projects/-Users-allans-code-polygram/memory/project_interpretability_bet.md)
needs.

## What Changes

A new internal-only `BlockView` value type that
`build_clustered_dictionary` uses in place of full `Dictionary`
instances for the per-block gram path. `BlockView` carries only the
data the analytic primitives actually need:

- `indices: tuple[int, ...]` — feature indices into the parent
  flat feature list.
- `decoder_slice: np.ndarray` — view (not copy) into the parent
  decoder matrix.
- `encoding: <encoding>` — shared with the parent.
- `feature_names: tuple[str, ...]` — preserved for downstream
  reporting (the rest of the Feature carriers — knobs, cluster
  names, etc. — are dereferenced from the parent when needed).

`ClusteredDictionary.blocks` continues to expose a `Dictionary`-shaped
view via a lazy `Dictionary`-construction property
(`block_dictionary(idx)`) so existing callers don't have to learn a
new API; the property materialises a `Dictionary` only when first
asked. The eager pre-materialised `Dictionary` list is still
available behind a `materialise_dictionaries=True` kwarg for
callers that genuinely need it (compression panel construction is
the main known consumer).

`Dictionary.gram()` already accepts the data it needs — feature
knobs + encoding — but the gram path doesn't actually require a
full `Feature` instance: it reads `(name, cluster, beta, alpha,
gamma, phi, theta?, theta_amp, psi_aux, theta_amp_b, psi_amp_b,
amp_knobs)`. The proposal adds a thin
`Dictionary.from_block_view(view, defaults)` adapter that produces
a real `Dictionary` only when the gram is called — and reuses the
parent's `Feature` instances by reference (not by `replace`) for
the user_declared strategy, since those callers already supply
real Feature carriers via `hierarchy`.

## Success criterion

Re-run the amortised benchmark at full N=24,576 on the same SAE
fixture used by #93 / #94 with `--op all`. Both gates must hold:

1. **Build time**: `clustered_block_formation_seconds` drops by at
   least **5×** on MPS K=8 (target: ~10 s, from the current ~57 s).
2. **Gram cross-over**: passes the original proposal's success
   criterion of `cross_over_n_repeats ≤ 8` on the `gram` op
   across MPS / Rung3 / Rung4 (currently 50 / 372 / `none`).

Either gate failing surfaces a different bottleneck — at which
point the impl PR's report explains what shifted and which
follow-up that points at.

## What this does NOT change

- Public API. `ClusteredDictionary.blocks` continues to return
  Dictionary-shaped instances; existing callers see no shape
  difference.
- The cosine partition algorithm. Block formation logic is
  unchanged; only its output materialisation gets the lighter
  container.
- The amortised benchmark or the recall experiments. Those
  re-run on the same scripts with no script-side changes.
- Q-OrCA emission. Per-block `.q.orca.md` writes need a real
  `Dictionary`; the lazy materialisation kicks in here naturally
  and the cost is one-time-per-emit (a small constant compared to
  the ~24k allocations the current path makes upfront).

## Impact

- **Affected specs**: `dictionary` (the ClusteredDictionary fields
  + lazy materialisation contract).
- **Affected code**: `polygram/clustered_dictionary.py`
  (`_materialise_blocks` rewrite + new `BlockView` type +
  `ClusteredDictionary` storage adjustment). No public API
  changes; the existing `blocks: list[Dictionary]` field becomes
  computed-on-access from an internal `_block_views` list.
- **Risk**: medium. Touches the load-bearing build path. The
  recall + amortised-benchmark + clustered-dictionary tests
  (everything that exercises `build_clustered_dictionary`)
  acts as the regression net.
- **Validation budget**: full-N re-run on the real SAE fixture
  is part of the success-criterion check. Wall-clock budget for
  the re-run is ~5 minutes across the three encodings.

## Out of scope

- A sub-O(N²) approximate block former (LSH / HNSW). That's the
  *other* §12.2 follow-up from `clustered-dictionary-analysis`;
  different problem, separate proposal.
- Memory-mapped decoder slices. Numpy slicing is already a view
  by default for contiguous slices; the proposal's `decoder_slice`
  field relies on that. If profiling shows real copy overhead
  here, a follow-up can swap to `np.lib.stride_tricks.as_strided`
  explicitly.
- Per-feature knob storage changes (compact `Feature` layout, etc.)
  — `Feature` itself remains untouched. The win is in *how many*
  Feature instances exist, not their per-instance cost.
