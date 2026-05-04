## Why

`add-sharing-graph-triage` (PR #10, merged) ships a pure-classical
graph builder: `build_sharing_graph` / `build_separation_graph` take a
`TriagePrediction` (closed-form, rung-1 Gram) and emit a
`FeatureGraph` whose edges name the pairs that look promising (sharing
graph) or stuck (separation graph) under phase-only knobs. That probe
is cheap — no q-orca simulator runs.

A researcher who wants to *act* on that prediction still has to:

1. Read the FeatureGraph JSON.
2. Pick the top-K edges by hand.
3. For each pair, build a `Cancellation`, choose knob paths, run it,
   collect `min_overlap`, `best_knobs`, materialized artifacts.
4. Fold the results back against the FeatureGraph predictions to check
   whether the closed-form gap actually showed up.

This change ships the smallest piece that closes that loop:

- A `BatchExperiment` that takes a `FeatureGraph` (input) plus a
  `Dictionary` and runs `Cancellation` on the graph's top-K edges,
  defaulting to cluster-shared `<cluster>.phi` knobs (the regime PR #9
  validated as bit-for-bit invariant-preserving on `MPSRung1`).
- A `BatchResults` artifact: the input FeatureGraph plus per-edge
  empirical fields (achieved overlap, best knobs, `cancellation_
  efficiency = (current_overlap − achieved) / predicted_gap`) so that
  downstream tools can compare prediction vs observation.
- A `polygram batch` CLI subcommand wiring `--feature-graph FILE.json`
  to `--dictionary REF`.

**Scope discipline.** The pre-`add-sharing-graph-triage` shape of this
proposal introduced a parallel `SharingGraph` artifact, ran
`Cancellation` on every `(N choose 2)` pair (or every cross-cluster
pair) of an arbitrary Dictionary, and exposed pair-selection strings
(`"all"`, `"cross_cluster"`, `"within_cluster"`). PR #10 superseded
that:

- The pure-classical FeatureGraph already does pair selection — by
  weight threshold, by cross-cluster/within-cluster gates, and by
  formula. The right input to BatchExperiment is the *result* of that
  selection, not a re-implementation.
- Running `Cancellation` on every pair of an SAE subset is exactly
  the O(N²) hazard that motivated the closed-form triage.

So this proposal is now a *consumer* shape: BatchExperiment takes the
graph as input, runs Cancellation on the graph's already-ranked top-K
edges, and emits a BatchResults that carries both the predicted and
the observed numbers side by side.

## What Changes

### `BatchExperiment` — FeatureGraph-driven runner

- New dataclass `BatchExperiment` in a new module `polygram/batch.py`.
- Fields:
  - `feature_graph: FeatureGraph` — the input graph (typically from
    `build_sharing_graph` or `build_separation_graph`, but any
    `FeatureGraph` instance works).
  - `dictionary: Dictionary` — the encoded dictionary the graph was
    built against.
  - `top_k: int = 8` — number of input-graph edges to run, taken in
    the input graph's existing edge order (sorted by descending
    `weight`).
  - `knobs: Literal["cluster_shared", "per_feature"] =
    "cluster_shared"` — knob path style passed to per-pair
    `Cancellation`. Default is the cluster-shared regime that the
    `add-cluster-shared-knobs` archive demonstrated is bit-for-bit
    invariant-preserving on `MPSRung1` `<cluster>.phi` paths.
  - `output_dir: Path | None = None` — if set, every pair writes its
    `Cancellation` artifact bundle into
    `output_dir/{source}_x_{target}/`, and the aggregated
    `batch_results.json` lands at the top level.
  - `cancellation_kwargs: dict | None = None` — forwarded to each
    per-pair `Cancellation` (e.g. `tolerance`, `max_steps`).
- A safety rail: `__post_init__` SHALL raise `ValueError` if
  `top_k > 16` (cluster-shared `Cancellation` runs at ~seconds per pair
  on grid backend; 16 caps wall time at a couple of minutes). Override
  by clipping `top_k` explicitly — there is no `force=True` flag,
  because the FeatureGraph already encodes the user's pair-selection
  decision.
- `BatchExperiment.run() -> BatchResults` — runs `Cancellation` on
  each of the input graph's first `min(top_k, len(graph.edges))`
  edges, populates `BatchRun` records, and assembles a `BatchResults`.

### `BatchResults` — empirical artifact

- New dataclass `BatchResults` in `polygram/batch.py`. Frozen.
- Fields:
  - `source_graph: FeatureGraph` — the input graph, carried verbatim.
    Round-trip-stable.
  - `dictionary_name: str`
  - `knobs: str` — one of `"cluster_shared"`, `"per_feature"`.
  - `created_at: str` — ISO 8601 at run start.
  - `runs: tuple[BatchRun, ...]` — one record per pair run, in input
    graph edge order.
- `BatchRun` — frozen dataclass with the per-pair fields:
  - `source: str`, `target: str` — copied from `FeatureEdge`.
  - `predicted_floor: float` — `FeatureEdge.floor` (the rung-1
    structural floor from triage).
  - `predicted_gap: float` — `FeatureEdge.gap` (the rung-1
    `current_overlap − floor`).
  - `current_overlap: float` — measured at default knobs by
    `Cancellation` before optimization (sanity check against the
    prediction's `current_overlap`).
  - `achieved_overlap: float` — `Cancellation.run().min_overlap`.
  - `cancellation_efficiency: float` — `(current_overlap −
    achieved_overlap) / predicted_gap` if `predicted_gap > 1e-12`,
    else `0.0`. Values near 1.0 mean φ-search realized the predicted
    gap; values near 0.0 mean the prediction was right that there was
    nothing to find (separation kind) or the search got stuck (sharing
    kind).
  - `best_knobs: dict[str, float]` — `Cancellation.run().best_knobs`.
  - `tier_separation_after: float | None` — tier separation at the
    optimum, when computable; `None` otherwise.
  - `artifact_subpath: str | None` — relative path
    `f"{source}_x_{target}"` under `output_dir`, or `None` if
    `output_dir is None`.
- `BatchResults.to_json(path)` — deterministic JSON: nested
  `source_graph` is emitted via the existing
  `FeatureGraph.to_json()`; runs are sorted in input graph edge order;
  floats formatted to 6 sig figs; `None` → JSON `null`.
- `BatchResults.from_json(path) -> BatchResults` — inverse.

### `polygram batch` CLI subcommand

- Extends `polygram/cli.py` with `batch`:
  ```
  polygram batch --feature-graph FILE.json \
                 --dictionary REF \
                 [--top-k N] \
                 [--knobs cluster_shared|per_feature] \
                 [--output-dir DIR]
  ```
- `--feature-graph FILE.json` (required) — path to a JSON document
  produced by `FeatureGraph.to_json()`. The CLI reads its `kind`
  field for reporting only; runtime behavior is independent of kind.
- `--dictionary REF` (required) — either a `.q.orca.md` file path
  (parsed via the existing q-orca round-trip) or a `module:callable`
  reference whose callable returns a `Dictionary`.
- `--top-k N` (default `8`) — forwarded to `BatchExperiment.top_k`.
  Hard cap of 16 enforced as on the dataclass.
- `--knobs` (default `cluster_shared`) — forwarded.
- `--output-dir DIR` (default: a fresh temp directory; the resolved
  path is printed to stdout).

The CLI prints the resolved `batch_results.json` path on stdout and
exits 0 on success; non-zero with a clear stderr message on missing
inputs, malformed graph JSON, dictionary/graph node mismatch, or any
error raised by `BatchExperiment.run()`.

### Tests

- `tests/test_batch.py::TestBatchExperiment` — pair-set is the input
  graph's top-K edges in order; `top_k > 16` rejected; cluster-shared
  default produces knob paths of the form `<cluster>.phi`; per-pair
  artifact subdirs materialized when `output_dir` is set; dictionary
  whose feature names don't cover the input graph's nodes is rejected
  with a clear error.
- `tests/test_batch.py::TestBatchResults` — JSON round-trip preserves
  every field including the nested `source_graph`; deterministic
  ordering; `cancellation_efficiency` is `0.0` when
  `predicted_gap == 0`.
- `tests/test_cli.py::TestBatchSubcommand` — end-to-end on the toy
  SAE fixture: build a separation graph via `build_separation_graph`,
  serialize it to a temp file, invoke `polygram batch
  --feature-graph ... --dictionary examples.animals_hea:build_dictionary
  --top-k 2`, assert the produced `batch_results.json` parses and
  carries the input graph verbatim.

### Example

- `examples/batch_animals_hea.py` — runs `build_separation_graph` on
  the Animals dictionary's triage prediction, then feeds the resulting
  graph into `BatchExperiment(top_k=4, knobs="cluster_shared")`,
  writing the `BatchResults` JSON and a per-pair artifact tree to
  `examples/output/batch_animals_hea/`.

## Capabilities

### Modified Capabilities

- `cli` — new `batch` subcommand.

### New Capabilities

- `batch` — FeatureGraph-consuming experiment runner that emits a
  `BatchResults` artifact pairing predictions with observations.

## Out of Scope

- **Pair selection at the BatchExperiment layer.** The input
  FeatureGraph already encodes the user's threshold, cross-cluster
  gate, and ranking. Re-introducing `pairs="all"` /
  `pairs="cross_cluster"` would defeat the FeatureGraph's purpose and
  resurrect the O(N²) hazard.
- **A standalone `SharingGraph` dataclass.** The artifact shape this
  change introduces is `BatchResults`, which carries the input
  `FeatureGraph` verbatim. Adding a second graph type would be
  duplicate-with-drift.
- **`InterferenceSweep` orchestration in batch mode.** v0 runs only
  `Cancellation`. Aggregating a sweep across many pairs has no
  obvious shape; defer until concrete usage signal arrives.
- **Compression-algorithm derivation.** Going from `BatchResults` to
  a smaller classical model is the load-bearing research question
  that motivated the project. `BatchResults` is the empirical input
  to that question; what's done with it is downstream.
- **Encoding-invariance verification.** A separate, complementary
  question: do safe/unsafe classifications survive a change of
  encoding (MPS → HEA)? Captured as a research-track follow-up in
  `tech-debt-backlog`.
- **Distributed / parallel batch execution.** v0 runs pairs
  sequentially. Parallelism is a workload question that depends on
  real SAE-scale usage signals.
- **Cross-FeatureGraph batching.** v0 takes a single graph. Comparing
  results across graph kinds (sharing vs separation) is a downstream
  analysis layer.

## Impact

- `polygram/batch.py` — new module with `BatchExperiment`,
  `BatchResults`, `BatchRun`.
- `polygram/cli.py` — `batch` subcommand handler.
- `polygram/__init__.py` — re-export `BatchExperiment`,
  `BatchResults`, `BatchRun`.
- `pyproject.toml` — no new runtime deps.
- `tests/test_batch.py`, `tests/test_cli.py`, `tests/test_examples.py`
  — extended for the new surfaces.
- `examples/batch_animals_hea.py` — new walk-through.
- `tech-debt-backlog` — bullet recording the encoding-invariance spike
  as a research-track follow-up.
