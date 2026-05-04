# batch Specification

## Purpose
TBD - created by archiving change add-batch-experiment. Update Purpose after archive.
## Requirements
### Requirement: BatchExperiment runs Cancellation on a FeatureGraph's top-K edges

`polygram.BatchExperiment` SHALL be a dataclass that consumes a `FeatureGraph` (input) plus a `Dictionary` and runs `Cancellation` on the top-K edges of the input graph.

The dataclass exposes the following fields:

- `feature_graph: FeatureGraph` — required. The input graph,
  typically produced by `polygram.analysis.build_sharing_graph` or
  `polygram.analysis.build_separation_graph`. Any `FeatureGraph` whose
  `nodes` are a subset of `dictionary.feature_names()` is acceptable.
- `dictionary: Dictionary` — required. The encoded dictionary the
  graph was built against.
- `top_k: int = 8` — number of input-graph edges to run, taken in
  the input graph's existing edge order (sorted by descending
  `weight` per the `FeatureGraph` invariant).
- `knobs: Literal["cluster_shared", "per_feature"] =
  "cluster_shared"` — knob path style passed to the per-pair
  `Cancellation`. The cluster-shared default is the regime that the
  `add-cluster-shared-knobs` archive demonstrated is bit-for-bit
  invariant-preserving on `MPSRung1` `<cluster>.phi` paths.
- `output_dir: Path | None = None` — when set, the per-pair
  Cancellation artifact bundle SHALL be written to
  `output_dir/{source}_x_{target}/`, and the aggregated
  `batch_results.json` SHALL be written at the top of `output_dir`.
- `cancellation_kwargs: dict | None = None` — keyword arguments
  forwarded to each per-pair `Cancellation` constructor (e.g.
  `tolerance`, `max_steps`).

`__post_init__` SHALL:

1. Reject `top_k > 16` with `ValueError` naming the value and the
   16-pair cap. The cap exists because cluster-shared `Cancellation`
   runs at ~seconds per pair on grid backend; 16 caps wall time at a
   couple of minutes. There is no `force=True` override — the input
   FeatureGraph already encodes the user's pair-selection decision.
2. Reject `top_k < 1` with `ValueError`.
3. Reject `knobs` values outside `{"cluster_shared", "per_feature"}`
   with `ValueError`.
4. Reject input graphs whose node set is not a subset of
   `dictionary.feature_names()` with `ValueError` naming the missing
   feature(s).

`BatchExperiment.run() -> BatchResults` SHALL iterate the first
`min(self.top_k, len(self.feature_graph.edges))` edges of the input
graph, build a `Cancellation` per edge with `target_pair=(edge.source,
edge.target)` and a knob-path list determined by `self.knobs`, run it
with `self.cancellation_kwargs or {}`, and assemble a `BatchResults`.
When `output_dir is not None`, each per-pair `Cancellation` SHALL
materialize its artifact bundle under
`output_dir/{source}_x_{target}/` and `batch_results.json` SHALL be
written at the top level of `output_dir`.

`knobs == "cluster_shared"` SHALL produce knob paths
`<cluster>.phi` for `MPSRung1` dictionaries and
`<cluster>.theta[r,d,q]` for `HEA_Rung2`, applied to every cluster
mentioned by the pair's two endpoints (typically two clusters when
the pair is cross-cluster, one when within-cluster).
`knobs == "per_feature"` SHALL produce paths
`<feature>.phi` (MPS) or `<feature>.theta[r,d,q]` (HEA) for both
endpoints.

#### Scenario: pairs run are the input graph's top-K edges in order

- **GIVEN** a `FeatureGraph` whose `edges` field has length 5 and
  whose edges are sorted by descending `weight`
- **WHEN** `BatchExperiment(feature_graph=g, dictionary=d,
  top_k=3).run()` returns
- **THEN** the returned `BatchResults.runs` has exactly 3 entries,
  each `BatchRun.(source, target)` matching the corresponding input
  edge's `(source, target)` in the same order

#### Scenario: top_k larger than edge count is silently clamped

- **GIVEN** a `FeatureGraph` with 2 edges
- **WHEN** `BatchExperiment(feature_graph=g, dictionary=d,
  top_k=8).run()` returns
- **THEN** `BatchResults.runs` has exactly 2 entries — no error is
  raised for `top_k > len(g.edges)`

#### Scenario: top_k above 16 rejected

- **WHEN** `BatchExperiment(feature_graph=g, dictionary=d,
  top_k=17)` is constructed
- **THEN** `__post_init__` raises `ValueError` naming the value
  `17` and the 16-pair cap

#### Scenario: knobs default produces cluster-shared phi paths on MPS

- **GIVEN** an `MPSRung1`-encoded `Dictionary` with two clusters and
  a `FeatureGraph` whose top edge is cross-cluster
- **WHEN** `BatchExperiment(feature_graph=g, dictionary=d).run()`
  returns
- **THEN** the corresponding `BatchRun.best_knobs` keys are exactly
  the two `<cluster>.phi` paths for the two clusters touched by the
  pair

#### Scenario: dictionary missing a graph node rejected

- **GIVEN** a `FeatureGraph` whose `nodes` includes a feature
  `"ghost"` not declared by the dictionary
- **WHEN** `BatchExperiment(feature_graph=g, dictionary=d)` is
  constructed
- **THEN** `__post_init__` raises `ValueError` naming `"ghost"`

#### Scenario: per-pair artifacts written under output_dir

- **GIVEN** a `BatchExperiment` with `output_dir=tmp_path` and an
  input graph with 3 edges, `top_k=3`
- **WHEN** `run()` returns
- **THEN** for every input edge `(a, b)`, the directory
  `tmp_path / f"{a}_x_{b}"` exists and contains the standard
  `Cancellation` artifact bundle, AND
  `tmp_path / "batch_results.json"` exists at the top level

### Requirement: BatchResults pairs predictions with observations

`polygram.BatchResults` SHALL be a frozen dataclass exposing:

- `source_graph: FeatureGraph` — the input graph carried verbatim.
- `dictionary_name: str`
- `knobs: str` — one of `"cluster_shared"`, `"per_feature"`.
- `created_at: str` — ISO 8601 timestamp at run start.
- `runs: tuple[BatchRun, ...]` — one record per pair run, ordered as
  the input graph's first `top_k` edges.

`polygram.BatchRun` SHALL be a frozen dataclass with the per-pair
fields:

- `source: str`, `target: str` — the pair, copied from the input
  `FeatureEdge`.
- `predicted_floor: float` — `FeatureEdge.floor`.
- `predicted_gap: float` — `FeatureEdge.gap`.
- `current_overlap: float` — squared overlap at default knobs,
  measured by `Cancellation` before optimization.
- `achieved_overlap: float` — `Cancellation.run().min_overlap`.
- `cancellation_efficiency: float` — defined as
  `(current_overlap − achieved_overlap) / predicted_gap` if
  `predicted_gap > 1e-12`, else `0.0`. Values near 1.0 mean φ-search
  realized the closed-form prediction; values near 0.0 mean the
  prediction was right that there was nothing to find (separation
  kind), or the search got stuck (sharing kind).
- `best_knobs: dict[str, float]` — `Cancellation.run().best_knobs`.
- `tier_separation_after: float | None` — tier separation at the
  optimum, when the dictionary has computable tiers; `None`
  otherwise.
- `artifact_subpath: str | None` — relative path
  `f"{source}_x_{target}"` under the run's `output_dir`, or `None` if
  `output_dir` was not set.

`BatchResults.to_json(path)` SHALL write a deterministic JSON
document. Numeric values SHALL be formatted to 6 significant figures;
`None` SHALL serialize as JSON `null`. The nested `source_graph`
SHALL be emitted via the existing `FeatureGraph.to_json()`. The
output SHALL be byte-identical across repeated runs of the same
inputs.

`BatchResults.from_json(path) -> BatchResults` SHALL be the inverse:
`from_json(to_json(b)) == b` for every `b` reachable from
`BatchExperiment.run()`.

#### Scenario: cancellation_efficiency is zero when predicted_gap is zero

- **GIVEN** a `BatchRun` whose corresponding input edge has
  `gap == 0.0`
- **THEN** `BatchRun.cancellation_efficiency == 0.0` regardless of
  `current_overlap` and `achieved_overlap`

#### Scenario: source_graph is preserved verbatim

- **GIVEN** a `FeatureGraph g` passed to
  `BatchExperiment(feature_graph=g, ...).run()` returning `r`
- **THEN** `r.source_graph.to_json() == g.to_json()` byte-for-byte

#### Scenario: JSON round-trip preserves every field

- **GIVEN** a `BatchResults` `b` produced by
  `BatchExperiment.run()`
- **WHEN** `b.to_json(p)` is called and the result is read back via
  `BatchResults.from_json(p)`
- **THEN** the reconstructed object equals `b` field-for-field,
  including every nested `BatchRun` and the embedded `source_graph`

#### Scenario: JSON output is deterministic

- **GIVEN** the same `BatchExperiment` configuration run twice on
  inputs that produce identical `Cancellation` outputs
- **WHEN** both runs call `to_json` to the same path
- **THEN** the produced byte sequences are identical

### Requirement: BatchExperiment supports both encodings

`BatchExperiment` SHALL accept dictionaries with either `MPSRung1` or
`HEA_Rung2` encoding. Per-pair experiments dispatch on the encoding
via the existing `Dictionary.gram()` mechanism; no encoding-specific
code paths are introduced in this layer beyond knob-path selection.

#### Scenario: HEA dictionary produces a valid BatchResults

- **GIVEN** a `Dictionary` with `encoding=HEA_Rung2(depth=2)` and a
  `FeatureGraph` built against it
- **WHEN** `BatchExperiment(feature_graph=g, dictionary=d,
  knobs="cluster_shared", top_k=2).run()` returns
- **THEN** the returned `BatchResults.runs` has 2 entries with
  populated `current_overlap` and `achieved_overlap`, and
  `BatchRun.best_knobs` keys are `<cluster>.theta[r,d,q]` paths

