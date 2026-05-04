## ADDED Requirements

### Requirement: build_sharing_graph emits a SharingGraph from a TriagePrediction

`polygram.analysis.build_sharing_graph` SHALL produce a `SharingGraph`
from an existing `TriagePrediction` using only fields that are
already present on each `PairPrediction`. Signature:
`build_sharing_graph(prediction, *, threshold=0.5,
allow_cross_cluster=False) -> SharingGraph`. The function SHALL NOT
call any q-orca simulator, build any quantum state, or run any
`Cancellation` optimization.

For each `PairPrediction p`, the per-pair sharing weight SHALL be
computed as:

```
ratio    = p.cancellation_gap / max(p.current_overlap, 1e-12)
weight   = clip(ratio, 0.0, 1.0)
           × (0.0 if p.structural_floor > FLOOR_BLOCK else 1.0)
           × (0.0 if (p.is_cross_cluster and not allow_cross_cluster)
              else 1.0)
```

where `FLOOR_BLOCK = 0.5` is a module-level constant. The
`SharingGraph.edges` tuple SHALL contain one `SharingEdge` per pair
whose `weight >= threshold`; pairs with smaller weights SHALL NOT
appear in the edge list.

`SharingEdge` SHALL be a frozen dataclass exposing `source: str`,
`target: str`, `weight: float`, `floor: float`, `gap: float`,
`is_cross_cluster: bool`, and `reason: str`. The `reason` field SHALL
be a short stable identifier from a closed vocabulary:
`"high_gap_low_floor"` (kept edge with both factors clean),
`"phase_separable_low_overlap"` (kept edge whose `current_overlap`
was already small), or one of the gate identifiers used internally
for diagnostics.

`SharingGraph` SHALL be a frozen dataclass exposing:

- `nodes: tuple[str, ...]` — the feature names from the prediction,
  in the same order as `prediction.dictionary.features`.
- `edges: tuple[SharingEdge, ...]` — sorted by descending `weight`
  with ties broken by `(source, target)` lexicographically, so the
  output is deterministic.
- `clusters: tuple[tuple[str, ...], ...]` — connected components of
  the undirected graph induced by `edges`, each component sorted
  lexicographically; the components themselves sorted by descending
  size with ties broken lexicographically by first member.
  Singleton nodes (features that participate in no kept edge) SHALL
  appear as size-1 components.
- `metadata: dict[str, Any]` — at minimum `selection_method` (string),
  `total_features` (int), `threshold` (float), `allow_cross_cluster`
  (bool), and `formula` (the literal `EDGE_WEIGHT_FORMULA` string).

`EDGE_WEIGHT_FORMULA` SHALL be a module-level constant whose value
reproduces the `weight = ...` block above as a documentation string.
It SHALL be re-emitted via `SharingGraph.metadata["formula"]` so
consumers can quote the rule without duplicating it.

#### Scenario: edges respect the weight threshold

- **GIVEN** a `TriagePrediction` produced by
  `predict_cancellation_depth(toy_sae, [0, 1, 4, 5])`
- **WHEN** `build_sharing_graph(prediction, threshold=0.6)` is called
- **THEN** every `SharingEdge` in the returned graph has
  `weight >= 0.6`, and every pair from the prediction whose computed
  weight is below `0.6` is absent from the edge list

#### Scenario: cross-cluster edges gated by allow_cross_cluster

- **GIVEN** a `TriagePrediction` with at least one cross-cluster pair
  whose computed weight (ignoring the cross-cluster gate) would
  exceed the threshold
- **WHEN** `build_sharing_graph(prediction, allow_cross_cluster=
  False)` is called
- **THEN** no edge with `is_cross_cluster=True` appears in the result
- **AND WHEN** the same call is repeated with
  `allow_cross_cluster=True`
- **THEN** that same pair appears as a `SharingEdge` with
  `is_cross_cluster=True`

#### Scenario: high floors block edges regardless of phase headroom

- **GIVEN** a synthetic `TriagePrediction` where some pair has
  `structural_floor = 0.7` (above `FLOOR_BLOCK = 0.5`) and
  `cancellation_gap = 0.2`
- **WHEN** `build_sharing_graph(prediction, threshold=0.0)` is called
- **THEN** that pair does NOT appear in the edge list (the
  floor-block gate zeroes the weight before threshold filtering)

#### Scenario: clusters are connected components of kept edges

- **GIVEN** a `TriagePrediction` and a threshold for which the kept
  edge set forms a hand-computable adjacency
- **WHEN** `build_sharing_graph(prediction, threshold=...)` is called
- **THEN** `result.clusters` partitions every node in `result.nodes`
  exactly once
- **AND** features that share a path through kept edges are in the
  same component
- **AND** features with no kept incident edge appear as a size-1
  component

### Requirement: SharingGraph.to_json emits deterministic, parseable JSON

`SharingGraph.to_json` SHALL return a string that parses cleanly via
`json.loads` to a dict with keys `"nodes"`, `"edges"`, `"clusters"`,
and `"metadata"`. The output SHALL be byte-identical across repeated
calls on the same graph (sorted keys, fixed numeric formatting, no
timestamps, no random ordering).

Each edge SHALL be encoded as an object with the seven `SharingEdge`
fields named identically to the dataclass attributes. Each cluster
SHALL be encoded as a list of feature names in the same order as the
corresponding `clusters[i]` tuple.

#### Scenario: round-trips through json.loads

- **WHEN** `graph.to_json()` is called and the result passed to
  `json.loads`
- **THEN** the parsed dict's `"nodes"` list equals
  `list(graph.nodes)`, and every entry in `"edges"` exposes the
  fields `source`, `target`, `weight`, `floor`, `gap`,
  `is_cross_cluster`, and `reason`

#### Scenario: byte-identical across calls

- **WHEN** `graph.to_json()` is called twice on the same
  `SharingGraph` instance
- **THEN** both returned strings are equal byte-for-byte

### Requirement: render_sharing_graph_section returns a markdown section

`polygram.analysis.render_sharing_graph_section(graph) -> str` SHALL
return a markdown fragment that contains a `## Sharing graph`
heading, a one-line summary stating the threshold and the number of
kept edges and components, an `### Edges` table sorted by descending
weight with columns `(source, target, weight, floor, gap,
cross_cluster, reason)`, an `### Components` section listing each
cluster on its own line, and a `### Formula` footer that quotes
`EDGE_WEIGHT_FORMULA`.

The section SHALL be deterministic given the input. It SHALL be
*additive* — `render_report` continues to emit the per-pair table
unchanged; callers wishing to include the sharing graph append the
result of `render_sharing_graph_section` to `render_report`'s output.

#### Scenario: section contains required headings

- **WHEN** `render_sharing_graph_section(build_sharing_graph(
  predict_cancellation_depth(toy_sae, [0, 1, 4, 5])))` is called
- **THEN** the returned string contains the headings
  `"## Sharing graph"`, `"### Edges"`, `"### Components"`, and
  `"### Formula"`, and the formula text contains the substring
  `"FLOOR_BLOCK"`
