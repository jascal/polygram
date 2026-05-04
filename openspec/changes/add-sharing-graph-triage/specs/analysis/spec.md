## ADDED Requirements

### Requirement: build_sharing_graph emits a sharing FeatureGraph

`polygram.analysis.build_sharing_graph` SHALL produce a `FeatureGraph`
with `kind == "sharing"` from an existing `TriagePrediction`, using
only fields already present on each `PairPrediction`. Signature:
`build_sharing_graph(prediction, *, threshold=0.5,
allow_cross_cluster=False) -> FeatureGraph`. The function SHALL NOT
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
`FeatureGraph.edges` tuple SHALL contain one `FeatureEdge` per pair
whose `weight >= threshold`; pairs with smaller weights SHALL NOT
appear in the edge list.

The returned `FeatureGraph.metadata` SHALL include
`kind == "sharing"`, `threshold`, `allow_cross_cluster`, and
`formula == SHARING_EDGE_FORMULA`. `SHARING_EDGE_FORMULA` SHALL be a
module-level string constant whose value reproduces the `weight = ...`
block above as a documentation string.

#### Scenario: edges respect the weight threshold

- **GIVEN** a `TriagePrediction` produced by
  `predict_cancellation_depth(toy_sae, [0, 1, 4, 5])`
- **WHEN** `build_sharing_graph(prediction, threshold=0.6)` is called
- **THEN** every `FeatureEdge` in the returned graph has
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
- **THEN** that same pair appears as a `FeatureEdge` with
  `is_cross_cluster=True`

#### Scenario: high floors block edges regardless of phase headroom

- **GIVEN** a synthetic `TriagePrediction` where some pair has
  `structural_floor = 0.7` (above `FLOOR_BLOCK = 0.5`) and
  `cancellation_gap = 0.2`
- **WHEN** `build_sharing_graph(prediction, threshold=0.0)` is called
- **THEN** that pair does NOT appear in the edge list (the
  floor-block gate zeroes the weight before threshold filtering)

#### Scenario: result kind and formula are sharing

- **WHEN** `build_sharing_graph(prediction)` is called
- **THEN** `result.kind == "sharing"` and
  `result.metadata["formula"] == SHARING_EDGE_FORMULA`

### Requirement: build_separation_graph emits a separation FeatureGraph

`polygram.analysis.build_separation_graph` SHALL produce a
`FeatureGraph` with `kind == "separation"` from the same
`TriagePrediction`, flagging pairs whose squared overlap cannot be
driven low by phase tuning alone. Signature:
`build_separation_graph(prediction, *, threshold=0.2,
include_within_cluster=False) -> FeatureGraph`. The function SHALL
NOT call any q-orca simulator, build any quantum state, or run any
`Cancellation` optimization.

For each `PairPrediction p`, the per-pair separation weight SHALL be
computed as:

```
weight = clip(p.structural_floor, 0.0, 1.0)
         × (0.0 if (not p.is_cross_cluster
                    and not include_within_cluster) else 1.0)
```

The returned `FeatureGraph.edges` tuple SHALL contain one
`FeatureEdge` per pair whose `weight >= threshold`. The returned
`FeatureGraph.metadata` SHALL include `kind == "separation"`,
`threshold`, `include_within_cluster`, and
`formula == SEPARATION_EDGE_FORMULA`. `SEPARATION_EDGE_FORMULA` SHALL
be a module-level string constant reproducing the formula block above.

The semantic: a non-zero edge weight means the pair has irreducible
squared overlap that no φ tuning can pierce. Cross-cluster edges of
this kind flag pairs the phase-only triage cannot disambiguate; a
future disentanglement primitive operating on β/α/γ would target them.

#### Scenario: separation weights equal the structural floor on kept cross-cluster pairs

- **GIVEN** a `TriagePrediction` with at least one cross-cluster
  pair `p` whose `structural_floor` is in `(0.2, 1.0]`
- **WHEN** `build_separation_graph(prediction, threshold=0.2)` is
  called
- **THEN** the corresponding `FeatureEdge.weight` equals
  `min(p.structural_floor, 1.0)` to within numeric round-off
- **AND** `FeatureEdge.floor == p.structural_floor`

#### Scenario: within-cluster gated by include_within_cluster

- **GIVEN** a `TriagePrediction` with at least one within-cluster
  pair `p` whose `structural_floor >= 0.2`
- **WHEN** `build_separation_graph(prediction,
  include_within_cluster=False)` is called
- **THEN** no edge with `is_cross_cluster=False` appears in the
  returned graph
- **AND WHEN** the same call is repeated with
  `include_within_cluster=True`
- **THEN** the within-cluster pair appears as a `FeatureEdge`

#### Scenario: result kind and formula are separation

- **WHEN** `build_separation_graph(prediction)` is called
- **THEN** `result.kind == "separation"` and
  `result.metadata["formula"] == SEPARATION_EDGE_FORMULA`

### Requirement: FeatureGraph artifact shape and JSON serialization

`FeatureGraph` SHALL be a frozen dataclass exposing `kind: str`
(one of `"sharing"`, `"separation"`), `nodes: tuple[str, ...]`,
`edges: tuple[FeatureEdge, ...]`, `clusters: tuple[tuple[str, ...],
...]`, and `metadata: dict[str, Any]`.

- `nodes` SHALL list feature names in the same order as
  `prediction.dictionary.features`.
- `edges` SHALL be sorted by descending `weight` with ties broken by
  `(source, target)` lexicographically.
- `clusters` SHALL be the connected components of the undirected
  graph induced by `edges`, each component sorted lexicographically;
  the components themselves sorted by descending size with ties
  broken lexicographically by first member. Singleton nodes (no
  incident kept edge) SHALL appear as size-1 components.
- `metadata` SHALL include at minimum `kind`, `selection_method`,
  `total_features`, `threshold`, and `formula`. Builder-specific
  flags (`allow_cross_cluster`, `include_within_cluster`) SHALL be
  recorded under their dataclass-attribute names.

`FeatureEdge` SHALL be a frozen dataclass exposing `source: str`,
`target: str`, `weight: float`, `floor: float`, `gap: float`,
`is_cross_cluster: bool`, and `reason: str`. The `reason` field
SHALL be a short stable identifier; the closed vocabulary SHALL be
documented in the implementing module's docstring.

`FeatureGraph.to_json()` SHALL return a string that parses cleanly
via `json.loads` to a dict with keys `"kind"`, `"nodes"`, `"edges"`,
`"clusters"`, and `"metadata"`. The output SHALL be byte-identical
across repeated calls on the same graph (sorted keys, fixed numeric
formatting, no timestamps, no random ordering). Each edge SHALL be
encoded as an object with the seven `FeatureEdge` fields named
identically to the dataclass attributes.

#### Scenario: clusters are connected components of kept edges

- **GIVEN** a `TriagePrediction` and a threshold for which the kept
  edge set forms a hand-computable adjacency
- **WHEN** any of the builders is called with that threshold
- **THEN** `result.clusters` partitions every node in `result.nodes`
  exactly once
- **AND** features that share a path through kept edges are in the
  same component
- **AND** features with no kept incident edge appear as a size-1
  component

#### Scenario: to_json round-trips through json.loads

- **WHEN** `graph.to_json()` is called and the result passed to
  `json.loads`
- **THEN** the parsed dict's `"kind"` equals `graph.kind`,
  `"nodes"` equals `list(graph.nodes)`, and every entry in
  `"edges"` exposes the fields `source`, `target`, `weight`,
  `floor`, `gap`, `is_cross_cluster`, and `reason`

#### Scenario: to_json byte-identical across calls

- **WHEN** `graph.to_json()` is called twice on the same
  `FeatureGraph` instance
- **THEN** both returned strings are equal byte-for-byte

### Requirement: render_feature_graph_section returns a kind-aware markdown section

`polygram.analysis.render_feature_graph_section(graph) -> str` SHALL
return a markdown fragment whose top-level heading is
`"## Sharing graph"` when `graph.kind == "sharing"` and
`"## Separation graph"` when `graph.kind == "separation"`. The
section SHALL contain a one-line summary stating the threshold and
the number of kept edges and components, an `### Edges` table sorted
by descending weight with columns
`(source, target, weight, floor, gap, cross_cluster, reason)`, an
`### Components` section listing each cluster on its own line, and
an `### Formula` footer that quotes
`graph.metadata["formula"]`.

The section SHALL be deterministic given the input. It SHALL be
*additive* — `render_report` continues to emit the per-pair table
unchanged; callers wishing to include a feature graph append the
result of `render_feature_graph_section` to `render_report`'s output.

#### Scenario: section heading reflects graph kind

- **GIVEN** a `FeatureGraph` produced by `build_sharing_graph`
- **WHEN** `render_feature_graph_section(graph)` is called
- **THEN** the returned string contains `"## Sharing graph"` and
  does NOT contain `"## Separation graph"`
- **AND GIVEN** the same prediction's
  `build_separation_graph(...)` result
- **WHEN** `render_feature_graph_section(graph)` is called
- **THEN** the returned string contains `"## Separation graph"` and
  does NOT contain `"## Sharing graph"`

#### Scenario: section quotes the kind-specific formula

- **WHEN** `render_feature_graph_section(graph)` is called for a
  graph of either kind
- **THEN** the returned string contains the substring
  `graph.metadata["formula"]`
