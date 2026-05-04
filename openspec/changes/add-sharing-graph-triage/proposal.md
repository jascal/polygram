## Why

The `analysis-triage-layer` (archived 2026-05-03) ships
`predict_cancellation_depth`, which already computes the per-pair
`(M, V, structural_floor, cancellation_gap)` decomposition for any
selected SAE subset from `n_features + 1` closed-form Gram
evaluations. Researchers consume the per-pair table as markdown today
but the natural downstream artifact — a graph that tells you *which
features can safely be compressed onto overlapping dimensions* — is
left on the table.

The triage data already answers that question. For every pair `(A, B)`
the closed-form bound

```
|<A|B>|²(δ) = M + V·cos(δ),  δ ∈ [0, 2π)
```

defines a `cancellation_gap = current_overlap − (M − |V|)`: the
distance the squared overlap can be driven down by tuning the relative
phase. A pair with a large gap and a low floor is a *good sharing
candidate* (phase tuning gives us room); a pair with a small gap or a
high floor is dangerous to compress (interference can't separate them).

This change emits that information as a graph artifact (`SharingGraph`)
and a sibling renderer alongside `render_report`, with edge weights
defined entirely by the existing closed-form fields. No new physics, no
per-pair `Cancellation` optimizations, no q-orca calls.

**What this proposal explicitly does NOT do.** The user-supplied design
sketch suggested a composite score
`0.5·cancel_eff + 0.3·sensitivity + 0.2·coherence` driven by per-pair
`Cancellation` runs and Leiden/Louvain community detection. This
proposal rejects all three:

- **No per-pair Cancellation runs.** That is `O(N²)` quantum
  optimizations and reproduces the V signal we already have in closed
  form. The recently-archived `add-cluster-shared-knobs` change
  documented the *cluster-shattering hazard* of per-feature `.phi`
  cancellation: a pair-targeted optimization can drive `(A, B)` to
  zero while inverting siblings (`0.9999 → 0.5735` in the fixture
  experiment). The triage layer is the wrong place to introduce
  that hazard.
- **No arbitrary composite weights.** `0.5/0.3/0.2` is undefended.
  The `cancellation_gap` field already integrates phase headroom
  with current overlap; coherence enters as a hard constraint
  (cross-cluster edges flagged), not a soft weight.
- **No community detection in v0.** Triage targets fixture sizes
  ≤16; threshold graph + connected components is enough. Leiden
  and friends are valuable when N grows past where naive grouping
  fails — defer until a workload demands it.

## What Changes

### `analysis` capability — sharing graph artifact

Add three additive primitives to `polygram.analysis`:

- **`SharingEdge`** dataclass (frozen): `source: str`, `target: str`,
  `weight: float` (0.0 = unsafe to share, 1.0 = very safe), `floor:
  float` (the pair's `structural_floor`), `gap: float` (the pair's
  `cancellation_gap`), `is_cross_cluster: bool`, `reason: str` (short
  human-readable label, e.g. `"high_gap_low_floor"`,
  `"cross_cluster"`, `"high_floor"`).

- **`SharingGraph`** dataclass (frozen): `nodes: tuple[str, ...]`,
  `edges: tuple[SharingEdge, ...]`, `clusters: tuple[tuple[str, ...],
  ...]` (connected components on the threshold subgraph),
  `metadata: dict[str, Any]` (selection method, feature count,
  threshold value, formula reference). Provides:
  - `to_json() -> str` — deterministic, sorted-key JSON serialization
    (the artifact format the user's sketch proposed).
  - `to_markdown() -> str` — sibling to `render_report`: short
    summary, edge table sorted by descending weight, cluster
    listing.

- **`build_sharing_graph(prediction: TriagePrediction, *,
  threshold: float = 0.5, allow_cross_cluster: bool = False) ->
  SharingGraph`** — pure function over an existing
  `TriagePrediction`. Edge weight is computed from the per-pair
  fields by:

  ```
  weight(pair) = clip01(pair.cancellation_gap / max(pair.current_overlap, ε))
                  × (0.0 if pair.current_overlap > FLOOR_BLOCK else 1.0)
                  × (0.0 if (cross_cluster and not allow_cross_cluster) else 1.0)
  ```

  where `FLOOR_BLOCK = 0.5` (overlaps above this are too coincident
  to safely compress regardless of phase headroom) and `ε = 1e-12`
  guards the divide. Edges with `weight < threshold` are dropped
  before component analysis. The `reason` field records which factor
  dominated the weight (or, for dropped edges, which gate fired).

  This is the entire scoring rule. It is one closed-form expression
  over fields that already exist on `PairPrediction`; it does NOT
  introduce phase-sensitivity-only scoring (V alone confuses
  "swing magnitude" with "compressibility": a pair with V=0.4
  centered at M=0.6 has high swing but its floor is 0.2, not zero)
  and it does NOT use composite weights.

- **`render_sharing_graph_section(graph: SharingGraph) -> str`** —
  appends to the existing `render_report` output. Adds a "Sharing
  graph" heading, the threshold/formula footer, and the cluster
  list. The original `render_report` stays unchanged; callers opt
  into the addendum.

### Edge-weight rationale (for the spec body)

`cancellation_gap / current_overlap` is the *fractional phase
headroom*: how much of the current squared overlap is reachable by
phase tuning. A pair sitting at `current_overlap = 0.8, floor = 0.05`
has `gap = 0.75`, ratio `0.94` — almost all of the overlap can be
phased away, so the pair compresses well *if we control φ at use
time*. A pair at `current_overlap = 0.05, floor = 0.04` has `gap =
0.01`, ratio `0.2` — already separated, but no headroom; sharing it
isn't dangerous, but isn't load-bearing either, so we threshold it
out by default. The `FLOOR_BLOCK` cutoff handles the inverse: a pair
with `floor = 0.6` is too close even after maximal cancellation —
weight zero regardless.

### CLI integration (additive)

`polygram analyze` (existing subcommand) gains optional
`--sharing-graph <path.json>` and `--sharing-threshold <float>`
flags. When `--sharing-graph` is supplied, the CLI calls
`build_sharing_graph` and writes the JSON artifact alongside the
markdown report; when omitted, behavior is unchanged.

### Tests

- `tests/test_analysis.py::TestSharingGraph`:
  - `build_sharing_graph` over the toy SAE fixture produces an edge
    set that respects the threshold.
  - Edge `weight ∈ [0.0, 1.0]` for every emitted edge.
  - Cross-cluster edges are dropped when
    `allow_cross_cluster=False` and present (with `is_cross_cluster=
    True`) when `True`.
  - `floor > FLOOR_BLOCK` blocks an edge regardless of gap.
  - `to_json()` round-trips through `json.loads` cleanly and the
    output is byte-identical across two calls (deterministic).
  - `clusters` are exactly the connected components of the kept-edge
    subgraph (verified against a hand-computed adjacency on the
    fixture).
- `tests/test_cli.py::test_analyze_emits_sharing_graph` — invoking
  `polygram analyze --sharing-graph <out.json>` writes parseable JSON
  whose schema matches the dataclasses.

## Capabilities

### Modified Capabilities

- `analysis` — `SharingEdge`, `SharingGraph`, `build_sharing_graph`,
  `render_sharing_graph_section` added to the public API.
- `cli` — `analyze` subcommand gains `--sharing-graph` /
  `--sharing-threshold` flags.

### New Capabilities

*(none — additive on existing `analysis` and `cli` capabilities)*

## Out of Scope

- **Per-pair `Cancellation` runs as edge weights.** Scaling and
  cluster-shattering hazard documented above. Closed-form V already
  captures the relevant signal; running the optimization adds cost
  and risk for no information gain at this layer.
- **Community detection (Leiden / Louvain).** Threshold +
  connected-components handles ≤16-feature triage cleanly. When a
  workload arrives that breaks this assumption, propose a follow-up.
- **Composite weighted scoring.** No defensible empirical grounding
  for `0.5·a + 0.3·b + 0.2·c`-style weights yet. Single defensible
  signal (fractional gap + hard floor/cluster gates) is honest about
  what we know.
- **Cross-SAE-layer sharing.** This change scores sharing within a
  single SAE's selected subset. Cross-layer compression is a
  different problem (different feature spaces) and out of scope.
- **Force-directed visualization / D3 export.** The JSON artifact is
  the integration point; visualization belongs in a notebook or
  downstream tool, not this primitive.

## Impact

- `polygram/analysis/sharing_graph.py` — new module (~150 LOC) for
  `SharingEdge`, `SharingGraph`, `build_sharing_graph`,
  `render_sharing_graph_section`, edge-weight rule and threshold
  constants.
- `polygram/analysis/__init__.py` — re-exports the new symbols.
- `polygram/cli.py` — `_cmd_analyze` gains the two new flags.
- `tests/test_analysis.py` — new `TestSharingGraph` test class.
- `tests/test_cli.py` — new CLI flag test.
- No q-orca dependency change; no changes to `dictionary`,
  `experiment`, or `sae` capabilities.
- `tech-debt-backlog` — no entry needed; the deferred items
  (community detection, composite scoring) are explicitly named in
  Out of Scope above and live there as a record.
