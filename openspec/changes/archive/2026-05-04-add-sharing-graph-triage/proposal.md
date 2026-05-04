## Why

The `analysis-triage-layer` (archived 2026-05-03) ships
`predict_cancellation_depth`, which already computes the per-pair
`(M, V, structural_floor, cancellation_gap)` decomposition for any
selected SAE subset from `n_features + 1` closed-form Gram
evaluations. Researchers consume the per-pair table as markdown today
but the natural downstream artifacts — graphs that summarize *which
features can safely be compressed onto overlapping dimensions* and
*which pairs are dangerously coincident regardless of phase tuning* —
are left on the table.

The triage data already answers both questions. For every pair `(A, B)`
the closed-form bound

```
|<A|B>|²(δ) = M + V·cos(δ),  δ ∈ [0, 2π)
```

defines a `cancellation_gap = current_overlap − (M − |V|)`: the
distance the squared overlap can be driven down by tuning the relative
phase. A pair with a large gap and a low floor is a *good sharing
candidate* (phase tuning gives us room); a pair whose `structural_floor`
remains high regardless of phase tuning, especially across declared
clusters, is a *must-separate candidate* (no amount of φ tuning will
disambiguate it — the danger is in β/α/γ).

This change emits both as graph artifacts (`FeatureGraph`, with
`metadata["kind"]` distinguishing `"sharing"` from `"separation"`) and
sibling renderers alongside `render_report`, with edge weights defined
entirely by the existing closed-form fields. No new physics, no
per-pair `Cancellation` optimizations, no q-orca calls.

**What this proposal explicitly does NOT do.** A user-supplied design
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
  The `cancellation_gap` and `structural_floor` fields already carry
  the signal; coherence enters as a hard constraint (cross-cluster
  edges flagged), not a soft weight.
- **No community detection in v0.** Triage targets fixture sizes
  ≤16; threshold graph + connected components is enough. Leiden
  and friends are valuable when N grows past where naive grouping
  fails — defer until a workload demands it.

**What this proposal also defers.** A separate user sketch proposed an
*Uncompress / Disentanglement* primitive: a `DisentangleExperiment`
that fine-tunes SAE decoder weights via "quantum-informed gradients"
back-propagated from interference experiments. That direction is
exciting but the gradient path
`SAE_weights → from_sae_lens → Gram` is not currently differentiable
(KMeans + PCA in the import path), and Polygram has no training-loop
infrastructure (optimizer state, capability-preservation evaluator,
weight checkpointing). Captured as a research note at
`docs/research/spec-disentanglement-loop.md`; will earn its own
proposal when a toy fixture demonstrates the gradient signal exists.

## What Changes

### `analysis` capability — feature-graph artifacts

Add the following primitives to `polygram.analysis`:

- **`FeatureEdge`** dataclass (frozen): `source: str`, `target: str`,
  `weight: float` (semantics depend on `FeatureGraph.kind`),
  `floor: float` (the pair's `structural_floor`), `gap: float` (the
  pair's `cancellation_gap`), `is_cross_cluster: bool`,
  `reason: str` (short stable identifier from a closed vocabulary).

- **`FeatureGraph`** dataclass (frozen): `kind: str`
  (`"sharing"` or `"separation"`), `nodes: tuple[str, ...]`,
  `edges: tuple[FeatureEdge, ...]`, `clusters: tuple[tuple[str, ...],
  ...]` (connected components on the threshold subgraph),
  `metadata: dict[str, Any]` (selection method, feature count,
  threshold, formula). Provides:
  - `to_json() -> str` — deterministic, sorted-key JSON serialization.
  - The shape is the same for both kinds; the meaning of `weight`
    differs by `kind`.

- **`build_sharing_graph(prediction, *, threshold=0.5,
  allow_cross_cluster=False) -> FeatureGraph`** — pure function over
  an existing `TriagePrediction`. Edge weight is "sharing safety":

  ```
  ratio  = p.cancellation_gap / max(p.current_overlap, 1e-12)
  weight = clip(ratio, 0.0, 1.0)
           × (0.0 if p.structural_floor > FLOOR_BLOCK else 1.0)
           × (0.0 if (cross_cluster and not allow_cross_cluster) else 1.0)
  ```

  with `FLOOR_BLOCK = 0.5`. Edges with `weight < threshold` are
  dropped. Returns `FeatureGraph(kind="sharing", ...)`.

- **`build_separation_graph(prediction, *, threshold=0.2,
  include_within_cluster=False) -> FeatureGraph`** — pure function
  over the same `TriagePrediction`. Edge weight is "separation
  danger":

  ```
  weight = clip(p.structural_floor, 0.0, 1.0)
           × (0.0 if (not p.is_cross_cluster and not include_within_cluster)
              else 1.0)
  ```

  Rationale: `structural_floor` is the irreducible squared overlap
  no φ tuning can pierce. Cross-cluster pairs with high floor
  *cannot* be disambiguated by phase tuning alone — they are the
  pairs a future disentanglement primitive (β/α/γ adjustment) would
  need to target. Within-cluster floor is by-design (siblings share
  semantic similarity); flag only when explicitly requested. Edges
  with `weight < threshold` are dropped. Returns
  `FeatureGraph(kind="separation", ...)`.

- **`render_feature_graph_section(graph: FeatureGraph) -> str`** —
  appends to `render_report` output. Heading text is kind-aware
  (`"## Sharing graph"` vs `"## Separation graph"`); edges table,
  components section, and formula footer follow the same shape for
  both kinds. The original `render_report` stays unchanged; callers
  opt into the addendum.

### Edge-weight rationale (for the spec body)

**Sharing.** `cancellation_gap / current_overlap` is the *fractional
phase headroom*: how much of the current squared overlap is reachable
by phase tuning. A pair sitting at `current_overlap = 0.8, floor =
0.05` has `gap = 0.75`, ratio `0.94` — almost all of the overlap can
be phased away, so the pair compresses well *if we control φ at use
time*. A pair at `current_overlap = 0.05, floor = 0.04` has `gap =
0.01`, ratio `0.2` — already separated, but no headroom; sharing it
isn't dangerous, but isn't load-bearing either, so we threshold it
out by default. The `FLOOR_BLOCK` cutoff handles the inverse: a pair
with `floor = 0.6` is too close even after maximal cancellation —
weight zero regardless.

**Separation.** `structural_floor` directly *is* the irreducible
squared overlap. Pairs with high floor across cluster boundaries are
the pairs the user's "must-separate map" was reaching for. A
disentanglement primitive that searches β/α/γ rather than just φ —
the explicit out-of-scope #1 in `cancellation-phase-floor.md` — would
target precisely these edges. Until that primitive exists, the
separation graph is a *flag*, not an actionable repair: it tells the
user which pairs lie outside the phase-only triage's reach.

### CLI integration (additive)

`polygram analyze` (existing subcommand) gains optional flags:

- `--sharing-graph <path.json>` / `--sharing-threshold <float>`
- `--separation-graph <path.json>` / `--separation-threshold <float>`

When a `*-graph` flag is supplied, the CLI calls the corresponding
builder and writes the JSON artifact. Default behavior (neither flag
supplied) is unchanged.

### Tests

- `tests/test_analysis.py::TestSharingGraph`:
  - `build_sharing_graph` respects the threshold; weights in `[0, 1]`.
  - Cross-cluster edges gated by `allow_cross_cluster`.
  - `floor > FLOOR_BLOCK` blocks regardless of gap.
  - `clusters` are connected components of kept edges.
  - `to_json()` round-trips and is byte-identical across calls.
- `tests/test_analysis.py::TestSeparationGraph`:
  - `build_separation_graph` weights equal `structural_floor` on the
    cross-cluster pairs that pass threshold.
  - Within-cluster pairs are absent from the edge list with the
    default `include_within_cluster=False` and present (clamped to
    `[0, 1]`) when `True`.
  - `clusters` partition the kept-edge subgraph; isolated features
    appear as singletons.
  - `metadata["kind"] == "separation"` and the formula constant
    matches `SEPARATION_EDGE_FORMULA`.
- `tests/test_analysis.py::TestRenderFeatureGraphSection`:
  - Sharing graph renders `## Sharing graph`; separation renders
    `## Separation graph`; both contain the kind-appropriate
    formula constant.
- `tests/test_cli.py::test_analyze_emits_sharing_graph` and
  `::test_analyze_emits_separation_graph` — invoke the CLI with
  each flag and assert parseable JSON.

## Capabilities

### Modified Capabilities

- `analysis` — `FeatureEdge`, `FeatureGraph`, `build_sharing_graph`,
  `build_separation_graph`, `render_feature_graph_section`,
  `SHARING_EDGE_FORMULA`, `SEPARATION_EDGE_FORMULA`, `FLOOR_BLOCK`
  added to the public API.
- `cli` — `analyze` subcommand gains `--sharing-graph` /
  `--sharing-threshold` / `--separation-graph` /
  `--separation-threshold` flags.

### New Capabilities

*(none — additive on existing `analysis` and `cli` capabilities)*

## Out of Scope

- **Per-pair `Cancellation` runs as edge weights.** Scaling and
  cluster-shattering hazard documented above. Closed-form fields
  already capture the relevant signal.
- **Community detection (Leiden / Louvain).** Threshold +
  connected-components handles ≤16-feature triage cleanly. Defer
  until a workload demands it.
- **Composite weighted scoring.** No defensible empirical grounding
  for `0.5·a + 0.3·b + 0.2·c`-style weights yet.
- **Cross-SAE-layer sharing or separation.** Single-subset triage
  only. Cross-layer is a different problem (different feature
  spaces).
- **Disentanglement / uncompress primitive.** The "must-separate"
  graph this proposal emits is a *flag*, not a repair. Active
  disentanglement (β/α/γ adjustment, SAE-decoder fine-tuning,
  "quantum-informed gradients") is captured in
  `docs/research/spec-disentanglement-loop.md` and stays
  research-track until the gradient plumbing exists. See that note
  for the specific blockers.
- **Force-directed visualization / D3 export.** JSON artifact is
  the integration point; renderers belong downstream.

## Impact

- `polygram/analysis/feature_graph.py` — new module (~200 LOC) for
  `FeatureEdge`, `FeatureGraph`, `build_sharing_graph`,
  `build_separation_graph`, `render_feature_graph_section`, the
  edge-weight rules, and threshold constants.
- `polygram/analysis/__init__.py` — re-exports the new symbols.
- `polygram/cli.py` — `_cmd_analyze` gains four new flags.
- `tests/test_analysis.py` — new `TestSharingGraph`,
  `TestSeparationGraph`, `TestRenderFeatureGraphSection` test
  classes.
- `tests/test_cli.py` — new CLI flag tests.
- `docs/research/spec-disentanglement-loop.md` — new research note
  capturing the deferred uncompress direction.
- No q-orca dependency change; no changes to `dictionary`,
  `experiment`, or `sae` capabilities.
- `tech-debt-backlog` — no entry needed; deferred items are named
  in Out of Scope above and live there as a record.
