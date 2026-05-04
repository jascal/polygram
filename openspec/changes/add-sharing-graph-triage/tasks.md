# add-sharing-graph-triage — tasks

## 0. Proposal

- [x] 0.1 `proposal.md` — Why / What Changes / Capabilities / Out of
      Scope / Impact (covers both sharing and separation graphs;
      defers disentanglement to research note).
- [x] 0.2 `specs/analysis/spec.md` — ADDED Requirements for
      `build_sharing_graph`, `build_separation_graph`, `FeatureGraph`
      shape + `to_json`, `render_feature_graph_section`.
- [x] 0.3 `specs/cli/spec.md` — MODIFIED `analyze` with four new
      flags (`--sharing-graph`, `--sharing-threshold`,
      `--separation-graph`, `--separation-threshold`).
- [x] 0.4 `docs/research/spec-disentanglement-loop.md` — research
      note capturing the deferred uncompress direction and naming
      its blockers.
- [x] 0.5 `openspec validate add-sharing-graph-triage --strict` ✓.

## 1. Edge-weight primitives

- [x] 1.1 `polygram/analysis/feature_graph.py` —
      `SHARING_EDGE_FORMULA`, `SEPARATION_EDGE_FORMULA`, `FLOOR_BLOCK
      = 0.5` module-level constants.
- [x] 1.2 `_sharing_weight(p, allow_cross_cluster) -> tuple[float, str]`
      and `_separation_weight(p, include_within_cluster) ->
      tuple[float, str]` returning `(weight, reason)` tuples.
- [x] 1.3 Document the `reason` vocabulary in the module docstring
      (closed set of stable identifiers per kind).

## 2. Dataclasses

- [x] 2.1 `FeatureEdge` — frozen dataclass with `source`, `target`,
      `weight`, `floor`, `gap`, `is_cross_cluster`, `reason`.
- [x] 2.2 `FeatureGraph` — frozen dataclass with `kind`, `nodes`,
      `edges`, `clusters`, `metadata`. Edge ordering: descending
      weight, ties broken by `(source, target)`. Cluster ordering:
      descending size, ties broken by lexicographic first member.
- [x] 2.3 `FeatureGraph.to_json()` — `json.dumps` with
      `sort_keys=True`, `separators=(",", ":")`, deterministic field
      order via dataclass `asdict`. Verify byte-identical across two
      calls.

## 3. Builders

- [x] 3.1 `build_sharing_graph(prediction, *, threshold=0.5,
      allow_cross_cluster=False) -> FeatureGraph` — iterate
      `prediction.pairs`, compute `(weight, reason)`, drop edges
      where `weight < threshold`, build adjacency, run union-find
      for components. `metadata["kind"] = "sharing"`.
- [x] 3.2 `build_separation_graph(prediction, *, threshold=0.2,
      include_within_cluster=False) -> FeatureGraph` — same shape
      with the separation edge-weight rule.
      `metadata["kind"] = "separation"`.
- [x] 3.3 Singleton features (no kept edge) appear as size-1
      components for both kinds.

## 4. Renderer

- [x] 4.1 `render_feature_graph_section(graph) -> str` — kind-aware
      heading (`## Sharing graph` vs `## Separation graph`); shared
      shape for the `### Edges` table, `### Components` section,
      `### Formula` footer (which quotes the kind-specific formula).
- [x] 4.2 Section is deterministic given the input.

## 5. Public API

- [x] 5.1 `polygram/analysis/__init__.py` — re-export
      `FeatureEdge`, `FeatureGraph`, `build_sharing_graph`,
      `build_separation_graph`, `render_feature_graph_section`,
      `SHARING_EDGE_FORMULA`, `SEPARATION_EDGE_FORMULA`,
      `FLOOR_BLOCK`.

## 6. CLI

- [x] 6.1 `polygram/cli.py` — `_cmd_analyze` parser gains four
      flags (sharing-graph/threshold + separation-graph/threshold).
- [x] 6.2 Handler: when each `*-graph` flag is supplied, call the
      corresponding builder and write `graph.to_json()` to the path.
      Both flags can be supplied independently. Handle parse errors
      on the threshold flags with clean error messages.

## 7. Tests

- [x] 7.1 `tests/test_analysis.py::TestSharingGraph` —
      `test_edges_respect_threshold`, `test_weights_in_unit_interval`,
      `test_cross_cluster_gated_by_flag`,
      `test_high_floor_blocks_edge`,
      `test_clusters_are_connected_components`,
      `test_kind_and_formula_are_sharing`.
- [x] 7.2 `tests/test_analysis.py::TestSeparationGraph` —
      `test_weights_equal_floor_on_kept_pairs`,
      `test_within_cluster_gated_by_flag`,
      `test_clusters_are_connected_components`,
      `test_kind_and_formula_are_separation`.
- [x] 7.3 `tests/test_analysis.py::TestFeatureGraphSerialization` —
      `test_to_json_round_trips`,
      `test_to_json_byte_identical`.
- [x] 7.4 `tests/test_analysis.py::TestRenderFeatureGraphSection` —
      `test_sharing_section_headings_and_formula`,
      `test_separation_section_headings_and_formula`.
- [x] 7.5 `tests/test_cli.py::test_analyze_emits_sharing_graph` and
      `::test_analyze_emits_separation_graph` and
      `::test_analyze_emits_both_graphs` — invoke `polygram analyze`
      with each flag combination, assert parseable JSON with the
      right `"kind"`.
- [x] 7.6 `tests/test_cli.py::test_analyze_threshold_malformed` —
      invalid float on either threshold flag exits non-zero with a
      clear error.

## 8. Validate + commit

- [x] 8.1 Full pytest suite green; ruff clean.
- [x] 8.2 `openspec validate add-sharing-graph-triage --strict` ✓.
- [ ] 8.3 Commit + push, open impl PR, merge after review.

## 9. Archive

- [ ] 9.1 `openspec archive add-sharing-graph-triage` after merge —
      propagate the new requirements into
      `openspec/specs/{analysis,cli}/spec.md`.
