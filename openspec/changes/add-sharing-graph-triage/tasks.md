# add-sharing-graph-triage — tasks

## 0. Proposal

- [x] 0.1 `proposal.md` — Why / What Changes / Capabilities / Out of
      Scope / Impact.
- [x] 0.2 `specs/analysis/spec.md` — ADDED Requirements for
      `build_sharing_graph`, `SharingGraph.to_json`, and
      `render_sharing_graph_section`, with scenarios.
- [x] 0.3 `specs/cli/spec.md` — MODIFIED `analyze` subcommand with
      `--sharing-graph` and `--sharing-threshold` flags.
- [x] 0.4 `openspec validate add-sharing-graph-triage --strict` ✓.

## 1. Edge-weight primitive

- [ ] 1.1 `polygram/analysis/sharing_graph.py` — `EDGE_WEIGHT_FORMULA`
      module-level constant, `FLOOR_BLOCK = 0.5`, `_pair_weight(p,
      allow_cross_cluster) -> tuple[float, str]` returning
      `(weight, reason)` from a `PairPrediction`.
- [ ] 1.2 The reason vocabulary is a closed set of stable string
      identifiers (`"high_gap_low_floor"`,
      `"phase_separable_low_overlap"`, `"floor_blocked"`,
      `"cross_cluster_blocked"`, `"below_threshold"`); document the
      vocabulary in the module docstring.

## 2. Dataclasses

- [ ] 2.1 `SharingEdge` — frozen dataclass with `source`, `target`,
      `weight`, `floor`, `gap`, `is_cross_cluster`, `reason`.
- [ ] 2.2 `SharingGraph` — frozen dataclass with `nodes`, `edges`,
      `clusters`, `metadata`. Edge ordering: descending weight, ties
      broken by `(source, target)`. Cluster ordering: descending
      size, ties broken by lexicographic first member.
- [ ] 2.3 `SharingGraph.to_json()` — `json.dumps` with
      `sort_keys=True`, `separators=(",", ":")`, deterministic field
      order via dataclass `asdict`. Verify byte-identical across two
      calls in tests.

## 3. Builder

- [ ] 3.1 `build_sharing_graph(prediction, *, threshold=0.5,
      allow_cross_cluster=False) -> SharingGraph` — iterate
      `prediction.pairs`, compute `(weight, reason)`, drop edges
      where `weight < threshold`, build adjacency, run a small
      union-find to extract components, populate `metadata`.
- [ ] 3.2 Singleton features (no kept edge) appear as size-1
      components — verify against the toy fixture.

## 4. Renderer

- [ ] 4.1 `render_sharing_graph_section(graph) -> str` — markdown
      fragment with `## Sharing graph`, `### Edges`, `### Components`,
      `### Formula` headings. Edges table sorted by descending
      weight; components listed one per line; formula footer quotes
      `EDGE_WEIGHT_FORMULA`.
- [ ] 4.2 Section is deterministic given the input — no timestamps,
      no random ordering.

## 5. Public API

- [ ] 5.1 `polygram/analysis/__init__.py` — re-export
      `SharingEdge`, `SharingGraph`, `build_sharing_graph`,
      `render_sharing_graph_section`, `EDGE_WEIGHT_FORMULA`,
      `FLOOR_BLOCK`.

## 6. CLI

- [ ] 6.1 `polygram/cli.py` — `_cmd_analyze` parser gains
      `--sharing-graph` (path) and `--sharing-threshold` (float,
      default 0.5) flags.
- [ ] 6.2 Handler: when `--sharing-graph` is supplied, call
      `build_sharing_graph(prediction,
      threshold=args.sharing_threshold)` and write
      `graph.to_json()` to the path. Handle parse errors on
      `--sharing-threshold` with a clean error message.

## 7. Tests

- [ ] 7.1 `tests/test_analysis.py::TestSharingGraph` —
      `test_edges_respect_threshold`, `test_weights_in_unit_interval`,
      `test_cross_cluster_gated_by_flag`,
      `test_high_floor_blocks_edge`,
      `test_clusters_are_connected_components`,
      `test_to_json_round_trips`,
      `test_to_json_byte_identical`.
- [ ] 7.2 `tests/test_analysis.py::TestSharingGraphRender` —
      `test_section_contains_required_headings`,
      `test_section_quotes_formula`.
- [ ] 7.3 `tests/test_cli.py::test_analyze_emits_sharing_graph` —
      invoke `polygram analyze ... --sharing-graph ...`, assert the
      JSON parses and exposes the documented keys.
- [ ] 7.4 `tests/test_cli.py::test_analyze_sharing_threshold_malformed`
      — invalid float on `--sharing-threshold` exits non-zero with
      a clear error.

## 8. Validate + commit

- [ ] 8.1 Full pytest suite green; ruff clean.
- [ ] 8.2 `openspec validate add-sharing-graph-triage --strict` ✓.
- [ ] 8.3 Commit + push, open PR, merge after review.

## 9. Archive

- [ ] 9.1 `openspec archive add-sharing-graph-triage` after merge —
      propagate the new requirements into
      `openspec/specs/{analysis,cli}/spec.md`.
