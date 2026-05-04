## ADDED Requirements

### Requirement: BatchExperiment runs experiments across feature pairs

Polygram SHALL expose a `BatchExperiment` dataclass that orchestrates
multi-pair experiment runs over a single `Dictionary`. The class is
imported as `polygram.BatchExperiment`.

Fields: `dictionary: Dictionary`, `experiments: list[str]`,
`pairs: str | list[tuple[str, str]] = "all"`,
`output_dir: Path | None = None`,
`cancellation_kwargs: dict | None = None`,
`sweep_kwargs: dict | None = None`,
`force: bool = False`.

`experiments` SHALL be a non-empty list drawn from the supported set
`("sweep", "cancellation")`. Unknown values SHALL be rejected by
`__post_init__` with `ValueError` listing the supported kinds.

`pairs` SHALL select which feature pairs to run on:

- `"all"` — every unordered pair `(N choose 2)` of distinct
  features in `dictionary.features`.
- `"cross_cluster"` — pairs whose two features have different
  `cluster` values.
- `"within_cluster"` — pairs whose two features share a `cluster`
  value (excluding self-pairs).
- `list[tuple[str, str]]` — explicit list. Each tuple SHALL name
  two distinct features declared in `dictionary.features`. Order
  within each tuple is normalized alphabetically.

The resolved pair list SHALL be deduplicated and sorted
alphabetically by `(a, b)`.

A safety rail SHALL fire when the resolved pair count exceeds 50
unless `force=True`: `__post_init__` raises `ValueError` naming the
count and recommending the user narrow `pairs` or pass `force=True`.

`BatchExperiment.run() -> SharingGraph` SHALL run the requested
experiments on each resolved pair and assemble the results into a
`SharingGraph`. When `output_dir` is non-`None`, per-pair
sub-artifacts SHALL be materialized under
`output_dir/{a}_x_{b}/` and the aggregated `sharing_graph.json`
SHALL be written at the top level.

#### Scenario: default pair selection covers every distinct pair

- **GIVEN** a `Dictionary` with 4 features
- **WHEN** `BatchExperiment(dictionary=d, experiments=["cancellation"]).run()`
  returns
- **THEN** the returned `SharingGraph` contains exactly 6 edges
  (one per `(N choose 2)` pair) and every edge's `(a, b)` is in
  alphabetical order

#### Scenario: cross_cluster filter excludes within-cluster pairs

- **GIVEN** a `Dictionary` with `hierarchy = {"dogs":
  ["dog_poodle", "dog_beagle"], "birds": ["bird_hawk",
  "bird_sparrow"]}`
- **WHEN** `BatchExperiment(..., pairs="cross_cluster").run()`
  returns
- **THEN** the SharingGraph has exactly 4 edges (2×2 cross
  product) and no edge has both endpoints in `"dogs"` or both in
  `"birds"`

#### Scenario: explicit pair list is honored verbatim

- **GIVEN** a 4-feature dictionary
- **WHEN** `BatchExperiment(..., pairs=[("dog_poodle",
  "bird_hawk")]).run()` returns
- **THEN** the SharingGraph has exactly 1 edge with
  `(a, b) == ("bird_hawk", "dog_poodle")` (alphabetical
  normalization)

#### Scenario: safety rail rejects oversized batches

- **GIVEN** a 12-feature dictionary (66 pairs) with
  `pairs="all"`
- **WHEN** `BatchExperiment(...)` is constructed without
  `force=True`
- **THEN** `__post_init__` raises `ValueError` naming the count
  `66` and recommending the user pass `force=True` or narrow
  `pairs`

#### Scenario: unsupported experiment kind rejected

- **WHEN** `BatchExperiment(..., experiments=["bogus"])` is
  constructed
- **THEN** `__post_init__` raises `ValueError` listing the
  supported kinds

#### Scenario: per-pair sub-artifacts written under output_dir

- **GIVEN** a `BatchExperiment` with
  `experiments=["cancellation"]` and `output_dir=tmp_path`
- **WHEN** `run()` returns
- **THEN** for every resolved pair `(a, b)`,
  `tmp_path / f"{a}_x_{b}"` exists and contains a
  `<dictionary_name>_at_optimum.q.orca.md`, and
  `tmp_path / "sharing_graph.json"` exists at the top level

### Requirement: SharingGraph is the aggregated batch artifact

Polygram SHALL expose a `SharingGraph` dataclass and a
`SharingEdge` dataclass under `polygram.batch`, re-exported at the
top level as `polygram.SharingGraph` and `polygram.SharingEdge`.

`SharingGraph` fields: `nodes: list[str]` (feature names),
`clusters: dict[str, str]` (feature name → cluster name),
`edges: list[SharingEdge]`,
`experiment_kinds: list[str]`,
`dictionary_name: str`,
`created_at: str` (ISO 8601 timestamp at run start).

`SharingEdge` fields: `a: str`, `b: str` (alphabetically ordered
endpoints), `before_overlap: float`,
`after_overlap: float | None`,
`cancellation_gap: float | None`,
`optimized_knobs: dict[str, float] | None`,
`tier_separation_after: float | None`,
`phase_sensitivity_std: float | None`,
`structural_floor: float | None`.

`SharingEdge` SHALL be a frozen dataclass. Fields whose
corresponding experiment did not run SHALL be `None` (e.g.
`cancellation_gap is None` when `"cancellation"` was not in
`experiments`).

`SharingGraph.to_json(path)` SHALL write a deterministic JSON
document: nodes sorted alphabetically; edges sorted alphabetically
by `(a, b)`; floats formatted with up to 6 significant figures;
`None` represented as JSON `null`. The same input SHALL produce
byte-identical output across runs.

`SharingGraph.from_json(path) -> SharingGraph` SHALL reconstruct a
graph from a JSON document produced by `to_json`. Round-trip
property: `from_json(to_json(g)) == g` for every `g` reachable
from `BatchExperiment.run()`.

`SharingGraph.plot(path)` SHALL render a node-link diagram via
matplotlib. Edge width SHALL be proportional to
`cancellation_gap` (when populated); edge color SHALL encode
`tier_separation_after` (when populated). When neither is
populated for any edge, the method SHALL fall back to a per-edge
scatter showing `before_overlap`. Lazy `import matplotlib`; if
unavailable, `ImportError` SHALL name the `polygram[plot]` extra.

#### Scenario: every edge field is preserved by JSON round-trip

- **GIVEN** a SharingGraph `g` produced by
  `BatchExperiment.run()` with both `"sweep"` and
  `"cancellation"` requested
- **WHEN** `g.to_json(p)` is called and the result is read back
  via `SharingGraph.from_json(p)`
- **THEN** every field of every edge equals the original
  (`None`s preserved as `None`)

#### Scenario: JSON output is deterministic

- **GIVEN** the same `BatchExperiment` configuration run twice
- **WHEN** both runs call `to_json` to the same path
- **THEN** the produced byte sequences are identical

#### Scenario: edges are alphabetically ordered

- **WHEN** any `SharingGraph` is constructed via
  `BatchExperiment.run()`
- **THEN** for every edge, `edge.a < edge.b` lexicographically,
  and the edge list is sorted by `(a, b)` lexicographically

#### Scenario: plot writes a non-empty PNG

- **GIVEN** a SharingGraph with `cancellation_gap` populated on
  every edge
- **WHEN** `g.plot(tmp_path / "graph.png")` is called
- **THEN** the path exists and the file size is non-zero

### Requirement: BatchExperiment supports both encodings

`BatchExperiment` SHALL accept dictionaries with either
`MPSRung1` or `HEA_Rung2` encoding. Per-pair experiments
dispatch on the encoding via the existing
`Dictionary.gram()` mechanism; no encoding-specific code paths
are introduced in this layer.

#### Scenario: HEA dictionary produces a valid SharingGraph

- **GIVEN** a `Dictionary` with
  `encoding=HEA_Rung2(depth=2)` and 4 features
- **WHEN** `BatchExperiment(dictionary=d,
  experiments=["cancellation"]).run()` returns
- **THEN** the SharingGraph has 6 edges with populated
  `before_overlap` and `after_overlap` fields, and
  `structural_floor` is `None` on every edge (per the
  `Cancellation.structural_floor()` contract on HEA)
