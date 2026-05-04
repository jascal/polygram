## Why

`extend-cancellation-sweep-hea` (archived 2026-05-03) and the in-
flight `add-cluster-shared-knobs` (PR #7) make per-pair experiments
expressive: a researcher can ask "does target pair `(A, B)` cancel
cleanly under cluster-shared θ knobs, while preserving siblings?" and
get a precise answer with materialized artifacts. But the workflow is
still **single-pair-at-a-time** — to triage an SAE subset of even 8
features (28 pairs) the researcher writes a loop, juggles output
directories, and aggregates results by hand.

This change ships the smallest piece of multi-pair tooling that makes
that triage one command:

- A `BatchExperiment` class that takes a `Dictionary` and a list of
  experiment specs, runs them across a configurable pair selection,
  and writes a single aggregated artifact.
- A `SharingGraph` output: nodes = features, edges = experiment-
  derived signals (cancellation gap, tier separation, phase
  sensitivity, best knobs). JSON serialization for downstream tools;
  optional matplotlib visualization.
- A `polygram batch` CLI subcommand that wires up the SAE → Dictionary
  load path (already shipped as `from_sae_lens`) and runs the batch.

**Scope discipline.** Earlier scoping discussion proposed a much
larger "north-star" pipeline (sharing graph → automatic compression
algorithm generation → distillation benchmarks → research paper).
That is **deliberately not** the shape of this change. The
load-bearing assumption — that quantum-encoding interference patterns
under our chosen encoding generalize to compressibility properties of
the underlying SAE features — has not been tested. This proposal
ships the *probe* (batch runner + sharing graph) and the *artifact*
(serializable graph). It does not ship a compression algorithm, a
semantic interpretation of edges, nor a magic-formula "safe sharing
strength" scalar. Each edge keeps multiple separate signals and lets
downstream tools do their own weighting.

## What Changes

### `BatchExperiment` — multi-pair experiment runner

- New dataclass `BatchExperiment` in a new module `polygram/batch.py`.
- Fields: `dictionary: Dictionary`, `experiments: list[str]`,
  `pairs: str | list[tuple[str, str]] = "all"`, `output_dir: Path |
  None = None`, `cancellation_kwargs: dict | None = None`,
  `sweep_kwargs: dict | None = None`.
- `experiments` is a list of named experiment kinds drawn from
  `{"sweep", "cancellation"}`. v0 ships those two; the list is open
  for future extension (e.g. `"cluster_shared_cancellation"` once
  PR #7 lands).
- `pairs` selects which feature pairs to run:
  - `"all"` — every unordered pair `(N choose 2)`,
  - `"cross_cluster"` — pairs whose two features live in different
    clusters,
  - `"within_cluster"` — pairs whose two features live in the same
    cluster,
  - `list[tuple[str, str]]` — explicit list.
- A safety rail: when `len(self._resolved_pairs) > 50`,
  `__post_init__` SHALL raise `ValueError` recommending the user
  narrow `pairs` (or set `force=True` to override). Cancellation runs
  ~seconds per pair; 50 caps the worst-case wall time around a few
  minutes for grid backend at default `max_steps`.
- `BatchExperiment.run() -> SharingGraph` — runs every requested
  experiment on every selected pair, optionally writing per-pair
  sub-artifacts under `output_dir/{a}_x_{b}/` when `output_dir` is
  set, and aggregates the results into a `SharingGraph`.

### `SharingGraph` — aggregated artifact

- New dataclass `SharingGraph` in `polygram/batch.py`.
- Fields: `nodes: list[str]` (feature names),
  `clusters: dict[str, str]` (feature → cluster), `edges: list[
  SharingEdge]`, `experiment_kinds: list[str]`,
  `dictionary_name: str`, `created_at: str` (ISO8601).
- `SharingEdge` carries per-pair signals as separate fields, **not** a
  collapsed score. Fields populated when the corresponding experiment
  ran:
  - `a: str`, `b: str` — feature names (alphabetical)
  - `before_overlap: float` — overlap at default knobs
  - `after_overlap: float | None` — best overlap from cancellation;
    `None` if cancellation not requested
  - `cancellation_gap: float | None` — `before − after`; `None` if
    cancellation not requested
  - `optimized_knobs: dict[str, float] | None` — best knob values
  - `tier_separation_after: float | None` — tier separation at the
    cancellation optimum (None if not computable)
  - `phase_sensitivity_std: float | None` — `np.std` of overlap over
    a `<a>.phi` sweep at default resolution; `None` if sweep not
    requested
  - `structural_floor: float | None` — only when defined per the
    `Cancellation.structural_floor()` contract (canonical MPS rung-1
    2-φ); `None` otherwise
- `SharingGraph.to_json(path)` writes a deterministic JSON document
  (sorted keys; floats rounded to 6 sig figs).
- `SharingGraph.plot(path)` (optional matplotlib): a node-link
  diagram with edge weight = `cancellation_gap` (when populated).
  Falls back to a per-edge scatter when only `before_overlap` is
  present.

### `polygram batch` CLI subcommand

- Extends `polygram/cli.py` (existing entry point already supports
  `run` and `analyze`) with `batch`:
  ```
  polygram batch [--sae PATH] [--dictionary PATH] \
                 --features id1,id2,... \
                 [--experiments sweep,cancellation] \
                 [--pairs all|cross_cluster|within_cluster] \
                 [--output-dir DIR]
  ```
- Exactly one of `--sae` (loads SAE JSON via `from_sae_lens`) or
  `--dictionary` (loads a serialized `.q.orca.md` Dictionary or a
  Python module exposing `build_dictionary()`) is required.
- `--features` is required when `--sae` is used; ignored when
  `--dictionary` is used (the dictionary already declares its
  features).
- `--experiments` defaults to `"sweep,cancellation"`. `--pairs`
  defaults to `"all"`. `--output-dir` defaults to a temp directory
  with the SharingGraph JSON path printed to stdout.

### Tests

- `tests/test_batch.py::TestBatchExperiment` — pair selection
  filters; safety-rail rejection above 50 pairs; per-pair
  sub-artifact materialization; SharingGraph fields populated
  correctly for each `experiments` configuration.
- `tests/test_batch.py::TestSharingGraph` — JSON round-trip
  preserves every edge field; deterministic ordering (alphabetical by
  `(a, b)`); plot writes a non-empty PNG (matplotlib opt-in).
- `tests/test_cli.py::TestBatchSubcommand` — `polygram batch` end-
  to-end on the toy SAE fixture writes a SharingGraph JSON.

### Example

- `examples/batch_animals_hea.py` — runs `BatchExperiment` on the
  Animals HEA dictionary across all 6 pairs, writes the SharingGraph
  + visualization to `examples/output/batch_animals_hea/`.
- Module docstring documents the output layout.

## Capabilities

### Modified Capabilities

- `cli` — new `batch` subcommand.

### New Capabilities

- `batch` — multi-pair experiment orchestration and the SharingGraph
  artifact.

## Out of Scope

- **Compression-algorithm derivation.** Going from a SharingGraph to
  a smaller classical model is the load-bearing research question
  this change deliberately does not answer. The SharingGraph is the
  probe output; what's done with it is downstream.
- **A magic "safe sharing strength" scalar.** Each edge keeps
  multiple separate signals (cancellation gap, tier separation, phase
  sensitivity). Collapsing to one number reads as a guarantee but
  bakes in a weighting choice that has no empirical justification yet.
  Downstream tools weight as needed.
- **Encoding-invariance verification.** A separate, complementary
  question: do the safe/unsafe classifications survive a change of
  encoding (MPS → HEA)? The right test is its own focused spike, not
  a feature in this runner. Captured as a research-track follow-up
  in `tech-debt-backlog`.
- **Distributed / parallel batch execution.** v0 runs pairs
  sequentially. Parallelism is a workload question (process pool? ray?
  separate machines?) that depends on real SAE-scale usage signals.
- **Cross-SAE batching.** v0 takes a single Dictionary. Multi-SAE
  comparison is a separate orchestration layer.
- **Scoring or thresholding edges into "safe / unsafe" labels.**
  Same reason as the magic-scalar item — labels are downstream
  decisions.

## Impact

- `polygram/batch.py` — new module with `BatchExperiment`,
  `SharingGraph`, `SharingEdge`.
- `polygram/cli.py` — `batch` subcommand handler.
- `polygram/__init__.py` — re-export `BatchExperiment`,
  `SharingGraph`.
- `pyproject.toml` — no new runtime deps; matplotlib stays the
  optional plotting dep already in use.
- `tests/test_batch.py`, `tests/test_cli.py`, `tests/test_examples.py`
  — extended for the new surfaces.
- `examples/batch_animals_hea.py` — new walk-through.
- `tech-debt-backlog` — new bullet recording the encoding-invariance
  spike as a follow-up.
