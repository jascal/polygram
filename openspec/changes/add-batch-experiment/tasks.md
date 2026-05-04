# add-batch-experiment — tasks

## 1. BatchResults + BatchRun

- [x] 1.1 New module `polygram/batch.py` with `BatchRun` frozen
      dataclass holding the per-pair fields named in the proposal
      (`source`, `target`, `predicted_floor`, `predicted_gap`,
      `current_overlap`, `achieved_overlap`,
      `cancellation_efficiency`, `best_knobs`,
      `tier_separation_after`, `artifact_subpath`).
- [x] 1.2 `BatchResults` frozen dataclass with the field set named
      in the proposal (`source_graph`, `dictionary_name`, `knobs`,
      `created_at`, `runs`).
- [x] 1.3 `BatchResults.to_json(path)` — deterministic ordering
      (runs in input graph edge order); floats formatted to 6 sig
      figs via `format(v, ".6g")` then re-parsed; `None` preserved
      as JSON null; nested `source_graph` emitted via the existing
      `FeatureGraph.to_json()`.
- [x] 1.4 `BatchResults.from_json(path) -> BatchResults`
      round-trip helper. Round-trip property: `from_json(to_json(b))
      == b` for every `b` reachable from `BatchExperiment.run()`.
- [x] 1.5 `cancellation_efficiency` computed as `(current_overlap −
      achieved_overlap) / predicted_gap` when `predicted_gap >
      1e-12`, else `0.0`.

## 2. BatchExperiment runner

- [x] 2.1 `BatchExperiment` dataclass with the field set named in
      the proposal (`feature_graph`, `dictionary`, `top_k=8`,
      `knobs="cluster_shared"`, `output_dir=None`,
      `cancellation_kwargs=None`). `__post_init__` validates
      `top_k ∈ [1, 16]`, `knobs ∈ {"cluster_shared",
      "per_feature"}`, and that the input graph's nodes are a
      subset of `dictionary.feature_names()`.
- [x] 2.2 `_resolve_knob_paths(self, edge) -> list[str]` — for
      `knobs="cluster_shared"`, returns `<cluster>.phi` (MPS) or
      `<cluster>.theta[r,d,q]` (HEA) for the cluster(s) the pair
      touches; for `knobs="per_feature"`, returns
      `<feature>.phi` / `<feature>.theta[r,d,q]` for both endpoints.
- [x] 2.3 `BatchExperiment.run() -> BatchResults` — for the first
      `min(top_k, len(graph.edges))` edges, builds and runs a
      `Cancellation` with `target_pair=(edge.source, edge.target)`,
      knob paths from §2.2, and `cancellation_kwargs or {}`.
      Captures `current_overlap`, `min_overlap`, `best_knobs`, and
      assembles `BatchRun`s in input-graph edge order.
- [x] 2.4 Per-pair sub-artifact materialization: when
      `output_dir` is set, write each pair's `Cancellation`
      artifact bundle under `output_dir/{source}_x_{target}/` and
      the aggregated `batch_results.json` at the top level.
- [x] 2.5 Honest progress: print one line per pair to stdout
      (`pair_index/total: source x target — done in Xs`). No
      tqdm dep.

## 3. CLI subcommand

- [x] 3.1 `polygram/cli.py` — add the `batch` subparser with the
      argument set named in the proposal (`--feature-graph FILE.json`,
      `--dictionary REF`, `--top-k N` default 8 with
      `[1, 16]` argparse-level validation, `--knobs
      cluster_shared|per_feature` default `cluster_shared`,
      `--output-dir DIR` default temp).
- [x] 3.2 `--feature-graph` parsed via `FeatureGraph.from_json`
      with try/except to surface parse errors on stderr.
- [~] 3.3 `--dictionary` accepted as either a `.q.orca.md` file
      path (parse + reconstruct via existing q-orca round-trip) or
      a `module:callable` reference whose callable returns a
      `Dictionary`.
      Partial: `module:callable` fully implemented and tested.
      `.q.orca.md` paths are rejected today with a clear
      `SystemExit` pointing at the `module:callable` form — the
      rung-1 wire format does not carry feature `cluster`
      assignments, so a proper inverse round-trip needs either a
      Polygram-side Dictionary metadata header or an HEA-only
      restriction. Filed as a follow-up under `tech-debt-backlog`
      §3.1 (encoding-invariance + round-trip); the rejection path
      is covered by
      `tests/test_cli.py::TestBatchSubcommand::test_qorca_md_dictionary_path_rejected`.
- [x] 3.4 Construct `BatchExperiment`; surface `__post_init__`
      `ValueError`s on stderr with non-zero exit. Run, write
      `batch_results.json`, print path on stdout.

## 4. Re-exports

- [x] 4.1 `polygram/__init__.py` — re-export `BatchExperiment`,
      `BatchResults`, `BatchRun`. Keep alphabetized.

## 5. Example

- [x] 5.1 `examples/batch_animals_hea.py` — runs
      `predict_cancellation_depth` → `build_separation_graph` →
      `BatchExperiment(top_k=4, knobs="cluster_shared")` on the
      Animals dictionary, writes `BatchResults` JSON and per-pair
      artifacts to `examples/output/batch_animals_hea/`.
- [x] 5.2 Module docstring documents the output layout and the
      prediction-vs-observation comparison the user can run.

## 6. Tests

- [x] 6.1 `tests/test_batch.py::TestBatchRun` — frozen,
      `cancellation_efficiency` is `0.0` when `predicted_gap == 0`.
- [x] 6.2 `tests/test_batch.py::TestBatchResults` — JSON
      round-trip preserves every field including the nested
      `source_graph`; deterministic byte-identical output across
      repeated calls; `source_graph.to_json()` survives the
      round-trip byte-for-byte.
- [x] 6.3 `tests/test_batch.py::TestBatchExperiment` — pair set is
      the input graph's top-K edges in order; `top_k > 16`
      rejected; `top_k > len(edges)` silently clamped; cluster-
      shared default produces `<cluster>.phi` knob paths on MPS;
      per-feature mode produces `<feature>.phi` paths; per-pair
      artifact subdirs materialized when `output_dir` is set;
      dictionary missing a graph node rejected with a clear error;
      both MPS and HEA dictionaries supported.
- [x] 6.4 `tests/test_cli.py::TestBatchSubcommand` — end-to-end on
      the `tests/fixtures/toy_sae.json` fixture: build a separation
      graph via `build_separation_graph`, serialize, invoke
      `polygram batch --feature-graph ... --dictionary
      module:callable --top-k 2`. Asserts: produced
      `batch_results.json` parses, `runs` length is 2,
      `source_graph` matches the input. Plus argparse rejection
      cases (`--top-k 0`, `--top-k 17`, `--knobs bogus`,
      malformed `--feature-graph`).
- [x] 6.5 `tests/test_examples.py::test_batch_animals_hea_runs` —
      example produces the expected artifacts.

## 7. Backlog

- [x] 7.1 `tech-debt-backlog` — new bullet capturing the
      encoding-invariance spike (MPS vs HEA classification
      stability) as a research-track follow-up that should land
      before any compression-pipeline work.

## 8. Validate + commit

- [x] 8.1 `openspec validate add-batch-experiment --strict` ✓
- [x] 8.2 Full pytest suite green; ruff clean
- [ ] 8.3 Commit + push, open PR, merge after review

## 9. Archive

- [ ] 9.1 `openspec archive add-batch-experiment` after merge —
      propagate the new requirements into
      `openspec/specs/{batch,cli}/spec.md`.
