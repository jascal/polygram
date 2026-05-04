# add-batch-experiment — tasks

## 1. SharingGraph + SharingEdge

- [ ] 1.1 New module `polygram/batch.py` with `SharingEdge`
      dataclass holding the per-pair signal fields named in the
      proposal. Use `dataclass(frozen=True, slots=True)` for cheap
      hashing/serialization.
- [ ] 1.2 `SharingGraph` dataclass with the field set named in the
      proposal. `__post_init__` validates that every edge's
      `(a, b)` is alphabetically ordered, both names appear in
      `nodes`, and there are no duplicate edges.
- [ ] 1.3 `SharingGraph.to_json(path)` — deterministic ordering
      (nodes sorted; edges sorted by `(a, b)`); floats rounded to
      6 sig figs via `format(v, ".6g")` then re-parsed; `None`
      preserved as JSON null.
- [ ] 1.4 `SharingGraph.from_json(path)` round-trip helper.
      Round-trip property: `from_json(to_json(g)) == g` for every
      `g` reachable from `BatchExperiment.run()`.
- [ ] 1.5 `SharingGraph.plot(path)` — node-link layout via
      matplotlib. Edge width ∝ `cancellation_gap` (when populated);
      edge color ∝ `tier_separation_after` (when populated). Lazy
      matplotlib import behind the same install hint as
      `CancellationResult.plot`.

## 2. BatchExperiment runner

- [ ] 2.1 `BatchExperiment` dataclass with the field set named in
      the proposal. `__post_init__` validates `experiments` against
      `SUPPORTED_EXPERIMENTS = ("sweep", "cancellation")`,
      resolves `pairs` to a concrete list of tuples, applies the
      ≤50-pair safety rail (override via `force=True`).
- [ ] 2.2 `_resolve_pairs(self) -> list[tuple[str, str]]` — handles
      `"all"`, `"cross_cluster"`, `"within_cluster"`, and explicit
      list. Pairs are alphabetically ordered and deduplicated.
- [ ] 2.3 `BatchExperiment.run() -> SharingGraph` — for each pair,
      runs the requested experiments via the existing
      `Cancellation` and `Experiment` primitives, captures the
      relevant fields, and assembles the SharingGraph.
- [ ] 2.4 Per-pair sub-artifact materialization: when
      `output_dir` is set, write each pair's CancellationResult
      artifacts under `output_dir/{a}_x_{b}/` and the aggregated
      `sharing_graph.json` at the top level.
- [ ] 2.5 Honest progress: print one line per pair to stdout
      (`pair_index/total: a x b — done in Xs`). No tqdm dep.

## 3. CLI subcommand

- [ ] 3.1 `polygram/cli.py` — add the `batch` subparser with the
      argument set named in the proposal.
- [ ] 3.2 `--sae` path: load via existing `from_sae_lens` helper,
      build a Dictionary from the listed `--features`.
- [ ] 3.3 `--dictionary` path: accept either a `.q.orca.md` file
      (parse and reconstruct) or a `module:callable` reference
      that exposes `build_dictionary()`. Mutual exclusion with
      `--sae` enforced via `argparse` group.
- [ ] 3.4 Run `BatchExperiment.run()`, write
      `sharing_graph.json` to the resolved output dir, print the
      output path to stdout.

## 4. Re-exports

- [ ] 4.1 `polygram/__init__.py` — re-export `BatchExperiment`,
      `SharingGraph`, `SharingEdge`. Keep alphabetized.

## 5. Example

- [ ] 5.1 `examples/batch_animals_hea.py` — runs
      `BatchExperiment` on the Animals HEA dictionary, all 6
      pairs, both `sweep` and `cancellation` experiments. Writes
      to `examples/output/batch_animals_hea/`.
- [ ] 5.2 Module docstring documents the output layout.

## 6. Tests

- [ ] 6.1 `tests/test_batch.py::TestSharingEdge` — frozen,
      alphabetical pair ordering enforced.
- [ ] 6.2 `tests/test_batch.py::TestSharingGraph` — JSON
      round-trip preserves every edge field; deterministic ordering;
      `plot` writes a non-empty PNG (matplotlib opt-in).
- [ ] 6.3 `tests/test_batch.py::TestBatchExperiment` — pair
      selection filters; ≤50-pair safety rail rejected without
      `force`; per-pair sub-artifacts materialized; SharingGraph
      fields populated correctly for each `experiments`
      configuration; both MPS and HEA dictionaries supported.
- [ ] 6.4 `tests/test_cli.py::TestBatchSubcommand` — end-to-end
      on the `tests/fixtures/toy_sae.json` fixture; writes a
      valid SharingGraph JSON.
- [ ] 6.5 `tests/test_examples.py::test_batch_animals_hea_runs`
      — example produces the expected artifacts.

## 7. Backlog

- [ ] 7.1 `tech-debt-backlog` — new §3.1 bullet capturing the
      encoding-invariance spike (MPS vs HEA classification
      stability) as a research-track follow-up that should land
      before any compression-pipeline work.

## 8. Validate + commit

- [ ] 8.1 `openspec validate add-batch-experiment --strict` ✓
- [ ] 8.2 Full pytest suite green; ruff clean
- [ ] 8.3 Commit + push, open PR, merge after review

## 9. Archive

- [ ] 9.1 `openspec archive add-batch-experiment` after merge —
      propagate the new requirements into
      `openspec/specs/{batch,cli}/spec.md`.
