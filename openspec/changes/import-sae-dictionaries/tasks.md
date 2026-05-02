# import-sae-dictionaries — tasks

## 1. Toy fixture

- [x] 1.1 Write `tests/fixtures/toy_sae.json` — 16 features in 4
      clusters (mammals, birds, vehicles, fruits), 8-dim projection
      vectors with cluster-mean directions + small perturbations
      (σ=0.08, deterministic from numpy seed=42)
- [x] 1.2 Test: `load_toy_sae` round-trips the fixture (16 records,
      labels intact, projection arrays of length 8)

## 2. SAE import module

- [x] 2.1 `polygram/sae_import.py` with `SAEFeatureRecord`,
      `SelectionReport` frozen dataclasses
- [x] 2.2 `load_toy_sae(path)` JSON loader; validates required
      fields, casts projection list to `np.ndarray(float64)`
- [x] 2.3 `_kmeans(points, k, seed=0)` — pure numpy Lloyd's with
      k-means++ init; returns `(assignments, empty_cluster_indices)`
- [x] 2.4 `from_sae_lens(records, feature_ids, *, name,
      cluster_assignments, n_clusters, encoding, beta_range)` —
      capacity check (≤8), precedence (user → labels → kmeans),
      β spread across cluster means, returns `(Dictionary, report)`
- [x] 2.5 Re-exports from `polygram/__init__.py`

## 3. Tests

- [x] 3.1 `test_fixture_loads_clean`
- [x] 3.2 `test_select_too_many_features_rejected`
- [x] 3.3 `test_explicit_cluster_assignments_honored`
- [x] 3.4 `test_from_labels_path`
- [x] 3.5 `test_kmeans_default_separates_toy_fixture`
      (var_explained > 0.9 with k-means++ init)
- [x] 3.6 `test_beta_variance_in_unit_interval`
- [x] 3.7 `test_warning_on_overspecified_n_clusters`
- [x] 3.8 `test_returned_dictionary_is_valid_and_grams`
- [x] Plus: record-validation (2D / NaN), unknown id, empty
      selection, identical projections → 1.0, n_input_features
      surfaced — 15 tests total

## 4. Example

- [x] 4.1 `examples/import_from_sae.py` loads toy fixture, picks
      `[dog_poodle, dog_beagle, hawk_red, hawk_cooper]`, runs
      `hawk_red.phi` sweep, materializes + saves + plots
- [x] 4.2 Top-of-file docstring documents the swap path to a real
      SAE (sae_lens / safetensors pseudocode, "future loader" tag)
- [x] 4.3 `test_examples.py::test_import_from_sae_runs` —
      coarsened end-to-end + verifying `.q.orca.md`

## 5. Packaging + README

- [x] 5.1 `pyproject.toml` — `[project.optional-dependencies] sae`
      placeholder (empty in v0)
- [x] 5.2 README — install snippet, capacity-limits callout, SAE
      import section with code, example reference

## 6. Validate + commit

- [x] 6.1 `openspec validate import-sae-dictionaries --strict` ✓
- [x] 6.2 73 tests pass; ruff clean
- [ ] 6.3 Commit + push
