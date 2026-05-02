# cancellation-primitive — tasks

## 1. Cancellation core

- [ ] 1.1 `polygram/cancellation.py` — `Cancellation` dataclass with
      `dictionary`, `target_pair`, `tolerance=0.05`,
      `preserve_tiers=True`,
      `optimize={"method": "grid", "max_steps": 50}`,
      `optimize_all=False`
- [ ] 1.2 `Cancellation.__post_init__` validates: target_pair refs
      declared features, optimize dict has a known method, raises
      `NotImplementedError` if `optimize_all=True`
- [ ] 1.3 `Cancellation.run()` dispatches to `_run_grid` or
      `_run_scipy` and returns `CancellationResult`
- [ ] 1.4 `_run_grid(res)` — evaluate target-pair overlap on a
      `res × res` grid in `[0, 2π]²`; if `preserve_tiers`, mark
      cells where `hierarchical_ordering_preserved` is False as
      infeasible; pick the feasible argmin (or best infeasible if
      no feasible cells)
- [ ] 1.5 `_run_scipy(maxiter)` — lazy `from scipy.optimize import
      differential_evolution`; bounds `[(0, 2π), (0, 2π)]`,
      `seed=0`, `maxiter=maxiter`; objective adds a large penalty
      when `preserve_tiers` is violated
- [ ] 1.6 `_dictionary_at(phi_a, phi_b)` helper — clones dictionary
      with the two φs set on the target features

## 2. CancellationResult

- [ ] 2.1 `CancellationResult` dataclass: `optimized_phis`,
      `before_gram`, `after_gram`, `before_overlap`,
      `after_overlap`, `tolerance_met`, `method`, `trajectory`,
      `feasible_count`, `dictionary_at_optimum`, `target_pair`
- [ ] 2.2 `.plot(path)` — grid: imshow heatmap with infeasible mask
      and optimum star; scipy: line plot of objective vs
      evaluation count. Lazy matplotlib import; clear ImportError
- [ ] 2.3 `.materialize(output_dir)` — write
      `<name>.q.orca.md` (Dictionary at optimum), `<name>_summary.md`,
      `<name>_trajectory.csv`. Returns `dict[str, Path]`.

## 3. SAE polish

- [ ] 3.1 `SelectionReport` adds `reconstruction_error: dict[str,
      float]`, `tier_preservation: float | None`, `gamma_method: str`
- [ ] 3.2 `from_sae_lens` adds `assign_gamma: bool = False`,
      `gamma_range: tuple[float, float] = (-0.25, 0.25)`
- [ ] 3.3 When `assign_gamma=True`: per-cluster PCA on centered
      projection vectors, project each onto first PC, rescale to
      `gamma_range`, write into `Feature.gamma`
- [ ] 3.4 Compute `tier_preservation`: Pearson correlation of
      off-diagonal `|G|²` entries between projection-space
      cosine-overlap matrix and analytic Polygram Gram of the
      built Dictionary. None if N ≤ 1.
- [ ] 3.5 Compute `reconstruction_error[name]` as Euclidean
      distance from `feature.projection` to its assigned cluster
      centroid (in projection space)
- [ ] 3.6 `gamma_method = "zero"` (default) or `"projection_pca"`

## 4. Tests

- [ ] 4.1 `test_cancellation.py::test_grid_finds_drop_below_baseline`
      — Animals dictionary, target (dog_poodle, bird_hawk),
      tolerance=0.05; assert `after_overlap < before_overlap` and
      `feasible_count > 0`
- [ ] 4.2 `test_grid_with_preserve_tiers_false_can_be_lower`
      — running same case with preserve_tiers=False reaches at
      least as low as preserve_tiers=True
- [ ] 4.3 `test_optimize_all_not_yet_implemented`
- [ ] 4.4 `test_unknown_method_rejected`
- [ ] 4.5 `test_target_pair_must_reference_features`
- [ ] 4.6 `test_materialize_writes_optimized_qorca` — emitted
      `.q.orca.md` parses + verifies clean
- [ ] 4.7 `test_plot_grid_writes_png`
- [ ] 4.8 `test_scipy_backend_or_skip` — `pytest.importorskip("scipy")`
      and run a small case; assert `tolerance_met` reachable
- [ ] 4.9 `test_sae_import.py::test_assign_gamma_writes_nonzero_gammas`
- [ ] 4.10 `test_sae_import.py::test_reconstruction_error_per_feature`
- [ ] 4.11 `test_sae_import.py::test_tier_preservation_in_unit_interval`

## 5. Example

- [ ] 5.1 `examples/cancellation_example.py`: load toy SAE →
      Dictionary (4 features) → run an `InterferenceSweep` AND a
      `Cancellation`; save all artifacts to
      `<output_dir>/cancellation_example/`
- [ ] 5.2 Top-level `main(output_dir, n_points=None)` for CLI
      compat (`polygram run examples/cancellation_example.py
      --output-dir results/`)
- [ ] 5.3 `test_examples.py::test_cancellation_example_runs` —
      coarsened end-to-end + verifying `.q.orca.md`

## 6. Packaging + README

- [ ] 6.1 `pyproject.toml` — `[opt] = ["scipy"]`
- [ ] 6.2 `polygram/__init__.py` re-exports `Cancellation`,
      `CancellationResult`
- [ ] 6.3 README — quickstart adds Cancellation snippet; SAE section
      documents `assign_gamma` + new report fields

## 7. Validate + commit

- [ ] 7.1 `openspec validate cancellation-primitive --strict` ✓
- [ ] 7.2 All tests pass; ruff clean
- [ ] 7.3 Commit + push
