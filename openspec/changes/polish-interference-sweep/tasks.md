# polish-interference-sweep — tasks

## 1. Tier statistics

- [x] 1.1 Add `polygram._tier_stats.compute_tier_stats(gram, dictionary,
      target_pair)` returning `{"self": ..., "sibling": ...,
      "cross_cluster": ...}` arrays for a single Gram
- [x] 1.2 Wire `InterferenceSweep.run()` to populate
      `ExperimentResult.tier_stats: dict[str, np.ndarray]` with
      sweep-shaped arrays
- [x] 1.3 Test: matched-φ Animals dictionary → sibling tier ≈ 1.0
      (siblings share α/β/γ → identical states), cross_cluster tier ≈
      `cos(0.5)⁴ ≈ 0.5931`, self tier ≡ 1.0

## 2. Plotting

- [x] 2.1 `ExperimentResult.plot(path, kind="overlap")` — 1D line plot
      with tier baselines, 2D heatmap, ≥3D raises `NotImplementedError`
- [x] 2.2 Lazy matplotlib import; clear `ImportError` pointing to
      `polygram[plot]` extra if absent
- [x] 2.3 Tests: 1D PNG written, 2D PNG written, 3D raises

## 3. Summary report

- [x] 3.1 `Experiment.materialize()` writes `<name>_summary.md` with
      dictionary name, sweep axes, target pair, assertions list
- [x] 3.2 `ExperimentResult.write_summary()` appends tier rollup +
      target-overlap stats + assertion pass-rate
- [x] 3.3 Tests cover both materialize header and write_summary append

## 4. CLI

- [x] 4.1 `polygram/cli.py` parses `polygram run <target>
      [--output-dir DIR] [--n-points N]` and `polygram --version`
- [x] 4.2 Path or `pkg.mod:func` resolution; missing-`main` exits 2
      with clear error
- [x] 4.3 `[project.scripts] polygram = "polygram.cli:main"`
- [x] 4.4 Tests: path target writes to output_dir, missing-main errors,
      n-points forwarded when accepted, gracefully dropped when not

## 5. Animals example expansion

- [x] 5.1 `mode: Literal["single", "two_axis"]` knob on
      `build_experiment` (default `"single"`)
- [x] 5.2 `two_axis` mode: 12×12 default grid over `dog_poodle.phi` and
      `bird_hawk.phi`
- [x] 5.3 `main(output_dir, n_points, mode)` saves to
      `<output_dir>/animals_interference/` and calls `result.plot(...)`
- [x] 5.4 Notebook mirrors with both 1D and 2D cells

## 6. Tests + docs

- [x] 6.1 `test_experiment.py` — 2D sweep shape, tier-stats baselines,
      summary header + append, plot 1D/2D/3D, CSV tier columns
- [x] 6.2 README — quickstart, CLI snippet, 1D + 2D plot screenshots
- [x] 6.3 `openspec validate polish-interference-sweep --strict` ✓
- [x] 6.4 57 tests pass; `ruff check .` clean
