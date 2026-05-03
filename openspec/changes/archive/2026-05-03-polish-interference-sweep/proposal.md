## Why

The v0 milestone closed the core `InterferenceSweep` path: declare a
`Dictionary`, sweep `<feature>.phi`, get a verifiable `.q.orca.md` and
an `ExperimentResult` of overlaps + Gram tensors. What's missing for
researchers to actually *use* the primitive day-to-day:

1. **Tier statistics.** A 4×4 Gram per sweep point is too low-level —
   researchers want to see "how does each *tier* (self / sibling /
   cross-cluster) move under this phase steering?" That rollup belongs
   in `ExperimentResult`, not in every notebook.
2. **Plots.** Saving an `.npz` and a `.csv` is necessary but not
   sufficient — the most-asked-for output is "show me overlap vs φ".
   Polygram should ship a default plot renderer keyed off sweep
   dimensionality (1D → line, 2D → heatmap).
3. **Summary report.** `materialize()` writes machine + runner; it
   should also write a human-readable `summary.md` with sweep config,
   tier rollup, and assertion pass/fail counts so researchers can
   diff runs at a glance.
4. **Multi-axis confidence.** Multi-key sweep is implemented but
   under-tested — the Animals example only sweeps one axis. A 2D
   `(phi_dog, phi_bird)` sweep is the natural validation case and
   makes the heatmap renderer immediately useful.
5. **CLI.** `python examples/animals_interference.py` works, but
   `polygram run examples/animals_interference.py --output-dir DIR`
   gives a stable entry point for batch runs and CI without each
   example owning its own argparse boilerplate.

These changes are scope-bounded — no new physics primitives, no new
backends, no new file formats. They are the minimum that turns the v0
sweep machinery into something researchers can drive from a notebook
without writing glue.

## What Changes

- **MODIFIED** `experiment` capability:
  - `ExperimentResult` gains `tier_stats: dict[str, np.ndarray]` with
    keys `"self"`, `"sibling"`, `"cross_cluster"`. Each value has shape
    `(*sweep_dims,)` and holds the *mean* off-diagonal magnitude-squared
    overlap within that tier at each sweep point. Self tier collapses
    to constant 1.0 (kept for symmetry).
  - `ExperimentResult.plot(path, kind="overlap")` renders a default
    matplotlib figure: 1D sweep → line plot of target-pair overlap with
    in-cluster + cross-cluster baselines; 2D sweep → heatmap. Higher
    dimensions raise `NotImplementedError`. Returns the saved `Path`.
    `matplotlib` is an optional install (`polygram[plot]`); a clear
    `ImportError` fires if absent.
  - `Experiment.materialize(output_dir)` additionally writes
    `<name>_summary.md` containing: dictionary name, sweep axes +
    ranges, target pair, assertion list, tier-rollup table (min /
    mean / max per tier), and per-assertion pass-rate.
- **MODIFIED** `experiment` capability:
  - Multi-axis sweep is now an explicit, tested requirement (not an
    accidental property of the Cartesian-product implementation). 2D
    sweep with shape `(D1, D2, N, N)` is covered by tests and the
    Animals example optionally exercises it.
- **NEW** `cli` capability:
  - Add `polygram` console script entry-point. `polygram run <path>
    [--output-dir DIR] [--n-points N]` imports the target module
    (resolved as filesystem path or `module:function` form) and calls
    its `main(output_dir=...)` function. Module is required to expose
    a `main(output_dir: str | Path)` callable.
  - `polygram --version` prints the package version.
- Update `examples/animals_interference.py`:
  - Add a `mode` knob: `"single"` (current 40-pt φ_bird sweep) or
    `"two_axis"` (12×12 grid over φ_dog and φ_bird).
  - Save outputs into `examples/output/animals_interference/` (own
    subdir) and call `result.plot(...)` to drop a PNG.
- Update `examples/animals_interference.ipynb` to mirror.
- Tests:
  - `test_experiment.py` — add 2D sweep shape test, tier-stats
    correctness test (matched-φ baseline), and a plot-renders-and-
    saves test (smoke; assert PNG written).
  - `test_cli.py` — invoke the CLI on a tiny throwaway script under
    `tmp_path`, assert `--output-dir` is honored.
- README: add a screenshot/`![](...)` reference to the saved 1D plot
  and the 2D heatmap, plus the CLI invocation snippet.

## Capabilities

### New Capabilities

- `cli` — console-script entry point for running Polygram example
  modules and (eventually) batch experiments.

### Modified Capabilities

- `experiment` — tier statistics, default plotting, summary report,
  explicit multi-axis sweep guarantees.

## Impact

- `polygram/experiment.py` — tier stats, plot, summary
- `polygram/cli.py` — new module
- `pyproject.toml` — `[project.scripts] polygram = "polygram.cli:main"`,
  matplotlib already in `[plot]` extra
- `examples/animals_interference.py` + `.ipynb` — `mode` knob, plot calls
- `examples/output/animals_interference/` (gitignored)
- `tests/test_experiment.py`, `tests/test_cli.py` — new coverage
- `README.md` — plot screenshot + CLI snippet
- No q-orca dep version change. No physics changes.
