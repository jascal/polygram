# animals-example — tasks

## 1. Example script

- [x] 1.1 `examples/animals_interference.py` declares the 4-feature
      Animals dictionary with default β per cluster
- [x] 1.2 Builds an `Experiment` targeting `(dog_poodle, bird_hawk)` with
      `bird_hawk.phi` swept over 40 points in `[0, π]`
- [x] 1.3 Calls `experiment.materialize("examples/output/")` and
      `experiment.run()` (analytic backend); script is ≤ 80 lines

## 2. Notebook

- [x] 2.1 `examples/animals_interference.ipynb` mirrors the script with
      markdown context cells (motivation, what to look for in the plot)
- [x] 2.2 Final cell uses `matplotlib` to plot target-pair overlap vs
      `bird_hawk.phi`; output cells are cleared before commit

## 3. Index + tests

- [x] 3.1 `examples/README.md` — one-paragraph intro + links to the
      artifacts + a note that `examples/output/` is gitignored
- [x] 3.2 `tests/test_examples.py::test_animals_interference_runs`
      runs a coarsened version of the script (5 sweep points) and
      asserts: (a) `.q.orca.md` parses + verifies clean,
      (b) hierarchical ordering holds throughout the sweep,
      (c) result tensor shapes match. Destructive interference is
      *not* asserted — single-φ sweep on this geometry leaves the
      cross-cluster overlap above baseline; antisymmetric two-side
      φ steering is the future `Cancellation` primitive's job.
- [x] 3.3 Add `examples/output/` to `.gitignore`
