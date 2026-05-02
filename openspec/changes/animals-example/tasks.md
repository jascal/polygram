# animals-example — tasks

## 1. Example script

- [ ] 1.1 `examples/animals_interference.py` declares the 4-feature
      Animals dictionary with default β per cluster
- [ ] 1.2 Builds an `Experiment` targeting `(Dog_Poodle, Bird_Hawk)` with
      `phi_cross` swept over 40 points in `[0, π]`
- [ ] 1.3 Calls `experiment.materialize("examples/output/")` and
      `experiment.run()` (analytic backend); script is ≤ 60 lines

## 2. Notebook

- [ ] 2.1 `examples/animals_interference.ipynb` mirrors the script with
      markdown context cells (motivation, what to look for in the plot)
- [ ] 2.2 Final cell uses `matplotlib` to plot target-pair overlap vs
      `phi_cross`; output cells are cleared before commit

## 3. Index + tests

- [ ] 3.1 `examples/README.md` — one-paragraph intro + links to both
      artifacts + a note that `examples/output/` is gitignored
- [ ] 3.2 `tests/test_examples.py::test_animals_interference_runs`
      runs a coarsened version of the script (5 sweep points) and
      asserts: (a) `.q.orca.md` parses + verifies clean, (b) destructive
      assertion holds at the endpoint, (c) hierarchical ordering holds
      throughout the sweep
- [ ] 3.3 Add `examples/output/` to `.gitignore`
