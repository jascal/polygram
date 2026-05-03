## Why

`Dictionary.gram()` (from `core-dictionary-mpsrung1`) produces a single Gram
matrix at one point in parameter space. The whole point of Polygram, and the
reason q-orca-lang PR #51 added safe `Rz` knobs in the first place, is to
*sweep* a phase parameter and watch interference patterns develop on
specific feature pairs.

This change adds the experiment layer: a way to declare which parameter to
sweep, which features the sweep targets, what to measure, and what
invariants to assert at each sweep point. It also adds the file-emitter
that writes a verifiable `.q.orca.md` next to the Python script — the
artifact a researcher would commit alongside the experiment.

## What Changes

- Add `polygram.experiment.Experiment` — `name: str`, `dictionary`,
  `target_pair: tuple[str, str]`, `sweep: dict[str, np.ndarray]`,
  `measures: list[str]` (subset of `{"overlap", "gram_matrix",
  "schmidt_rank"}`), `assertions: list[str]`, `seed: int = 0`.
- Add `polygram.experiment.InterferenceSweep` — runs the sweep by, for each
  point, mutating the dictionary's `φ` knob(s), recomputing the Gram, and
  collecting the requested measures. Returns an `ExperimentResult` with a
  pandas-free in-house container (NumPy arrays + a small dict) so the only
  hard dep stays `numpy + q-orca`.
- Add `polygram.emit.write_qorca(dictionary, path)` — writes a
  larql-animals-interference-style `.q.orca.md` file with header comments
  pointing back to the Polygram source (file path + Dictionary name + git
  hash if available).
- Add `Experiment.materialize(output_dir)` — emits:
  - `<name>.q.orca.md` (the dictionary at sweep midpoint, for reference)
  - `run_<name>.py` (a reproducible runner script)
  - `<name>_result.npz` (after `.run()`)
- Built-in assertions for v0:
  - `"hierarchical_ordering_preserved"` — at every sweep point,
    same-cluster overlap ≥ cross-cluster overlap for the target pair's
    cluster vs the other cluster
  - `"target_pair_destructive_at_endpoint"` — overlap at the last
    sweep point is below a tunable threshold (default 0.1)

`Cancellation`, `EntanglementProbe`, `HybridSteer` are roadmap items —
not in this change.

## Capabilities

### New Capabilities

- `experiment` — declarative phase-sweep experiments on a `Dictionary`
- `emit` — Q-Orca `.q.orca.md` file emission with provenance comments

### Modified Capabilities

- `dictionary` — adds a `with_phi(name, value)` helper used internally by
  the sweep runner; no breaking API changes

## Impact

- `polygram/experiment.py` — new (`Experiment`, `InterferenceSweep`,
  `ExperimentResult`)
- `polygram/emit.py` — new (`write_qorca`)
- `polygram/_assertions.py` — new (built-in assertion checkers)
- `tests/test_experiment.py`, `tests/test_emit.py` — new
- `tests/test_examples.py` is **not** added here — that lands with the
  `animals-example` change
- Depends on `core-dictionary-mpsrung1`
