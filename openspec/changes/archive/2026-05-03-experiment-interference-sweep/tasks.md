# experiment-interference-sweep — tasks

## 1. Experiment + InterferenceSweep

- [x] 1.1 `Experiment` dataclass with the fields listed in the proposal;
      sweep keys use `<feature_name>.phi` form (only `.phi` knob in v0)
- [x] 1.2 `Experiment.run(backend="analytic", shots=0)` delegates to
      `InterferenceSweep` for the analytic path; non-analytic backends
      raise `NotImplementedError` pointing to roadmap
- [x] 1.3 `InterferenceSweep` walks the Cartesian product of `sweep`
      values, recomputes Gram per point, and accumulates measures
      (overlap, gram_matrix, schmidt_rank)
- [x] 1.4 `ExperimentResult` exposes `gram_matrices`, `overlaps`,
      `schmidt_ranks`, `sweep_axes`, `assertion_pass`, plus a `save()`
      to .npz

## 2. Q-Orca emitter

- [x] 2.1 `emit.write_qorca(dictionary, path)` writes a `.q.orca.md`
      with all required sections (context, events, states, transitions,
      actions, verification rules), larql-animals-interference style
- [x] 2.2 Emitter prepends a top HTML comment block: dict name, feature
      count, generation timestamp (UTC), git rev or "unversioned"
- [x] 2.3 Emitter only ever uses preparation form — every Polygram
      machine has prep-form `prepare_*` events into `prepared_*` states

## 3. Materialize

- [x] 3.1 `Experiment.materialize(output_dir)` writes
      `<name>.q.orca.md` (dictionary at sweep midpoint) and
      `run_<name>.py`. The result npz is written separately via
      `ExperimentResult.save()` after `.run()`
- [x] 3.2 Generated `run_<name>.py` is self-contained: imports polygram,
      reconstructs the Dictionary explicitly, runs the sweep, saves
      result. Compiles cleanly under py_compile (verified in test)

## 4. Built-in assertions

- [x] 4.1 `_assertions.hierarchical_ordering_preserved(gram, dictionary,
      target_pair)` checks per-cluster siblings dominate the cross-pair
      overlap; returns bool per sweep point
- [x] 4.2 `_assertions.target_pair_destructive_at_endpoint(gram,
      dictionary, target_pair, threshold=0.1)` checks the final-point
      overlap; result broadcast across the sweep dim

## 5. Tests

- [x] 5.1 `test_experiment.py` — 13 tests covering validation (target
      pair, measures, assertions, sweep keys), backend dispatch, sweep
      shape, baseline overlap at φ=0 (matches `cos⁴(0.5)`), assertion
      arrays, materialize artifacts, runner script compiles, .npz
      round-trip
- [x] 5.2 `test_emit.py` — 4 tests: file created, provenance block
      present (dict name + git rev), q-orca parser parses cleanly with
      no errors, prep-form structure (every feature has both
      `prepare_<slug>` and `prepared_<slug>`)
- [x] 5.3 `test_state.py` — 5 tests for the local statevector simulator
      (zero-angles → |000>, normalization, baseline overlap, Schmidt
      rank 1 for product / 2 for entangled)
