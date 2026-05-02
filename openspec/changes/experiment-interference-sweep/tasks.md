# experiment-interference-sweep — tasks

## 1. Experiment + InterferenceSweep

- [ ] 1.1 `Experiment` dataclass with the fields listed in the proposal
- [ ] 1.2 `Experiment.run(shots: int = 0, backend: str = "analytic")`
      delegates to `InterferenceSweep` for the analytic path; `shots`
      and shot-based backends are stubbed (NotImplementedError) for v0
- [ ] 1.3 `InterferenceSweep` walks the Cartesian product of `sweep`
      values, recomputes Gram per point, and accumulates measures
- [ ] 1.4 `ExperimentResult` exposes `gram_matrices: np.ndarray` (shape
      `[*sweep_dims, N, N]`), `overlaps: np.ndarray` for the target
      pair, and `assertion_pass: dict[str, np.ndarray[bool]]`

## 2. Q-Orca emitter

- [ ] 2.1 `emit.write_qorca(dictionary, path)` writes a `.q.orca.md`
      with all `## context`, `## events`, `## state`, `## transitions`,
      `## actions`, `## verification rules` sections matching
      larql-animals-interference style
- [ ] 2.2 Emitter prepends a top comment block linking back to the
      Polygram source: dict name, generation timestamp, git rev (if in
      a repo, else "unversioned")
- [ ] 2.3 Emitter uses `form="preparation"` style (4 prep call sites in
      the transitions table) — never inverse form when φ ≠ 0 anywhere

## 3. Materialize

- [ ] 3.1 `Experiment.materialize(output_dir)` creates the dir if needed,
      writes `<name>.q.orca.md`, `run_<name>.py`, and a placeholder
      `<name>_result.npz` only after `.run()`
- [ ] 3.2 Generated `run_<name>.py` is self-contained: imports polygram,
      reconstructs the dictionary, runs the sweep, writes the result npz

## 4. Built-in assertions

- [ ] 4.1 `_assertions.hierarchical_ordering_preserved(gram, dictionary,
      target_pair)` returns bool per sweep point
- [ ] 4.2 `_assertions.target_pair_destructive_at_endpoint(gram,
      target_pair, threshold=0.1)` returns bool for the last sweep point

## 5. Tests

- [ ] 5.1 `test_experiment.py` — toy 2×2 dictionary, sweep φ in `[0, π/2]`
      at 5 points, verify `gram_matrices.shape`, target overlap is
      monotonic-ish, assertions evaluate without raising
- [ ] 5.2 `test_emit.py` — emit a dictionary, parse the result back with
      `q_orca.parser.parse_q_orca_markdown`, assert state count and
      transition count match expected
- [ ] 5.3 `test_emit.py` — provenance comment block contains the
      Polygram dict name and a non-empty git rev field
