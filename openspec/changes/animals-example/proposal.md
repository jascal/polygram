## Why

To validate the v0 stack end-to-end and give researchers a copy-pasteable
starting point, Polygram should ship a worked example that mirrors the
proposal's "PoodleHawk_Interference" story. The example exercises every
piece of the stack — `Dictionary`, `MPSRung1`, `Experiment`,
`InterferenceSweep`, the Q-Orca emitter, and the built-in assertions.

It also serves as documentation: anyone reading
`examples/animals_interference.py` should see the full Polygram API
in idiomatic form, ≤ 60 lines.

## What Changes

- Add `examples/animals_interference.py` — declares a 4-feature, 2-cluster
  Animals dictionary (Dog_Poodle, Dog_Beagle, Bird_Hawk, Bird_Sparrow),
  defines a `phi_cross` sweep targeting `(Dog_Poodle, Bird_Hawk)` over
  40 points in `[0, π]`, and runs it.
- Add `examples/animals_interference.ipynb` — same content as the script
  with explanatory markdown cells and a final `matplotlib` plot showing
  the target-pair overlap vs `phi_cross`.
- Add `examples/README.md` — short index pointing to both artifacts.
- Add `tests/test_examples.py::test_animals_interference_runs` — runs the
  example end-to-end with a coarsened sweep (5 points instead of 40),
  asserts the `.q.orca.md` artifact validates under
  `q_orca.verifier.verify`, and asserts the destructive-endpoint
  assertion holds at `phi = π`.
- Wire `examples/` into the test suite via the new test; do **not**
  ship the notebook output cells in version control (notebooks are
  cleared via a pre-commit-style script not added in this change).

## Capabilities

### New Capabilities

*(none — this change is example + integration test only)*

### Modified Capabilities

*(none)*

## Impact

- `examples/` directory created
- `tests/test_examples.py` — new
- Depends on `experiment-interference-sweep`
- Closes the v0 milestone — after this change archives, the next round
  of work moves to the "Roadmap" primitives (Cancellation,
  EntanglementProbe, HybridSteer) in their own OpenSpec changes
