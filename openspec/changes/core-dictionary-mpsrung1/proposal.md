## Why

The first piece of real functionality Polygram needs is a way to declare a
polysemantic dictionary in Python and turn it into a set of rung-1 MPS
preparation parameters that Q-Orca can act on.

A dictionary is a set of named **features** (e.g. `"Dog_Poodle"`,
`"Bird_Hawk"`) organized into a shallow **hierarchy** (cluster →
sibling). Each feature must map onto a tuple of MPS staircase angles
`(α, β, γ)` plus an optional phase knob `φ`. The rung-1 staircase is

    Ry(qs[0], α); CNOT(qs[0], qs[1]);
    Ry(qs[1], α + β); Rz(qs[1], φ);
    CNOT(qs[1], qs[2]); Ry(qs[2], β + γ)

— matching `examples/larql-animals-interference.q.orca.md` in q-orca-lang.

Researchers should not have to compute angles by hand. Polygram should pick
defaults that produce a strictly ordered Gram (cluster siblings closer than
cross-cluster pairs) and let users override per-feature.

## What Changes

- Add `polygram.dictionary.Feature` — `name: str`, `cluster: str`,
  `alpha: float = 0.0`, `beta: float`, `gamma: float = 0.0`,
  `phi: float = 0.0`.
- Add `polygram.dictionary.Dictionary` — `name: str`,
  `features: list[Feature]`, `hierarchy: dict[str, list[str]]`
  (cluster name → list of feature names). Validates that every feature
  belongs to exactly one cluster declared in `hierarchy`.
- Add `polygram.encoding.MPSRung1` — `bond_dim: int = 2`,
  `phase_knobs: bool = True`. Acts as a tagged config marker for now;
  later changes use it to dispatch encoder behavior.
- Add a default-angle helper: given a hierarchy with `K` clusters, assign
  `β` values evenly across `[-0.5, 0.5]` so cluster centers are visually
  separated; users can override per-`Feature` to fine-tune.
- Add `Dictionary.gram() -> np.ndarray` that builds an in-memory Q-Orca
  `QMachineDef` (no `.q.orca.md` file written) and calls
  `q_orca.compiler.concept_gram_mps.compute_concept_gram_mps` to return the
  analytic Gram.

The `Experiment` / `InterferenceSweep` pieces are deferred to the next change.

## Capabilities

### New Capabilities

- `dictionary` — declarative API for feature dictionaries with hierarchy
- `encoding` — config marker for the rung-1 MPS encoding (with phase
  knobs); a single class today, leaves room for future encodings

### Modified Capabilities

*(none)*

## Impact

- `polygram/dictionary.py` — new (`Feature`, `Dictionary`)
- `polygram/encoding.py` — new (`MPSRung1`)
- `polygram/_qorca_emit.py` — new internal helper that builds an in-memory
  `QMachineDef` from a `Dictionary` (used by `Dictionary.gram()`; the
  `.q.orca.md` file emitter lands in the next change)
- `tests/test_dictionary.py`, `tests/test_encoding.py`,
  `tests/test_gram.py` — new
- Depends on `bootstrap-package` landing first (CI + smoke test).
