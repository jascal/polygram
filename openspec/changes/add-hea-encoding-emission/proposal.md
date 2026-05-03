## Why

q-orca-lang shipped first-class rung-2 HEA support in v0.9.0
(2026-05-03): explicit `## encoding` + `## theta` grammar plus a
declarative `concept_gram_tier_separation` invariant that the verifier
checks at Stage 4b. Polygram currently emits rung-1 MPS only — every
`Dictionary` ends up as a 3-qubit cross-coupled CNOT-staircase with at
most a single `Rz` phase knob. That's enough for the Animals example
but not for richer hierarchical feature dictionaries (deeper trees,
larger registers, multi-axis polysemy).

This change lifts Polygram to emit HEA-encoded machines as a second
encoding option, alongside the existing `MPSRung1`. Users opt in by
constructing a `Dictionary` with `encoding=HEA_Rung2(...)`; the
emitter then produces the new `## encoding` + `## theta` blocks,
populates a `cluster` column from each `Feature`'s declared cluster,
and wires in a default `concept_gram_tier_separation` invariant so
the q-orca verifier catches dictionaries whose θ tensors don't
deliver the cohesion/separation they claim.

The Cancellation primitive's structural-floor diagnostic was derived
for rung-1's single canonical δ. The multi-knob HEA generalization is
a research question — closed-form M ± V·cos(δ) bounds don't carry
over directly. That work is **deliberately deferred** to a follow-up
research-track proposal once we have a concrete SAE workload that
needs it. This change keeps Cancellation rung-1-only and only
extends the encoding/emission surface.

## What Changes

- Bump the q-orca dependency pin from `q-orca>=0.7.1` to
  `q-orca>=0.9.0` (the first PyPI release with HEA + tier-invariant
  support).
- Add `polygram.encoding.HEA_Rung2` — a frozen dataclass mirroring
  `MPSRung1` with fields `depth: int`, `entangler: str = "ring"`,
  `rotations: tuple[str, ...] = ("Ry", "Rz")`, and
  `tier_separation_bound: float = 0.025` (the recommended-default
  declared-invariant bound; users can override or pass `None` to
  suppress invariant emission).
- Add a per-feature θ tensor to `Feature` — optional
  `theta: np.ndarray | None = None` of shape
  `(|rotations|, depth, n_qubits)`. When `None`, the emitter
  generates a default tensor from the existing `(α, β, γ, φ)`
  knobs by laying them across the first layer (small numbers in
  every slot keeps the cohesion contract; outsiders use larger
  magnitudes). Users with their own optimization pipeline can
  pass arbitrary tensors directly.
- `Dictionary.encoding` accepts both `MPSRung1` and `HEA_Rung2`.
- The internal emitter (`polygram/_qorca_emit.py`) dispatches on
  `dictionary.encoding`. The HEA branch produces:
  - A `## encoding` table with `kind: hea`, `depth`, `entangler`,
    `rotations`.
  - A `## theta` table with three columns
    `| concept | tensor | cluster |` — the cluster column carries
    each `Feature`'s `cluster` field verbatim, giving the q-orca
    Stage 4b verifier the tier-grouping it needs.
  - A `## invariants` section declaring
    `concept_gram_tier_separation >= <bound>` when
    `tier_separation_bound is not None`.
- `Dictionary.gram()` dispatches on encoding too: rung-1 keeps
  calling `compute_concept_gram_mps`; HEA calls
  `compute_concept_gram_hea`.
- `polygram.emit.write_qorca` is unchanged in signature — its output
  changes only by virtue of the renderer now handling HEA.
- The Animals example gains a sibling
  `examples/animals_hea.py` that builds the same dictionary with
  `encoding=HEA_Rung2(depth=2)` and asserts the emitted machine
  verifies clean (Stage 4b green, including the tier-separation
  invariant). The original rung-1 example is untouched.

## Capabilities

### Modified Capabilities

- `dictionary` — `HEA_Rung2` encoding marker added; `Feature` gains
  optional θ tensor; `Dictionary.gram()` dispatches on encoding
- `experiment` — emitted `.q.orca.md` files now include `## encoding`
  + `## theta` + tier-separation invariant for HEA dictionaries

### New Capabilities

*(none — `encoding` is part of `dictionary` per the existing v0
layout)*

## Out of Scope

The following items appeared in early scoping discussions and are
explicitly **not** part of this change:

- **Multi-knob structural-floor / cancellation generalization.**
  Deferred to a separate research-track proposal once we have a
  concrete SAE workload demanding it. Cancellation stays rung-1-only.
- **Auto-generation of a multi-layer θ tensor from a sweep range.**
  Sweep semantics over an `(|rotations|, depth, n_qubits)` parameter
  space is non-trivial and out of scope here. Users who want to
  sweep on HEA can pass a fixed θ tensor and sweep over a single
  named scalar that the emitter substitutes — that thin extension
  can ship as a small follow-up.
- **`QFT_Rung3` and other algebraic encoding families.**
  Speculative until SAE evidence justifies them.
- **`Dictionary.from_sae_with_hea(...)` heuristic.** The depth /
  entangler choice is an empirical question; we'd be guessing without
  a real workload.

## Impact

- `polygram/encoding.py` — new `HEA_Rung2` class
- `polygram/dictionary.py` — `Feature.theta` field; `Dictionary` accepts
  the new encoding; `Dictionary.gram()` dispatches on encoding
- `polygram/_qorca_emit.py` — HEA branch added, with three-column
  `## theta` table and tier-separation invariant emission
- `polygram/emit.py` — no signature change; output format extends
- `pyproject.toml` — `q-orca` pin bumped to `>=0.9.0`
- `tests/test_encoding.py`, `tests/test_dictionary.py`,
  `tests/test_qorca_emit.py`, `tests/test_emit.py` — extended
- `examples/animals_hea.py` — new
- Depends on q-orca-lang v0.9.0 being on PyPI (shipped 2026-05-03).
