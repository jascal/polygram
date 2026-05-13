## Why

`Rung3` saturates at **16 linearly-independent features**, not the
32 the qubit count alone suggests. Per the empirical rank probe in
[`docs/research/rung3-rank-bound.md`](../../docs/research/rung3-rank-bound.md),
the bottleneck is the amp branch's parameterisation
`|amp(θ, ψ)⟩ = cos(θ)|00⟩ + e^(iψ) sin(θ)|11⟩`, which restricts the
amp-branch state to the 2-dim subspace `span{|00⟩, |11⟩}` of C⁴. The
2 missing basis directions (`|01⟩`, `|10⟩`) are structurally
unreachable.

`Rung4` replaces the entangled-Bell-pattern amp branch with a
**product amplitude branch**: two independent single-qubit amps on
qubits 3 and 4. Each single-qubit amp lives in C² (and its
parameter family spans the full C²), so the product spans
`C² ⊗ C² = C⁴`, raising the per-feature Hilbert dim from 16 to **32**.

The change keeps Rung3 intact (its viability spike's conclusions stay
valid) and ships Rung4 as a parallel opt-in encoding.

Empirically motivated:

| Encoding | amp Hilbert dim | total dim | empirical cap |
|---|---|---|---|
| MPSRung1 | n/a | 8 | 8 (saturated) |
| Rung3 | 2 (restricted subspace) | 16 | 16 (saturated, 15-order gap) |
| **Rung4 (this change)** | 4 (full C⁴ span) | **32** | empirically verified |

This bundles encoding + cancellation extension + Q-OrCA emission +
worked example + viability spike, mirroring the
`add-rung3-encoding-mvp` shape from 2026-05-05.

## What Changes

- **New encoding class `Rung4`** in `polygram/encoding.py`. Same
  frozen-dataclass shape as `Rung3` (`bond_dim: int = 2`). Adds
  `max_features = 32` (via `per-encoding-feature-cap`'s abstraction).
- **New per-feature amp knobs** on `Feature`: `theta_amp_b: float`
  and `psi_amp_b: float` (q4 single-qubit amp), additive on top of
  the existing `theta_amp` and `psi_aux` (which become the q3
  single-qubit amp under Rung4's interpretation). Backwards-
  compatible: defaulting the new fields preserves Rung3's per-feature
  data layout.
- **New analytic overlap functions** `rung4_amp_overlap` (complex)
  and `rung4_amp_overlap_squared` (real). Each is the product of two
  single-qubit overlaps, factoring through a shared
  `_single_qubit_overlap` helper.
- **`Dictionary.gram()` dispatch** for `Rung4` (same elementwise-
  product factorisation as Rung3 with the new overlap).
- **Cancellation primitive extension**: `Cancellation(encoding="rung4")`
  with a 6-knob default joint optimiser
  `[a.phi, b.phi, b.theta_amp, b.psi_aux, b.theta_amp_b, b.psi_amp_b]`.
  Same `min_amp_overlap` constraint shape against the product-amp
  overlap.
- **Q-OrCA emission** for Rung4 dictionaries. Amp-branch action
  signature accepts 4 amp knobs per feature; circuit is two
  independent single-qubit preparations on q3, q4 (no CNOT between
  them — simpler than Rung3's Bell-pattern amp).
- **Worked example** `examples/rung4_viability_spike.py` mirroring
  the Rung3 spike methodology; emits `data/rung4_viability_spike.json`.
- **Research note** `docs/research/rung4-viability-spike.md` with the
  four-criterion (A/B/C/D) bucket analysis and a recommendation on
  whether Rung4 should default-on or stay opt-in like Rung3.

## Capabilities

### New Capabilities

- `rung4-encoding`: 5-qubit encoding parallel to `Rung3` with a
  product (rather than Bell-pattern) amp branch. Supports up to 32
  linearly-independent features. Analytic gram via the elementwise-
  product factorisation reusing the existing single-qubit overlap
  math.

### Modified Capabilities

- `cancellation`: `Cancellation.encoding` accepts `"rung4"`. The
  Rung4 joint optimiser handles 6 knobs (φ_a, φ_b, plus B's four amp
  knobs). `min_amp_overlap` applies to the product-amp overlap.
  `structural_floor` reduces to the MPS-phase-only floor of the same
  (α, β, γ), matching the Rung3 pattern.
- `sae`: `Feature` gains `theta_amp_b: float = 0.0` and
  `psi_amp_b: float = 0.0` fields. Rung3 dictionaries that don't set
  them get the default (which under Rung3's amp factorisation has no
  effect; under Rung4 they parameterise the q4 single-qubit amp).

## Impact

- `polygram/encoding.py` — new `Rung4` class, new `_single_qubit_overlap`
  helper, new `rung4_amp_overlap` / `rung4_amp_overlap_squared`,
  new `Rung4State`.
- `polygram/dictionary.py` — `Feature` gains the two new fields;
  `Dictionary.gram()` adds a `Rung4` branch.
- `polygram/cancellation.py` — `SUPPORTED_ENCODINGS` adds `"rung4"`;
  `__post_init__` adds the Rung4 default-knobs list; new
  `_run_rung4_joint` (or refactor `_run_rung3_joint` into a shared
  `_run_amp_joint` parameterised by the amp knob list).
- `polygram/_qorca_emit.py` — Rung4 emission path (additive next to
  Rung3); action signature accepts 4 amp knobs.
- `examples/rung4_viability_spike.py` — new file.
- `examples/rung4_rank_verification.py` — small fixture verifying
  empirical saturation at 32 (parameterised re-run of the existing
  `examples/rung3_rank_probe.py`).
- `docs/research/rung4-viability-spike.md` — new research note.
- `docs/research/data/rung4_viability_spike.json` — artifact.
- `tests/test_encoding.py`, `tests/test_dictionary.py`,
  `tests/test_cancellation.py`, `tests/test_examples.py` — new test
  cases for the Rung4 paths.

**Depends on** `per-encoding-feature-cap` for `Rung4.max_features = 32`
to be enforced at the loader. Order: `per-encoding-feature-cap` ships
first; this change adds `Rung4` and uses the established mechanism.

No breaking changes. Default encoding remains `MPSRung1`. Rung3
users are unaffected (the new `Feature` fields default to values that
have no effect on Rung3 grams).
