## Why

`Rung4` caps at **32 linearly-independent features** because its
product amp branch is fixed at exactly 2 amp qubits (q3, q4),
giving per-feature Hilbert dim `8 · 4 = 32`. For sae-forge pareto
sweeps that vary feature count along the compression axis, 32 is
already small relative to typical SAE widths and forces a hard
encoding-switch step in any sweep that wants to push N above 32.

`Rung5` generalises Rung4's product amp branch to **k independent
single-qubit amps on qubits q3..q3+k−1**, with `k` fixed at
`Rung5` construction time. Per-feature Hilbert dim becomes
`8 · 2^k`, so a single `Rung5(n_amp_qubits=k)` family parameterises
a clean ladder of feature caps (`k=2 → 32`, `k=3 → 64`, `k=4 → 128`,
…) without changing the underlying math: the amp overlap is still a
product of `_single_qubit_overlap` calls, and the default-knob
property still collapses the gram to the MPSRung1-equivalent gram.

`k` is dictionary-wide and immutable after `Rung5` construction.
Per-feature variable-k is explicitly out of scope; sae-forge selects
a single `k` per pareto point.

Empirically motivated:

| Encoding | amp qubits | per-feature dim | empirical cap |
|---|---|---|---|
| MPSRung1 | 0 | 8 | 8 |
| Rung4 | 2 (fixed) | 32 | 32 |
| **Rung5(k)** (this change) | **k (configurable)** | **8 · 2^k** | empirically verified per-k at sweep time |

`Rung4` (the `k=2` case) stays as a separate, frozen encoding — this
change does not deprecate or absorb it.

## What Changes

- **New encoding class `Rung5`** in `polygram/encoding.py`. Frozen
  dataclass with two fields: `bond_dim: int = 2` (validated like
  every prior rung — `bond_dim != 2` raises) and
  `n_amp_qubits: int` (required, no default; positive int). Declares
  `max_features` as `8 * 2 ** n_amp_qubits` per instance (computed,
  not class-level — first encoding where `max_features` varies by
  instance).
- **New per-feature amp parameterisation** on `Feature`: a single
  field `amp_knobs: tuple[tuple[float, float], ...] = ()` carrying
  the k pairs `((θ_amp_0, ψ_amp_0), …, (θ_amp_{k−1}, ψ_amp_{k−1}))`.
  Rung4-only and Rung3-only dictionaries are unaffected — they
  continue to use `theta_amp`, `psi_aux`, `theta_amp_b`, `psi_amp_b`.
  `Rung5` dispatch SHALL ignore the Rung3/Rung4 amp fields and read
  exclusively from `amp_knobs`. Length-validation against the
  encoding's `n_amp_qubits` happens in `Dictionary.__post_init__`.
- **New analytic overlap functions** `rung5_amp_overlap` (complex)
  and `rung5_amp_overlap_squared` (real). Each takes two equal-length
  tuples of `(θ, ψ)` pairs and returns the product of
  `_single_qubit_overlap` calls — one per amp qubit.
- **`Dictionary.gram()` dispatch** for `Rung5` (elementwise-product
  factorisation identical in shape to Rung3/Rung4, with the new
  k-factor overlap).
- **Cancellation primitive extension**: `Cancellation(encoding="rung5")`
  with a `(2 + 2k)`-knob joint optimiser
  `[a.phi, b.phi, b.amp_knobs[0].θ, b.amp_knobs[0].ψ, …,
   b.amp_knobs[k−1].θ, b.amp_knobs[k−1].ψ]`. `min_amp_overlap`
  applies to the k-fold product overlap. `structural_floor` reduces
  to the MPS-phase-only floor of the same (α, β, γ), matching
  Rung3/Rung4.
- **Q-OrCA emission** for Rung5 dictionaries. Amp-branch action
  signature accepts `2k` amp knobs per feature; circuit is `k`
  independent single-qubit preparations on q3..q3+k−1 (no
  entanglement across amp qubits — same shape as Rung4 at `k=2`).
- **Worked example** `examples/rung5_rank_verification.py`
  parameterised over a small k-ladder (e.g. k ∈ {2, 3, 4}) emitting
  empirical rank data and confirming saturation at `8 · 2^k` for each.

## Capabilities

### New Capabilities

- `rung5-encoding`: 5-qubit-plus encoding parallel to `Rung4` but
  with the amp register width `k` configurable at construction time.
  Per-feature Hilbert dim is `8 · 2^k`. Analytic gram via the
  elementwise-product factorisation, with the amp overlap factor
  being the product of k single-qubit overlaps. Reuses
  `_single_qubit_overlap`; no new math primitives.

### Modified Capabilities

- `cancellation`: `Cancellation.encoding` accepts `"rung5"`. The
  Rung5 joint optimiser handles `(2 + 2k)` knobs where `k` is read
  from the encoding instance. `min_amp_overlap` applies to the
  k-fold product overlap. `structural_floor` reduces to the
  MPS-phase-only floor of the same (α, β, γ), matching the
  Rung3/Rung4 pattern.
- `dictionary` (renamed in tree as `sae`): `Feature` gains
  `amp_knobs: tuple[tuple[float, float], ...] = ()`. Rung3 and Rung4
  dictionaries that don't populate it get the empty-tuple default.
  `Dictionary.__post_init__` SHALL validate, for `Rung5`-encoded
  dictionaries, that every feature's `amp_knobs` has length exactly
  `encoding.n_amp_qubits`.

## Impact

- `polygram/encoding.py` — new `Rung5` class, new `rung5_amp_overlap`
  / `rung5_amp_overlap_squared`, new `Rung5State`. Reuses the
  existing `_single_qubit_overlap` helper introduced by the Rung4
  change.
- `polygram/dictionary.py` — `Feature` gains the `amp_knobs` field;
  `Dictionary.__post_init__` adds Rung5 length-validation;
  `Dictionary.gram()` adds a `Rung5` branch.
- `polygram/cancellation.py` — `SUPPORTED_ENCODINGS` adds `"rung5"`;
  joint optimiser reads `k` from the encoding and assembles the
  `(2 + 2k)`-knob default list dynamically.
- `polygram/_qorca_emit.py` — Rung5 emission path; action signature
  parameterised over `k`.
- `examples/rung5_rank_verification.py` — new file confirming
  empirical rank `8 · 2^k` at N = `8 · 2^k` and saturation at
  N > `8 · 2^k` for k ∈ {2, 3, 4}.
- `docs/research/rung5-encoding.md` — short note documenting the
  generalisation, the design choice to fix `k` at construction, and
  the link to the sae-forge pareto-sweep use case.
- `tests/test_encoding.py`, `tests/test_dictionary.py`,
  `tests/test_cancellation.py`, `tests/test_qorca_emit.py`,
  `tests/test_examples.py` — new test cases for the Rung5 paths.

**Depends on** `per-encoding-feature-cap`'s instance-level
`max_features` resolution. Today's spec exposes `max_features` as a
ClassVar; Rung5 needs it readable from the instance (Python lookup
already supports this when implemented as an `@property` or computed
in `__post_init__`). If the loader currently reads `type(encoding).max_features`,
that read site SHALL be updated to read from the instance.

No breaking changes for existing Rung3/Rung4/MPSRung1 users. Default
encoding remains `MPSRung1`. The new `Feature.amp_knobs` field
defaults to `()`, which is ignored by every non-Rung5 dispatch.
