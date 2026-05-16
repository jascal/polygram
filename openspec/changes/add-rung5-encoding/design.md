## Context

`Rung4` (the `k=2` case of a generalised product-amp encoding) ships
as a fixed 5-qubit encoding: MPSRung1 on q0–q2 + two independent
single-qubit amps on q3 and q4. Per-feature Hilbert dim is `8 · 4 = 32`,
so dictionaries cap at 32 linearly-independent features.

The sae-forge pareto-sweep workflow needs to vary feature count along
the compression axis. At today's encoding ladder
(MPSRung1: 8, Rung3: 16, Rung4: 32) any sweep that wants N > 32 has
no encoding to target, and any sweep that wants to compare neighbour
points like N=24 vs N=48 vs N=96 has to step across an arbitrary
encoding boundary.

The Rung4 architecture already factorises cleanly: the amp overlap is
a product of two independent single-qubit overlaps via the
`_single_qubit_overlap` helper. Generalising the product width from 2
to k is mechanical — the math primitive doesn't change, just the
fold width.

Stakeholders: sae-forge (consumer; drives the k-ladder requirement);
polygram cancellation primitive (consumer; needs k-parameterised
joint optimiser); Q-OrCA emitter (consumer; needs k-parameterised
action signature).

## Goals / Non-Goals

**Goals:**

- Ship `Rung5(n_amp_qubits=k)` as a frozen encoding with `k`
  fixed at construction time and `max_features = 8 · 2^k` derived
  per-instance.
- Reuse the existing `_single_qubit_overlap` math primitive
  unchanged. No new analytic surface besides the k-fold product
  wrapper.
- Preserve the "default knobs → MPSRung1-equivalent gram" property
  for every choice of k.
- Cancellation primitive accepts `"rung5"` and reads k from the
  encoding instance to size its joint optimiser.
- Q-OrCA emission supports Rung5 with a k-parameterised action
  signature.
- Empirical rank verification at multiple k values (≥ 3 points on
  the ladder).

**Non-Goals:**

- Per-feature variable k. `n_amp_qubits` is dictionary-wide and
  immutable after construction. Mixing different amp widths inside
  one dictionary is out of scope.
- Auto-growing k as features are added. sae-forge selects k per
  pareto point; the dictionary commits to that k at construction
  and stays there.
- Deprecating, absorbing, or aliasing Rung4. Rung4 stays as a
  separate, frozen encoding; users who already target Rung4 see no
  change.
- Generalising the MPS core. The MPSRung1 block on q0–q2 stays at
  bond_dim=2, n_qubits=3. Any future work to widen the MPS core is
  a separate change.
- Entangling amp qubits. Rung5's amp branch is product (no CNOTs
  between amp qubits), matching Rung4's geometry. Bell-pattern or
  more exotic amp topologies are out of scope.
- Tuning-time selection of k. Picking the right k for a given SAE
  is sae-forge's job; polygram only enforces the contract.

## Decisions

### Decision 1: `k` is a constructor parameter, not a class-level constant

`Rung5(n_amp_qubits: int)` carries k as an instance field, parallel
to how `HEA_Rung2(n_qubits=...)` carries its qubit count. `Rung5`
SHALL NOT have a default `n_amp_qubits` — every site that constructs
a Rung5 must explicitly state its k.

`max_features` is exposed as an `@property` returning
`8 * 2 ** n_amp_qubits` (mirroring `HEA_Rung2.max_features`'s
property shape). This is already the pattern the loader expects:
`_encoding_max_features` in `clustered_dictionary.py` uses
`getattr(encoding, "max_features", None)`, which resolves through
instance lookup for properties. `sae_import.py` reads
`target_encoding.max_features` (instance attribute), which also
resolves through the property. No loader changes required.

**Alternatives considered:**
- *Class-level constant per (Rung5, k)*: would force a `Rung5K2`,
  `Rung5K3`, … hierarchy and either explode the class count or
  abuse a metaclass factory. Rejected — adds zero expressiveness
  vs. a property.
- *Default k=2*: would make `Rung5()` mean "Rung5 with k=2", which
  collides semantically with `Rung4`. Rejected — forcing the caller
  to state k makes the choice explicit and avoids two ways to spell
  the same encoding.

### Decision 2: Per-feature amp knobs become a length-k tuple of pairs

Add `Feature.amp_knobs: tuple[tuple[float, float], ...] = ()`. Rung5
reads exclusively from `amp_knobs`. Rung3 and Rung4 continue to read
their existing `theta_amp` / `psi_aux` / `theta_amp_b` / `psi_amp_b`
fields and ignore `amp_knobs`.

`Dictionary.__post_init__` SHALL validate, when
`isinstance(encoding, Rung5)`, that every feature's `amp_knobs`
length equals `encoding.n_amp_qubits` and that every pair is a
2-tuple of floats. Errors raise `ValueError` with the feature name,
expected length, and actual length.

**Alternatives considered:**
- *Two parallel tuples* `theta_amps: tuple[float, ...]` and
  `psi_amps: tuple[float, ...]`: marginally easier to mutate one
  axis at a time but loses the per-qubit pairing in the data model
  and forces every read site to zip them. Rejected — the pair is
  the conceptual unit (one (θ, ψ) per amp qubit).
- *Extend Rung4's named fields with `theta_amp_c`, `psi_amp_c`, …*:
  doesn't scale. Rejected on its face.
- *Separate `Rung5Feature` subclass*: violates the existing pattern
  where `Feature` is a single concrete dataclass that all encodings
  share. Rejected — the empty-tuple default makes this additive for
  non-Rung5 users.

### Decision 3: Cancellation joint optimiser reads k from the encoding

`Cancellation(encoding="rung5", dictionary=...)` SHALL, in
`__post_init__`, read `k = dictionary.encoding.n_amp_qubits` and
assemble the default-knob list as
`[a.phi, b.phi, b.amp_knobs[0].θ, b.amp_knobs[0].ψ, …,
 b.amp_knobs[k−1].θ, b.amp_knobs[k−1].ψ]` — a `(2 + 2k)`-knob list.

The grid backend's `GRID_KNOB_LIMIT = 4` constraint already excludes
the joint-amp path for Rung3/Rung4 at default knob counts of 4 / 6;
Rung5 continues this — `k=2` gives 6 knobs (over the grid limit),
`k=3` gives 8, etc. Joint optimisation SHALL route to the scipy
backend; the grid backend SHALL raise a clear error if a Rung5
joint solve is requested. Single-knob `.phi`-only cancellations
remain available on the grid backend.

`min_amp_overlap` applies to the k-fold product overlap.
`structural_floor` reduces to the MPS-phase-only floor of the same
(α, β, γ), identical in form to the Rung3/Rung4 case.

**Alternative considered:** parameterising `Cancellation` with an
explicit `k` argument. Rejected — `k` is already encoded in the
dictionary's encoding instance, so passing it again invites drift.

### Decision 4: New `Dictionary.with_knob` grammar for `amp_knobs`

Extend the knob-path grammar with
`<feature_or_cluster_name>.amp_knobs[i].theta` and
`<feature_or_cluster_name>.amp_knobs[i].psi` where `i ∈ [0, k)`.
Out-of-range `i` raises `ValueError` with the encoding's k and the
attempted index.

Rationale: every other per-feature knob has a `with_knob` path; the
cancellation primitive and tuning surface both depend on the
grammar for parameterising solves. Skipping this would force Rung5
callers to construct `Feature` objects by hand.

### Decision 5: Q-OrCA emission generalises Rung4's amp action

The Rung4 emitter writes one amp-branch action accepting 4 amp
knobs (`θ_a, ψ_a, θ_b, ψ_b`). The Rung5 emitter SHALL write one
amp-branch action accepting `2k` amp knobs and produce k
independent single-qubit preparations on q3..q3+k−1. No CNOTs
across amp qubits.

The emitter SHALL read `n_amp_qubits` from the encoding instance
to size the action and the circuit. The MPS block on q0–q2 stays
unchanged.

## Risks / Trade-offs

- **[Cancellation optimiser dimension at large k]** — at k=10, joint
  cancellation lives in a 24-dim parameter space per pair. scipy's
  differential evolution will scale superlinearly. → Mitigation:
  document the recommended k range in the research note; sae-forge
  pareto sweeps that push k high should pre-screen with default-knob
  φ-only cancellation before invoking the full joint solver. No
  explicit cap in code — the encoding doesn't know whether
  cancellation will be applied.

- **[Feature.amp_knobs default is a data-model footgun]** — a Rung5
  feature with the default `amp_knobs = ()` will fail validation in
  `Dictionary.__post_init__`. Existing Rung3/Rung4 features don't
  notice. → Mitigation: validation raises with a clear message
  pointing at the encoding's `n_amp_qubits`; tests assert the error
  text. Add a helper `Feature.with_default_amp_knobs(encoding)` that
  returns a feature with `amp_knobs` padded to the right k with
  zeros (matching the "default reduces to MPSRung1" property).

- **[Naming collision risk: `Rung5` vs. future k-aware Rung*]** — if
  a future encoding wants e.g. a configurable MPS core *and*
  configurable amp width, `Rung5` having `n_amp_qubits` may be
  confusing. → Mitigation: document in the research note that
  `Rung5` specifically generalises the amp-register width with a
  fixed 3-qubit MPS core; future work on MPS-core width is a
  separate encoding (likely `Rung6` or a parameterised
  `RungMPS(n_mps, n_amp)` if the design space justifies it).

- **[Per-feature data layout asymmetry between Rung3/Rung4 and Rung5]**
  — Rung3/Rung4 read named fields; Rung5 reads a tuple. Two
  read-paths inside `Dictionary.gram()` and the cancellation
  primitive. → Mitigation: a thin adapter `_feature_amp_knobs(
  feature, encoding) -> tuple[tuple[float, float], ...]` that
  returns the right tuple shape for any encoding (length 0 for
  MPSRung1/HEA, length 1 for Rung3 — wrapping (θ_amp, ψ_aux),
  length 2 for Rung4 — wrapping the two named pairs, length k for
  Rung5 — passing `amp_knobs` through). Single read-path
  downstream.

  - *Note*: this adapter is internal — it does not change Rung3's or
    Rung4's published Hilbert-dim semantics. Rung3's amp branch is
    *not* a product of single-qubit amps (it's the Bell-pattern
    restricted-subspace amp), so the adapter for Rung3 returns a
    1-element tuple that the Rung3 gram path explicitly understands
    as Rung3 amp knobs; the adapter is shape-only, not semantic
    equivalence.

- **[max_features property vs. ClassVar inconsistency]** — MPSRung1
  and Rung3/Rung4 expose `max_features` as a ClassVar; HEA_Rung2 and
  now Rung5 expose it as an `@property`. Both work for instance
  reads (and both work for `type(encoding).max_features` reads in
  Python's MRO), but ClassVar is faster and more discoverable.
  → Mitigation: today's read sites use `getattr(encoding, …)` or
  `encoding.max_features`, both of which resolve through the
  property. No code changes. A future cleanup pass could unify on
  the property pattern for all encodings; out of scope here.

## Migration Plan

No data migration. The change is additive:

1. `Feature` gains `amp_knobs: tuple[tuple[float, float], ...] = ()`
   with default empty tuple. Existing Rung3/Rung4/MPSRung1
   serialisations round-trip unchanged.
2. `Rung5` class lands behind the existing import surface
   (`polygram.encoding.Rung5`). Nothing dispatches to it until a
   `Dictionary` is constructed with `encoding=Rung5(n_amp_qubits=k)`.
3. Cancellation, Q-OrCA emission, and `with_knob` dispatch each gain
   a Rung5 branch. Non-Rung5 callers see no behaviour change.
4. `sae_import.py` and `clustered_dictionary.py` already read
   `max_features` instance-attribute-style; no change.

Rollback: revert the change. No persisted state references Rung5
until sae-forge starts emitting Rung5 dictionaries.

## Open Questions

- **Bounds on `n_amp_qubits`?** A hard upper bound (e.g. k ≤ 16,
  giving max_features = 524288) keeps the validation honest and
  prevents accidental `Rung5(n_amp_qubits=1_000_000)` calls. Tasks
  should set a defensible cap with a clear error message. Open:
  what value, and whether to expose the cap as a module-level
  constant for sae-forge to read. Lean: `RUNG5_MAX_N_AMP_QUBITS = 16`
  unless there's a sae-forge-driven reason to go higher.

- **Should `examples/rung5_rank_verification.py` sweep k or accept
  a CLI arg?** Both work; the rung4 verification example is
  single-k. Lean: parameterised script that defaults to
  `k ∈ {2, 3, 4}` and accepts `--k <int>` overrides, since sae-forge
  will want to re-run at arbitrary k.

- **Equivalence assertion: does `Rung5(n_amp_qubits=2)` produce the
  same gram as `Rung4` when fed the same knobs?** Mathematically
  yes (both are 2-fold products of `_single_qubit_overlap`). Worth
  a test, but it's not a *requirement* — Rung4 and Rung5 stay as
  separate encoding classes with separate dispatch paths. Open:
  document this equivalence in the research note as an internal
  consistency check, but don't expose it in the user API.
