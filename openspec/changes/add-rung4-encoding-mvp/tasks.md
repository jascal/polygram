## 1. `Rung4` encoding class

- [x] 1.1 Add `Rung4` frozen dataclass to `polygram/encoding.py`, parallel to `Rung3`. Single field `bond_dim: int = 2` with the same `__post_init__` validation. Class-level `max_features: ClassVar[int] = 32` (consumes the `per-encoding-feature-cap` abstraction).
- [x] 1.2 Extract `_single_qubit_overlap(theta_a, psi_a, theta_b, psi_b) -> complex` helper. The body is the existing `rung3_amp_overlap` math, no behaviour change.
- [x] 1.3 Refactor `rung3_amp_overlap` to call `_single_qubit_overlap` with the existing args. Regression: byte-identical numerical output on a fixed-seed Rung3 fixture (test via existing test suite — should produce no diffs).
- [x] 1.4 Add `rung4_amp_overlap(theta_a3, psi_a3, theta_a4, psi_a4, theta_b3, psi_b3, theta_b4, psi_b4) -> complex` returning the product of two `_single_qubit_overlap` calls (q3 amp and q4 amp).
- [x] 1.5 Add `rung4_amp_overlap_squared(...)` returning `abs(rung4_amp_overlap(...)) ** 2`. Verify against the Rung3 amp-squared math on the slice where the q4 amp knobs are both at default.
- [x] 1.6 Add `Rung4State` frozen dataclass parallel to `Rung3State`. Fields: alpha, beta, gamma, phi, theta_amp, psi_aux (q3 amp), theta_amp_b, psi_amp_b (q4 amp). Methods: `amp_overlap_squared(other)`, `from_mps_knobs(alpha, beta, gamma, phi, *, theta_amp=0, psi_aux=0, theta_amp_b=0, psi_amp_b=0)`.
- [x] 1.7 Unit tests: default-knob `Rung4State.amp_overlap_squared(other)` equals 1 when both features hold defaults; otherwise equals the product of two single-qubit squared overlaps.

## 2. `Feature` extension

- [x] 2.1 Add `theta_amp_b: float = 0.0` and `psi_amp_b: float = 0.0` to `Feature` in `polygram/dictionary.py`. Default values chosen so Rung3 grams are unaffected (Rung3 doesn't read them).
- [x] 2.2 Regression: existing `tests/test_dictionary.py` `Feature.__eq__` / `Feature.__hash__` / serialisation cases continue to pass with the new fields defaulting.
- [x] 2.3 SAE-import JSON round-trip: import a Rung3 dictionary saved before this change, confirm the new fields reconstruct at 0.0.

## 3. `Dictionary.gram()` Rung4 dispatch

- [x] 3.1 In `polygram/dictionary.py:294` add an `isinstance(self.encoding, Rung4)` branch above the Rung3 branch (or share with Rung3 via a common `_amp_branch_gram` helper). Same elementwise-product factorisation; calls `rung4_amp_overlap`.
- [x] 3.2 Unit test: 4-feature Rung4 dictionary with default amp knobs produces a gram bit-identical to its MPSRung1 equivalent on the same (α, β, γ, φ).
- [x] 3.3 Unit test: 4-feature Rung4 dictionary with non-default amp knobs produces a gram different from the MPSRung1 equivalent in the expected directions (amp factors apply per pair).

## 4. Rank verification

- [x] 4.1 Add `examples/rung4_rank_verification.py` (small fixture script paralleling `examples/rung3_rank_probe.py`). Default sweep N ∈ {4, 8, 16, 24, 32, 40} on Rung4 across two seeds.
- [x] 4.2 Run and confirm saturation at N=32 (singular value at index 32 below 1e-12 relative). Commit `docs/research/data/rung4_rank_verification.json`.
- [x] 4.3 Add to `tests/test_examples.py` smoke list with a fast-mode flag (small sizes).

## 5. Cancellation extension

- [x] 5.1 Add `"rung4"` to `SUPPORTED_ENCODINGS` in `polygram/cancellation.py`.
- [x] 5.2 Extend `_infer_encoding_string` to return `"rung4"` for `Rung4` instances.
- [x] 5.3 In `Cancellation.__post_init__`, define the Rung4 canonical knob list `[a.phi, b.phi, b.theta_amp, b.psi_aux, b.theta_amp_b, b.psi_amp_b]` (6 knobs). Reject non-canonical lists with a clear error, matching the Rung3 stance.
- [x] 5.4 Refactor `_run_rung3_joint` into a generalised `_run_amp_joint(amp_knob_count: int)` parameterised by the number of amp knobs on feature B. Rung3 → 2 amp knobs; Rung4 → 4 amp knobs.
- [x] 5.5 In `run()`, dispatch `encoding == "rung4"` to `_run_amp_joint(amp_knob_count=4)`.
- [x] 5.6 `min_amp_overlap` constraint: apply against `rung4_amp_overlap_squared` (computed from feature A's amp knobs and feature B's amp knobs). Same semantics as Rung3 — rejects amp configurations whose overlap factor falls below the threshold.
- [x] 5.7 `structural_floor()` for Rung4 reduces to the MPS-phase-only floor of (α, β, γ) — same pattern as Rung3 via `_mps_equivalent_floor`.
- [x] 5.8 Determine sensible default for `grid_outer` on Rung4 (4D outer grid is `grid_outer**4` cells). Default to `(3, 3, 3, 3)` = 81 cells to keep wall-clock comparable to Rung3's 25-cell default; expose for tuning via `CancellationConfig`.
- [x] 5.9 Unit tests: Rung4 cancellation on a 2-feature toy dictionary lowers the target-pair overlap; `tolerance_met` reports correctly; `cancellation_efficiency` is non-None.
- [x] 5.10 Plot dispatch: `kind="grid"` on Rung4 raises `NotImplementedError` (6D > 2D); `kind="before_after"` works.

## 6. Q-OrCA emission

- [x] 6.1 Extend `polygram/_qorca_emit.py` to handle `Rung4` dictionaries. Reuse the Rung3 MPS preparation on q0–q2 unchanged; replace the Rung3 Bell-pattern amp branch on q3–q4 with two independent single-qubit preparations (`Ry(q3, θ_a)`, `Rz(q3, ψ_a)`, `Ry(q4, θ_b)`, `Rz(q4, ψ_b)`).
- [x] 6.2 Action signature: amp preparation takes 4 knobs per feature (vs Rung3's 2). Action label can be reused (`prepare_concept`) since shape is encoded via the encoding tag in the `.q.orca.md` `## encoding` section.
- [x] 6.3 Emitter parses the new amp knob columns from `Feature`'s `theta_amp_b` / `psi_amp_b` and writes them into the per-feature `## theta` (or equivalent) table.
- [x] 6.4 Round-trip test: 2-feature Rung4 dictionary emits a `.q.orca.md`, parses back via q-orca, computes gram via the q-orca path. Confirm equality to the analytic `Dictionary.gram()` to 1e-10.
- [x] 6.5 Document the Rung4 amp-branch circuit shape in a comment block referencing the dimensional analysis from `docs/research/rung3-rank-bound.md`.

## 7. Worked example — `examples/rung4_viability_spike.py` (deferred to follow-up PR)

§7 is **deferred** out of this implementation PR. The viability
spike requires running on a real GPT-2-small SAE with the
`[behavioural]` extra installed (torch + transformers) — that's
heavier ground than the encoding+cancellation+emit work this PR
already covers. The spike methodology is preserved exactly (mirror
`examples/rung3_viability_spike.py`'s four-criterion A/B/C/D
bucket); landing it cleanly belongs in a research-track follow-up
PR alongside the resulting `docs/research/rung4-viability-spike.md`
note.

- [x] 7.1 Mirror `examples/rung3_viability_spike.py` structure → shipped as `examples/rung4_viability_spike.py`.
- [x] 7.2 Two output JSONs → `docs/research/data/rung4_viability_spike.json` (unconstrained) + `rung4_viability_spike_constrained.json` (ε = 0.5).
- [x] 7.3 Define the decision rule → `make-rung4-default` is dead; Rung4 stays opt-in. See `docs/research/rung4-viability-spike.md` §Decision.
- [ ] 7.4 ~~Plot output~~ — skipped; the result is a single constant (median residual = ε at constraint boundary), no plot adds information.

## 8. Tests

- [x] 8.1 `tests/test_encoding.py::test_rung4_state_defaults_reduce_to_mps`.
- [x] 8.2 `tests/test_encoding.py::test_rung4_product_amp_overlap_factorisation`.
- [x] 8.3 `tests/test_dictionary.py::test_rung4_gram_shape_and_invariants`.
- [x] 8.4 `tests/test_cancellation.py::test_rung4_canonical_knob_list_required`.
- [x] 8.5 `tests/test_cancellation.py::test_rung4_joint_optimizer_lowers_overlap`.
- [x] 8.6 `tests/test_cancellation.py::test_rung4_min_amp_overlap_constraint`.
- [x] 8.7 `tests/test_qorca_emit.py::test_rung4_emit_round_trip`.
- [x] 8.8 `tests/test_examples.py::test_rung4_rank_verification_smoke`.
- [x] 8.9 `tests/test_examples.py::test_rung4_viability_spike_smoke`.

## 9. Closing

- [x] 9.1 Run `pytest` full suite; verify no regressions.
- [x] 9.2 Run `openspec validate add-rung4-encoding-mvp --strict`.
- [x] 9.3 Update `CHANGELOG.md`: new `Rung4` encoding, 32-feature cap, opt-in by default. Reference `docs/research/rung3-rank-bound.md` and the new `docs/research/rung4-viability-spike.md`.
- [x] 9.4 Update `README.md`: Rung4 mentioned in the encoding line-up; capacity-limits section reflects the new 32 option.

## 10. Findings PR (follows this change)

- [x] 10.1 Run `examples/rung4_viability_spike.py` — both runs landed (analytic phase only; behavioural phase deferred pending torch-enabled host).
- [x] 10.2 Write `docs/research/rung4-viability-spike.md` — A/B/C/D table mirrors Rung3 note. B/C/D rows show `skipped` (behavioural phase pending); A alone is sufficient for the verdict. Final decision: Rung4 stays opt-in.
- [x] 10.3 Commit the spike's JSON artifacts under `docs/research/data/`.
- [ ] 10.4 Open a findings PR (in progress).
