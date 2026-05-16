## 1. `Rung5` encoding class

- [ ] 1.1 Add `Rung5` frozen dataclass to `polygram/encoding.py`, parallel to `Rung4`. Fields: `bond_dim: int = 2`, `n_amp_qubits: int` (no default). `__post_init__` validates `bond_dim == 2` (same error message shape as `Rung4`) and `1 <= n_amp_qubits <= RUNG5_MAX_N_AMP_QUBITS`.
- [ ] 1.2 Add module-level `RUNG5_MAX_N_AMP_QUBITS: int = 16` constant in `polygram/encoding.py` with a docstring naming the cap and the resulting `max_features = 8 · 2^16 = 524288`.
- [ ] 1.3 Expose `Rung5.max_features` as an `@property` returning `8 * 2 ** self.n_amp_qubits` (mirrors `HEA_Rung2.max_features`).
- [ ] 1.4 Re-export `Rung5` and `RUNG5_MAX_N_AMP_QUBITS` from `polygram/__init__.py`.
- [ ] 1.5 Add `rung5_amp_overlap(amp_a, amp_b) -> complex` where each input is `tuple[tuple[float, float], ...]`. Returns the product of `_single_qubit_overlap` calls — one per amp-qubit index. Raises `ValueError` on length mismatch with a message naming both lengths.
- [ ] 1.6 Add `rung5_amp_overlap_squared(amp_a, amp_b) -> float` returning `abs(rung5_amp_overlap(amp_a, amp_b)) ** 2`.
- [ ] 1.7 Add `Rung5State` frozen dataclass parallel to `Rung4State`. Fields: `alpha`, `beta`, `gamma`, `phi`, `amp_knobs: tuple[tuple[float, float], ...]`. Methods: `amp_overlap_squared(other)`, `from_mps_knobs(alpha, beta, gamma, phi, *, amp_knobs=())`.
- [ ] 1.8 Unit tests: `Rung5(n_amp_qubits=k)` for k ∈ {1, 2, 3, 4, 5, 16} produces the expected `max_features`; k ∈ {0, -1, 17} raises `ValueError` with messages matching the spec.

## 2. `Feature.amp_knobs` extension

- [ ] 2.1 Add `amp_knobs: tuple[tuple[float, float], ...] = ()` to `Feature` in `polygram/dictionary.py`. Coerce list inputs to tuples in `__post_init__` for ergonomics; reject non-2-tuples with `ValueError`.
- [ ] 2.2 `Feature.with_default_amp_knobs(encoding) -> Feature` returns a copy with `amp_knobs` padded to `((0.0, 0.0),) * encoding.n_amp_qubits` when `isinstance(encoding, Rung5)`, and returns `self` unchanged otherwise.
- [ ] 2.3 Persist `amp_knobs` in `Dictionary` serialisation only when non-empty (omit the field on serialise when `()`). Confirms Rung3/Rung4 round-trip is byte-identical.
- [ ] 2.4 Regression: existing `tests/test_dictionary.py` `Feature.__eq__` / `Feature.__hash__` / serialisation cases continue to pass with the new field defaulting to `()`.
- [ ] 2.5 SAE-import JSON round-trip: import a Rung4 dictionary saved before this change, confirm the new `amp_knobs` field reconstructs at `()` and `Dictionary.gram()` matches pre-change.

## 3. `Dictionary` Rung5 dispatch and validation

- [ ] 3.1 In `Dictionary.__post_init__`, when `isinstance(self.encoding, Rung5)`, validate every feature's `amp_knobs` length equals `self.encoding.n_amp_qubits` and each entry is a 2-tuple of floats. Raise `ValueError` naming the feature, expected length, actual length.
- [ ] 3.2 In `Dictionary.gram()`, add an `isinstance(self.encoding, Rung5)` branch above the Rung4 branch. Same elementwise-product factorisation; calls `rung5_amp_overlap` on each pair's `amp_knobs`.
- [ ] 3.3 Internal helper `_feature_amp_knobs(feature, encoding) -> tuple[tuple[float, float], ...]`: returns `()` for MPSRung1/HEA, `((feature.theta_amp, feature.psi_aux),)` for Rung3, `((feature.theta_amp, feature.psi_aux), (feature.theta_amp_b, feature.psi_amp_b))` for Rung4, `feature.amp_knobs` for Rung5. Shape-only — does not change Rung3 semantics (which still uses `rung3_amp_overlap`).
- [ ] 3.4 Extend `Dictionary.with_knob` path grammar (`_parse_knob_path` in `polygram/dictionary.py`) to accept `<name>.amp_knobs[i].theta` and `<name>.amp_knobs[i].psi`. Reject on non-Rung5 encodings with `ValueError`. Validate `0 <= i < encoding.n_amp_qubits`.
- [ ] 3.5 `with_knob`: when the target feature's `amp_knobs == ()`, materialise the default-padded tuple via `Feature.with_default_amp_knobs(encoding)` before setting the slot.
- [ ] 3.6 Cluster-shared variants `<cluster>.amp_knobs[i].theta` and `<cluster>.amp_knobs[i].psi` work identically to the per-feature variant: set the same `(i, axis)` slot on every feature in `dictionary.hierarchy[cluster]`.
- [ ] 3.7 Unit test: 4-feature Rung5 dictionary at k=3 with all-zero `amp_knobs` produces a gram bit-identical to the MPSRung1 equivalent on the same (α, β, γ, φ).
- [ ] 3.8 Unit test: 4-feature Rung5 dictionary at k=3 with non-zero `amp_knobs` produces a gram different from MPSRung1 in the expected directions (per-pair amp factors apply).
- [ ] 3.9 Unit test: `Rung5(n_amp_qubits=2)` produces grams numerically equal to `Rung4` when fed identical (α, β, γ, φ, θ, ψ pairs) — internal consistency check, documented in `docs/research/rung5-encoding.md` (not a published API equivalence).

## 4. Rank verification

- [ ] 4.1 Add `examples/rung5_rank_verification.py` parameterised over k. Default sweep `k ∈ {2, 3, 4}`; accept `--k <int>` override. For each k, sweep `N ∈ {8·2^k, 2·8·2^k}` across two seeds.
- [ ] 4.2 Run and confirm saturation at `N = 8 · 2^k` for each k (singular value at index `8 · 2^k` below 1e-12 relative). Commit `docs/research/data/rung5_rank_verification.json` keyed by k.
- [ ] 4.3 Add to `tests/test_examples.py` smoke list with a fast-mode flag (k ∈ {2, 3} at small N).

## 5. Cancellation extension

- [ ] 5.1 Add `"rung5"` to `SUPPORTED_ENCODINGS` in `polygram/cancellation.py`.
- [ ] 5.2 Extend `_infer_encoding_string` to return `"rung5"` for `Rung5` instances.
- [ ] 5.3 In `Cancellation.__post_init__`, read `k = dictionary.encoding.n_amp_qubits` and assemble the canonical knob list `[a.phi, b.phi, b.amp_knobs[0].theta, b.amp_knobs[0].psi, …, b.amp_knobs[k-1].theta, b.amp_knobs[k-1].psi]`. Reject non-canonical lists with a clear error, matching the Rung3/Rung4 stance.
- [ ] 5.4 Generalise the existing `_run_amp_joint` (added in the Rung4 change) to accept arbitrary amp-knob count, including reading `k` from the encoding when not already plumbed.
- [ ] 5.5 In `run()`, dispatch `encoding == "rung5"` to `_run_amp_joint(amp_knob_count=2*k)` with `k` read from the dictionary's encoding.
- [ ] 5.6 `min_amp_overlap` constraint: apply against `rung5_amp_overlap_squared(amp_a, amp_b)` using both features' `amp_knobs`.
- [ ] 5.7 `structural_floor()` for Rung5 reduces to the MPS-phase-only floor of (α, β, γ) — reuse `_mps_equivalent_floor`. Test that the floor is independent of k.
- [ ] 5.8 Grid backend: raise `ValueError` with grid-knob-limit message + scipy-backend recommendation when the resolved Rung5 default-knob list exceeds `GRID_KNOB_LIMIT`. φ-only knob lists on Rung5 dictionaries continue to route through the grid backend normally.
- [ ] 5.9 Reject explicit `n_amp_qubits` / `k` kwargs on `Cancellation` (no new field — rely on Python `TypeError` from the unchanged signature).
- [ ] 5.10 Unit tests: Rung5 cancellation at k=2 and k=3 on toy dictionaries lowers the target-pair overlap; `tolerance_met` reports correctly; `cancellation_efficiency` is non-None; grid backend rejection works for default knob list at k ≥ 2.

## 6. Q-OrCA emission

- [ ] 6.1 Extend `polygram/_qorca_emit.py` to handle `Rung5` dictionaries. Reuse the MPS preparation on q0–q2 unchanged; emit `k` independent single-qubit preparations on qubits q3..q3+k−1.
- [ ] 6.2 Action signature: amp preparation takes `2k` knobs per feature, in interleaved `(θ_0, ψ_0, θ_1, ψ_1, …)` order. Action label `prepare_concept` reused.
- [ ] 6.3 Emitter parses the new `amp_knobs` field from `Feature` and writes the `(θ_i, ψ_i)` pairs into the per-feature `## amp branch` (or equivalent) table.
- [ ] 6.4 Round-trip test: 2-feature Rung5 dictionaries at k ∈ {2, 3, 4} each emit a `.q.orca.md`, parse back via q-orca, compute gram via the q-orca path. Confirm equality to the analytic `Dictionary.gram()` to 1e-10.
- [ ] 6.5 Sanity assertion: emitted `.q.orca.md` SHALL contain no two-qubit gates between any pair of amp qubits q3..q3+k−1.
- [ ] 6.6 Document the Rung5 amp-branch circuit shape in a comment block referencing the dimensional analysis from `docs/research/rung3-rank-bound.md` and the Rung4 precedent.

## 7. Research note

- [ ] 7.1 Add `docs/research/rung5-encoding.md` documenting: the generalisation from Rung4's fixed k=2 to configurable k; the per-feature Hilbert dim formula; the design choice to fix k at construction time (deferring per-feature variable k to a future change if sae-forge ever demands it); the link to the sae-forge pareto-sweep use case; the empirical rank-verification protocol and observed saturation at each k.
- [ ] 7.2 Include a short section flagging that the cancellation joint optimiser dimension grows as `2 + 2k` and that sae-forge sweeps pushing k high should pre-screen with φ-only cancellation before invoking the full joint solver.
- [ ] 7.3 Commit `docs/research/data/rung5_rank_verification.json` as the empirical-bound artifact.

## 8. Tests

- [ ] 8.1 `tests/test_encoding.py::test_rung5_state_defaults_reduce_to_mps` — k ∈ {1, 2, 3, 4}.
- [ ] 8.2 `tests/test_encoding.py::test_rung5_amp_overlap_kfold_factorisation`.
- [ ] 8.3 `tests/test_encoding.py::test_rung5_amp_overlap_length_mismatch_rejected`.
- [ ] 8.4 `tests/test_encoding.py::test_rung5_max_features_scales_with_k`.
- [ ] 8.5 `tests/test_encoding.py::test_rung5_n_amp_qubits_bounds`.
- [ ] 8.6 `tests/test_dictionary.py::test_rung5_gram_shape_and_invariants` — symmetric, PSD, unit diagonal at k ∈ {2, 3}.
- [ ] 8.7 `tests/test_dictionary.py::test_rung5_amp_knobs_length_validation`.
- [ ] 8.8 `tests/test_dictionary.py::test_rung5_with_knob_amp_knobs_path` — including out-of-range index rejection, non-Rung5 encoding rejection, and default-padding on first set.
- [ ] 8.9 `tests/test_dictionary.py::test_rung5_cluster_shared_amp_knobs_with_knob`.
- [ ] 8.10 `tests/test_dictionary.py::test_rung5_matches_rung4_at_k2` — internal consistency check confirming Rung5(k=2) and Rung4 produce numerically equal grams on the same knobs.
- [ ] 8.11 `tests/test_dictionary.py::test_feature_amp_knobs_default_round_trip` — existing Rung3/Rung4/MPSRung1 dictionaries round-trip with `amp_knobs == ()`.
- [ ] 8.12 `tests/test_cancellation.py::test_rung5_canonical_knob_list_required`.
- [ ] 8.13 `tests/test_cancellation.py::test_rung5_joint_optimizer_lowers_overlap` — at k=2 and k=3.
- [ ] 8.14 `tests/test_cancellation.py::test_rung5_min_amp_overlap_constraint`.
- [ ] 8.15 `tests/test_cancellation.py::test_rung5_structural_floor_independent_of_k`.
- [ ] 8.16 `tests/test_cancellation.py::test_rung5_grid_backend_rejects_default_knob_list`.
- [ ] 8.17 `tests/test_qorca_emit.py::test_rung5_emit_round_trip` — at k ∈ {2, 3, 4}.
- [ ] 8.18 `tests/test_qorca_emit.py::test_rung5_emit_no_amp_entanglement`.
- [ ] 8.19 `tests/test_examples.py::test_rung5_rank_verification_smoke`.

## 9. Closing

- [ ] 9.1 Bump version in `pyproject.toml` (suggest `0.7.0` — new opt-in encoding, additive `Feature.amp_knobs` field; no behaviour change for existing encodings).
- [ ] 9.2 Update `polygram/__init__.py` `__all__` to include `Rung5`, `Rung5State`, `rung5_amp_overlap`, `rung5_amp_overlap_squared`, `RUNG5_MAX_N_AMP_QUBITS`.
- [ ] 9.3 Run the full test suite and confirm no regressions in MPSRung1/HEA/Rung3/Rung4 paths.
- [ ] 9.4 Manual smoke: build a `Rung5(n_amp_qubits=3)` dictionary from a toy fixture, compute gram, run a φ-only cancellation solve, emit q-orca, confirm round-trip.
- [ ] 9.5 Mark the change ready for archive once tests pass and sae-forge has confirmed the configurable-k interface meets its pareto-sweep needs (sae-forge integration is out of scope here).
