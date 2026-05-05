# add-rung3-encoding-mvp — tasks

## 1. `Rung3` encoding class

- [x] 1.1 Add `Rung3` frozen dataclass to `polygram/encoding.py`
      (or extract to subpackage if the file grows past ~300
      lines) with fields named in the spec
      (`alpha, beta, gamma, phi, theta_amp=π/4, psi_aux=0.0`).
      Implementation note: `Rung3` is a *config tag* parallel to
      `MPSRung1` / `HEA_Rung2` (carries `bond_dim` only); per-
      feature data lives on `Feature` (which gained `theta_amp`
      and `psi_aux` defaulting to π/4 and 0.0). `Rung3State` is
      the per-feature analytic helper.
- [x] 1.2 `Rung3.compute_concept_gram(other)` — analytic squared
      overlap. Reuse `MPSRung1.compute_concept_gram` on (α, β,
      γ, φ); add the amplitude branch contribution closed-form.
      Implementation: `Dictionary.gram()` dispatches to a
      Rung3 path that builds the MPSRung1-equivalent dictionary,
      computes its gram via the existing q-orca path, and
      multiplies by the analytic complex amp factor matrix
      (`rung3_amp_overlap` per pair). `Rung3State.amp_overlap_squared`
      is the per-feature analytic helper.
- [x] 1.3 `Rung3.from_mps(mps, theta_amp=π/4, psi_aux=0.0)`
      classmethod (provided as `Rung3State.from_mps_knobs`).
- [x] 1.4 Default-knob equivalence test: see
      `tests/encoding/test_rung3.py::TestDefaultKnobEquivalence`
      (covers `test_dictionary_gram_matches_mps_at_default_knobs`
      with random sampling and `test_grid_sweep_phi_equivalence`
      with a 5×5×8 (phi_a, phi_b)×(α, β, γ) sweep).
- [x] 1.5 `Dictionary` learns `encoding=Rung3()` dispatch in
      `polygram/dictionary.py::Dictionary.gram()`.
- [x] 1.6 Public exports added to `polygram/__init__.py`
      (`Rung3`, `Rung3State`).

## 2. `Cancellation(encoding="rung3")` joint optimizer

- [x] 2.1 `Cancellation` accepts `encoding="rung3"` (auto-inferred
      from `dictionary.encoding` when not passed). Validated
      against `SUPPORTED_ENCODINGS = ("mps", "hea", "rung3")`.
- [x] 2.2 Outer grid 5×5 default via `grid_outer=(M, N)`. Inner
      reuses the canonical 2-φ phase grid via
      `_phi_only_grid_search`.
- [x] 2.3 Scipy `Nelder-Mead` refine over
      `(φ_a, φ_b, θ_b, ψ_b)` starting from the best outer cell
      (feature A's amp knobs anchored at the default `(π/4, 0)`).
- [x] 2.4 `CancellationResult.theta_amp_optimum` and
      `psi_aux_optimum` (NaN for `mps`/`hea` results).
- [x] 2.5 Rung3's `structural_floor` carries the MPS-phase-only
      floor `M − |V|` of the same (α, β, γ) — implemented via
      `_mps_equivalent_floor(...)` which constructs the
      MPSRung1-equivalent dictionary and calls its
      `structural_floor()`.
- [x] 2.6 `CancellationResult.structural_floor` docstring
      updated to note the rung-3 semantics ("baseline being
      broken", not "bound").

## 3. Q-Orca emission for Rung3

> **Deferred for v0.** The validator and the §4.5 spike consume
> the live `Dictionary` object directly, not a materialized
> `.q.orca.md`. Q-Orca emission for the 5-qubit Rung3 machine
> stays a follow-up that can land alongside or after the spike's
> findings PR; nothing in the v0 spike pipeline blocks on it.

- [ ] 3.1 *(deferred)* `polygram/_qorca_emit.py` learns
      `_emit_rung3_machine` — 5-qubit context, prepared-state
      declarations, `apply_amp_branch` action signatures.
- [ ] 3.2 *(deferred)* The emitted `.q.orca.md` parses +
      verifies clean against the shipped q-orca verifier with
      no new verification rules.
- [ ] 3.3 *(deferred)* `Dictionary.materialize` dispatches to
      `_emit_rung3_machine` when `encoding="rung3"`. The spike
      script writes a JSON snapshot of the rung-3 master
      dictionary's per-feature knobs in lieu of the machine
      file (`rung3_master_knobs.json`).

## 4. Worked example — `examples/rung3_viability_spike.py`

- [x] 4.1 New script with CLI args (`--feature-ids`,
      `--sae-checkpoint`, `--output-dir`, `--n-prompts`,
      `--quiet`). Default `--feature-ids` is the §4.4 selection.
- [x] 4.2 Runs `Cancellation(encoding="mps")` and
      `Cancellation(encoding="rung3")` over all 28 pairs;
      records per-pair `structural_floor`, pre/post-overlap,
      `cancellation_efficiency`, `theta_amp_optimum`,
      `psi_aux_optimum`, `rung3_residual_ratio`.
- [x] 4.3 Materializes the baseline (MPS) optimized dictionary
      to `{output_dir}/baseline/` as a verifiable `.q.orca.md`
      and the Rung3 optimized dictionary as
      `{output_dir}/rung3/rung3_master_knobs.json` (q-orca
      emission for Rung3 is §3, deferred).
- [x] 4.4 Runs `BehaviouralValidator` on each master dictionary;
      records per-pair `gate_pass`, `jaccard`,
      `polygram_overlap`, `n_both_fire`.
- [x] 4.5 Computes the four criteria (A, B, C, D) per
      `proposal.md`'s calibrated table.
- [x] 4.6 Prints decision banner with bucket
      (`strong_pass | partial_pass | fail`) and per-criterion
      breakdown.
- [x] 4.7 Emits `rung3_viability_spike.json`; schema in the
      script docstring.
- [x] 4.8 Skip path: SAE checkpoint or torch absent → exit 0
      with hint message.

## 5. Tests

- [x] 5.1 `tests/encoding/test_rung3.py` — default-knob
      equivalence (5×5×8 sweep), knob-perturbation sanity (θ
      and ψ smoothness, ψ=π zero-amp test), torch-free import
      check.
- [x] 5.2 `tests/test_cancellation.py::TestRung3Cancellation::
      test_cancellation_rung3_smoke`.
- [x] 5.3 `tests/test_cancellation.py::TestRung3Cancellation::
      test_cancellation_rung3_breaks_floor_synthetic`.
- [ ] 5.4 *(deferred — q-orca emission is §3, deferred.)*
      `test_emit_rung3_machine_verifies`.
- [x] 5.5 `tests/test_examples.py::test_rung3_viability_spike_smoke`
      — exit 0 on the SAE-or-torch-absent branch.

## 6. Closing

- [x] 6.1 README "Rung3 encoding (experimental)" section landed
      between SAE-import and validator sections.
- [x] 6.2 `tech-debt-backlog/tasks.md` §5.2 lists the Rung3
      spike as the upstream prerequisite for compression's
      production encoding choice.
- [x] 6.3 Full test suite green (317 passed, 0 failed). Ruff
      clean.
- [x] 6.4 Squash-merge to main; archive this change directory
      under `openspec/changes/archive/`.

## 7. Findings PR (separate, follows this change)

> Tracked here for visibility but **not** part of this PR's
> deliverable. After this change merges, run the worked
> example against the real GPT-2-small SAE and write up the
> findings.

- [x] 7.1 Run `examples/rung3_viability_spike.py` against the
      §4.4 selection on real data.
- [x] 7.2 Write `docs/research/rung3-viability-spike.md`
      following the §4.2 / §4.3 / §4.4 research-note shape:
      TL;DR table, 5-qubit circuit diagram, per-pair before/
      after, decision bucket, next steps.
- [x] 7.3 Commit the spike's `rung3_viability_spike.json`
      artifact under `docs/research/data/` for downstream
      reproducibility.
- [ ] 7.4 Open the findings PR. Title: `tech-debt-backlog: §4.5
      — Rung3 viability spike findings`. Body summarizes the
      decision bucket and links the research note.
- [ ] 7.5 Verdict was `strong_pass` per the calibrated rule, but
      the optimum is the trivial amp-zeroing solution
      `(θ_b = π/4, ψ_b = π)` against the anchored A defaults — see
      `docs/research/rung3-viability-spike.md` "Caveats". Hold
      `make-rung3-default` until either (a) a non-degenerate-amp
      constraint is added to the joint optimizer and the spike re-
      run, or (b) symmetric anchoring (6-knob search) is tested.
      Rung3 stays opt-in until then.
