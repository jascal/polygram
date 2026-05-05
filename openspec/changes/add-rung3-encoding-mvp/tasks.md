# add-rung3-encoding-mvp — tasks

## 1. `Rung3` encoding class

- [ ] 1.1 Add `Rung3` frozen dataclass to `polygram/encoding.py`
      (or extract to subpackage if the file grows past ~300
      lines) with fields named in the spec
      (`alpha, beta, gamma, phi, theta_amp=π/4, psi_aux=0.0`).
- [ ] 1.2 `Rung3.compute_concept_gram(other)` — analytic squared
      overlap. Reuse `MPSRung1.compute_concept_gram` on (α, β,
      γ, φ); add the amplitude branch contribution closed-form.
- [ ] 1.3 `Rung3.from_mps(mps, theta_amp=π/4, psi_aux=0.0)`
      classmethod.
- [ ] 1.4 Default-knob equivalence test: `Rung3.from_mps(mps)`
      .compute_concept_gram(other) matches MPSRung1's value to
      1e-12 absolute, sweeping a 20×20 grid of (φ_a, φ_b) and a
      5×5 grid of (α, β, γ).
- [ ] 1.5 `Dictionary` learns `encoding="rung3"` dispatch — same
      shape as the `"mps"` / `"hea"` paths in
      `polygram/dictionary.py`. Existing paths unchanged.
- [ ] 1.6 Public exports added to `polygram/__init__.py`
      (`Rung3`).

## 2. `Cancellation(encoding="rung3")` joint optimizer

- [ ] 2.1 `Cancellation` accepts `encoding="rung3"`.
      `__post_init__` validates the encoding string against the
      supported set.
- [ ] 2.2 Outer grid helper — 5×5 default; configurable via
      `Cancellation(grid_outer=(M, N))`. Inner reuses the
      existing 2-φ MPSRung1 optimizer.
- [ ] 2.3 Scipy refine over (φ_a, φ_b, θ_amp, ψ_aux) starting
      from the best outer cell. `Nelder-Mead` (the existing
      default for Cancellation's scipy step).
- [ ] 2.4 `CancellationResult` gains `theta_amp_optimum` and
      `psi_aux_optimum` fields. NaN for `mps` / `hea` results.
- [ ] 2.5 Rung3 result's `structural_floor` carries the
      MPS-phase-only floor `M − |V|` of the same (α, β, γ)
      tuple. Internal implementation: construct the equivalent
      MPSRung1 and call its `structural_floor()`.
- [ ] 2.6 Docstring update on `CancellationResult.structural_floor`
      noting the Rung3 semantics ("baseline being broken", not
      "bound").

## 3. Q-Orca emission for Rung3

- [ ] 3.1 `polygram/_qorca_emit.py` learns `_emit_rung3_machine`
      — 5-qubit context, prepared-state declarations,
      `apply_amp_branch` action signatures.
- [ ] 3.2 The emitted `.q.orca.md` parses + verifies clean
      against the shipped q-orca verifier with no new
      verification rules.
- [ ] 3.3 `Dictionary.materialize` dispatches to
      `_emit_rung3_machine` when `encoding="rung3"`.

## 4. Worked example — `examples/rung3_viability_spike.py`

- [ ] 4.1 New script with CLI args mirroring
      `examples/behavioural_validate.py`: `--feature-ids`,
      `--sae-checkpoint`, `--output-dir`, `--n-prompts`,
      `--quiet`. Default `--feature-ids` is the §4.4 selection
      `(12999, 19398, 4192, 23625, 8371, 2287, 68, 13737)`.
- [ ] 4.2 Runs `Cancellation(encoding="mps")` and
      `Cancellation(encoding="rung3")` over all 28 pairs of the
      8-feature panel. Records per-pair `structural_floor`,
      pre-overlap, post-overlap, `cancellation_efficiency`,
      `theta_amp_optimum`, `psi_aux_optimum`.
- [ ] 4.3 Materializes both optimized dictionaries to disk
      (`{output_dir}/baseline/`, `{output_dir}/rung3/`).
- [ ] 4.4 Runs `BehaviouralValidator` on each materialized
      dictionary; records per-pair `gate_pass`, `jaccard`,
      `polygram_overlap`, `n_both_fire`.
- [ ] 4.5 Computes the four criteria (A, B, C, D) per
      `proposal.md`'s table.
- [ ] 4.6 Prints decision banner with bucket
      (`strong_pass | partial_pass | fail`) and the per-criterion
      pass/fail breakdown. Format mirrors `behavioural_validate`'s
      banner.
- [ ] 4.7 Emits `rung3_viability_spike.json` with all per-pair
      numbers + the four criteria + bucket. Schema documented in
      a JSON-schema-style block in the script docstring.
- [ ] 4.8 Skip path: SAE checkpoint or torch absent → exit 0
      with hint message (matches §4.2 / §4.3 / §4.4 / §4.5
      pattern).

## 5. Tests

- [ ] 5.1 `tests/encoding/test_rung3.py` — default-knob
      equivalence (Requirement 1.4); knob-perturbation sanity
      (varying θ_amp moves the gram value smoothly); torch-free
      import check.
- [ ] 5.2 `tests/test_cancellation.py` gains
      `test_cancellation_rung3_smoke` — synthesize a tiny
      dictionary, run `Cancellation(encoding="rung3")`, assert
      result has the new fields populated, `structural_floor`
      matches the MPSRung1 value of the same (α, β, γ).
- [ ] 5.3 `tests/test_cancellation.py` gains
      `test_cancellation_rung3_breaks_floor_synthetic` — a
      hand-crafted pair with known M, V, and a non-trivial
      θ_amp optimum that demonstrably reaches below `M − |V|`.
- [ ] 5.4 `tests/test_qorca_emit.py` (or existing analog) gains
      `test_emit_rung3_machine_verifies` — emit + parse +
      verify clean.
- [ ] 5.5 `tests/test_examples.py` gains
      `test_rung3_viability_spike_smoke` — exit 0 on the
      SAE-or-torch-absent branch.

## 6. Closing

- [ ] 6.1 README "Library tour" gains a one-paragraph entry for
      Rung3 (between encoding section and validator section)
      noting the spike status — "experimental; default remains
      MPS pending §4.5 verdict."
- [ ] 6.2 Update `tech-debt-backlog/tasks.md` §5 with the
      Rung3 spike as the upstream prerequisite for compression
      defaulting (the Rung3 vs MPS verdict feeds compression's
      production encoding choice).
- [ ] 6.3 Run the full test suite end-to-end. CI green.
- [ ] 6.4 Squash-merge to main; archive this change directory
      under `openspec/changes/archive/`.

## 7. Findings PR (separate, follows this change)

> Tracked here for visibility but **not** part of this PR's
> deliverable. After this change merges, run the worked
> example against the real GPT-2-small SAE and write up the
> findings.

- [ ] 7.1 Run `examples/rung3_viability_spike.py` against the
      §4.4 selection on real data.
- [ ] 7.2 Write `docs/research/rung3-viability-spike.md`
      following the §4.2 / §4.3 / §4.4 research-note shape:
      TL;DR table, 5-qubit circuit diagram, per-pair before/
      after, decision bucket, next steps.
- [ ] 7.3 Commit the spike's `rung3_viability_spike.json`
      artifact under `docs/research/data/` for downstream
      reproducibility.
- [ ] 7.4 Open the findings PR. Title: `tech-debt-backlog: §4.5
      — Rung3 viability spike findings`. Body summarizes the
      decision bucket and links the research note.
- [ ] 7.5 If the verdict is **strong-pass**: open a follow-up
      change (`make-rung3-default`) flipping the default
      encoding to Rung3 across `Dictionary` / `Cancellation` /
      `BehaviouralValidator` (validator stays unchanged in
      surface; the dictionaries it consumes change). If
      **partial** or **fail**: skip the follow-up; Rung3 stays
      opt-in.
