## Why

Polygram's two shipped encodings — `MPSRung1` (3-qubit, single-φ
phase knob per feature) and `HEA_Rung2` (depth-1 hardware-efficient
ansatz, single rotation knob) — share a load-bearing limitation:
the per-pair squared overlap that the cancellation primitive can
reach by phase-only optimization is bounded by an analytic floor
`|⟨A|B⟩|² ≥ M − |V|`, where M and V are determined by the encoding
geometry and the dictionary's β / γ assignment. Phase optimization
*navigates between* M − |V| and M + |V|; it cannot push *below* the
floor. For pairs whose floor is high (e.g., M = 0.6, V = 0.3 → floor
0.3), phase-only cancellation tops out well above the Polygram gate
threshold (0.7), and the post-cancellation overlap stays in the
"flagged-as-redundant" range even when the structural collision is
mostly artifactual.

Rung3 introduces a 5-qubit encoding with two new knobs — `θ_amp`
(amplitude) and `ψ_aux` (auxiliary-qubit phase) — that, in
principle, modulate M and V themselves rather than just the
phase-relative position between them. If the modulation is real,
joint optimization over (φ, θ_amp, ψ_aux) breaks below the
phase-only floor and gives the compression pipeline a tighter knob
on which collisions are *structural* (irreducible) vs *parametric*
(removable by encoding choice).

The §4.5 viability spike asks four grounded questions:

- **A. Floor-breaking** — does Rung3's joint optimizer push
  post-cancellation overlap meaningfully below MPS's phase-only
  floor `M − |V|` on a representative pair set?
- **B. Gate discrimination** — at the Polygram gate (≥ 0.7),
  does Rung3 raise the true-positive rate (high-overlap pairs that
  also gate-pass behaviourally) above the §4.4 baseline of 8/12
  ≈ 66%?
- **C. Ranker preservation** — does Rung3 preserve §4.4's Spearman
  ranker signal `+0.637` against Jaccard, or does it destroy the
  ranking? (If destroyed, Rung3 fails as a ranker regardless of
  magnitude.)
- **D. Coverage** — does Rung3's gate continue to catch ≥ 90% of
  pairs with Jaccard ≥ 0.30? If Rung3 hides real redundancies,
  it fails regardless of A/B/C.

This change ships the **infrastructure to answer those four
questions on a real selection** (the §4.4 8-feature panel at
GPT-2-small `blocks.10`), plus the decision rule for what to do
with the answer. It does *not* ship Rung3 as the production default
— that is a follow-up change conditional on the spike outcome.

## What Changes

### `encoding` capability — new `Rung3` class

Add `polygram.encoding.Rung3`, a 5-qubit encoding parallel to
`MPSRung1` and `HEA_Rung2`:

- `Rung3` dataclass carries `(alpha, beta, gamma, phi, theta_amp,
  psi_aux)` per feature. `theta_amp` defaults to `π/4` and
  `psi_aux` to `0.0` so a Rung3 dictionary with default knobs
  reduces to MPSRung1 behaviour for backwards-compatible
  comparison.
- `Rung3.compute_concept_gram(other)` returns the analytic squared
  overlap. The implementation reuses MPSRung1's core on qubits
  0–2 and adds the amplitude branch (qubits 3–4) analytically;
  no simulator round-trip needed.
- `Dictionary` learns to dispatch on `encoding="rung3"` the same
  way it dispatches on `"mps"` and `"hea"`.

### `cancellation` capability — joint amp + aux optimizer

`polygram.cancellation.Cancellation` learns an
`encoding="rung3"` mode whose optimizer jointly searches
`(phi, theta_amp, psi_aux)`:

- Outer grid over `(theta_amp, psi_aux)` at 5×5 (configurable).
- Inner phase optimization per `(theta_amp, psi_aux)` cell uses
  the existing `MPSRung1` 2-φ optimizer.
- Final scipy refinement over all three knobs.
- The result's `structural_floor` field carries the *MPS phase-only
  floor* of the same pair (computed by running an MPS-encoded
  Cancellation alongside) so Rung3's residual is comparable to
  the floor it is trying to break.

### Q-Orca emission for Rung3

`polygram._qorca_emit` learns the 5-qubit Rung3 template. The
emitter produces a `.q.orca.md` machine that parses + verifies
clean against `q-orca`. Verification rule additions are
out-of-scope for this change; the existing rules (unitarity,
Schmidt rank) are sufficient for the 5-qubit shape.

### `examples/rung3_viability_spike.py` — the probe harness

A worked example that:

1. Loads the §4.4 8-feature selection (same `feature_ids` as
   `examples/behavioural_validate.py`).
2. Runs all 28 pairs through both `Cancellation(encoding="mps")`
   and `Cancellation(encoding="rung3")`, recording per-pair
   `structural_floor`, post-cancellation overlap, and
   `cancellation_efficiency`.
3. Materializes both optimized dictionaries to disk (`baseline/`
   and `rung3/`).
4. Runs `BehaviouralValidator` on each (zero validator changes
   needed; it is encoding-agnostic).
5. Computes the four criteria (A, B, C, D) and prints a
   decision-bucket banner.
6. Emits a `rung3_viability_spike.json` with all per-pair numbers
   for the research-note writeup.

### Non-goals for this change

- **Shipping Rung3 as default.** That is a follow-up change
  contingent on the spike outcome. The current default
  (`encoding="mps"`) stays in place after this change merges.
- **Rung3 on Gemma / Pythia / Llama.** The probe runs on GPT-2
  small only — same constraint as §4.2 / §4.3 / §4.4.
- **Depth-4 HEA comparison.** A separate probe will benchmark
  Rung3 against deeper HEA expressivity. Calling that out now
  prevents the conclusion "Rung3 wins vs MPS" being mis-read
  as "Rung3 is the lever."
- **Compression action wired to Rung3.** The compression spec
  (PR #28) is encoding-agnostic; it consumes `ValidationReport`
  regardless of which encoding produced the dictionary. No
  changes to `polygram.compression` are required.
- **Auto-tuning the (θ_amp, ψ_aux) grid.** 5×5 + scipy refine
  is the contract. Future work may explore better initialization
  heuristics, but the spike measures the lever's existence, not
  its tuned ceiling.

## Sequencing relative to PR #28

The compression-action spec stays valid: the `Compressor`
consumes a `ValidationReport`, which is encoding-agnostic. PR
#28 can land independently of this change.

The *implementation* of compression should wait for the §4.5
verdict only if the verdict is **strong-pass** — in which case
Rung3 becomes the production encoding for the validator-then-
compress workflow. If the verdict is **partial-pass** or
**fail**, compression-impl proceeds with MPS as the default
encoding, and Rung3 stays as an optional comparison tool.

## What this proposal explicitly does NOT do

- Modify the `BehaviouralValidator` surface (it is encoding-
  agnostic; passing a Rung3-encoded `Dictionary` works without
  validator changes).
- Modify the `Compressor` surface (encoding-agnostic for the
  same reason).
- Add new optional extras to `pyproject.toml`. Rung3's analytic
  Gram function uses the same numpy + safetensors stack as the
  rest of `polygram`.
- Touch `polygram.analysis.triage` — `cancellation_gap` and the
  pair-ranking surface stay defined against `MPSRung1`. Rung3
  feeds the same triage path through the same `Dictionary`
  abstraction.

## Decision rule (the four criteria, calibrated)

| Criterion | Strong | Partial | Fail |
|:---|:---:|:---:|:---:|
| **A. Floor-breaking** — median `post_rung3 / floor_mps_phase_only` across 28 pairs | ≤ 0.3 | 0.3 < r ≤ 0.7 | > 0.7 |
| **B. Gate true-positive rate** — fraction of Polygram ≥ 0.7 pairs that gate-pass behaviourally | ≥ 80% | 70–80% | < 66% (regression vs §4.4) |
| **C. Ranker preservation** — Spearman(post-cancellation overlap, Jaccard) | ≥ +0.65 | +0.50 ≤ s < +0.65 | < +0.50 (destroyed §4.4 signal) |
| **D. Coverage** — fraction of Jaccard ≥ 0.30 pairs caught by gate | ≥ 90% | 80–90% | < 80% (hides real redundancies) |

**Decision rule (applied in this order):**

1. If **D fails** → pivot regardless of A/B/C. Hiding real
   redundancies is a deal-breaker.
2. If **A fails** → pivot. The floor wasn't broken; Rung3 isn't
   the lever.
3. If **A is strong** AND (**B or C is strong**) → green-light.
   Follow-up change makes Rung3 the production encoding.
4. Otherwise → partial. Rung3 stays opt-in; calibration follow-up
   probes the (θ_amp, ψ_aux) initialization.

The four criteria together are designed so that no single metric
can drive the decision. A: physical lever; B + C: complementary
behavioural signals (B is gate quality, C is rank order); D:
safety floor.
