## Why

A user-supplied design sketch proposed a `suggest_safe_knobs()` /
`suggest_phi_candidates()` heuristic helper that returns a curated
list of "safe" knob paths (e.g. "last-layer Rz", "middle-layer Rz",
"cross-layer shared φ"). The recently-archived
`add-cluster-shared-knobs` change explicitly rejected that helper —
it is the *third* bullet in that proposal's Out-of-Scope section,
verbatim:

> **`suggest_safe_knobs()` heuristic helper.** An external task spec
> proposed a helper that returns "safe" path lists. Empirically there
> is no defensible per-feature safe choice on HEA (Rz layer-0 has
> zero leverage on `|0⟩` initial states; Ry knobs are exactly the
> cluster-shatterer). This proposal supersedes that idea: the
> principled answer is a binding mechanic, not a curated path list.

The user's intuition behind the helper was real, though: there *are*
empirical findings about which knobs have leverage and which carry
hazard, and they are scattered across multiple archived proposals
and research notes. A reader asking "OK then how *do* I choose
knobs?" today has to grep four archives. That is the friction the
helper was reaching for.

This change replaces the rejected helper with a *documentation*
intervention: a "Choosing knobs" section inside the existing
`render_report` output that names the empirical findings, points at
the principled grammar (`<feature>.phi`, `<cluster>.phi`,
`<feature>.theta[r,d,q]`, `<cluster>.theta[r,d,q]`), and tells the
reader the order in which to reach for them. No new API, no curated
list, no callable helper. The text lives in one place
(`polygram/analysis/triage.py`) and the rendered report quotes it.

## What Changes

### `analysis` capability — render_report adds a "Choosing knobs" section

`render_report` SHALL emit a new `## Choosing knobs` section between
the existing `## Per-feature sensitivity` and `## Encoding
suitability` sections. The section's content SHALL be a fixed
documentation paragraph synthesizing the empirical findings already
captured in `docs/research/cancellation-phase-floor.md` and
`openspec/changes/archive/2026-05-04-add-cluster-shared-knobs/`,
namely:

1. **Default starting knob.** A single `<feature>.phi` (or
   `<target_a>.phi` and `<target_b>.phi` for the default
   `Cancellation` configuration) on either encoding. This is the
   "last-layer Rz on MPS" the sketch named — already the default
   knob; no new helper required. On HEA it is also the cleanest
   single axis because the final `Rz(qs[1], phi)` factors out
   regardless of θ.

2. **Multi-feature binding.** When multiple features should be
   tuned coherently within a cluster, prefer the cluster-shared
   path (`<cluster>.phi` or `<cluster>.theta[r,d,q]`) over a list
   of per-feature paths. Bit-for-bit Gram preservation holds for
   `MPSRung1 <cluster>.phi` (final-Rz factorization); HEA
   cluster-shared paths ship as a search-space dimensionality
   reduction (one axis per cluster). Per-feature θ on diverse-
   sibling HEA fixtures is the *cluster-shatterer* — the empirical
   experiment archived under `add-cluster-shared-knobs` showed a
   4-θ Ry knob set drove `(dog_poodle, bird_hawk)` to ≈0 while
   inverting `(dog_poodle, dog_beagle)` from `0.9999 → 0.5735`.

3. **HEA Pauli leverage.** Rz at depth 0 has *zero* leverage on
   `|0⟩` initial states (`Rz |0⟩` is a global phase). Rz at later
   depths takes effect only after entanglers rotate states off the
   Z basis; leverage is therefore depth- and entangler-dependent
   rather than uniform. Ry has across-the-board leverage but is
   the cluster-shatterer above. The "middle-layer Rz" intuition
   from the user sketch is *partly* right — there is leverage
   there — but it is not principled enough to recommend by default.

4. **When pure-phase search hits the floor.** `Cancellation` is a
   *constraint solver* over the M+V·cos(δ) decomposition, not a
   universal overlap minimizer. The structural floor at `M − |V|`
   cannot be pierced by any φ tuning; β/α/γ adjustment or a
   richer encoding is needed. See
   `docs/research/cancellation-phase-floor.md` and the deferred
   disentanglement direction in
   `docs/research/spec-disentanglement-loop.md`.

5. **Sensitivity ranking is two lines.** The proposed
   `prefer_high_sensitivity=True` flag reduces to:

   ```python
   top = sorted(prediction.feature_sensitivity.items(),
                key=lambda kv: kv[1], reverse=True)[:n]
   ```

   No helper warranted; the per-feature sensitivity table emitted
   in the section above already gives the reader the same
   information sorted.

The section SHALL be a fixed string (no per-prediction
parameterization beyond what the surrounding report already
contains). The intent is documentation that travels with every
analysis report, not a per-call recommendation engine.

### Module-level constant

The section text SHALL be exposed as a module-level string constant
`KNOB_SELECTION_GUIDANCE` so consumers (notebooks, CLI invocations
that don't render the full report) can quote it directly without
reaching into the renderer.

## Capabilities

### Modified Capabilities

- `analysis` — `render_report` emits a `## Choosing knobs` section;
  `KNOB_SELECTION_GUIDANCE` exposed as a module-level constant.

### New Capabilities

*(none — additive on existing `analysis` capability)*

## Out of Scope

- **`suggest_safe_knobs` / `suggest_phi_candidates` / any callable
  helper that returns a curated knob list.** Already rejected in
  `add-cluster-shared-knobs`; this change documents *why*, not what.
- **Per-prediction recommendation logic** ("you should run with
  these specific knobs given your input"). The empirical findings
  are about the encoding, not the dictionary state. Variation by
  fixture would require empirical evidence we don't have.
- **Code changes to `Cancellation` or `Dictionary`.** The
  recommendation is documentation-only; the existing knob grammar
  already supports every path the section recommends.

## Impact

- `polygram/analysis/triage.py` — `KNOB_SELECTION_GUIDANCE`
  module-level constant; `render_report` emits the new section
  (~30 LOC, mostly text).
- `polygram/analysis/__init__.py` — re-export
  `KNOB_SELECTION_GUIDANCE`.
- `tests/test_analysis.py` — extend `test_render_report_*` to
  assert the new heading and a stable substring from the guidance
  text are present.
- No q-orca dependency change; no other module changes.
