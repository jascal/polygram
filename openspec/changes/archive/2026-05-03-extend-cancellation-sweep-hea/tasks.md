# extend-cancellation-sweep-hea — tasks

## 1. Dictionary knob plumbing

- [x] 1.1 `polygram/dictionary.py` — `Dictionary.with_knob(path, value)`
      method. Path syntax: `<feature>.phi` (both encodings) or
      `<feature>.theta[r,d,q]` (HEA only). Returns a new `Dictionary`
      (uses `replace` like `with_phi`) with the named slot set.
- [x] 1.2 `<feature>.phi` on `MPSRung1` updates `Feature.phi`
      identically to `with_phi`. On `HEA_Rung2`, sets `Feature.phi`
      (the `_default_hea_theta` synthesizer reads it on the next
      `gram()` call) and leaves any explicit `Feature.theta`
      untouched.
- [x] 1.3 `<feature>.theta[r,d,q]` is rejected on `MPSRung1` with
      `ValueError` naming the encoding.
- [x] 1.4 `<feature>.theta[r,d,q]` on `HEA_Rung2`: when
      `Feature.theta is None`, materialize the default via
      `_default_hea_theta`, copy, set the slot, then return a
      `Dictionary` whose feature carries the patched tensor. Index
      out-of-range raises `ValueError` naming the offending slot and
      the encoding's `theta_shape`.
- [x] 1.5 Malformed paths raise `ValueError` describing the expected
      grammar.

## 2. Cancellation knob list

- [x] 2.1 `polygram/cancellation.py` — add
      `knobs: list[str] | None = None` field on the `Cancellation`
      dataclass. `__post_init__` resolves `None` to
      `[f"{a}.phi", f"{b}.phi"]`.
- [x] 2.2 `__post_init__` validates each knob path via the same
      grammar as `Dictionary.with_knob`; rejects `.theta[...]` paths
      on `MPSRung1` with `ValueError` naming the encoding.
- [x] 2.3 Bounds: per-knob bounds derived from path type — `.phi` →
      `(0.0, 2π)`, `.theta[r,d,q]` → `(-π, π)`.
- [x] 2.4 Grid backend rejects `len(knobs) > 4` with `ValueError`
      recommending `method="scipy"`. Scipy backend has no such limit.
- [x] 2.5 `_dictionary_at(*values)` becomes variadic; walks
      `self.knobs` applying each value via `Dictionary.with_knob`.
- [x] 2.6 `trajectory` shape becomes `(M, len(knobs) + 1)`. Last
      column is target-pair overlap. `feasible_mask` shape `(M,)`
      unchanged.
- [x] 2.7 `optimized_phis` field on `CancellationResult` renamed to
      `optimized_knobs: dict[str, float]` keyed by knob path.
      Materialized summary + plot code updated.

## 3. structural_floor() contract

- [x] 3.1 `Cancellation.structural_floor()` raises
      `NotImplementedError` whenever (a) `dictionary.encoding` is
      not `MPSRung1`, OR (b) `knobs` is not the canonical
      `[f"{a}.phi", f"{b}.phi"]` pair. The message names the
      configuration and points at this proposal's deferral
      paragraph.
- [x] 3.2 `cancellation_efficiency` is `None` when
      `structural_floor()` raises (in addition to the existing
      `before − floor < 1e-9` case). `Cancellation.run()` catches
      the `NotImplementedError` and stores
      `structural_floor=float("nan")`,
      `cancellation_efficiency=None`.
- [x] 3.3 `CancellationResult` materialized summary distinguishes
      "structural floor undefined for this configuration" from
      "structural floor reached" in the human-readable summary.

## 4. InterferenceSweep generalization

- [x] 4.1 `polygram/experiment.py` — `Experiment._parse_sweep_key`
      accepts both `<feature>.phi` and `<feature>.theta[r,d,q]`.
      `.theta` paths on `MPSRung1` raise `ValueError`.
- [x] 4.2 `_dictionary_at_sweep_index` uses `Dictionary.with_knob`
      to apply each axis value.
- [x] 4.3 `InterferenceSweep.run()` populates a
      `tier_separation: np.ndarray | None` per sweep point. `None`
      when the dictionary's clusters are all singletons.
- [x] 4.4 `ExperimentResult.tier_separation` field added; `.save()`
      and `.to_csv()` extended to include it (CSV column
      `tier_separation`; `.npz` key `tier_separation`). Skipped
      cleanly when `None`.

## 5. Tier-separation assertion

- [x] 5.1 `polygram/_assertions.py` — new
      `concept_gram_tier_separation_bound_holds(gram, dictionary)`
      checker. Returns `True` when
      `compute_tier_separation(gram, clusters) >=
      encoding.tier_separation_bound` (per sweep point); `False`
      otherwise. Raises `ValueError` if called on a
      dictionary whose encoding lacks a bound (defensive — the
      Experiment validator should catch this earlier).
- [x] 5.2 `Experiment.__post_init__` rejects the assertion at
      construction time when the dictionary's encoding has no
      `tier_separation_bound` (either MPS rung-1, or HEA with the
      bound set to `None`).
- [x] 5.3 `SUPPORTED_ASSERTIONS` extended; `_assertions.py`
      `__all__` updated.

## 6. Combined before/after plot

- [x] 6.1 `polygram/cancellation.py` — `CancellationResult.plot`
      accepts `kind: str | None = None`. `None` → existing per-
      method default (`"grid"` or `"scipy"`). `"before_after"` →
      new three-panel figure: before Gram, after Gram (shared
      colorbar), bar chart with `before/after/floor`.
- [x] 6.2 Tier-separation indicator on the bar chart: if the
      dictionary has any cluster with ≥ 2 features, render a
      secondary axis with the `tier_separation` before vs after.
      Skipped silently otherwise.
- [x] 6.3 Both Gram heatmaps highlight the target pair cell with
      a marker so it's recognizable at a glance.

## 7. Tests

### Dictionary
- [x] 7.1 `tests/test_dictionary.py::TestWithKnob` — `.phi` on MPS
      and HEA, `.theta[r,d,q]` on HEA happy path, `.theta` rejected
      on MPS, out-of-range slot rejected, malformed path rejected.

### Cancellation
- [x] 7.2 `tests/test_cancellation.py::TestKnobsList` — default
      resolves to 2-φ; explicit list with HEA `.theta[r,d,q]`
      runs; `len(knobs) > 4` rejected on grid; trajectory shape
      matches `(M, len(knobs) + 1)`.
- [x] 7.3 `tests/test_cancellation.py::TestStructuralFloorContract`
      — MPS default 2-φ returns float as before; HEA default 2-φ
      raises `NotImplementedError`; MPS with non-canonical knob
      list raises `NotImplementedError`; `result.structural_floor`
      is NaN and `cancellation_efficiency` is `None` when raised.

### InterferenceSweep
- [x] 7.4 `tests/test_experiment.py::TestSweepKnobs` — `.phi` axis
      on HEA dictionary works; `.theta[r,d,q]` axis on HEA works;
      `.theta` axis on MPS rejected.
- [x] 7.5 `tests/test_experiment.py::TestTierSeparationMeasure` —
      ExperimentResult carries `tier_separation` shaped per sweep;
      all-singleton dictionary yields `None`; CSV + npz round-
      trip preserves the field.
- [x] 7.6 `tests/test_experiment.py::TestTierBoundAssertion` —
      assertion passes on a clearly-tiered HEA dictionary;
      construction rejected when encoding lacks a bound.

### Plot
- [x] 7.7 `tests/test_cancellation.py::TestBeforeAfterPlot` —
      `kind="before_after"` writes a non-empty PNG; existing
      default kinds unchanged.

### Example
- [x] 7.8 `tests/test_examples.py::test_animals_hea_example_runs`
      extended to assert the example writes the InterferenceSweep
      artifacts, the Cancellation artifacts, and the before/after
      figure.

## 8. Example update

- [x] 8.1 `examples/animals_hea.py` — extend `main()` to (a) run
      InterferenceSweep over `dog_poodle.phi`, (b) run Cancellation
      on the `(dog_poodle, bird_hawk)` pair with the default 2-φ
      knobs, (c) emit the before/after figure, (d) print a tier-
      separation rollup. Existing emit-and-verify path stays
      first.
- [x] 8.2 Example output directory layout documented in the
      module docstring.

## 9. Validate + commit

- [x] 9.1 `openspec validate extend-cancellation-sweep-hea --strict`
      ✓
- [x] 9.2 Full pytest suite green; ruff clean
- [ ] 9.3 Commit + push, open PR, merge after review

## 10. Archive

- [ ] 10.1 `openspec archive extend-cancellation-sweep-hea` after
      merge — propagate the new requirements into
      `openspec/specs/{dictionary,experiment}/spec.md`.
