# Implementation tasks

## 0. Design pre-locks (blocking)

- [ ] 0.1 Lock the readout subspace construction: `R = top-r right singular directions of (gain ⊙ U)`
  (`U`: `(vocab, d_model)` unembed; `gain`: `(d_model,)` final-norm, default ones). `r` defaults to a
  capped rank (e.g. `min(64, d_model)`); the geometry is built on `projections @ Rᵀ`. Confirm this matches
  the R2 / sae-forge `_readout_aligned_order` construction.
- [ ] 0.2 Lock that the projection happens **inside** `ReadoutAlignedKnobAssignment.assign(...)` so the
  `from_sae_lens` call site (`sae_import.py:813`) and the `GeometricProfile`/registry plumbing are unchanged.
- [ ] 0.3 Lock scope: knob assignment (β/γ) only; Q-Orca emission + `Dictionary.gram()` consume final knobs and
  are transparent. No host-model load in the core.

## 1. `polygram/geometry/readout_aligned.py` — the profile

- [ ] 1.1 `ReadoutAlignedKnobAssignment` mirroring `ClusteredKnobAssignment` (`clustered.py:143`), carrying
  `u_matrix` + `gain` (+ `readout_rank`); `.assign(projections, ...)` projects onto the readout subspace, then
  runs the existing k-means / centroid / residual-variance / γ-PCA path on the projected vectors.
- [ ] 1.2 `ReadoutAlignedFidelity` (implements the `GeometricFidelity` protocol) + `readout_aligned()` factory
  (mirroring `clustered()` at `clustered.py:258`).
- [ ] 1.3 Register `readout_aligned()` in `polygram/geometry/__init__.py` (+ `__all__`); ensure
  `get_profile("readout-aligned")` resolves.

## 2. `polygram/sae_import.py` — thread `u_matrix` / `gain`

- [ ] 2.1 Add `u_matrix: np.ndarray | None = None`, `gain: np.ndarray | float | None = None` to `from_sae_lens`
  (`:615`). When the resolved profile is `readout-aligned`, **require** `u_matrix` (2-D, second axis == d_model)
  — raise a clear `ValueError` otherwise. Inject `u_matrix`/`gain` into the strategy before `.assign` (`:813`).
- [ ] 2.2 Default profile unchanged (`clustered`); a run without `profile="readout-aligned"` is byte-identical.

## 3. Tests

- [ ] 3.1 Synthetic: a feature set whose raw-decoder clustering differs from its readout-aligned clustering
  (construct `U` so the readout subspace re-orders the directions) → the two profiles assign different β/γ /
  cluster labels; `readout-aligned` without `u_matrix` raises `ValueError`.
- [ ] 3.2 `clustered` default path byte-identical (existing geometry tests green); `get_profile("readout-aligned")`
  registered.
- [ ] 3.3 Determinism: same `(projections, U, gain, seed)` → identical assignment.

## 4. Behavioural gate (blocking — the decomposition IS the result)

- [ ] 4.1 Extend `examples/behavioural_gram_scaleup.py` to run `clustered` vs `readout-aligned` head-to-head on
  GPT-2-small (supply `U`/`gain` from the host the probe already loads); emit
  `polygram_overlap_readout_aligned` into `docs/research/data/scaleup_pairs.csv`.
- [ ] 4.2 Record the head-to-head Spearman table in `docs/research/behavioural-scaleup-probe.md` (in place, no
  new page). **Descriptive verdict, pre-committed both ways:** WIN (Spearman ≥ 0.70) or NO-IMPROVEMENT
  (basis isn't the limit → Reckoning #3 prune more likely). No necessity claims.
