## Why

[`docs/research/rung4-viability-spike-v2.md`](../../../docs/research/rung4-viability-spike-v2.md) surfaced a load-bearing finding while running Axis 2 of the v2 viability methodology:

> At default knobs, Rung3 and Rung4 produce **bit-identical** gram to MPSRung1 at the same K. This is a designed property of the encodings (per Rung4's own docstring: *"Default-knob Rung4 reduces to MPSRung1-equivalent gram on the same (α, β, γ, φ)"*). `from_sae_lens` doesn't assign non-default amp-branch knobs from decoder geometry, so real consumers never see the higher rungs' larger state space at the K=8 baseline.

The implication is structural: **every production Rung3/Rung4 dictionary that flows through `from_sae_lens` is gram-equivalent to MPSRung1**. The amp-branch knobs (`theta_amp`, `psi_aux`, `theta_amp_b`, `psi_amp_b`) carry the encoding's extra state-space dimensions; with the loader leaving them at default `(π/4, 0, π/4, 0)`, the amp factors collapse to 1 and the gram reduces to the MPS-only path.

This change extends the `KnobAssignment` strategy to optionally populate amp-branch knobs from decoder geometry. With the new path enabled, Rung3 and Rung4 dictionaries actually exercise their state-space capacity rather than aliasing MPSRung1.

This is **P0 from the [post-#61 strategic review](../../../docs/research/rung-viability-methodology.md)**: until it lands, the entire rung ladder (Rung3, Rung4, future rungs) is unreachable for real consumers under default loading. After it lands, Axis 1 (compression coverage) and Axis 4 (sae-forge faithfulness) become meaningful experiments because they'd actually be testing the higher rungs rather than MPS-equivalents.

## What Changes

### Scope (single-flag opt-in)

- New `assign_amp_knobs: bool = False` kwarg on `from_sae_lens`. Default `False` preserves byte-identical behavior (every existing test pins this).
- `KnobAssignmentResult` gains four optional fields: `theta_amps`, `psi_auxes`, `theta_amp_bs`, `psi_amp_bs`. Each is `list[float] | None`. `None` (default) → consumer uses encoding defaults.
- `KnobAssignment.assign` signature extended with `assign_amp_knobs: bool = False` and `encoding` kwargs. Strategies that don't implement amp assignment return `None` for the four new result fields; strategies that do populate them with per-feature values derived from decoder geometry.
- Profile registry: built-in `clustered` and `uniform-sphere` profiles both gain an amp-knob assignment path (PCA-axis extension — described in design.md). When `assign_amp_knobs=True`, both profiles populate the encoding's amp-branch knobs from higher PCA axes; when `False`, both behave exactly as today.
- `from_sae_lens` propagates the new per-feature amp values into `Feature` objects when the result fields are non-None.

### Encoding-awareness

The assignment is encoding-aware in the sense that the strategy knows how many amp knobs the encoding consumes:
- **MPSRung1** has no amp branch → `assign_amp_knobs=True` is a no-op (logged at debug level so users don't think their flag did something).
- **Rung3** has 2 amp knobs (`theta_amp`, `psi_aux`) → strategy populates `theta_amps` and `psi_auxes`; leaves `theta_amp_bs` and `psi_amp_bs` as `None`.
- **Rung4** has 4 amp knobs → strategy populates all four arrays.
- **HEA_Rung2** has its own knob structure (`theta_shape = (|rotations|, depth, n_qubits)`) — out of scope for v1; flag is a no-op with a clear log message.

### Falsifiable test

After this change, a Rung4 dictionary built via `from_sae_lens(records, encoding=Rung4(), assign_amp_knobs=True)` MUST produce a gram **measurably different** from the same dictionary built with `assign_amp_knobs=False`. Specifically:

- `mean(|gram_amp_on|² - |gram_amp_off|²)` is non-zero by more than 1e-6 on the toy SAE fixture (sanity: the change exists somewhere).
- `||gram_amp_on|² - |gram_amp_off||²_F > 1e-3` (full-matrix Frobenius distance well above FP noise floor).
- **Off-diagonal-only Frobenius distance** `||off_diag(|gram_amp_on|² - |gram_amp_off|²)||_F > 1e-3` — stronger assertion that the change isn't concentrated on the diagonal (a degenerate impl that perturbed only the on-diagonal terms would pass the previous check but fail this one).
- The byte-identical default (`assign_amp_knobs=False`) test continues to pass — every existing differential regression test is unchanged.

If the new path produces bit-identical gram to the old path, the change is broken and the test fails loudly.

## Impact

### Affected specs
`sae` (the loader spec). New requirement: `from_sae_lens` accepts `assign_amp_knobs`. Modified requirement: `KnobAssignment.assign` extended signature; `KnobAssignmentResult` extended fields.

### Affected code
- `polygram/geometry/protocols.py` — `KnobAssignmentResult` extended; `KnobAssignment.assign` signature extended
- `polygram/geometry/clustered.py` — `ClusteredKnobAssignment.assign` gains amp-knob path
- `polygram/geometry/uniform_sphere.py` — `UniformSphereKnobAssignment.assign` gains amp-knob path
- `polygram/sae_import.py` — `from_sae_lens` gains `assign_amp_knobs` kwarg; threads through to strategy; populates Feature objects
- `polygram/dictionary.py` — `Feature` already has the amp-knob fields (added in PR #52). No new fields needed.
- `tests/test_sae_import.py` (+ new file `test_amp_knob_assignment.py`) — coverage for both False and True paths
- `CHANGELOG.md`
- `docs/research/rung4-viability-spike-v2.md` — append a "Resolved" section once the impl ships

### Closes
- The "Add encoding-specific knob assignment to `from_sae_lens`" follow-up identified in `docs/research/rung4-viability-spike-v2.md`.

### What this change explicitly does NOT do

- **Doesn't change the default behavior.** `assign_amp_knobs=False` is the default; every existing call site is byte-identical.
- **Doesn't add new profiles.** Both `clustered` and `uniform-sphere` profiles gain the new path; no new profile name needed.
- **Doesn't extend to HEA_Rung2.** HEA's knob structure (`theta_shape = (|rotations|, depth, n_qubits)`) is fundamentally different; covered in a separate change if/when a consumer needs it.
- **Doesn't optimize the amp knobs.** The assignment is a *single forward pass* over the projection PCA — it doesn't search for "best" knob values. Optimization is a much bigger scope and not needed to validate the rung ladder.
- **Doesn't auto-enable `assign_amp_knobs` for Rung3/Rung4.** Conservative: opt-in only. Tests pin the byte-identical default. A future change could flip the default once enough downstream evidence accumulates.
