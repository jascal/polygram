## Context

`polygram/sae_import.py::from_sae_lens` constructs a `Dictionary` from a set of `SAEFeatureRecord`s. The knob-assignment path:

1. Compute projection vectors (decoder rows, optionally re-normalized).
2. Resolve a `GeometricProfile` (defaults to `clustered`).
3. Invoke `profile.knob_assignment.assign(...)` → returns `KnobAssignmentResult(cluster_per_feature, betas, gammas, cluster_method, beta_variance_explained)`.
4. Build per-feature `Feature` objects with the assigned `(cluster, beta, gamma)`. `α` and `φ` default to `0.0`. **Amp-branch knobs (`theta_amp`, `psi_aux`, `theta_amp_b`, `psi_amp_b`) default to `π/4, 0, π/4, 0` respectively — and `from_sae_lens` never overrides them.**

The encodings are designed to reduce to MPSRung1-equivalent gram at these defaults, so the higher rungs' state-space dimensions go unused for any feature produced by `from_sae_lens`.

The fix is structurally simple: extend the strategy to populate the amp-branch knob fields when the caller opts in. The interesting question is *what values* to assign.

## Goals / Non-Goals

**Goals**:
- Provide a single user-facing flag to opt into encoding-aware knob assignment.
- Preserve byte-identical behavior at the default `False` setting.
- Produce *measurably different* gram from the default-knob path when the flag is on — the falsifying invariant.
- Work with both shipped profiles (`clustered`, `uniform-sphere`).

**Non-goals**:
- A "best" knob-assignment strategy. We pick PCA-axis-extension because it's a natural generalization of the existing uniform-sphere β strategy; we don't claim it's optimal.
- Optimization / search over knob values. Single forward pass from projection geometry.
- HEA_Rung2 amp assignment. Different knob shape; separate change.
- Auto-enable for higher-rung encodings. Opt-in only.

## Decisions

### Decision 1: PCA-axis extension as the assignment strategy

For each profile, when `assign_amp_knobs=True`:

- Compute top-K PCA components of the projection vectors (where K depends on how many amp knobs the encoding consumes — 2 for Rung3, 4 for Rung4).
- Assign each feature's per-amp-knob value from its coordinate on the corresponding PCA axis.
- Rescale coordinates into the knob's natural range:
  - `theta_amp` → `[0, π/2]` (positive interval, sinusoidal sensitivity)
  - `psi_aux` → `[0, 2π]` (full phase circle)
  - `theta_amp_b` → `[0, π/2]`
  - `psi_amp_b` → `[0, 2π]`

The β strategy already uses the top-1 PCA axis; this extends naturally to higher axes. For features with degenerate decoder geometry (fewer PCA components than amp knobs requested), the remaining knobs are filled with the encoding's default.

**Why PCA, not k-means cluster ordinals?**
- PCA gives continuous per-feature values that vary smoothly with decoder geometry — likely to give a smoother gram landscape than discrete ordinals.
- The β strategy already uses PCA; extending it is the lowest-friction path.
- k-means cluster ordinals would require a second clustering pass (or a hierarchical / sub-cluster scheme), which is more code for arguably less geometric meaning.

**Why not random with deterministic seed?**
- Deterministic random is the simplest fallback but provides no decoder-geometry signal. The amp values would be feature-id-aliased noise, not capacity-utilizing structure.

### Decision 2: Opt-in via a single boolean flag

`from_sae_lens` gains `assign_amp_knobs: bool = False`. The flag is propagated to the strategy via the `assign` signature.

Alternatives considered:
- **Per-knob flags** (`assign_theta_amp`, `assign_psi_aux`, etc.): more granular but encoding-specific. Worse UX.
- **Profile-level toggle**: would require a new profile (e.g., `"clustered-amp"`); worse versioning story.
- **Auto-on for Rung3/Rung4**: breaks byte-identity for existing call sites that explicitly pass `encoding=Rung3()`.

Single flag is the cleanest.

### Decision 3: `KnobAssignment.assign` signature extension is backward-compatible

The new kwargs (`assign_amp_knobs`, `encoding`) have defaults (`False`, `None`). Any third-party `KnobAssignment` implementation that doesn't take them keeps working at the default-False path. Strategies that want to support the new path opt in by accepting the new kwargs.

In Python, this means widening the protocol's signature. Existing implementations satisfying `KnobAssignment` will continue to satisfy it as long as they accept arbitrary kwargs (via `**kwargs`) or are updated. The two shipped implementations (`ClusteredKnobAssignment`, `UniformSphereKnobAssignment`) get explicit updates.

### Decision 4: `KnobAssignmentResult` extension uses `None` sentinels

`KnobAssignmentResult` is `frozen=True`. Adding fields is a breaking change to construction calls — but only inside polygram (the two strategies) and any third-party consumer subclassing the result type. The new fields default to `None`:

```python
theta_amps: list[float] | None = None
psi_auxes: list[float] | None = None
theta_amp_bs: list[float] | None = None
psi_amp_bs: list[float] | None = None
```

Strategies that don't populate them leave them as `None`. `from_sae_lens` checks: when non-None, write into the Feature; when None, use the encoding's default.

### Decision 5: Encoding-awareness is "soft" — flag is a no-op for MPSRung1/HEA

When `assign_amp_knobs=True` and the encoding is `MPSRung1` (no amp branch) or `HEA_Rung2` (different knob structure), the strategy logs a debug message and proceeds without populating any amp knobs. The flag is **not** an error — users may want a uniform call site across encodings.

When the encoding is `Rung3`, populate `theta_amps` and `psi_auxes` only. When `Rung4`, populate all four.

### Decision 6: Falsifiable test — non-trivial gram difference

The test that proves the amp assignment is *actually doing something* is the cornerstone of this change. Without it, the flag could land as a no-op and we wouldn't know.

```python
def test_assign_amp_knobs_changes_gram():
    # Same SAE, same features, encoding=Rung4()
    records = load_toy_sae(...)
    
    d_off, _ = from_sae_lens(records, encoding=Rung4(), assign_amp_knobs=False)
    d_on,  _ = from_sae_lens(records, encoding=Rung4(), assign_amp_knobs=True)
    
    g_off = np.abs(d_off.gram()) ** 2
    g_on  = np.abs(d_on.gram())  ** 2
    
    # Frobenius distance well above FP noise.
    assert np.linalg.norm(g_off - g_on, ord='fro') > 1e-3
    
    # And it isn't just FP noise on the diagonal:
    iu = np.triu_indices(g_off.shape[0], k=1)
    assert np.linalg.norm(g_off[iu] - g_on[iu]) > 1e-3
```

This is the load-bearing assertion. If it fails, the impl is wrong.

## Risks / Trade-offs

- **PCA-axis assignment is a research choice, not a derivation.** We're picking it because it's a natural extension of the existing β strategy and gives non-trivial gram changes. Whether it produces *better* compression / forging outcomes than other strategies is an Axis-1 / Axis-4 question — out of scope here. The goal of P0 is to un-dormant the rungs, not to find the optimal amp assignment.
- **Degenerate decoder geometry** (fewer non-zero singular values than amp knobs requested) is an edge case. We fall back to encoding defaults for the missing axes — documented in the assign() docstring.
- **Cross-encoding consistency.** A user switching `encoding=Rung3() → encoding=Rung4()` with `assign_amp_knobs=True` gets a *different* gram even on the shared (MPS) subspace, because Rung4's higher PCA axes shift slightly when re-fit. Acceptable: this is the price of using more state space.

## Open Questions

- **Should the rescaling be linear or sinusoidal?** Top-K PCA coords are not necessarily uniformly distributed; a uniform linear rescale into `[0, π/2]` or `[0, 2π]` could produce clustering near the endpoints (PCA coords on real SAE projections tend to be heavy-tailed, so a `coord / abs_max` linear map will push the bulk of features toward zero with a long tail toward the boundary). A sinusoidal rescale (`asin(coord / abs_max) * (π/2) / (π/2)` or similar) would spread samples more uniformly across the interior. **For v1: linear rescale** — simpler, matches the existing β strategy, defers the distribution-shape question until Axis-1 measurements show pathology. If early Axis-1 runs find the endpoint-clustering is biting (e.g., gram condition number doesn't improve as much as predicted), a sinusoidal variant is a cheap follow-up (one-line change in the rescale helper). Pinning this as a known-deferred decision rather than a "maybe later" item.
- **Should we expose the assignment strategy as configurable?** A user might want random-seed assignment for an ablation, or k-means-cluster-ordinal for a different geometric story. v1 hardcodes PCA-axis; the API leaves room for a strategy plug-in if a consumer asks. Out of scope for v1.

## Migration Plan

- **Default behavior unchanged**: `assign_amp_knobs=False` is the default; existing `from_sae_lens` callers see no difference.
- **Opt-in for downstream consumers**: `EpochCompressor`, `Cancellation`, `ForgePipeline` (sae-forge) etc. can either:
  - Keep the current default-knob behavior (status quo).
  - Pass `assign_amp_knobs=True` to actually use the higher rungs.

A future change might flip the default to `True` for `encoding=Rung3()` or `encoding=Rung4()` once empirical evidence (Axis 1 / Axis 4) confirms the amp-knob path produces better downstream outcomes. That's a separate decision and a separate change.
