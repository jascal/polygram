## Why

External bug report against polygram 0.4.0 on jbloom/GPT-2-small layer-1 SAE (2026-05-15) surfaced four issues blocking Axis 4 (sae-forge faithfulness) sweeps. The root cause: **MPSRung1.gram() saturates on activation-uncorrelated features** — 12 of 28 off-diagonal pairs hit |G|² ≥ 0.9, mean 0.758, on features whose actual pairwise Jaccard mean is 0.05.

Reproduced locally on the bundled toy SAE fixture with the default loader path (clustered profile, k=2):

| Metric | Value |
|---|---|
| Mean off-diagonal \|G\|² | **0.7567** (≈ the reporter's 0.758) |
| Max off-diagonal \|G\|² | 0.9998 |
| Pairs \|G\|² ≥ 0.9 | **12 of 28** (≈ the reporter's 12 of 28) |
| Per-feature α | 0 (always) |
| Per-feature φ | 0 (always) |
| Per-feature β | only ±0.5 (k=2 cluster ordinal) |

**This is the same dormancy that PR #63 fixed for Rung3/Rung4 amp-branch knobs, at a lower level**: `from_sae_lens` assigns β (PC1) and γ (per-cluster PCA when `assign_gamma=True`), but α and φ default to 0 and are never assigned. MPSRung1's prepare-form circuit uses all four knobs; loading via `from_sae_lens` uses only two. The algebraic feature cap of 8 is real; the loader's effective discriminating capacity is closer to 2.

Sanity check confirms the fix works: populating α and φ per-feature drops the gram metrics dramatically:

| Metric | Default (α=φ=0) | With α, φ populated |
|---|---|---|
| Mean off-diag | 0.7567 | **0.2807** (63% drop) |
| Pairs ≥ 0.9 | 12 / 28 | **1 / 28** |
| Frobenius `‖g_phase − g_default‖` | — | 4.22 |

This change is the natural extension of PR #63's `encoding-aware-knob-assignment` work: a parallel `assign_phase_knobs` flag for α/φ that applies to *any* encoding with those knobs (MPSRung1, Rung3, Rung4 — all share the MPS-substrate phase knobs). The amp-branch flag stays as it was; the new flag is independent.

**PCA-component notation throughout this proposal**: PC_k denotes the *k*-th principal component (1-indexed; PC1 is the top component). In code, these correspond to `vt[k-1]` rows (0-indexed array access). Standard ML convention; sticky enough to avoid the "PC2 vs axis 1" confusion that mixed numbering creates.

## What Changes

### Scope (small, mirrors PR #63 shape)

- **`from_sae_lens`** gains `assign_phase_knobs: bool = False` kwarg. Default `False` preserves byte-identical behavior.
- **`SAEImportConfig`** gains `assign_phase_knobs: bool = False` field. Same precedence as `assign_gamma`.
- **`KnobAssignmentResult`** gains two new optional fields: `alphas: list[float] | None = None`, `phis: list[float] | None = None`.
- **New helper**: `polygram/geometry/phase_assignment.py::assign_phase_knobs_pca(projections, encoding)` returns a dict with `alphas` and `phis`, populated from PC2 and PC3 of the projection vectors. Encoding-agnostic — applies to MPSRung1, Rung3, Rung4 (all share the MPS-substrate α and φ).
- **`KnobAssignment.assign` signature**: extended with `assign_phase_knobs: bool = False`. Both shipped strategies (`ClusteredKnobAssignment`, `UniformSphereKnobAssignment`) honor the flag via the helper.
- **`EpochCompressor`, `Compressor`** gain `assign_phase_knobs: bool = False` field plumbed through to internal `from_sae_lens` calls — same plumbing pattern as PR #64 did for `assign_amp_knobs`.
- **Existing `assign_amp_knobs` PCA-component allocation shifts** from PC2-PC5 to PC4-PC7. The two flags don't collide; phase always gets the lowest PCs after β. This is a backward-compat break for the PR-#63-era exact numbers in `docs/research/rung_gram_condition_*_amp_on.json` — qualitative findings still hold but exact values change. The v2.1 results note flags this with a regenerate-and-update.

### Falsifying invariant

After the impl lands, `from_sae_lens(records, encoding=MPSRung1(), assign_phase_knobs=True)` MUST produce a gram measurably different from `assign_phase_knobs=False`. The cornerstone test:

```python
def test_phase_knobs_activate_mpsrung1_capacity():
    records, ids = _load_toy_records(8)
    d_off, _ = from_sae_lens(records, ids, encoding=MPSRung1(), assign_phase_knobs=False)
    d_on,  _ = from_sae_lens(records, ids, encoding=MPSRung1(), assign_phase_knobs=True)
    g_off = np.abs(d_off.gram()) ** 2
    g_on  = np.abs(d_on.gram()) ** 2
    # Frobenius distance well above FP noise.
    assert np.linalg.norm(g_off - g_on, ord="fro") > 1.0
    # Mean off-diagonal must drop materially (sanity-check confirms ~63% drop).
    n = g_off.shape[0]
    iu = np.triu_indices(n, k=1)
    assert g_on[iu].mean() < 0.5 * g_off[iu].mean()
```

The 0.5× factor is calibrated against the local sanity check (0.76 → 0.28, a 63% drop). The test fails loudly if the impl produces bit-identical gram on both flag values, or if the drop is materially smaller than predicted.

### What this change explicitly does NOT do

- **Doesn't change the default**: `assign_phase_knobs=False` stays the default; every existing call site is byte-identical.
- **Doesn't auto-enable for higher-rung encodings**: explicit opt-in. A future change might flip the default once Axis 1 / Axis 4 measurements support it.
- **Doesn't address the other 3 issues** in the bug report (#2 `plan_pareto` ignoring `gate_pass`; #3 EpochCompressor cross-panel leak into ForgePipeline; #4 uniform-sphere docstring). Those are filed as separate issues.
- **Doesn't merge phase + amp into a single combined flag**: keeps two distinct surfaces for users who want fine-grained control.

## Impact

### Affected specs

`sae`. New requirement: `from_sae_lens` accepts `assign_phase_knobs`. Modified requirement: `KnobAssignment.assign` signature extended with the new kwarg.

### Affected code

- `polygram/geometry/protocols.py` — `KnobAssignmentResult` gains `alphas`, `phis` fields; `KnobAssignment.assign` signature extended
- `polygram/geometry/phase_assignment.py` (NEW) — the PCA helper
- `polygram/geometry/amp_assignment.py` — shift axis allocation from 2-5 to 4-7
- `polygram/geometry/clustered.py` + `uniform_sphere.py` — both strategies honor the new flag
- `polygram/sae_import.py` — `from_sae_lens` plumbing
- `polygram/config.py` — `SAEImportConfig.assign_phase_knobs` field
- `polygram/compression/epoch.py` + `compressor.py` — `assign_phase_knobs` field + plumbing (mirrors PR #64's `assign_amp_knobs` work)
- `examples/rung_gram_condition.py`, `examples/rung_compression_coverage.py` — `--assign-phase-knobs` CLI flag
- `tests/test_phase_knob_assignment.py` (NEW) — falsifying invariant + edge cases
- `CHANGELOG.md`
- `docs/research/rung4-viability-spike-v2.md` — note the axis shift and regenerate the `_amp_on` data if exact reproducibility matters

### Closes

The root-cause finding from the 2026-05-15 GPT-2 bug report (#1). The other three issues filed as separate work.

### Effort estimate

Same shape as PR #63 (encoding-aware-knob-assignment) + PR #64 (compression plumbing). One openspec proposal + one impl PR. Maybe 200 LOC + tests.
