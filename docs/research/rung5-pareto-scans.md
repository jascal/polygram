# Rung5 Pareto scans

**Date:** 2026-05-16
**Status:** scripted in `examples/rung5_pareto_scans.py`
**Artifacts:** `docs/research/data/rung5_pareto/{capacity_quality,saturation_density,pca_amp_ablation}.json`

Three CPU-only sweeps probing the polygram 0.7.0 Rung5 encoding on a Mac
host (no torch / no behavioural validator). All numerical results come
from the analytic Gram path; the goal is to characterise the
configurable-amp-width axis added by this release.

## Headline findings

1. **Gram condition number drops ~5 orders of magnitude as k grows at
   fixed N.** At N=64, Rung5(k=3) has gram cond ≈ 1.47e7 while
   Rung5(k=5) has cond ≈ 2.4e2 — a 60000× improvement in numerical
   conditioning. This is the load-bearing benefit of going past the
   smallest-cap encoding that fits your feature count.
2. **Decoder-Gram fidelity (Spearman) does *not* track conditioning.**
   At fixed N, Spearman is essentially constant across encoding
   choices that have headroom. The k-axis buys you headroom and
   conditioning, not fidelity gains.
3. **Saturation transition at N=cap is sharp.** Gram is full-rank and
   numerically well-conditioned at N ≤ 0.91·cap, marginally
   well-conditioned at N = cap, and effectively singular at any
   N > cap (cond jumps to ~1e17). Practical recommendation: pick a
   k such that the target N is ≤ 0.9·cap for headroom.
4. **PCA-based `assign_amp_knobs=True` buys conditioning, not
   Spearman.** Turning on the Rung5 PCA branch added in this release
   drops gram cond by 10–14 orders of magnitude (1e18 → 1e8 at k=3,
   1e18 → 1e4 at k=4) while Spearman moves by ≤ 0.03. The new
   helper's value is numerical, not geometric.

## Scan 1: capacity-vs-quality across the encoding ladder

Sweeps `N ∈ {8, 16, 32, 64, 128, 256}` × every encoding with cap ≥ N
(MPSRung1, Rung3, Rung4, Rung5(k=3..5)). For each cell, builds a
dictionary via `from_sae_lens(assign_amp_knobs=True,
assign_phase_knobs=True)` on a synthetic clustered SAE (64 clusters of
4 in 64-dim, isotropic-noise siblings of unit-norm cluster centroids).
Features are stride-selected across clusters to keep the inter-cluster
signal alive at small N.

Reads from
[`docs/research/data/rung5_pareto/capacity_quality.json`](data/rung5_pareto/capacity_quality.json):

| N | Best Spearman (encoding) | Best cond (encoding) | Cond range across encodings |
|---|---|---|---|
| 8 | +0.0755 (Rung3) | 8.47 (Rung4/Rung5 — tied) | 1.3e4 → 8.5 |
| 16 | +0.1966 (Rung5 k=5) | 4.42 (Rung5 k=5) | 4.8e4 → 4.4 |
| 32 | +0.0889 (Rung5 k=5) | 27.9 (Rung5 k=5) | 1.7e6 → 28 |
| 64 | +0.0450 (Rung5 k=5) | 239 (Rung5 k=5) | 1.5e7 → 239 |
| 128 | +0.0943 (Rung5 k=5) | 1.44e4 (Rung5 k=5) | 5.7e8 → 1.4e4 |
| 256 | +0.0738 (Rung5 k=5) | 4.19e10 (k=5, saturated) | — |

**Take-away:** Rung5(k=5) is consistently the best-conditioned choice
at every N. Spearman is within noise across encodings that have
headroom. The Pareto-optimal pick for a target N is "smallest k such
that 8·2^k ≥ N · 1.1", i.e. pick a 10% headroom buffer above N.

## Scan 2: saturation transition around N = cap

For k ∈ {2, 3, 4}, scans `N ∈ {0.25, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0} ×
cap` with two seeds. Reads from
[`docs/research/data/rung5_pareto/saturation_density.json`](data/rung5_pareto/saturation_density.json).

The cliff at N = cap is clean across all k:

```
k=4 (cap=128):
  N=115 (0.90× cap): rank=115, σ_min>0=4.31e-03, cond=8.5e2   ← well-conditioned
  N=128 (1.00× cap): rank=128, σ_min>0=3.17e-05, cond=1.2e5   ← on the edge
  N=141 (1.10× cap): rank=128, σ_min>0=4.26e-03, cond=1.5e17  ← effectively singular
```

The smallest non-zero singular value tracks roughly `(1 − N/cap)^2` in
the under-saturated regime, then collapses by 2 decades right at
N = cap. Practical takeaway: targeting N ≤ 0.9·cap leaves enough
headroom that downstream Gram-inversion solves stay tractable.

## Scan 3: PCA amp-knob ablation

For k ∈ {3, 4}, imports the same synthetic SAE twice — once with
`assign_amp_knobs=False` (all amp knobs default to zero — Rung5 gram
reduces to MPSRung1 gram), once with `assign_amp_knobs=True` (PCA axes
4..(4 + 2k − 1) populate per-feature `amp_knobs`). Reads from
[`docs/research/data/rung5_pareto/pca_amp_ablation.json`](data/rung5_pareto/pca_amp_ablation.json).

```
k=3 (N=64, cap=64):
  assign_amp_knobs=False: Spearman +0.1349, cond 2.88e18   ← rank-deficient
  assign_amp_knobs=True:  Spearman +0.1039, cond 1.75e08
  Δ Spearman: -0.0310

k=4 (N=64, cap=128):
  assign_amp_knobs=False: Spearman +0.1349, cond 2.88e18   ← rank-deficient
  assign_amp_knobs=True:  Spearman +0.1162, cond 1.18e04
  Δ Spearman: -0.0187
```

**Take-away:** The new `_assign_amp_knobs_pca_rung5` branch is
load-bearing for *conditioning* — without it, every feature gets
identical amp_knobs = ((0,0),...,(0,0)) and the Rung5 gram is exactly
the MPSRung1 gram, which is rank-deficient at N > 8. With PCA-derived
amp_knobs, the gram becomes full rank (cond drops 10–14 decades).

Spearman against decoder cosine drops slightly, because the PCA axes
beyond the cluster basis (PC4+) are noise dimensions in this synth, so
they spread features across the amp-Hilbert-space *orthogonally* to
the cluster structure. That's the intended behaviour — the amp branch
is supposed to add capacity, not refine cluster-geometry fidelity.

## Bug fix: `from_sae_lens` default-pads `amp_knobs` for Rung5

Running scan 3 surfaced a real polygram bug: `from_sae_lens` with a
Rung5 encoding and `assign_amp_knobs=False` left `Feature.amp_knobs`
empty (`()`), causing `Dictionary.__post_init__` to reject the
mismatched length. Fixed at the same time as adding these scans —
`sae_import.py` now default-pads to `((0.0, 0.0),) * n_amp_qubits`
when the strategy doesn't produce `amp_knobs_list`. This is the
"default reduces to MPS" property at the loader level.

## Reproduction

```bash
# Scan 1 — encoding-ladder Pareto (~ 2 min on a Mac)
python examples/rung5_pareto_scans.py capacity-quality \
    --json-out docs/research/data/rung5_pareto/capacity_quality.json

# Scan 2 — saturation density (~ 1 min, k ∈ {2, 3, 4})
python examples/rung5_pareto_scans.py saturation-density \
    --json-out docs/research/data/rung5_pareto/saturation_density.json

# Scan 3 — PCA amp-knob ablation (~ 15 sec, k ∈ {3, 4})
python examples/rung5_pareto_scans.py pca-amp-ablation \
    --json-out docs/research/data/rung5_pareto/pca_amp_ablation.json
```

All three are deterministic (seeded) and torch-free.

## Open follow-ups

- **Real-SAE replication.** Run scan 1 against a Gemma-Scope SAE
  loaded via `load_sae_safetensors`. The synth here is intentionally
  isotropic; a real SAE's higher-order PCA structure may give Rung5's
  amp-knob assignment more signal to ride on.
- **Cancellation-cost sweep.** Wall-clock of `_run_rung5_joint` as a
  function of k at fixed N. Not measured here — scan 1 only times
  `gram()`, not joint cancellation. If joint cancellation cost grows
  superlinearly with k, the conditioning win may be outweighed by
  solver cost at large k.
- **Saturation transition shape.** The cliff at N = cap is sharp but
  empirical. A small theory note relating the σ_min>0 decay rate to
  the `2 + 2k`-dim parameter freedom around the manifold (referencing
  the theoretical treatment's Prop 4.1 codimension result) would
  close the loop.
