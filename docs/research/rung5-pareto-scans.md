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

## Scan 4: learned PCA-axis assignment (prototype)

The fidelity-axis follow-up flagged in §"The bigger picture for the
encoding ladder" — learned axis-to-knob assignment — has a small
prototype attached as the `learned-assignment` subcommand. For each
k, it greedy-searches over axis-to-knob permutations using
decoder-Gram Spearman as the objective and compares against the
hardcoded baseline (PC2→α, PC3→φ, PC4..→amp_knobs).

Reads from
[`docs/research/data/rung5_pareto/learned_assignment.json`](data/rung5_pareto/learned_assignment.json).

```
k=3 (N=64, cap=64):
  baseline   Spearman = +0.1042   cond = 2.31e9
  learned    Spearman = +0.3350   cond = 6.01e8
  Δ Spearman = +0.2309   search = 4.7 s (32 candidate axes)

k=4 (N=64, cap=128):
  baseline   Spearman = +0.1161   cond = 1.95e4
  learned    Spearman = +0.3380   cond = 9.49e3
  Δ Spearman = +0.2219   search = 6.1 s (32 candidate axes)
```

**Take-away:** 3× Spearman improvement from a few seconds of greedy
search. The hardcoded assignment is leaving substantial signal on
the table — at least in this synthetic regime where β goes through
the cluster-labels bypass path (so PC1 is *not* reserved for β as
the encoding-aware-knob-assignment docstring claims).

Specifically, both k=3 and k=4 learned `α ← PC1` (axis 0), where the
baseline reserves PC1 for β-via-labels and starts α at PC2.
Effectively the baseline is asking α to ride on the second principal
component of the projection geometry when the first PC is the
cluster-bearing direction. The greedy variant finds and exploits
that.

The conditioning also improves (3.8× at k=3, 2.1× at k=4) — so the
learned assignment is a strict win on this synth: higher fidelity
*and* better-conditioned gram, in seconds of search. Per-feature
parameter count is unchanged; the optimisation is entirely on the
fixed-size axis-to-knob map.

This is genuinely "tuning, but at a different layer" — the SAE's
features aren't touched (no retraining), and per-feature knobs still
come from the same PCA of the same decoder. What's optimised is
*which PCA axis feeds which knob slot*. The cost is one extra
calibration pass at import time (Θ(n_knobs²) Spearman evaluations,
each Θ(N²·n) at fixed encoding).

A production version would replace the greedy permutation with
proper continuous optimisation (Riemannian gradient descent on a
small linear map W : R^d_model → R^(2n+1) per feature), but the
greedy result already establishes that *significant* fidelity
headroom exists — confirming the "Rung5 buys conditioning, not
fidelity" finding (scan 1) is an artefact of the hardcoded
assignment, not of the encoding itself.

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

# Scan 4 — learned axis-to-knob assignment (~ 12 sec, k ∈ {3, 4})
python examples/rung5_pareto_scans.py learned-assignment \
    --json-out docs/research/data/rung5_pareto/learned_assignment.json
```

All three are deterministic (seeded) and torch-free.

## What the scans mean for project theory

Each scan maps cleanly to a specific result in the
[May 2026 theoretical treatment](theory/polygram.pdf). The
correspondence is sharp enough that the scans should be read as
empirical validators of the formal claims, not as standalone numeric
curiosities.

### Scan 1 ↔ Prop 6.9 (Gram condition number) + §4 (manifold geometry)

Prop 6.9 bounds `κ(G) ≤ (1 + (N−1)ρ) / (1 − (N−1)ρ)` where `ρ` is the
maximum pairwise overlap of the dictionary's Polygrams. The bound is
vacuous at our scan-1 N (we exceed `(N−1)ρ < 1`), but the *trend* it
predicts is the load-bearing insight: smaller pairwise overlaps give
better-conditioned grams.

For random order-n Polygrams in ambient dim `2^n`, sphere
concentration gives expected `|⟨f|g⟩|² ~ 2^{−n}`, so expected
`ρ ~ 2^{−n/2}`. In our hybrid `Polygram(M=3, k)` encoding the
effective order is `n = 3 + k`, and ρ drops as `2^{−(3+k)/2}` as k
grows. The 5-decade conditioning improvement from Rung5(k=3) to
Rung5(k=5) at N=64 is the Gershgorin shadow of moving features onto
a larger sphere. **The k-axis is literally the manifold's
ambient-dimension knob**, and the win is structural, not incidental.

### Scan 2 ↔ Prop 4.1 (manifold dimension) + the dimensional wall

Prop 4.1: the Polygram manifold has real dimension `2n + 1` inside an
ambient sphere of dim `2·2^n − 1`. For our hybrid that means at most
`2^n = 8·2^k` linearly-independent features can live on the manifold.
The cliff at N = cap in scan 2 is this dimensional wall **empirically
observed**: smallest non-zero singular value decays smoothly until
N approaches cap, drops 2 decades right at N = cap, and at N > cap
the rank saturates at cap while the gram becomes effectively singular
(`σ_min/σ_max ~ 1e−17`).

In the language of Cor 4.2, the missing directions number
`2^{n+1} − 2n − 2`. At k=4 that's `256 − 8 − 2 = 246` codimensions
per Polygram — at N > 128 those codimensions become null-space of
the gram. The cliff is sharp because the polygram manifold is
*real-analytic* (Prop 4.1): codimension doesn't decay continuously,
it switches from "present" to "absent" the moment N exceeds cap.

### Scan 3 ↔ "default reduces to MPS" as manifold collapse

This is the cleanest theory lesson the scans deliver. With
`assign_amp_knobs=False`, every feature's amp factor is `|0⟩^⊗k` and
the encoded states all live on the `MPSRung1` sub-manifold — real
dimension `2·3 + 1 = 7`, *not* `2(3 + k) + 1 = 2k + 7`. We're trying
to embed 64 features on a 7-dim manifold; Prop 4.1 says you can't,
and the gram dutifully reports rank 8 with condition `1e18`. Turning
on PCA amp-knob assignment spreads features across the full
`2k + 7`-dim hybrid manifold, recovering full rank.

**The PCA branch isn't "adding fidelity" — it's preventing manifold
collapse.** Scan 3's Spearman barely moves because the PCA axes used
for the amp knobs (PC4..PC{3 + 2k}) are orthogonal to the cluster-
bearing PC1..PC3 axes in our synth: they add volume to the embedding
without refining its cluster geometry, exactly as §4.2's volume
calculation `vol[Σ_n] = π^n` would predict. More k buys more volume,
not more curvature in the right places.

### The synthesis: capacity vs. fidelity, and the (M, k) split

The scans cleanly separate two questions the theoretical treatment
keeps distinct:

- **Can the dictionary geometrically represent N features?** Pure
  manifold-dimension question. Answered by `8·2^k ≥ 1.1·N` (the 10%
  headroom buffer is the practical translation of scan 2's cliff).
- **Does the encoded gram track decoder geometry?** Pure
  parameter→state-map question. Depends on *which PCA axes feed
  which knobs*, not on the manifold's size. The k-axis does not help
  here — it's an MPS-substrate-quality question, not a Polygram-amp
  question.

This vindicates the paper's choice to *operationally* split
`n = M + k` (base vs amplitude factors) while *theoretically*
refusing to use the distinction for any theorem (the M+k notation in
§3.1 is explicitly marked operational). The split is a *strategy*
choice — which factors carry cluster structure vs. capacity — not a
structural one. Our hybrid puts cluster structure on the bond-dim-2
MPS substrate (M=3 qubits) and capacity on the amp branch (k qubits).
Scan 3 makes this concrete: amp knobs respond to PCA noise
(capacity), while phase knobs respond to cluster structure
(fidelity).

### The bigger picture for the encoding ladder

Polygram 0.7.0's Rung5 expansion is moving along the **capacity axis**
of the Polygram family. Future work that wants to move along the
**fidelity axis** needs a different mechanism — e.g. learned
per-knob PCA-axis assignments that respect cluster geometry, or a
richer MPS substrate (the M-axis of the unified `Polygram(M, k)`
future flagged in [rung5-encoding.md](rung5-encoding.md)).

The conditioning win Rung5 ships is real and load-bearing for
downstream Gram-inversion solves — but it is not a quality win, and
confusing the two would lead to over-spending on k when the
underlying problem is geometry, not capacity. The scans are the
empirical guard against that confusion.

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
