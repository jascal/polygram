# Learned axis-to-knob assignment (`LearnedKnobAssignment`)

**Date:** 2026-05-16
**Status:** shipped in polygram 0.8.0 (this change)
**Capability:** [`learned-axis-assignment`](../../openspec/changes/add-learned-axis-assignment/)
**Driver:** scan 4 of [`rung5-pareto-scans.md`](rung5-pareto-scans.md) —
greedy axis-to-knob permutation lifted decoder-Gram Spearman by ~3×
on a synthetic 64-feature SAE in seconds of search.

## Summary

`polygram.geometry.LearnedKnobAssignment` is a `KnobAssignment`
strategy that **calibrates the PCA-axis-to-polygram-knob projection
from data** instead of using the hardcoded `PC2 → α / PC3 → φ /
PC4..→ amp_knobs` permutation in `assign_phase_knobs_pca` +
`assign_amp_knobs_pca`.

| | Baseline (hardcoded) | LearnedKnobAssignment (greedy) |
|---|---|---|
| k=3 Spearman vs decoder cos² | `+0.18` | `+0.36` (Δ `+0.19`) |
| k=4 Spearman vs decoder cos² | `+0.18` | `+0.37` (Δ `+0.19`) |
| k=3 gram condition number    | `~9e18` (rank-deficient) | `~1.8e8` |
| k=4 gram condition number    | `~9e18` (rank-deficient) | `~7.0e3` |
| Search wall-clock            | n/a                       | `~6 s` |

Numbers from `examples/learned_axis_assignment_demo.py` on the
scan-4 synthetic fixture, seed 0. The committed JSON artifact lives
at [`data/learned_axis_assignment_demo.json`](data/learned_axis_assignment_demo.json).

The strategy is **strict opt-in** (default off). Existing callers
see byte-identical behaviour.

## Quick start

```python
from polygram import from_sae_lens
from polygram.geometry import LearnedKnobAssignment

# Simplest: pass `True` to use the default strategy.
dictionary, report = from_sae_lens(
    records, ids, encoding=Rung4(),
    learn_axis_assignment=True,
)
# Inspect what was learned and how much it bought:
info = report.learned_axis_assignment
print(f"baseline Spearman: {info['objective_baseline']:+.4f}")
print(f"learned  Spearman: {info['objective_value']:+.4f}")
print(f"Δ = {info['objective_value'] - info['objective_baseline']:+.4f}")
print(f"axis_assignment = {info['axis_assignment']}")

# Or pass a configured instance for control over solver / objective.
strategy = LearnedKnobAssignment(
    solver="greedy",         # or "scipy" (requires polygram[opt])
    objective=spearman_objective,
    max_axes=32,
    validation_fraction=0.0,
    early_stop_eps=1e-4,
)
dictionary, report = from_sae_lens(
    records, ids, encoding=Rung4(),
    learn_axis_assignment=strategy,
)

# Equivalent via config:
from polygram.config import SAEImportConfig
cfg = SAEImportConfig(learn_axis_assignment=True)
dictionary, report = from_sae_lens(records, ids, encoding=Rung4(), config=cfg)

# CLI:
#   polygram analyze sae.json --features 0,1,2,3 --learn-axis-assignment
```

## What's learned

Given a decoder projection matrix `P ∈ R^{N × d_model}` and an
encoding (e.g. `Rung5(n_amp_qubits=k)`), the strategy:

1. Computes the PCA of mean-centered `P` once.
2. For each polygram knob slot (α, φ, then amp pairs in qubit
   order), picks the PCA axis whose addition maximises the
   objective (default: Spearman of `|gram|²` vs decoder cos²).
3. Returns a `KnobAssignmentResult` with the chosen knob → axis map
   in `axis_assignment`, plus the achieved `objective_value` and the
   hardcoded-baseline `objective_baseline` for comparison.

The shape of `axis_assignment` depends on the solver:

- `solver="greedy"` (default) → `dict[str, int]` mapping each knob
  name to a single chosen PCA-axis index. One-to-one permutation.
- `solver="scipy"` → `dict[str, list[float]]` mapping each knob
  name to a per-axis coefficient vector. The integer applier picks
  the dominant axis (`argmax`); the full weight vector is surfaced
  for inspection.

In both cases `objective_value` is the validation-set score when
`validation_fraction > 0` and the training-set score otherwise;
`training_objective_value` always carries the training score.

The greedy solver locks each knob's axis sequentially with early-stop
on flat marginal gains. The scipy solver (opt-in, `polygram[opt]`)
runs continuous optimisation on a small linear map `W ∈ R^{n_knobs ×
n_axes}` initialised from the greedy result — useful when no single
PCA axis cleanly dominates a knob's signal (noisy real SAEs).

## Promote-to-default gate

Strict criteria for flipping `learn_axis_assignment=True` to the
default (per the openspec proposal):

1. Replication on ≥ 2 distinct real SAEs (Gemma-Scope + GPT-2-small
   or equivalent), with positive Δ on training and held-out
   validation Spearman, seeds 0 and 42.
2. ≥ 1 downstream metric improvement (post-cancellation KL on a
   behavioural prompt set, or Frobenius reconstruction fidelity).
3. Calibration cost budget — greedy ≤ 5 s / scipy ≤ 60 s at N ≤ 1000
   on commodity CPU.

Until all three hold, the strategy stays opt-in.

## Theoretical context

The strategy operates at a **different layer** than sae-forge's
compress/regrow/train cycle: the SAE's `W_dec` is fixed input;
only the import-time projection into polygram-knob space is
recalibrated. In the language of the [May 2026 theoretical
treatment](theory/polygram.pdf), the strategy learns the *parameter
map* (decoder-PCA → Bloch-angle coordinates) without touching the
*polygram manifold* itself.

§9 of the paper discusses fitting algorithms (coordinate descent,
Riemannian gradient flow). The learned-axis-assignment strategy is
a calibration variant — it's a one-pass optimisation that runs
*before* dictionary construction, not during. §11's open problem
on sample-complexity bounds for axis recovery (the analogue of
Thm 7.5's 4n+1 parameter-recovery bound for the axis-recovery
sub-problem) is the natural theoretical follow-up.

## Reproducing the headline

```bash
python examples/learned_axis_assignment_demo.py \
    --json-out docs/research/data/learned_axis_assignment_demo.json
```

Output should match the table above to within float noise (k-means
on the synth is deterministic given the seed).

## Related

- **Proposal & specs:**
  `openspec/changes/add-learned-axis-assignment/`
- **Empirical motivation:**
  [`rung5-pareto-scans.md`](rung5-pareto-scans.md) scan 4 (prototype
  using ad-hoc helpers); this change ships the production strategy
  with the same headline result.
- **Theoretical treatment:** [`theory/polygram.pdf`](theory/polygram.pdf)
  §9 (Algorithms) and §11 (Open Problems).

## Future-proofing: scaling beyond N ~ 1000 features

The greedy solver is `Θ(n_knobs × n_axes × N²)` per import — fine
for the current target regime (≤ 32 knobs, ≤ 1000 features), but
the `N²` factor is the gram-rebuild cost and dominates at large N.
If real-SAE workloads start pushing N above ~1000, a cached-PCA
or low-rank-Gram-approximation variant of `_build_analytic_gram`
would lift the headroom without changing the strategy's
public-facing semantics. Tracked as a follow-up; the scipy solver
already pays attention to dimension-aware cost via the `n_knobs ≥ 8`
branch.

## Known limitations

- **HEA_Rung2 fallback:** the strategy detects `HEA_Rung2` and
  delegates to the hardcoded helpers via `ClusteredKnobAssignment`.
  HEA's per-feature θ tensor has a different shape; learned
  assignment for it is out of scope for v1.
- **Synthetic-only headline:** the 3× lift is on a clustered
  synthetic SAE. Real-SAE replication (Gemma-Scope / GPT-2-small)
  is the next research-track follow-up; will need a torch-equipped
  host to run the behavioural validator path.
- **Greedy vs. scipy on real SAEs:** the scipy solver may pay off
  more on noisy real SAEs where no single PCA axis dominates a
  knob's signal. Empirically open; the in-repo demo only exercises
  greedy.
