## Why

`Compressor` currently picks cluster representatives via geometric proxies — `n_fires` (highest firing count) or `scale_aware` (decoder norm × firing count). Both are **encoding-geometry approximations of what we actually want**: the cluster member whose retention preserves the most *downstream model behaviour*.

### Where this matters and where it doesn't

Rep selection is a quality knob that **pays off in structurally feasible regimes** — when the kept-features basis has enough rank to span the host model's residual stream and the forge can plausibly recover the host's behaviour given a good basis. In those regimes, picking the cluster's behaviourally most-load-bearing rep vs. its geometrically-largest rep changes the post-forge KL meaningfully.

Rep selection does **not** rescue structurally infeasible setups. The live observation that initially motivated this proposal — sae-forge PR #33's N=32 Rung4 smoke producing KL≈7 across all K — is *not* primarily a rep_selection failure: it's a structural-feasibility failure. Forging GPT-2-small (768-dim residual) from bases of rank 1–4 produces near-random output regardless of which features are chosen. Sae-forge PR #35 (`add-forge-quality-diagnostics`) added the tooling to recognise this regime: rows tagged `quality_tier="degenerate"` are structurally doomed, and no rep_selection change moves them. The K=4-vs-K=8 KL inversion observed in that smoke is curiosity-level noise, not an empirical motivator for this change.

The honest motivator is conceptual: **the existing rep_selection options are encoding-geometry proxies for a behavioural question, and a directly-behavioural answer is sitting unused in the existing ValidationReport surface**. Whether kl_attribution Pareto-dominates `scale_aware` on real forge runs is an open research question; this change ships the option so the question can be asked.

### The data is already in scope

In `add-pareto-target-compression` (polygram 0.4.0) I deferred `rep_selection="recon_proxy"` with the note "requires Compressor to accept activations or a per-feature attribution vector — real interface widening." **That framing was wrong.** The behavioural signal we need is **already in `CandidatePair`** as `kl_ablate_i` / `kl_ablate_j` — single-feature behavioural-ablation KL values that `BehaviouralValidator` measures during its forward passes. The Compressor has full access via `validation_report.pairs`; no new interface, no new dependencies, no activation pass-through. Correcting that earlier mis-framing is part of the value of this change.

### What the change adds

`rep_selection="kl_attribution"` — opt-in third option (default stays `scale_aware`; existing callers byte-identical). Picks the cluster rep with the highest mean `kl_ablate` — the feature whose ablation hurts model behaviour the most, hence the feature the cluster can least afford to drop. The natural test bed is a sae-forge Axis-4 sweep with `add-forge-quality-diagnostics` (sae-forge PR #35) enabled, filtered to `quality_tier in {"good", "saturated"}` rows where rep choice can meaningfully move the post-forge KL.

## What Changes

### Public API

- **`CompressionConfig.rep_selection` gains a third supported value: `"kl_attribution"`**. Validated in `__post_init__`. Defaults unchanged (`"scale_aware"`); existing call paths byte-identical.
- **`_SUPPORTED_REP_SELECTIONS` in `compression/compressor.py`** extended from `("n_fires", "scale_aware")` to `("n_fires", "scale_aware", "kl_attribution")`. Validates at construction.
- **`Compressor._pick_representative` gains a `kl_attribution` branch** that dispatches to a new private `_score_kl_attribution(cluster, pair_lookup)` helper.

### Algorithm

For each cluster `C` of features touched by confirmed pairs:

1. **Per-feature behavioural-importance score.** For each `f ∈ C`, look up `kl_ablate_f` from any `CandidatePair` containing `f` (the value is single-feature, not pair-local, so all measurements of the same feature should be approximately equal across its pairs — we take the mean for noise robustness). Result: `kl_ablate(f)`.
2. **Pick the rep with maximum `kl_ablate`.** The feature whose ablation produces the largest KL divergence between the original model and the ablated-model output distribution is the one the cluster can least afford to drop.
3. **Tiebreak** on `n_fires_total` descending (more-fired features have more reliable measurements), then feature_id ascending (deterministic).
4. **NaN handling**: features whose `kl_ablate` is `NaN` (e.g. very-low-firing features the validator's KL measurement couldn't pin reliably) fall back to `scale_aware` rep scoring for that individual feature. If the *whole cluster* has all-NaN `kl_ablate` (e.g. came through `DecoderGeometryConfirmer`, which leaves behavioural fields NaN — see `add-confirmation-strategies`), `_pick_representative` raises `ValueError` with an actionable message naming the supported rep_selection values for geometry-only reports.

### What this change explicitly does NOT do

- **No activation pass-through to Compressor.** The recon-aware signal is consumed from the existing `validation_report.pairs` surface; Compressor stays pure-numpy.
- **No change to `n_fires` / `scale_aware`.** Both remain available; `kl_attribution` is opt-in via `CompressionConfig(rep_selection="kl_attribution")`.
- **No change to the default.** `scale_aware` remains the default so existing test fixtures and downstream byte-equivalence holds.
- **No new strategies for `compression.strategy` (zero / merge).** Rep selection is orthogonal to compression strategy; both `zero` and `merge` accept `kl_attribution`.
- **No new fields on `ValidationReport` or `CandidatePair`.** The existing `kl_ablate_i` / `kl_ablate_j` fields suffice.

## Capabilities

### New Capabilities

- `recon-aware-rep-selection`: documents the `kl_attribution` algorithm, NaN-handling contract, tiebreak rule, and the relationship between `CandidatePair.kl_ablate_*` and the rep score. Public API surface is `CompressionConfig(rep_selection="kl_attribution")`.

### Modified Capabilities

- `tuning-config`: `CompressionConfig.rep_selection` supported value set extended from `("n_fires", "scale_aware")` to `("n_fires", "scale_aware", "kl_attribution")`. Existing fields and defaults unchanged.

## Impact

- **Modified**:
  - `polygram/compression/compressor.py` — `_SUPPORTED_REP_SELECTIONS` extended; new `_score_kl_attribution` helper; `_pick_representative` dispatch branch; the `_pick_representative` call site in `_build_plan` is unchanged (the new branch flows through the existing extension point).
  - `polygram/config.py` — `_SUPPORTED_REP_SELECTIONS` constant in `CompressionConfig` extended to match.
- **New**: tests under `tests/test_compression_rep_selection.py` covering pick-correctness, NaN fallback, all-NaN raise, byte-identity for the default path.
- **No breaking changes**: existing `Compressor(rep_selection="scale_aware")` / `"n_fires"` paths byte-identical.
- **No new dependencies**: numpy-only, same as the rest of compression.
- **Out of scope** (future work):
  - **`rep_selection="recon_loss_direct"`** — measures per-feature reconstruction loss against a held-out activation tensor rather than reading from `kl_ablate`. Would require Compressor to accept activations; defer until empirical evidence shows `kl_attribution` is insufficient.
  - **Rep selection that joint-optimises across the cluster** (e.g. greedy minimisation of total cluster reconstruction loss). The current per-feature scoring assumes the cluster's "best rep" is well-approximated by a single-feature behavioural score; if KL-attribution-of-cluster-mean is the right loss instead, that's a follow-up.
  - **Auto-selection** of rep_selection based on what fields are populated in the report (e.g. switch to `scale_aware` automatically for geometry-only reports). Today the caller picks explicitly; auto-selection adds magic that's hard to debug.
