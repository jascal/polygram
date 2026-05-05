## Why

Four shipped probes have settled four constraints on any compression
or disentanglement loop that consumes Polygram's predictions:

- **§4.1 / PR #18** — Polygram is a *ranker*, not a magnitude
  predictor. Spearman 0.94 against decoder cosine on the Real GPT-2
  SAE; per-pair magnitudes diverge by up to 0.44 squared-overlap
  units. Any loop must derive magnitudes from real-model metrics, not
  from `cancellation_gap` / `structural_floor` composites.
- **§4.2 / PR #20** — High decoder overlap ≠ behavioural redundancy.
  The highest-overlap pair in PR #18's selection (Polygram 0.987,
  decoder 0.992) co-fires on only ~30% of token positions where
  either fires (Jaccard 0.30). Loops must gate on real co-firing.
- **§4.3 / PR #23** — `blocks.0.hook_resid_pre` is a structural dead
  zone for ablation-KL on GPT-2 small (~5e-5 nats per single-feature
  ablation). At `blocks.5+` the per-feature ablation-KL jumps four
  orders of magnitude to ~1 nat and plateaus through `blocks.10`.
  Loops must hook at `blocks.5+`.
- **§4.4 / PR #25** — `Spearman(Polygram_overlap, Jaccard_co_fire) =
  +0.637` across 28 pairs at `blocks.10`, above the 0.6 "loop
  unblocked" threshold. Per-bucket Jaccard separates cleanly:
  mid-overlap (Polygram 0.4–0.7) → 0.145 [0.10, 0.19] vs high
  (≥0.7) → 0.621 [0.43, 0.82], non-overlapping CIs. Natural co-firing
  threshold τ ≈ 0.30 sits between the buckets.

Today these constraints live in research notes and a single-purpose
`examples/behavioural_gram_scaleup.py` script. There is no library
surface a researcher can call to apply the four constraints to *their*
Dictionary on *their* model and get back a ranked list of
"Polygram-flagged candidates that survive the behavioural gate."

The natural next step is **not** to start zeroing weights and
performing compression — that's a downstream change with much larger
blast radius. The natural next step is to ship the *read-only
validator* that runs the four-constraint pipeline end-to-end and
emits a structured report. Once the validator is in main, the
compression action becomes a follow-up change that consumes the
validator's report.

This change ships the validator.

## What Changes

### `behavioural` capability — new subpackage

Add `polygram.behavioural`, a new subpackage hosting:

- **`BehaviouralValidator`** dataclass: takes a `Dictionary`, an SAE
  checkpoint, a list of SAE feature ids matching the dictionary's
  features in order, a model name (default `"gpt2"`), a layer
  index, a prompt sequence, and four threshold knobs whose defaults
  encode the §4.4 findings.
- **`ValidationReport`** dataclass: per-pair `CandidatePair` rows
  carrying `polygram_overlap`, `decoder_overlap`, `jaccard`,
  `pearson_activation`, `kl_ablate_i`, `kl_ablate_j`,
  `kl_ratio_paired`, `n_fires_i/j/both`, `gate_pass`, plus a
  `ValidationSummary` with Spearman / Pearson and per-bucket
  Jaccard means with 95% bootstrap CIs.
- **Two-stage API**: `predict()` runs the cheap Polygram-only stage
  (no model needed; returns `CandidatePair`s with behavioural fields
  = NaN); `validate(candidates=None)` runs the expensive behavioural
  stage; `run()` is the convenience wrapper for both.
- **JSON + CSV round-trip**: `ValidationReport.to_json(path)` /
  `from_json(path)` matches the `BatchResults` pattern; `to_csv(path)`
  writes the per-pair table in the same column order
  `examples/behavioural_gram_scaleup.py` already produces, so any
  existing analysis on `docs/research/data/scaleup_pairs.csv`
  continues to work against validator output.

### `cli` capability — new `polygram validate` subcommand

`polygram validate` wraps `BehaviouralValidator.run()` with file-based
inputs (`--dictionary REF`, `--sae-checkpoint PATH`, `--feature-ids
7836,11978,...`, `--prompts PATH`) and writes JSON + optional CSV
artifacts. The subcommand stays out of the way for users who want the
Python API; it exists so that the "validate this candidate set"
operation becomes one line in a shell script for someone who isn't
writing Python.

### Optional `[behavioural]` extra

`pyproject.toml` gains a `[behavioural]` extra pulling
`torch>=2.0` + `transformers>=4.40`. The `BehaviouralValidator`
imports both lazily inside `validate()`; users who only need
`predict()` (the Polygram-only stage) can stay on the no-extras
install.

## What this proposal explicitly does NOT do

This change is the **read-only validator**. It does not:

- **Compress, merge, or modify SAE weights.** The validator emits a
  ranked candidate report; it never writes back to `W_dec`. The
  weight-modifying compression action is a separate change.
- **Train, fine-tune, or backprop through anything.** Same reason
  PR #10 (`add-sharing-graph-triage`) declined to host a
  "DisentangleExperiment" — Polygram's `from_sae_lens` path is not
  differentiable (KMeans + PCA), and the validator doesn't need
  gradients to do its job.
- **Bake in a single model architecture.** The four-constraint
  pipeline is GPT-2-small-shaped today (the `blocks.5+` cutoff is
  empirical for one model family); the validator takes `model_name`
  as a parameter so a future researcher can run it on Pythia / Llama
  / Gemma without touching the package, but the spec only ships the
  GPT-2-small path with a real test fixture. Other models work but
  ship without empirical layer-cutoff guidance.
- **Re-derive the §4.4 thresholds from the user's own data.** The
  defaults (`polygram_overlap_threshold=0.7`,
  `jaccard_threshold=0.30`) come from §4.4's bucket separation on
  GPT-2 small. Users who run the validator on other model families,
  or with a substantially different prompt distribution, may want
  per-workload calibration — the `predict()` stage's output gives
  them the data to do so without forcing a different default.
- **Add multi-Dictionary stitching.** The validator runs against a
  single Dictionary (≤8 features). The "stitch 3-4 Dictionaries for
  ~24 features" workaround the §4.4 research note named is a
  separate workhorse; this change doesn't preempt it.

## Discussion

### One vs two stages

The natural API question is whether to expose `predict()` and
`validate()` separately or fold both into a single `run()`. We expose
both because:

- The Polygram-only `predict()` stage is *cheap* (one Gram
  evaluation). The behavioural `validate()` stage is *expensive*
  (eight forward passes of GPT-2 small per ablation, ~3 minutes on
  CPU at the cap-imposed 8 features × 12 prompts).
- A user who just wants "show me which pairs Polygram thinks are
  candidates above τ" should not have to load torch, transformers,
  and the SAE checkpoint to find out.
- The two-stage pattern matches the existing `BatchExperiment` shape:
  the FeatureGraph predicts where the action is; `BatchExperiment.run()`
  pays the cost. Same separation of concerns.

`run()` is the convenience wrapper for the common case where you
want both.

### Per-feature vs per-pair ablation

§4.4's implementation pattern: run one ablation forward pass per
selected feature (N passes for N features), then aggregate per pair
from cached per-token KLs. The naive alternative — one ablation per
*pair* — would scale as N(N-1)/2 forward passes (28 vs 8 for the
§4.4 panel; quadratic vs linear). The per-feature pattern is
strictly cheaper and gives identical pair-level statistics.

The spec encodes the per-feature pattern as the contract: the
validator MAY NOT run more than `n_features` ablation forward passes.
This forces any future implementation to retain the cost guarantee.

### Layer-0 refusal default

`§4.3` settled empirically that ablation-KL at `blocks.0.hook_resid_pre`
on GPT-2 small is structural noise. The spec encodes this as a hard
default: `BehaviouralValidator.__post_init__` rejects `layer == 0`
unless the user passes `allow_layer_zero=True`. The escape hatch
exists because the dead-zone finding is GPT-2-small-specific —
researchers running the validator on a different family may
legitimately want layer 0 — but the default refuses, with the error
message pointing at the research note.

### Defaults come from §4.4

Three thresholds carry §4.4 numbers as defaults:

- `polygram_overlap_threshold = 0.7` — the lower bound of §4.4's
  "high overlap" bucket; pairs above this had mean Jaccard 0.62.
- `jaccard_threshold = 0.30` — between §4.4's mid-overlap upper
  CI (0.19) and high-overlap lower CI (0.43); admits ≥ 90% of
  high-overlap pairs while excluding ≥ 90% of mid.
- `min_firing_rate = 0.01` — matches §4.4's eligibility filter.

Defaults are explicit because **changing them silently breaks the
constraint chain**. Users should override consciously, not
absent-mindedly.
