## Context

The `examples/behavioural_gram_scaleup.py` shipped in PR #25 already
contains every piece of the validator pipeline: SAE residual capture
hook, full-SAE encode for firing-rate selection, projection-similarity
selection, `from_sae_lens` Dictionary build, per-feature ablation hook,
per-pair Jaccard / Pearson / KL-ratio computation, Spearman + Pearson
+ per-bucket aggregation, CSV emission. The script's job is "spike
this once, write up the findings, settle the constraint."

The validator's job is "be the public surface that lets a researcher
run that pipeline against their dictionary and get back a structured
report." The pieces are the same; the shape is different:

- The script hard-codes feature selection (a seed-stratified strategy
  appropriate for the §4.4 calibration question). The validator takes
  feature selection as an *input* — the user has already decided which
  features matter, often via the `polygram.analysis` triage layer or
  a paper's published feature list.
- The script writes one CSV and prints to stdout. The validator
  emits a structured `ValidationReport` (JSON + CSV round-trip) that
  downstream tools (a future compression action, a graph renderer)
  can consume programmatically.
- The script runs once. The validator's `predict()` stage runs in
  milliseconds and is intended to be called repeatedly (e.g., as a
  user iterates threshold knobs).

## Goals / Non-Goals

**Goals:**

- A `polygram.behavioural.BehaviouralValidator` that runs the §4.4
  pipeline end-to-end and emits a structured `ValidationReport`.
- Two-stage API: `predict()` (cheap, no torch) and `validate()`
  (expensive, lazy torch import) with `run()` as the convenience
  wrapper.
- JSON + CSV round-trip on `ValidationReport`, matching `BatchResults`
  for the JSON shape and the §4.4 CSV column layout for the CSV.
- A `polygram validate` CLI subcommand that wraps `run()` with
  file-based inputs.
- One end-to-end test that exercises the full pipeline on a real SAE
  checkpoint when present, with the same skip pattern as §4.2 / §4.3
  / §4.4 smoke tests when the checkpoint or torch is absent.

**Non-Goals:**

- Running the validator on multiple models simultaneously.
- Compressing weights, training, fine-tuning, or backpropagating.
- Multi-Dictionary stitching for > 8 features.
- Auto-discovering which layer to hook on a new model. The
  `blocks.5+` finding is GPT-2-small-specific; `layer` is a required
  user input. Defaults exist only for the model name (`"gpt2"`).

## Decisions

### Decision 1 — New `polygram.behavioural` subpackage; new `[behavioural]` extra

Rejected alternatives:

- **Add the validator to `polygram.analysis`.** Misleading; analysis
  is the closed-form triage layer (cheap, no model). Behavioural is
  the validation layer (expensive, model-bound). The two have
  different runtime cost profiles and different optional-extra
  dependencies; conflating them obscures both.
- **Top-level `polygram.validator`.** Too generic; "validation" is
  q-orca's word for state-machine soundness. `behavioural` is the
  word the research notes already use.

Choice: new `polygram.behavioural/` subpackage with
`validator.py`, `report.py`, `runtime.py` modules. New `[behavioural]`
extra pulls `torch>=2.0` + `transformers>=4.40`; both imports are
lazy inside `validate()`.

### Decision 2 — Two-stage API (`predict` then `validate`)

Public methods:

```python
def predict(self) -> list[CandidatePair]:  # No torch needed.
    "Polygram-only stage: compute predicted Gram, identify pairs."

def validate(
    self, candidates: list[CandidatePair] | None = None
) -> ValidationReport:
    "Behavioural stage: run forward + ablation passes; gate; aggregate."

def run(self) -> ValidationReport:  # Convenience wrapper.
    "predict() + validate() in one call."
```

`predict()` returns one `CandidatePair` per *all* pairs (not just
those above the threshold), with behavioural fields = `float("nan")`
and `gate_pass = False`. The Polygram threshold filter is applied
inside `validate()`, not `predict()`, because:

- `predict()` is observation-only — it shows you the geometry the
  Dictionary predicts, regardless of where the threshold sits.
- A user iterating on threshold values should not need to re-run
  `predict()` between iterations.

`validate()` accepts an optional `candidates` argument so a user can
hand-edit the candidate list (e.g., "skip pair (i,j); I already
know it's a label collision") between stages.

### Decision 3 — Per-feature ablation, per-pair aggregation; cap encoded as a contract

The validator MUST run no more than `len(self.feature_ids)` ablation
forward passes (one per selected feature). The pair-level KL ratio
on both-fire tokens is derived from the cached per-token KL arrays,
not from per-pair ablation runs.

Why this is a contract not a recommendation: the alternative (one
ablation per pair) scales as N²/2; for N=8 that's 28 forward passes
vs 8, ~3.5× wall-clock cost. For larger panels (when multi-Dictionary
stitching lands) the gap compounds. Encoding the per-feature pattern
as a contract prevents quiet regressions.

### Decision 4 — Layer 0 refused by default; `allow_layer_zero=True` escape hatch

`__post_init__` raises `ValueError` if `layer == 0` and
`allow_layer_zero is False`. The error message names the research
note path:

```
BehaviouralValidator: layer 0 is the structural dead zone for
GPT-2 small (per docs/research/deeper-layer-ablation-probe.md
finding: ~5e-5 nats KL per single-feature ablation, four orders of
magnitude below blocks.5). Use layer >= 5 (recommended: 10), or
pass allow_layer_zero=True if your model family has been
empirically shown to be informative at layer 0.
```

`layer < 0` raises `ValueError` unconditionally; `layer == 0` with
the override produces a runtime warning (not an error) so the
researcher sees a paper-trail of the choice in their logs.

### Decision 5 — Defaults from §4.4, no auto-tuning

All four thresholds carry §4.4 numbers as defaults:

| Field | Default | Source |
|:---|---:|:---|
| `polygram_overlap_threshold` | 0.7 | §4.4 high-bucket lower bound |
| `jaccard_threshold` | 0.30 | between §4.4 buckets' CIs |
| `min_firing_rate` | 0.01 | §4.4 eligibility filter |
| `min_both_fire` | 5 | §4.4 KL-ratio definability gate |

The validator does not auto-tune thresholds from the user's data.
Auto-tuning would silently couple the validator's gate to whatever
distribution the user supplied, which is exactly the failure mode
the §4.4 calibration was meant to prevent. Users who want different
thresholds set them explicitly.

### Decision 6 — `ValidationReport.to_json` matches `BatchResults` round-trip pattern

JSON layout:

```json
{
  "schema_version": 1,
  "dictionary_name": "...",
  "model_name": "gpt2",
  "layer": 10,
  "n_prompts": 12,
  "n_tokens": 654,
  "polygram_overlap_threshold": 0.7,
  "jaccard_threshold": 0.30,
  "min_firing_rate": 0.01,
  "min_both_fire": 5,
  "feature_ids": [12999, 19398, ...],
  "pairs": [
    {
      "i": 12999, "j": 19398,
      "polygram_overlap": 0.9939,
      "decoder_overlap": 0.3639,
      "jaccard": 0.5072,
      "pearson_activation": 0.9893,
      "kl_ablate_i": 0.2364,
      "kl_ablate_j": 0.1293,
      "kl_ratio_paired": 1.8296,
      "kl_log_ratio_abs": 0.6041,
      "n_fires_i": 311,
      "n_fires_j": 215,
      "n_both_fire": 177,
      "n_either_fire": 349,
      "gate_pass": true
    },
    ...
  ],
  "summary": {
    "spearman_polygram_jaccard": 0.6371,
    "spearman_decoder_jaccard": -0.0542,
    "spearman_polygram_log_kl_abs": -0.3295,
    "pearson_polygram_jaccard": 0.6943,
    "pearson_decoder_jaccard": -0.0753,
    "buckets": {
      "low_overlap":  {"polygram_range": "≤ 0.4",        "n_pairs": 0,  "jaccard_mean": null, "jaccard_ci_95": [null, null]},
      "mid_overlap":  {"polygram_range": "(0.4, 0.7)",   "n_pairs": 16, "jaccard_mean": 0.1445, "jaccard_ci_95": [0.0963, 0.1924]},
      "high_overlap": {"polygram_range": "≥ 0.7",        "n_pairs": 12, "jaccard_mean": 0.6209, "jaccard_ci_95": [0.4267, 0.8228]}
    },
    "outcome": "high_spearman_loop_unblocked"
  },
  "confirmed": [[12999, 19398], ...]
}
```

Round-trip: `ValidationReport.from_json(report.to_json(...))` is
equality-true. Floats formatted to 6 sig figs via `format(v, ".6g")`
then reparsed (matches `BatchResults`).

### Decision 7 — CSV column layout matches `examples/behavioural_gram_scaleup.py`

The CSV emitted by `ValidationReport.to_csv(path)` uses the same
column order the spike already uses in
`docs/research/data/scaleup_pairs.csv`:

```
i,j,polygram_overlap,decoder_overlap,jaccard,pearson_activation,
n_fires_i,n_fires_j,n_both_fire,n_either_fire,
kl_ablate_i_on_both_fire,kl_ablate_j_on_both_fire,
kl_ratio_i_over_j,kl_log_ratio_abs
```

Plus one new column `gate_pass` (bool, written as `true`/`false`).
Existing analysis on the scaleup CSV continues to work; new analysis
gets the gate column for free.

### Decision 8 — CLI: `polygram validate` with file-based inputs

```
polygram validate \
  --dictionary path/to/dictionary.json \
  --sae-checkpoint path/to/sae_weights.safetensors \
  --feature-ids 12999,19398,4192,23625,8371,2287,68,13737 \
  --layer 10 \
  --prompts path/to/prompts.txt \
  --polygram-threshold 0.7 \
  --jaccard-threshold 0.30 \
  --min-firing-rate 0.01 \
  --output path/to/validation.json \
  --csv path/to/validation_pairs.csv
```

`--dictionary` accepts a JSON file in the existing toy-SAE schema
(the `polygram analyze` chain already produces this). `--prompts` is
a text file with one prompt per non-empty line; lines starting with
`#` are ignored as comments. `--feature-ids` is comma-separated; the
list length must match the dictionary's feature count.

The CLI does **not** support a non-default `--model`; passing one
emits a warning that the validator's empirical defaults
(`blocks.5+`, threshold values) were calibrated on GPT-2 small.

### Decision 9 — One end-to-end test, real-checkpoint-aware

Mirroring §4.2 / §4.3 / §4.4: a single smoke test
`test_behavioural_validator_smoke` exercises `predict()` + `validate()`
with a tiny `n_prompts=1, n_features=4` configuration when the SAE
checkpoint is present, and falls back to a clean skip when either
the checkpoint or torch is absent. No new fixtures are checked in.

The validator gets one additional round-trip test
`test_validation_report_json_roundtrip` that builds a fixture
`ValidationReport` from in-memory data (no model needed, no SAE) and
asserts `from_json(to_json(r)) == r`.
