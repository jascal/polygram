## ADDED Requirements

### Requirement: BehaviouralValidator runs the four-constraint pipeline against a Dictionary

`polygram.behavioural.BehaviouralValidator` SHALL be a dataclass that runs the four-constraint compression-loop pipeline (Polygram-as-ranker + co-firing gate + ablation-KL impact metric + layer ≥ 1 hook) against a `Dictionary` of SAE features and emits a structured `ValidationReport`.

The dataclass exposes the following fields:

- `dictionary: Dictionary` — required. The Polygram Dictionary built
  via `from_sae_lens(records, feature_ids, ...)`. Must hold ≤
  `MAX_FEATURES_PER_DICTIONARY` features.
- `sae_checkpoint: Path` — required. On-disk `.safetensors`
  checkpoint with `W_enc / b_enc / W_dec / b_dec` tensors at the same
  `d_model` as the model named by `model_name`. The validator does
  not download.
- `feature_ids: list[int]` — required. SAE feature indices in the
  same order as `dictionary.features`. Length MUST equal
  `len(dictionary.features)`.
- `prompts: Sequence[str]` — required. Non-empty.
- `layer: int` — required. The transformer block whose `forward_pre`
  hook the validator registers. Must satisfy `layer >= 0`. By default
  `layer == 0` is rejected (see Requirement: BehaviouralValidator
  rejects layer-0 hooks unless explicitly allowed).
- `model_name: str = "gpt2"` — Hugging Face model identifier passed
  to `transformers.GPT2LMHeadModel.from_pretrained` (or future
  equivalent). Anything other than `"gpt2"` is accepted but the
  validator's threshold defaults are calibrated on GPT-2 small only.
- `polygram_overlap_threshold: float = 0.7` — pairs with predicted
  squared overlap below this are rejected by gate 1.
- `jaccard_threshold: float = 0.30` — pairs with co-firing Jaccard
  below this are rejected by gate 2.
- `min_firing_rate: float = 0.01` — features with firing rate (the
  fraction of tokens where the SAE feature has post-ReLU activation
  > 0) below this emit a `RuntimeWarning` during `validate()` but
  the run proceeds.
- `min_both_fire: int = 5` — pairs with fewer than this many
  both-fire tokens have undefined paired-KL ratio. Such pairs receive
  `kl_ratio_paired = float("nan")` and `gate_pass = False`.
- `allow_layer_zero: bool = False` — see Requirement: BehaviouralValidator
  rejects layer-0 hooks unless explicitly allowed.

`__post_init__` SHALL validate every field constraint named above and SHALL raise `ValueError` (with field name and offending value) on any violation.

### Requirement: BehaviouralValidator rejects layer-0 hooks unless explicitly allowed

When `layer == 0` and `allow_layer_zero is False`, `__post_init__` SHALL raise `ValueError` with a message that:

1. Names `layer == 0` as the offending value.
2. References `docs/research/deeper-layer-ablation-probe.md` as the
   source.
3. States the GPT-2-small-specific finding ("~5e-5 nats KL per
   single-feature ablation, four orders of magnitude below blocks.5").
4. Names `layer >= 5` (recommended `layer == 10`) as the corrective
   choice.
5. Names `allow_layer_zero=True` as the explicit override.

When `layer == 0` and `allow_layer_zero is True`, `__post_init__` SHALL emit a `RuntimeWarning` containing the same message body so the choice is visible in user logs.

When `layer < 0`, `__post_init__` SHALL raise `ValueError` unconditionally; `allow_layer_zero` does not gate negative layers.

### Requirement: predict() runs the cheap Polygram-only stage without torch

`BehaviouralValidator.predict() -> list[CandidatePair]` SHALL:

1. Compute `dictionary.gram()` and take its element-wise squared
   magnitude as the predicted squared-overlap matrix.
2. Read decoder column rows for each `feature_ids[i]` from
   `sae_checkpoint` via `polygram.load_sae_safetensors` (which is
   safetensors-only, no torch).
3. Compute decoder squared cosine for every `(i, j)` pair with
   `i < j`.
4. Return one `CandidatePair` per pair (in stable `(i, j)` ascending
   order) with:
   - `polygram_overlap`, `decoder_overlap` populated.
   - `jaccard`, `pearson_activation`, `kl_ablate_i`, `kl_ablate_j`,
     `kl_ratio_paired`, `kl_log_ratio_abs` set to `float("nan")`.
   - `n_fires_i`, `n_fires_j`, `n_both_fire`, `n_either_fire` set
     to 0.
   - `gate_pass` set to `False`.

`predict()` SHALL NOT import torch or transformers. Calling `predict()` on a system without those installed MUST succeed.

### Requirement: validate() runs the behavioural stage and emits a ValidationReport

`BehaviouralValidator.validate(candidates: list[CandidatePair] | None = None) -> ValidationReport` SHALL:

1. When `candidates is None`, default to `self.predict()`.
2. Lazy-import torch and transformers via
   `polygram.behavioural.runtime._import_torch_and_transformers`.
   On `ImportError`, raise with a hint pointing at
   `pip install polygram[behavioural]`.
3. Load `model_name` via `transformers.GPT2LMHeadModel.from_pretrained`
   (or the model-family-appropriate auto-class for non-GPT-2
   `model_name` values), put it in `eval()` mode, freeze gradients.
4. Register a single `forward_pre_hook` on
   `model.transformer.h[layer]` (or the equivalent block-input hook
   for non-GPT-2 architectures) that captures the residual stream
   on baseline forwards. Forward each prompt; collect residuals and
   next-token baseline logits.
5. Encode the captured residuals through the SAE (`f =
   relu((x - b_dec) @ W_enc + b_enc)`) for every feature in
   `self.feature_ids`. Compute per-feature firing rates. For any
   feature with rate below `self.min_firing_rate`, emit a
   `RuntimeWarning` naming the feature id and the rate.
6. Run exactly `len(self.feature_ids)` ablation forward-pass-batches:
   one per feature. Each batch loops over prompts, registers a
   forward-pre-hook on the same block that subtracts the feature's
   `f · W_dec[fid, :]` contribution at every token where it fires,
   forwards the prompt, captures the ablated logits, and records
   per-token KL between baseline and ablated next-token
   distributions. The validator MUST NOT run more than
   `len(self.feature_ids)` ablation batches.
7. For every input candidate `(i, j)`, populate the behavioural
   fields per Requirement: per-pair statistics carry the §4.4 schema.
8. Compute the summary (Spearman / Pearson / per-bucket Jaccard /
   outcome) per Requirement: ValidationSummary aggregates across
   the candidate set.
9. Assemble and return a `ValidationReport` whose `confirmed` field
   is `[(p.i, p.j) for p in pairs if p.gate_pass]` in the same
   order as `pairs`.

### Requirement: per-pair statistics carry the §4.4 schema

For every `(i, j)` candidate, `validate()` SHALL populate the `CandidatePair` fields as follows:

- `jaccard = n_both_fire / max(n_either_fire, 1)`, where "fires"
  means post-ReLU activation > 0.
- `pearson_activation = np.corrcoef(act_i, act_j)[0, 1]`, with NaN
  when either activation has zero variance.
- `kl_ablate_i = mean(KL(baseline || ablate_i)[t] for t where
  feature i fires)`. NaN if i never fires. Same shape for j.
- `kl_ratio_paired = kl_ablate_i_on_both / kl_ablate_j_on_both` when
  `n_both_fire >= self.min_both_fire` AND both both-fire KL means
  are positive. NaN otherwise.
- `kl_log_ratio_abs = abs(log(kl_ratio_paired))` when
  `kl_ratio_paired` is finite and positive. NaN otherwise.
- `gate_pass = True` iff all three of:
  - `polygram_overlap >= self.polygram_overlap_threshold`,
  - `jaccard >= self.jaccard_threshold`,
  - `n_both_fire >= self.min_both_fire`.

KL values SHALL be clamped at 0 from below (the algebraic guarantee may be violated by float32 noise on near-identical distributions).

### Requirement: ValidationSummary aggregates across the candidate set

`ValidationSummary` SHALL carry the following fields, computed across the full candidate set:

- `spearman_polygram_jaccard: float` — Spearman rank correlation
  between `polygram_overlap` and `jaccard` columns.
- `spearman_decoder_jaccard: float` — same against `decoder_overlap`.
- `spearman_polygram_log_kl_abs: float` — same against
  `kl_log_ratio_abs`, computed only on pairs with finite
  `kl_log_ratio_abs`.
- `pearson_polygram_jaccard: float`, `pearson_decoder_jaccard: float`.
- `buckets: dict[str, BucketStats]` with three keys:
  `"low_overlap"` (Polygram ≤ 0.4), `"mid_overlap"` (0.4 < polygram
  < 0.7), `"high_overlap"` (≥ 0.7). Each `BucketStats` carries
  `polygram_range: str`, `n_pairs: int`, `jaccard_mean: float`,
  `jaccard_ci_95: tuple[float, float]` (1000-resample bootstrap CI,
  RNG seed = 0). Empty buckets emit NaN means and (NaN, NaN) CIs.
- `outcome: str` — one of:
  - `"high_spearman_loop_unblocked"` when
    `spearman_polygram_jaccard >= 0.6`,
  - `"medium_spearman_loop_needs_calibration"` when
    `0.3 <= spearman_polygram_jaccard < 0.6`,
  - `"low_spearman_loop_blocked"` when
    `spearman_polygram_jaccard < 0.3`,
  - `"undefined"` when `spearman_polygram_jaccard` is NaN.

The thresholds 0.3 and 0.6 are the same gate values §4.4 used; they are part of the spec, not a knob.

### Requirement: ValidationReport supports JSON round-trip

`ValidationReport.to_json(path)` SHALL:

1. Write a deterministic JSON representation with pairs sorted by
   ascending `(i, j)`.
2. Format every float via `format(v, ".6g")` then re-parse, matching
   `BatchResults.to_json`.
3. Preserve `None` and `float("nan")` as JSON `null`.
4. Include the schema named in `design.md` Decision 6 verbatim.

`ValidationReport.from_json(path) -> ValidationReport` SHALL be the inverse of `to_json`. The round-trip property `from_json(to_json(r)) == r` SHALL hold for any `r` reachable from `BehaviouralValidator.run()` after canonicalizing NaN equality (NaN compares equal to NaN under the report's custom equality).

### Requirement: ValidationReport supports CSV emission for re-analysis

`ValidationReport.to_csv(path)` SHALL emit a CSV with columns in this order:

```
i, j, polygram_overlap, decoder_overlap, jaccard, pearson_activation,
n_fires_i, n_fires_j, n_both_fire, n_either_fire,
kl_ablate_i_on_both_fire, kl_ablate_j_on_both_fire,
kl_ratio_i_over_j, kl_log_ratio_abs, gate_pass
```

The first 14 columns and their order SHALL match `docs/research/data/scaleup_pairs.csv`. The last column `gate_pass` is new; it is written as the literal string `true` or `false`. Existing analysis written against the §4.4 CSV continues to load validator output without change.

### Requirement: run() is the convenience wrapper for predict() + validate()

`BehaviouralValidator.run() -> ValidationReport` SHALL be exactly equivalent to `self.validate(self.predict())`.

### Requirement: validator caps ablation cost at one forward batch per feature

`validate()` MUST NOT run more than `len(self.feature_ids)` ablation forward-pass-batches. Pair-level KL statistics MUST be derived from cached per-token KL arrays produced by the per-feature batches; the validator MUST NOT run a separate forward pass per pair.

### Requirement: behavioural extra is optional

The `polygram.behavioural` subpackage SHALL be importable without `torch` or `transformers` installed. Calling `validate()` (or `run()`, which calls `validate()`) on a system without those installed SHALL raise `ImportError` with a message naming `pip install polygram[behavioural]` as the resolution. `predict()` SHALL succeed without those installed.
