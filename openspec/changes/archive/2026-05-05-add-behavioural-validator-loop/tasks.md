# add-behavioural-validator-loop — tasks

## 1. Subpackage scaffold + report types

- [ ] 1.1 New subpackage `polygram/behavioural/` with
      `__init__.py`, `validator.py`, `report.py`, `runtime.py`.
- [ ] 1.2 `CandidatePair` frozen dataclass in
      `polygram/behavioural/report.py` with the field set named in
      the design (`i`, `j`, `polygram_overlap`, `decoder_overlap`,
      `jaccard`, `pearson_activation`, `kl_ablate_i`, `kl_ablate_j`,
      `kl_ratio_paired`, `kl_log_ratio_abs`, `n_fires_i`,
      `n_fires_j`, `n_both_fire`, `n_either_fire`, `gate_pass`).
- [ ] 1.3 `ValidationSummary` frozen dataclass with the field set
      named in the design (`spearman_*`, `pearson_*`, `buckets`,
      `outcome`).
- [ ] 1.4 `ValidationReport` frozen dataclass with the field set
      named in the design (`schema_version`, `dictionary_name`,
      `model_name`, `layer`, `n_prompts`, `n_tokens`, threshold
      knobs, `feature_ids`, `pairs`, `summary`, `confirmed`).
- [ ] 1.5 `ValidationReport.to_json(path)` — deterministic ordering
      (pairs sorted by `(i, j)`); floats formatted to 6 sig figs;
      `None` preserved as JSON null; matches the JSON layout in
      `design.md`.
- [ ] 1.6 `ValidationReport.from_json(path) -> ValidationReport`
      round-trip helper. Round-trip property: `from_json(to_json(r))
      == r` for any `r` reachable from `BehaviouralValidator.run()`.
- [ ] 1.7 `ValidationReport.to_csv(path)` — emits the column layout
      named in `design.md` Decision 7 (matches
      `docs/research/data/scaleup_pairs.csv` plus `gate_pass`).
- [ ] 1.8 Public exports added to `polygram/__init__.py`
      (`BehaviouralValidator`, `ValidationReport`, `ValidationSummary`,
      `CandidatePair`).

## 2. BehaviouralValidator — predict() stage (cheap)

- [ ] 2.1 `BehaviouralValidator` dataclass with the field set named
      in the design (`dictionary`, `sae_checkpoint`, `feature_ids`,
      `prompts`, `layer`, `model_name="gpt2"`,
      `polygram_overlap_threshold=0.7`, `jaccard_threshold=0.30`,
      `min_firing_rate=0.01`, `min_both_fire=5`,
      `allow_layer_zero=False`).
- [ ] 2.2 `__post_init__` validation:
      - `len(feature_ids) == len(dictionary.features)` else
        `ValueError`.
      - `len(feature_ids) <= MAX_FEATURES_PER_DICTIONARY` else
        `ValueError` (echoes the `from_sae_lens` check eagerly).
      - All threshold knobs in `[0, 1]` else `ValueError`.
      - `min_both_fire >= 1` else `ValueError`.
      - `layer < 0` raises unconditionally.
      - `layer == 0` and `allow_layer_zero is False` raises with the
        exact message named in `design.md` Decision 4.
      - `layer == 0` and `allow_layer_zero is True` issues
        `RuntimeWarning` with the same message.
      - `prompts` empty raises.
      - `sae_checkpoint` not on disk raises (the validator does not
        download).
- [ ] 2.3 `predict() -> list[CandidatePair]` — computes Polygram's
      predicted Gram via `dictionary.gram()`, computes decoder
      squared cosine via the SAE checkpoint's `W_dec` rows for each
      feature_id, returns one `CandidatePair` per `(i, j)` pair
      with `i < j`, with `jaccard / pearson_activation / kl_*` =
      NaN and `gate_pass = False`. SHALL NOT load torch or
      transformers.

## 3. BehaviouralValidator — validate() stage (expensive)

- [ ] 3.1 `polygram/behavioural/runtime.py` — lazy imports of torch
      and transformers behind `_import_torch_and_transformers()`,
      same shape as `examples/behavioural_gram_scaleup.py`. Raises
      `ImportError` with the install hint on missing extras.
- [ ] 3.2 `validate(candidates: list[CandidatePair] | None = None)
      -> ValidationReport` — defaults `candidates` to `predict()`.
      Loads model + tokenizer, hooks `model.transformer.h[layer]`
      (capture pre-hook), forwards every prompt, captures residuals
      and baseline next-token logits.
- [ ] 3.3 SAE encode pass — encode captured residuals through the
      checkpoint's `W_enc / b_enc / b_dec` for every feature in
      `feature_ids`. Compute per-feature firing rates; if *any*
      feature has firing rate < `min_firing_rate`, emit a
      `RuntimeWarning` naming the feature(s) but still proceed
      (low-firing features just produce NaN-or-near-zero Jaccard
      rows, which is the user's signal to revise selection).
- [ ] 3.4 Per-feature ablation pass — for each `fid` in
      `feature_ids`, register a forward-pre-hook on
      `model.transformer.h[layer]` that subtracts
      `f_fid · W_dec[fid, :]` at every token where `fid` fires; run
      one forward pass per prompt; compute per-token KL between
      baseline next-token logits and the ablated logits. Cache the
      per-token KL array per feature. SHALL run exactly
      `len(feature_ids)` ablation forward-pass-batches (one per
      feature; each batch loops over prompts).
- [ ] 3.5 Per-pair aggregation — for every input candidate
      `(i, j)`, compute `jaccard`, `pearson_activation`,
      `kl_ablate_i / kl_ablate_j` (mean on the *firing-feature's*
      tokens), `kl_ratio_paired` and `kl_log_ratio_abs` on
      both-fire tokens with `n_both_fire >= min_both_fire`, populate
      `n_*` counters. `gate_pass = (polygram >= polygram_threshold)
      and (jaccard >= jaccard_threshold) and (n_both_fire >=
      min_both_fire)`.
- [ ] 3.6 Summary aggregation — Spearman + Pearson between the
      pair-set's Polygram / decoder / Jaccard / log-KL columns;
      per-bucket Jaccard means with 95% bootstrap CIs (1000
      resamples, seed=0); outcome bucket
      (`high_spearman_loop_unblocked` for Spearman ≥ 0.6;
      `medium_spearman_loop_needs_calibration` for [0.3, 0.6);
      `low_spearman_loop_blocked` for < 0.3; `undefined` for NaN).
- [ ] 3.7 Assemble and return `ValidationReport`. `confirmed` is
      `[(p.i, p.j) for p in pairs if p.gate_pass]`.
- [ ] 3.8 `run() -> ValidationReport` — `validate(predict())`.

## 4. CLI — `polygram validate` subcommand

- [ ] 4.1 New subcommand registered in `polygram/cli.py`:
      `polygram validate` with the flag set named in `design.md`
      Decision 8.
- [ ] 4.2 `--dictionary` accepts a JSON file in the toy-SAE schema
      (loaded via existing `load_toy_sae`); the `Dictionary` is
      built via the same `from_sae_lens` path the rest of the CLI
      uses.
- [ ] 4.3 `--feature-ids` comma-separated parser; rejects mismatch
      with dictionary feature count with a non-zero exit code.
- [ ] 4.4 `--prompts` parser — reads the file, strips empty lines
      and `#`-prefixed lines.
- [ ] 4.5 `--model` defaults to `"gpt2"`; warns to stderr when the
      user passes anything else (the empirical defaults are
      GPT-2-small-specific).
- [ ] 4.6 Writes JSON to `--output`; if `--csv` is set, also writes
      CSV. Honest progress: prints one line per major stage
      (`predict → validate: loading model → forwarding prompts → SAE
      encode → ablation 1/N ... → aggregation → done`).

## 5. Optional `[behavioural]` extra

- [ ] 5.1 `pyproject.toml` adds a `[behavioural]` extra pulling
      `torch>=2.0` and `transformers>=4.40`. README's "Optional
      extras" section gains a one-line entry.
- [ ] 5.2 `polygram/behavioural/runtime.py`'s
      `_import_torch_and_transformers()` raises `ImportError` with a
      hint pointing at `pip install polygram[behavioural]` on
      missing imports.

## 6. Tests

- [ ] 6.1 `tests/behavioural/test_validator_predict.py` — unit
      tests for the cheap stage. Build a tiny Dictionary +
      synthesized SAE checkpoint via the existing
      `examples/sae_safetensors.py` fixture path; assert
      `predict()` returns the right number of pairs (N(N-1)/2),
      Polygram and decoder overlaps in `[0, 1]`, behavioural
      fields all NaN, `gate_pass=False`. No torch needed.
- [ ] 6.2 `tests/behavioural/test_report_roundtrip.py` —
      `ValidationReport.from_json(r.to_json()) == r` on a
      hand-built fixture report (no model, no SAE).
- [ ] 6.3 `tests/behavioural/test_validator_postinit.py` — exercise
      every `__post_init__` rejection path named in §2.2.
- [ ] 6.4 `tests/test_examples.py` gains
      `test_behavioural_validator_smoke` mirroring the §4.4 smoke:
      tiny `n_prompts=1, n_features=4` configuration; success path
      asserts the report banner; skip path asserts the standard
      checkpoint-missing message.
- [ ] 6.5 CLI smoke test in `tests/cli/test_validate_cli.py` —
      exercises argument parsing + dictionary loading + skip path
      when SAE is absent (matches the existing CLI test pattern).

## 7. Worked example + research note

- [ ] 7.1 New `examples/behavioural_validate.py` showing the full
      workflow: build a Dictionary from the §4.4 selection, run the
      validator, dump JSON + CSV, print confirmed candidates. Should
      reproduce the §4.4 numbers byte-for-byte (the validator and
      the §4.4 spike share a pipeline).
- [ ] 7.2 New `docs/research/behavioural-validator-design.md` (one
      page) — mostly a pointer to this change's `design.md` plus
      a paragraph on why the validator was the right scope to ship
      next vs jumping to the compression action.
- [ ] 7.3 Update `tech-debt-backlog/tasks.md` §4.5+ stub: fold the
      "loop spec writes itself" sentence in §4.4's closure block
      into a §5 stub naming the validator change as the loop's
      first half.

## 8. Closing

- [ ] 8.1 Add the new subpackage to the README's "Library tour"
      section (one paragraph + link to `examples/behavioural_validate.py`).
- [ ] 8.2 Run the full test suite end-to-end. CI green.
- [ ] 8.3 Squash-merge to main; archive this change directory under
      `openspec/changes/archive/`.
