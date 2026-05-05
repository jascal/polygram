# add-compression-regrow — tasks

## 1. Subpackage scaffold + report types

- [ ] 1.1 New module `polygram/compression/regrow.py` for the
      `Regrower` dataclass and the strategy dispatcher.
- [ ] 1.2 New module `polygram/compression/regrow_report.py` for
      the report types (`SlotPopulation`, `RegrowPlan`,
      `RegrowReport`, `RegrowResult`). Mirrors the JSON-shape
      conventions of `report.py`: six-sigfig float formatting via
      `format(v, ".6g")`, sorted-key serialization, NaN-aware
      equality.
- [ ] 1.3 New module `polygram/compression/strategies/residual_kmeans.py`
      for the implemented strategy. Uses `sklearn.cluster.KMeans`
      with the parameters fixed in `design.md` Decision 2
      (`n_init` configurable; `algorithm='lloyd'` pinned;
      `random_state=seed`).
- [ ] 1.4 `SlotPopulation` frozen dataclass: `feature_id: int`,
      `cluster_size: int`, `decoder_norm: float`,
      `encoder_norm: float`.
- [ ] 1.5 `RegrowPlan` frozen dataclass: `strategy: str`,
      `n_residual_tokens: int`, `zeroed_input: tuple[int, ...]`,
      `feature_ids: tuple[int, ...]`, `slots:
      tuple[SlotPopulation, ...]`.
- [ ] 1.6 `RegrowReport` frozen dataclass with the field set
      named in `design.md` Decision 8 (`schema_version`,
      `source_checkpoint`, `source_checkpoint_sha256`,
      `output_checkpoint`, `output_checkpoint_sha256`,
      `strategy`, `n_slots_repopulated: int`,
      `n_slots_left_zero: int`, `plan: RegrowPlan`,
      `strategy_params: dict[str, int | float]`,
      `provenance: dict[str, str]`).
- [ ] 1.7 `RegrowResult` frozen dataclass: `plan`, `report`,
      `output_checkpoint: Path`, `dictionary: Dictionary`.
- [ ] 1.8 `RegrowReport.to_json(path)` / `from_json(path)`
      round-trip matching the `CompressionReport` schema
      discipline.
- [ ] 1.9 Public exports added to
      `polygram/compression/__init__.py` and
      `polygram/__init__.py` (`Regrower`, `RegrowPlan`,
      `RegrowReport`, `RegrowResult`, `SlotPopulation`,
      `RegrowStrategy`).

## 2. Strategy dispatcher + residual extraction

- [ ] 2.1 `RegrowStrategy` `StrEnum` with three members
      (`residual_kmeans`, `high_decoder_norm_random`,
      `orthogonal_noise_scaled`). Only `residual_kmeans` has an
      implementation body; the other two raise
      `NotImplementedError("strategy <name> reserved for a
      future change; supply 'residual_kmeans'")`.
- [ ] 2.2 Helper `_extract_residual_stream(state_dict, residuals)
      -> np.ndarray` — runs the SAE forward pass on cached
      residuals per `design.md` Decision 2 step 1. Pure numpy,
      no torch.
- [ ] 2.3 Helper `_capture_residuals(prompts, model_name, layer,
      device) -> np.ndarray` — runs ONE GPT-2 forward per prompt
      with a `forward_pre_hook` at `model.transformer.h[layer]`,
      returns concatenated residuals. Lazy-imports torch via
      `_import_torch_and_transformers`; resolves device via
      `_resolve_device`. Mirrors the validator's existing capture
      flow.
- [ ] 2.4 Strategy implementation `_apply_residual_kmeans(
      state_dict, zeroed_sorted, residual_stream, seed, n_init)
      -> tuple[dict[str, np.ndarray], list[SlotPopulation]]`
      following `design.md` Decision 2 steps 2–5. Empty-cluster
      handling: assigned slots whose `cluster_size == 0` are left
      zero in the rewritten state-dict.

## 3. Regrower — plan() stage

- [ ] 3.1 `Regrower` dataclass with the field set named in
      `proposal.md`'s What Changes section.
- [ ] 3.2 `__post_init__` validation per `design.md` Decision 3:
      - exactly one of `prompts`, `cached_residuals` (XOR);
      - `sae_checkpoint` exists;
      - `strategy` is in `RegrowStrategy`;
      - `seed >= 0`, `n_init >= 1`;
      - `prompts` (if supplied) is non-empty, `layer >= 0`;
      - `zeroed` is a set of non-negative ints all in
        `[0, n_features)` per the SAE's W_dec shape (read from
        safetensors metadata, no full file load).
- [ ] 3.3 `from_compression_report` classmethod per
      `design.md` Decision 3: extracts `zeroed = union over
      report.plan.clusters[*].zeroed`, populates
      `_provenance` dict on the instance.
- [ ] 3.4 `plan() -> RegrowPlan`:
      1. Resolve residuals: `cached_residuals` if supplied,
         else call `_capture_residuals`.
      2. Pre-flight checks per `design.md` Decision 6:
         residual std < 1e-9 → raise `RuntimeError`;
         n_residual_tokens < K → raise `ValueError`.
      3. Compute residual stream via `_extract_residual_stream`.
      4. Run strategy's planning sub-step (k-means for
         `residual_kmeans`); capture per-slot diagnostics.
      5. Return `RegrowPlan` with deterministic slot ordering
         (sorted by `feature_id` ascending).
- [ ] 3.5 Caching: `plan()` is idempotent; calling twice on the
      same `Regrower` returns the same `RegrowPlan` (cached
      internally via `_cached_plan`).

## 4. Regrower — apply() stage

- [ ] 4.1 `apply(plan=None, output_checkpoint=...) ->
      RegrowResult` per `design.md` Decision 4:
      - reject `output_checkpoint == self.sae_checkpoint`;
      - load source state-dict via `safetensors.numpy.load_file`;
      - run strategy's apply sub-step on a copy of the
        state-dict;
      - write atomically (temp file + `os.replace`);
      - compute output sha256;
      - rebuild `Dictionary` via `from_sae_lens(load_sae_safetensors(
        output_checkpoint, feature_ids=plan.feature_ids), ...)`.
- [ ] 4.2 SHA256 helper reused from `compressor.py`; consider
      lifting to `polygram/compression/_hash.py` so both
      modules share it.
- [ ] 4.3 `run(output_checkpoint) -> RegrowResult` —
      `apply(plan(), output_checkpoint=output_checkpoint)`.
- [ ] 4.4 Cleanup on exception: temp file unlinked best-effort.

## 5. CLI — `polygram regrow` subcommand

- [ ] 5.1 New subparser registered in `polygram/cli.py`:
      `polygram regrow` with the flag set named in
      `proposal.md`'s What Changes section.
- [ ] 5.2 Mutually-exclusive groups via `argparse`:
      `--zeroed-list` vs `--compression-report` (one required);
      `--prompts` vs `--cached-residuals` (one required when
      `--prompts` is also accompanied by `--layer` / `--model`).
- [ ] 5.3 `--zeroed-list` parser: comma-separated ints; reuses
      the `_parse_feature_ids` helper.
- [ ] 5.4 `--compression-report` loads via
      `CompressionReport.from_json(path)`; on parse failure,
      exit 2 with a clear message.
- [ ] 5.5 `--cached-residuals` loads via `np.load(path)`; on
      load failure or wrong dtype/shape (must be 2D float32),
      exit 2.
- [ ] 5.6 Stage progress to stderr per `cli/spec.md`: load
      report/zeroed → load checkpoint → capture or load
      residuals → run k-means → rewrite → write report. Final
      line includes truncated source + output sha256.
- [ ] 5.7 Exit code 2 on missing input files, output =
      source path collision, malformed `--zeroed-list`, both
      `--zeroed-list` and `--compression-report` supplied,
      both `--prompts` and `--cached-residuals` supplied,
      strategy not in supported set.

## 6. Tests

- [ ] 6.1 `tests/compression/test_regrow_strategy_residual_kmeans.py`
      — strategy unit tests on a hand-built synthetic fixture
      (16 features, 8 d_model, 4 zeroed slots, 100 cached
      residual tokens):
      - assert `n_slots_repopulated == 4` on a well-conditioned
        residual stream;
      - assert decoder norm == 1.0 on every populated slot;
      - assert encoder column equals decoder row transpose;
      - assert encoder bias == 0 on every populated slot;
      - assert b_dec untouched;
      - assert non-zeroed slots' tensors are byte-equal to source;
      - degenerate input: residual std < 1e-9 → RuntimeError;
      - n_tokens < K → ValueError.
- [ ] 6.2 `tests/compression/test_regrow_determinism.py` — two
      `Regrower(strategy='residual_kmeans', seed=0,
      cached_residuals=R)` runs on identical inputs produce
      byte-identical output checkpoints (modulo
      safetensors-metadata mtime; assert tensor equality and
      output sha256 equality).
- [ ] 6.3 `tests/compression/test_regrow_postinit.py` —
      `__post_init__` rejection paths: missing checkpoint,
      bad strategy, both prompts and cached_residuals supplied,
      neither supplied, zeroed contains out-of-range fid,
      seed < 0, n_init < 1.
- [ ] 6.4 `tests/compression/test_regrow_from_compression_report.py`
      — chained constructor:
      - given a fixture `CompressionReport`, the resulting
        `Regrower.zeroed` equals the union of every cluster's
        `zeroed`;
      - the resulting `RegrowReport.provenance` carries
        `compression_report_source_sha256` and
        `compression_report_output_sha256` matching the
        `CompressionReport`'s fields;
      - direct constructor produces empty `provenance`.
- [ ] 6.5 `tests/compression/test_regrow_apply.py` — apply()
      end-to-end on a synthetic fixture:
      - source bytes unchanged after run;
      - output_checkpoint == source raises;
      - rebuilt Dictionary has the right feature count and
        non-zero projections on previously-zeroed slots.
- [ ] 6.6 `tests/compression/test_regrow_report_roundtrip.py` —
      `RegrowReport.from_json(r.to_json()) == r` on a
      hand-built fixture.
- [ ] 6.7 `tests/cli/test_regrow_cli.py` — CLI argument paths +
      end-to-end on a synthetic SAE:
      - happy path with `--zeroed-list` + `--cached-residuals`;
      - happy path with `--compression-report` (chained);
      - exit 2 on missing checkpoint, missing report, both
        zeroed-source flags supplied, both residual-source
        flags supplied;
      - exit 2 on output == source.
- [ ] 6.8 `tests/test_examples.py` gains
      `test_regrow_validated_smoke` mirroring the validator /
      compress smoke patterns: tiny configuration, success
      path asserts the regrow banner, skip path asserts a
      clear message.

## 7. Worked example + research note

- [ ] 7.1 New `examples/regrow_validated.py` showing the chained
      workflow: load a `CompressionReport` from
      `examples/compress_validated.py`'s output, construct
      `Regrower.from_compression_report`, run, dump the
      `RegrowReport`. Same skip-path semantics as
      `examples/compress_validated.py`.
- [ ] 7.2 New `docs/research/compression-regrow-design.md`
      (one page) — pointer to this change's `design.md` plus a
      paragraph explaining why `residual_kmeans` first (the
      §4.1 finding that decoder directions live in a structured
      sub-manifold; cluster centroids of failure-mode residuals
      are a principled population strategy).
- [ ] 7.3 Update `tech-debt-backlog/tasks.md` §5: append the
      regrow primitive as the loop's *fourth* component
      (validator → compressor → epoch → regrower).

## 8. Closing

- [ ] 8.1 README "Library tour" gains a one-paragraph entry for
      the `Regrower` (between the compression-action and
      Development sections, after the epoch entry once that
      lands).
- [ ] 8.2 Run `openspec validate add-compression-regrow
      --strict` — clean.
- [ ] 8.3 Run the full test suite end-to-end. CI green.
- [ ] 8.4 Squash-merge to main; archive this change directory
      under `openspec/changes/archive/`.
