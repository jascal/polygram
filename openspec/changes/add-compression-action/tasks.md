# add-compression-action — tasks

## 1. Subpackage scaffold + report types

- [ ] 1.1 New subpackage `polygram/compression/` with
      `__init__.py`, `compressor.py`, `report.py`,
      `strategies/__init__.py`, `strategies/zero.py`.
- [ ] 1.2 `ClusterPlan` frozen dataclass in
      `polygram/compression/report.py` with fields named in the
      design (`cluster_id`, `members`, `representative`,
      `zeroed`).
- [ ] 1.3 `CompressionPlan` frozen dataclass with `clusters`,
      `feature_ids`. JSON round-trip via the same `format(v, ".6g")`
      pattern as `ValidationReport`.
- [ ] 1.4 `CompressionReport` frozen dataclass with the field
      set named in `design.md` Decision 6 (`schema_version`,
      `source_checkpoint`, `source_checkpoint_sha256`,
      `output_checkpoint`, `output_checkpoint_sha256`,
      `validation_report_dictionary_name`,
      `validation_report_schema_version`, `strategy`, `plan`,
      `n_features_zeroed`, `n_features_kept`, `n_clusters`).
- [ ] 1.5 `CompressionResult` frozen dataclass with `plan`,
      `report`, `output_checkpoint`, `dictionary`.
- [ ] 1.6 `CompressionReport.to_json(path)` /
      `from_json(path)` round-trip, matching the schema in
      `design.md` Decision 6.
- [ ] 1.7 Public exports added to `polygram/__init__.py`
      (`Compressor`, `CompressionPlan`, `CompressionReport`,
      `CompressionResult`).

## 2. Compressor — plan() stage (cheap)

- [ ] 2.1 `Compressor` dataclass with the field set named in
      the spec (`validation_report`, `sae_checkpoint`,
      `strategy`, `representatives`).
- [ ] 2.2 `__post_init__` validation:
      - `sae_checkpoint` exists and is a file else `ValueError`.
      - `strategy == "zero"` else `ValueError` (other values
        listed for future strategies).
      - When `representatives` is not None, every cluster id
        named must exist in the plan and every fid must be a
        member of its named cluster.
- [ ] 2.3 Union-Find helper for connected-component analysis
      on `validation_report.confirmed`.
- [ ] 2.4 `plan() -> CompressionPlan` — builds clusters,
      assigns cluster ids by ascending min-fid, picks
      representative per cluster (highest summed `n_fires`,
      tiebreak lowest fid), respects `self.representatives`
      override, returns deterministic plan. Singletons
      excluded.

## 3. Compressor — apply() stage

- [ ] 3.1 `apply(plan=None, output_checkpoint=...) ->
      CompressionResult` — reads source `.safetensors` via
      `safetensors.numpy.load_file`, applies the `zero`
      strategy in-memory on numpy arrays, writes output
      atomically (temp file + `os.replace`).
- [ ] 3.2 Reject `output_checkpoint == sae_checkpoint`
      (resolved-path comparison).
- [ ] 3.3 SHA256 of source bytes (computed at read time) and
      output bytes (computed at write time) populated on the
      `CompressionReport`.
- [ ] 3.4 Rebuild `Dictionary` via
      `polygram.from_sae_lens(load_sae_safetensors(output, feature_ids=...), feature_ids, assign_gamma=True, name=...)` with the same name as the source `ValidationReport`'s `dictionary_name`.
- [ ] 3.5 `run(output_checkpoint) -> CompressionResult` —
      `apply(plan(), output_checkpoint=output_checkpoint)`.

## 4. CLI — `polygram compress` subcommand

- [ ] 4.1 New subparser registered in `polygram/cli.py`:
      `polygram compress` with the flag set named in
      `design.md` Decision 7.
- [ ] 4.2 `--validation-report` loads via
      `ValidationReport.from_json(path)`.
- [ ] 4.3 `--strategy` validates membership in supported set;
      exits 2 on unknown value.
- [ ] 4.4 `--representatives` parser: comma-separated
      `cluster_id=fid` pairs into
      `dict[int, int]`; exits 2 on malformed entries or
      cluster-id / fid mismatches.
- [ ] 4.5 Stage progress to stderr per `cli/spec.md` (load
      report → load checkpoint → plan → rewrite → write report).
      Final line includes truncated source + output SHA256.
- [ ] 4.6 Exit code 2 on missing input files, output =
      source path collision, malformed `--representatives`.

## 5. Tests

- [ ] 5.1 `tests/compression/test_compressor_plan.py` — unit
      tests for the cheap stage. Hand-build a synthetic
      `ValidationReport` with three confirmed-pair clusters
      (two singletons + one 4-clique). Assert: 3 clusters in
      plan, ascending min-fid order, representatives picked
      per `n_fires` rule, tiebreak by lowest fid, override
      honored, override-not-in-cluster raises.
- [ ] 5.2 `tests/compression/test_compressor_apply.py` —
      synthesize an SAE checkpoint via the
      `tests/behavioural/test_validator_predict.py::_synth_sae`
      helper (move helper to a shared fixture module), run
      `apply()` on a hand-built plan, assert: zeroed rows are
      0, representative rows untouched, singleton rows
      untouched, output checkpoint readable and parses cleanly,
      source checkpoint bytes unchanged on disk after run.
- [ ] 5.3 `tests/compression/test_compressor_postinit.py` —
      every `__post_init__` rejection path: missing
      checkpoint, bad strategy, override fid not in cluster,
      output path equals source path.
- [ ] 5.4 `tests/compression/test_report_roundtrip.py` —
      `CompressionReport.from_json(r.to_json()) == r` on a
      hand-built fixture report.
- [ ] 5.5 `tests/cli/test_compress_cli.py` — exercise argument
      parsing, missing-file exits, strategy-validation exit,
      `--representatives` parser, end-to-end with a synthetic
      checkpoint + report fixture.
- [ ] 5.6 `tests/test_examples.py` gains
      `test_compress_validated_smoke` mirroring the validator
      smoke pattern: tiny configuration; success path asserts
      the compression banner; skip path asserts a clear message.

## 6. Worked example + research note

- [ ] 6.1 New `examples/compress_validated.py` showing the
      full workflow: load a `ValidationReport` (or run the
      validator inline), construct `Compressor(...)`, call
      `run(output_checkpoint=...)`, dump the
      `CompressionReport`. Should consume the §4.4 selection
      worked example's output as input.
- [ ] 6.2 New `docs/research/compression-action-design.md`
      (one page) — pointer to this change's `design.md` plus a
      paragraph on why component-first beats pair-first
      (drawn from the live PR #27 run's far-cluster
      observation).
- [ ] 6.3 Update `tech-debt-backlog/tasks.md` §5: append the
      compression action as the loop's *second* half (the
      validator was named the first half in PR #26).

## 7. Closing

- [ ] 7.1 README "Library tour" gains a one-paragraph entry
      for the `compression` subpackage (between the validator
      section and Development).
- [ ] 7.2 Run the full test suite end-to-end. CI green.
- [ ] 7.3 Squash-merge to main; archive this change directory
      under `openspec/changes/archive/`.
