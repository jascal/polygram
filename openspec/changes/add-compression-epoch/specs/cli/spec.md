## ADDED Requirements

### Requirement: polygram compress-epoch wraps EpochCompressor.run() with file-based inputs

A new `polygram compress-epoch` subparser SHALL register on the
top-level `polygram` CLI parser. It SHALL accept the file-based
inputs and orchestrator parameters listed in
`add-compression-epoch/proposal.md`'s "What Changes — CLI" section.

Required flags (no defaults):

- `--sae-checkpoint`
- `--prompts`
- `--output-checkpoint`
- `--output` (path for the `EpochReport` JSON)

Optional flags with defaults from `EpochCompressor`:

- `--layer` (default 10)
- `--model` (default "gpt2")
- `--strategy` (default "zero"; `choices=("zero",)`)
- `--device` (default "auto"; `choices=("auto", "cuda", "mps", "cpu")`)
- `--coverage-target` (default 0.95)
- `--cosine-threshold` (default 0.30)
- `--n-visits-per-feature` (default 3)
- `--n-panels-max` (default 1000)
- `--min-firing-rate` (default 0.01)
- `--max-iterations` (default 5)
- `--quality-delta-multiplier` (default 2.0)
- `--polygram-threshold` (default 0.7)
- `--jaccard-threshold` (default 0.30)
- `--min-both-fire` (default 5)
- `--save-intermediate-reports` (boolean flag, default off)
- `--allow-layer-zero` (boolean flag, default off)

The CLI handler SHALL load the prompts file via the same helper
`polygram validate` uses (one prompt per non-empty, non-`#`-prefixed
line), construct an `EpochCompressor` with the parsed parameters,
invoke `run(output_checkpoint=...)`, and write the resulting
`EpochReport` to `--output`.

#### Scenario: end-to-end CLI invocation on a synthetic SAE

- **GIVEN** a synthetic SAE checkpoint at `tmp_path/sae.safetensors`
  (built via `tests._synth_sae.synth_sae`)
- **AND** a 1-prompt file at `tmp_path/prompts.txt`
- **WHEN** `polygram.cli.main(["compress-epoch", "--sae-checkpoint",
  str(sae), "--prompts", str(prompts), "--output-checkpoint",
  str(out_ckpt), "--output", str(out_report), "--n-panels-max",
  "2", "--max-iterations", "1"])` is called
- **THEN** the call SHALL return exit code 0
- **AND** `out_ckpt` SHALL exist and parse cleanly via
  `safetensors.numpy.load_file`
- **AND** `out_report` SHALL exist and parse cleanly via
  `EpochReport.from_json`

### Requirement: polygram compress-epoch prints per-iteration progress to stderr

The CLI handler SHALL emit per-iteration progress lines to stderr
covering: panels selected, coverage achieved, current iteration's
confirmed-pair count, features zeroed this iteration, and the
running cross-entropy delta. The final stderr line SHALL include
the source/output sha256s (truncated to 12 hex characters), the
total `n_features_zeroed_total`, the number of iterations run, and
the `convergence_reason`.

#### Scenario: stderr carries iteration progress and the final summary

- **GIVEN** a successful CLI invocation per the previous scenario
- **WHEN** stderr output is captured
- **THEN** stderr SHALL contain the substring `"epoch_compress:
  iter 1"` (zero-indexed iteration label) somewhere in its
  per-iteration progress lines
- **AND** the final stderr line SHALL contain the substring
  `"convergence_reason="` followed by the orchestrator's chosen
  termination reason

### Requirement: polygram compress-epoch exits 2 on missing inputs or invalid arguments

The CLI handler SHALL return exit code 2 (without raising) on any of:

- `--sae-checkpoint` file does not exist
- `--prompts` file does not exist or contains no non-empty,
  non-`#`-prefixed lines
- `--output-checkpoint` path resolves to the same path as
  `--sae-checkpoint`
- `--strategy` value is not in the supported set (`argparse`
  `choices` enforcement, exits 2 directly)
- `--device` value is not in `("auto", "cuda", "mps", "cpu")`
  (argparse choices)
- Any numeric flag has a value outside its valid range (caught by
  `EpochCompressor.__post_init__`'s `ValueError`, surfaced as a
  stderr message and exit code 2)

#### Scenario: missing sae-checkpoint exits 2

- **GIVEN** `--sae-checkpoint /tmp/does-not-exist.safetensors`
- **WHEN** the CLI handler runs
- **THEN** the return value SHALL be 2
- **AND** stderr SHALL contain a message naming the missing path

#### Scenario: output equal to source exits 2 cleanly

- **GIVEN** `--sae-checkpoint path` and `--output-checkpoint path`
  (the same path)
- **WHEN** the CLI handler runs
- **THEN** the return value SHALL be 2
- **AND** stderr SHALL contain a message stating both paths
  resolved to the same file

#### Scenario: out-of-range coverage-target exits 2

- **GIVEN** `--coverage-target 1.5`
- **WHEN** the CLI handler runs
- **THEN** the return value SHALL be 2
- **AND** stderr SHALL contain a message naming the
  `coverage_target` field
