## ADDED Requirements

### Requirement: polygram regrow wraps Regrower.run() with file-based inputs

A new `polygram regrow` subparser SHALL register on the top-level
`polygram` CLI parser. It SHALL accept the file-based inputs and
strategy parameters listed in `add-compression-regrow/proposal.md`'s
What Changes section.

Required flags (no defaults):

- `--sae-checkpoint`
- `--output-checkpoint`
- `--output` (path for the `RegrowReport` JSON)
- `--strategy` (`choices=("residual_kmeans",
  "high_decoder_norm_random", "orthogonal_noise_scaled")`)

Mutually-exclusive flag groups (each group requires exactly one
member):

- Zeroed-source: `--zeroed-list` (comma-separated ints) OR
  `--compression-report` (path to a `CompressionReport` JSON).
- Residual-source: `--cached-residuals` (path to a `.npy` file
  loadable as a 2D float32 array) OR (`--prompts` AND `--layer`
  AND `--model`).

Optional flags with defaults:

- `--seed` (default 0)
- `--n-init` (default 4)
- `--device` (default "auto"; `choices=("auto", "cuda", "mps",
  "cpu")`)

The CLI handler SHALL construct a `Regrower` (or
`Regrower.from_compression_report` when `--compression-report` is
supplied), invoke `run(output_checkpoint=...)`, and write the
resulting `RegrowReport` to `--output`.

#### Scenario: end-to-end CLI invocation with --zeroed-list and --cached-residuals

- **GIVEN** a synthetic SAE checkpoint at `tmp/sae.safetensors`
- **AND** a residuals `.npy` file at `tmp/residuals.npy` with
  shape `(100, 8)`
- **WHEN** `polygram.cli.main(["regrow", "--sae-checkpoint",
  str(sae), "--output-checkpoint", str(out_ckpt), "--output",
  str(out_report), "--strategy", "residual_kmeans",
  "--zeroed-list", "2,5,9,13", "--cached-residuals",
  str(residuals)])` is called
- **THEN** the call SHALL return exit code 0
- **AND** `out_ckpt` SHALL exist and parse cleanly via
  `safetensors.numpy.load_file`
- **AND** `out_report` SHALL exist and parse cleanly via
  `RegrowReport.from_json`
- **AND** the parsed `RegrowReport.provenance` SHALL equal `{}`

#### Scenario: end-to-end CLI invocation with --compression-report

- **GIVEN** a `CompressionReport` JSON at `tmp/compression.json`
  whose clusters' `zeroed` lists union to `{2, 5, 9, 13}`
- **AND** a residuals `.npy` file
- **WHEN** the CLI is called with `--compression-report` instead
  of `--zeroed-list`
- **THEN** the call SHALL return exit code 0
- **AND** the parsed `RegrowReport.provenance` SHALL contain
  `compression_report_source_sha256` and
  `compression_report_dictionary_name` matching the upstream
  report

### Requirement: polygram regrow prints stage progress to stderr

The CLI handler SHALL emit progress lines to stderr covering:
load report/zeroed → load checkpoint → capture or load
residuals → run strategy → write checkpoint → write report.
The final stderr line SHALL include the truncated source +
output sha256 hashes (12 hex chars each) plus
`n_slots_repopulated` and `n_slots_left_zero`.

#### Scenario: stderr carries stage progress and final summary

- **GIVEN** a successful CLI invocation per the previous
  scenario
- **WHEN** stderr output is captured
- **THEN** stderr SHALL contain the substring `"polygram
  regrow:"` on every progress line
- **AND** the final stderr line SHALL contain the substring
  `"n_slots_repopulated="` and a 12-char-truncated sha256 hex
  for both source and output

### Requirement: polygram regrow exits 2 on missing inputs or invalid arguments

The CLI handler SHALL return exit code 2 (without raising) on any
of:

- `--sae-checkpoint` file does not exist
- `--output-checkpoint` resolves to the same path as
  `--sae-checkpoint`
- `--zeroed-list` and `--compression-report` BOTH supplied
- Neither `--zeroed-list` nor `--compression-report` supplied
- `--cached-residuals` and `--prompts` BOTH supplied (or both
  via `--prompts` and `--layer` triple AND `--cached-residuals`)
- Neither `--cached-residuals` nor `--prompts` supplied
- `--compression-report` file does not exist or fails to parse
  via `CompressionReport.from_json`
- `--cached-residuals` file does not exist, fails to load, or
  has wrong shape (must be 2D) or wrong dtype (must be float32
  or float64)
- `--zeroed-list` contains non-integer or negative values
- `--strategy` value is not in the supported set (argparse
  `choices` enforcement, exits 2 directly)

#### Scenario: missing sae-checkpoint exits 2

- **GIVEN** `--sae-checkpoint /tmp/does-not-exist.safetensors`
- **WHEN** the CLI handler runs
- **THEN** the return value SHALL be 2
- **AND** stderr SHALL contain a message naming the missing path

#### Scenario: both zeroed-source flags supplied exits 2

- **GIVEN** `--zeroed-list 1,2,3` AND `--compression-report
  /path/to/report.json` both supplied
- **WHEN** the CLI handler runs
- **THEN** the return value SHALL be 2
- **AND** stderr SHALL contain a message naming both
  `--zeroed-list` and `--compression-report` and stating
  exactly one must be supplied

#### Scenario: output equal to source exits 2

- **GIVEN** `--sae-checkpoint path` AND `--output-checkpoint path`
- **WHEN** the CLI handler runs
- **THEN** the return value SHALL be 2
- **AND** stderr SHALL contain a message stating both paths
  resolved to the same file

#### Scenario: malformed --zeroed-list exits 2

- **GIVEN** `--zeroed-list "1,foo,3"`
- **WHEN** the CLI handler runs
- **THEN** the return value SHALL be 2
- **AND** stderr SHALL contain a message naming the offending
  token `foo`
