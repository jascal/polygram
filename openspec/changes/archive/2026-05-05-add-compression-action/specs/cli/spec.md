## ADDED Requirements

### Requirement: polygram compress wraps Compressor.run() with file-based inputs

`polygram compress` SHALL be a new CLI subcommand that wraps `Compressor.run()` with file-based inputs. The flag set:

- `--validation-report PATH` — required. Path to a JSON file
  produced by `ValidationReport.to_json()` (or by
  `polygram validate --output`).
- `--sae-checkpoint PATH` — required. Source SAE
  `.safetensors` file. Must exist on disk.
- `--output-checkpoint PATH` — required. Where the rewritten
  `.safetensors` is written. Must differ from
  `--sae-checkpoint`.
- `--strategy STR` — required. Must be `"zero"` (the only
  currently-implemented strategy). Other values exit with code
  2 and a message naming the supported set.
- `--output PATH` — required. Where the
  `CompressionReport` JSON is written.
- `--representatives SPEC` — optional. Comma-separated
  `cluster_id=fid` pairs (e.g.,
  `0=12999,1=4192,2=8371`). Cluster ids referenced must exist
  in the plan; fids must be members of their named cluster.

#### Scenario: end-to-end CLI invocation on a synthetic SAE

- **GIVEN** a synthetic SAE checkpoint at `tmp/sae.safetensors`
- **AND** a `ValidationReport` JSON at `tmp/validation.json` with non-empty `confirmed`
- **WHEN** `polygram.cli.main(['compress', '--validation-report', str(report_path), '--sae-checkpoint', str(sae_path), '--output-checkpoint', str(out_ckpt), '--strategy', 'zero', '--output', str(out_report)])` is called
- **THEN** the call SHALL return exit code `0`
- **AND** `out_ckpt` SHALL exist and parse cleanly via `safetensors.numpy.load_file`
- **AND** `out_report` SHALL exist and parse cleanly via `CompressionReport.from_json`

#### Scenario: --representatives override is parsed and applied

- **GIVEN** a `ValidationReport` whose plan would default cluster 0's rep to fid 1
- **WHEN** the CLI is invoked with `--representatives 0=0`
- **THEN** the resulting `CompressionReport.plan.clusters[0].representative` SHALL equal `0`

### Requirement: polygram compress prints honest stage progress

`polygram compress` SHALL print one line per major stage to stderr:

```
polygram compress: loading validation report ...
polygram compress: loading source SAE checkpoint ...
polygram compress: building compression plan ... (3 clusters, 5 features to zero)
polygram compress: rewriting checkpoint ...
polygram compress: writing compression report ...
polygram compress: done. Source SHA256 abcd... → Output SHA256 wxyz...
```

The final line SHALL include both checkpoint hashes (truncated to 12 hex chars for legibility) so the operation's identity is visible in shell logs.

#### Scenario: stderr carries the stage progress and final summary

- **GIVEN** a successful CLI invocation per the previous scenario
- **WHEN** stderr output is captured
- **THEN** stderr SHALL contain the substring `'polygram compress:'` on every progress line
- **AND** the final stderr line SHALL contain a 12-char-truncated source sha256 hex AND a 12-char-truncated output sha256 hex

### Requirement: polygram compress exits non-zero on missing inputs or invalid arguments

`polygram compress` SHALL exit with code `2` and a message on stderr when:

- `--validation-report` does not exist or is not a file.
- `--sae-checkpoint` does not exist or is not a file.
- `--output-checkpoint` resolves to the same path as
  `--sae-checkpoint`.
- `--strategy` is not one of the supported values.
- `--representatives` references a cluster id not in the plan.
- `--representatives` references a fid not in its named cluster.

The message SHALL name the offending argument and the offending value.

#### Scenario: missing validation-report exits 2

- **WHEN** the CLI is invoked with `--validation-report /tmp/does-not-exist.json`
- **THEN** the return value SHALL be `2`
- **AND** stderr SHALL contain a message naming the missing path

#### Scenario: output equal to source exits 2

- **WHEN** the CLI is invoked with `--sae-checkpoint path` and `--output-checkpoint path` (same path)
- **THEN** the return value SHALL be `2`
- **AND** stderr SHALL contain a message stating both paths resolved to the same file

#### Scenario: unknown strategy exits 2 via argparse

- **WHEN** the CLI is invoked with `--strategy merge`
- **THEN** the return value SHALL be `2` (argparse `choices` enforcement)

#### Scenario: representatives override referencing an unknown cluster exits 2

- **GIVEN** a `ValidationReport` whose plan has cluster ids `{0, 1}`
- **WHEN** the CLI is invoked with `--representatives 99=0`
- **THEN** the return value SHALL be `2`
- **AND** stderr SHALL contain a message naming the offending cluster id `99`
