## ADDED Requirements

### Requirement: analyze subcommand triages an SAE feature subset

The CLI SHALL register an `analyze` subcommand that takes a positional
SAE-path argument, a required `--features` option (comma-separated
feature ids), and an optional `--output` option (default
`analysis_report.md`). The handler SHALL load the SAE JSON file
(schema matching `tests/fixtures/toy_sae.json`), select the requested
feature ids, build a rung-1 Polygram Dictionary via `from_sae_lens`,
run `polygram.analysis.predict_cancellation_depth`, write the
rendered markdown report to the output path, and print the
suitability score to stdout.

The subcommand SHALL exit 0 on success and non-zero with a clear
error message on any of: missing/invalid `<sae_path>`, malformed
`--features` argument, or any error raised by
`predict_cancellation_depth`.

#### Scenario: analyze writes a report and prints a score

- **WHEN** the CLI is invoked as `polygram analyze
  tests/fixtures/toy_sae.json --features 0,1,4,5 --output /tmp/r.md`
- **THEN** the process exits 0
- **AND** `/tmp/r.md` exists and contains a markdown report with the
  expected section headings
- **AND** stdout contains a line naming the suitability score

#### Scenario: malformed features argument is rejected

- **WHEN** the CLI is invoked with `--features not-a-number`
- **THEN** the process exits non-zero with an error mentioning the
  malformed feature id
