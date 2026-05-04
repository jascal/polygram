## MODIFIED Requirements

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

The subcommand SHALL accept four additional optional flags:

- `--sharing-graph <path.json>` — when supplied, the handler SHALL
  invoke `polygram.analysis.build_sharing_graph` on the prediction
  and write the result of `FeatureGraph.to_json()` to the named
  path. Default behavior unchanged when omitted.
- `--sharing-threshold <float>` — threshold forwarded to
  `build_sharing_graph(threshold=...)`. Defaults to `0.5`. Ignored
  unless `--sharing-graph` is also supplied.
- `--separation-graph <path.json>` — when supplied, the handler
  SHALL invoke `polygram.analysis.build_separation_graph` on the
  prediction and write the result of `FeatureGraph.to_json()` to
  the named path. Default behavior unchanged when omitted.
- `--separation-threshold <float>` — threshold forwarded to
  `build_separation_graph(threshold=...)`. Defaults to `0.2`.
  Ignored unless `--separation-graph` is also supplied.

The two graph flags are independent — the user MAY supply neither,
either, or both. When both are supplied, the handler SHALL emit two
JSON files.

The subcommand SHALL exit 0 on success and non-zero with a clear
error message on any of: missing/invalid `<sae_path>`, malformed
`--features` argument, malformed `--sharing-threshold` or
`--separation-threshold` value, or any error raised by
`predict_cancellation_depth`, `build_sharing_graph`, or
`build_separation_graph`.

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

#### Scenario: sharing-graph flag emits a parseable JSON artifact

- **WHEN** the CLI is invoked as `polygram analyze
  tests/fixtures/toy_sae.json --features 0,1,4,5 --output /tmp/r.md
  --sharing-graph /tmp/g.json --sharing-threshold 0.4`
- **THEN** the process exits 0
- **AND** `/tmp/g.json` exists and parses cleanly via `json.loads`
- **AND** the parsed dict has the keys `"kind"`, `"nodes"`,
  `"edges"`, `"clusters"`, and `"metadata"`, with
  `"kind" == "sharing"`

#### Scenario: separation-graph flag emits a parseable JSON artifact

- **WHEN** the CLI is invoked as `polygram analyze
  tests/fixtures/toy_sae.json --features 0,1,4,5 --output /tmp/r.md
  --separation-graph /tmp/s.json --separation-threshold 0.15`
- **THEN** the process exits 0
- **AND** `/tmp/s.json` exists and parses cleanly via `json.loads`
- **AND** the parsed dict's `"kind"` field equals `"separation"`

#### Scenario: both graph flags supplied emit both artifacts

- **WHEN** the CLI is invoked with both `--sharing-graph` and
  `--separation-graph` paths
- **THEN** both JSON files exist and parse cleanly, with the
  expected `"kind"` values
