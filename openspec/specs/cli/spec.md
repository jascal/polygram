# cli Specification

## Purpose

The `polygram` console-script entry point and its subcommands. Covers
how Polygram artifacts are produced from the command line: `run`
loads and executes example modules; `analyze` triages SAE feature
subsets via the analytic Gram. New subcommands SHALL extend this
spec rather than introduce parallel entry points.

## Requirements
### Requirement: Console-script entry point

Polygram SHALL register a `polygram` console script. Invocation forms:

- `polygram --version` — print `polygram <__version__>` and exit 0
- `polygram run <target> [--output-dir DIR] [--n-points N]` — load
  the target module and invoke its `main(output_dir=...)` callable

`<target>` SHALL accept either a filesystem path to a `.py` file or
a `pkg.module:callable` reference. When the path form is used,
Polygram loads the module via `importlib.util` and looks up
`main`. The CLI SHALL exit 2 with a clear error if the target does
not expose `main(output_dir=...)`.

#### Scenario: filesystem-path target runs and writes to output dir

- **WHEN** the CLI is invoked as `polygram run /tmp/myexample.py
  --output-dir /tmp/out` and `myexample.py` defines
  `def main(output_dir): Path(output_dir).joinpath("hello").write_text("hi")`
- **THEN** the process exits 0 and `/tmp/out/hello` contains `"hi"`

#### Scenario: missing main raises clear error

- **WHEN** the CLI is invoked on a module with no `main` function
- **THEN** the process exits non-zero and stderr names the missing
  `main(output_dir=...)` callable

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

