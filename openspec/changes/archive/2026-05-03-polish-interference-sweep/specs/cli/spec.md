## ADDED Requirements

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
