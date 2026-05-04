## ADDED Requirements

### Requirement: sae-import subcommand converts a safetensors file to the toy-SAE JSON schema

The `polygram` CLI SHALL register a `sae-import` subcommand that loads a `.safetensors` file via `polygram.load_sae_safetensors`, optionally selects a subset, and writes the result as JSON in the same schema as `tests/fixtures/toy_sae.json`. The emitted JSON SHALL be directly readable by `polygram.load_toy_sae` and consumable by every downstream surface that already accepts that schema (notably `polygram analyze`).

Argument set:

- `<path>` (positional, required) — path to the source `.safetensors` file.
- `--features 0,12,1042` — optional comma-separated list of feature ids to keep. When omitted, every loaded feature is emitted. Out-of-range or non-integer entries SHALL exit non-zero with stderr naming the offending entry.
- `--names path.json` — optional path to a JSON file mapping either `{feature_id_int: name_str}` or `{name_str: feature_id_int}`. The handler SHALL detect the mapping direction by inspecting the first value's type: string-valued maps are interpreted as `id → name`; int-valued maps as `name → id` and inverted before being passed to `load_sae_safetensors`. Mixed-type values SHALL exit non-zero.
- `--output path.json` — optional output path. When omitted, the JSON document SHALL be written to stdout.

The handler SHALL:

1. Resolve `<path>`. Missing or unreadable files SHALL exit non-zero with stderr naming the path.
2. Parse `--names` if supplied; surface JSON parse errors and ambiguous-shape errors to stderr.
3. Call `load_sae_safetensors(path, names=resolved_names_or_None)`.
4. If `--features` is supplied, validate every requested id exists in the loaded record set and otherwise exit non-zero with stderr naming missing ids.
5. Emit a JSON document with shape `{"schema_version": 1, "description": "<short auto-generated note>", "features": [...]}` where each feature object carries the same fields as `tests/fixtures/toy_sae.json` entries (`feature_id`, `name`, `projection`, optional `label`, optional `activation_mean`, optional `activation_std`). Records whose optional fields are `None` SHALL be omitted from the JSON object (matching `load_toy_sae` semantics).
6. Print nothing to stdout when `--output` is supplied; print the resolved output path on stderr regardless. When `--output` is omitted, print the JSON document on stdout.

#### Scenario: end-to-end safetensors → JSON → analyze

- **GIVEN** a `.safetensors` file at `/tmp/sae.safetensors` with `W_dec` shape `(16, 8)`
- **WHEN** the CLI is invoked as `polygram sae-import /tmp/sae.safetensors --features 0,1,4,5 --output /tmp/picked.json`
- **THEN** the process exits 0
- **AND** `/tmp/picked.json` exists and is readable by `polygram.load_toy_sae`
- **AND** the loaded dict has exactly 4 entries keyed by `0`, `1`, `4`, `5`
- **AND** invoking `polygram analyze /tmp/picked.json --features 0,1,4,5` succeeds (exit 0)

#### Scenario: --features with unknown id rejected

- **GIVEN** a `.safetensors` file with 4 features (ids `0..3`)
- **WHEN** the CLI is invoked with `--features 0,5`
- **THEN** the process exits non-zero with stderr naming `5` and the valid id range

#### Scenario: --names accepts {id: name}

- **GIVEN** a `.safetensors` file at path `/tmp/sae.safetensors` and a labels file `/tmp/labels.json` with content `{"0": "dog_poodle", "2": "bird_hawk"}`
- **WHEN** the CLI is invoked as `polygram sae-import /tmp/sae.safetensors --names /tmp/labels.json --output /tmp/picked.json`
- **THEN** the process exits 0
- **AND** the emitted JSON's feature with `feature_id=0` has `name="dog_poodle"` and the feature with `feature_id=2` has `name="bird_hawk"`

#### Scenario: --names accepts {name: id} and inverts

- **GIVEN** a labels file `/tmp/labels.json` with content `{"dog_poodle": 0, "bird_hawk": 2}`
- **WHEN** the CLI is invoked with `--names /tmp/labels.json`
- **THEN** the inversion produces the same effect as the `{id: name}` form
- **AND** the emitted feature with `feature_id=0` has `name="dog_poodle"`

#### Scenario: --names with mixed value types rejected

- **GIVEN** a labels file containing `{"0": "dog_poodle", "1": 5}` (mixed string and int values)
- **WHEN** the CLI is invoked with that file as `--names`
- **THEN** the process exits non-zero with stderr naming the file and the ambiguous-shape error

#### Scenario: --output omitted writes to stdout

- **WHEN** the CLI is invoked without `--output`
- **THEN** the JSON document is written to stdout
- **AND** stderr does NOT echo the document

#### Scenario: missing source file rejected

- **WHEN** the CLI is invoked with `<path>` pointing at a file that does not exist
- **THEN** the process exits non-zero with stderr naming the missing path
