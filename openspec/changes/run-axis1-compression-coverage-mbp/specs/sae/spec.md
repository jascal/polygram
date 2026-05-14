# run-axis1-compression-coverage-mbp Specification

## Purpose

This change is primarily a measurement run, not a code change. The single user-visible interface addition is the `--assign-amp-knobs` CLI flag on `examples/rung_compression_coverage.py` — exposing the `EpochCompressor.assign_amp_knobs` field (PR #64) at the example-script level so callers can produce the Axis 1 comparison artifacts.

## ADDED Requirements

### Requirement: `rung_compression_coverage.py` exposes `--assign-amp-knobs`

`examples/rung_compression_coverage.py` SHALL accept a `--assign-amp-knobs` command-line flag (no argument; action="store_true"). When present, the script SHALL pass `assign_amp_knobs=True` to its `EpochCompressor` construction. When absent, the script SHALL pass `assign_amp_knobs=False` — the v2.1 measurement default.

The flag's value SHALL be recorded in the output JSON's top-level `assign_amp_knobs` field so the cell identity is reproducible from the artifact alone.

#### Scenario: flag present propagates to EpochCompressor

- **WHEN** the script is invoked with `--encoding rung4 --assign-amp-knobs`
- **THEN** the constructed `EpochCompressor` has `assign_amp_knobs=True`
- **AND** the output JSON's top-level `assign_amp_knobs` field is `true`

#### Scenario: flag absent defaults to False

- **WHEN** the script is invoked without `--assign-amp-knobs`
- **THEN** the constructed `EpochCompressor` has `assign_amp_knobs=False`
- **AND** the output JSON's top-level `assign_amp_knobs` field is `false`
- **AND** the resulting compression metrics are unchanged from the pre-change measurement on the same fixture (back-comparability with the v2.1 results note)
