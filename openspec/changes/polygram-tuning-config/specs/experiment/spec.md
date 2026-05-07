## ADDED Requirements

### Requirement: Cancellation accepts CancellationConfig

`Cancellation` SHALL accept an optional keyword argument
`config: CancellationConfig | None = None`. When supplied,
fields from `config` (`tolerance`, `preserve_tiers`, `optimize`,
`grid_outer`, `min_amp_overlap`) SHALL provide values for the
corresponding constructor fields. An explicit per-field keyword
argument SHALL override the config; the config SHALL override the
existing per-field defaults declared on `Cancellation`. When
`config is None`, behaviour SHALL be identical to today's
per-field-default behaviour.

`config` SHALL NOT replace the `dictionary`, `target_pair`,
`knobs`, or `optimize_all` fields — those remain explicit
constructor inputs because they describe the *target* of the
search, not its *tuning*.

The per-field defaults on `Cancellation` itself SHALL remain
unchanged in this change (`tolerance=0.05`,
`preserve_tiers=True`, `optimize={"method": "grid",
"max_steps": 50}`); the dataclass values in `CancellationConfig`
mirror them.

#### Scenario: config supplies tolerance and preserve_tiers

- **GIVEN** `cfg = CancellationConfig(tolerance=0.01,
  preserve_tiers=False)` and a valid HEA dictionary `d` with
  features `("a", "b")`
- **WHEN** `Cancellation(dictionary=d, target_pair=("a", "b"),
  config=cfg)` is constructed
- **THEN** the resulting instance has `tolerance == 0.01` and
  `preserve_tiers is False`

#### Scenario: per-field kwarg overrides config

- **GIVEN** `cfg = CancellationConfig(tolerance=0.01)`
- **WHEN** `Cancellation(dictionary=d, target_pair=("a", "b"),
  config=cfg, tolerance=0.001)` is constructed
- **THEN** `tolerance == 0.001`

#### Scenario: no-config call preserves legacy behaviour

- **WHEN** `Cancellation(dictionary=d, target_pair=("a", "b"))`
  is constructed with no `config`
- **THEN** `tolerance == 0.05`, `preserve_tiers is True`, and
  `optimize == {"method": "grid", "max_steps": 50}` — exactly
  matching the legacy defaults
