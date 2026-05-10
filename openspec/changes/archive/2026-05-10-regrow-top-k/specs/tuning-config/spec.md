## ADDED Requirements

### Requirement: RegrowConfig accepts an optional top_k cap

`RegrowConfig` SHALL declare an optional `top_k: int | None = None` field. When `None` (the default), the regrower regrows every zeroed slot in the supplied `CompressionReport` — preserving the pre-change byte-identical behavior. When set to a non-negative integer, the regrower regrows only the first `top_k` slots in `RegrowPlan.populations` order, and the remaining slots stay zero in the output checkpoint.

`RegrowConfig.__post_init__` SHALL reject negative values with a clear `ValueError` naming the field and the offending value. Zero is a valid value (equivalent to skipping the regrower entirely).

The field SHALL round-trip cleanly through `RegrowConfig.to_dict()` / `RegrowConfig.from_dict()` so downstream consumers (e.g. sae-forge's FSM context plumbing) can serialize it without special handling.

#### Scenario: default None preserves byte-identical behavior

- **GIVEN** a `RegrowConfig(model_name="pythia-160m", layer=10, strategy="residual_kmeans", seed=0)` (no `top_k` set)
- **WHEN** `Regrower.from_compression_report(...).run(out)` is called against a fixture compression report with `n_features_zeroed = 12`
- **THEN** the output checkpoint's SHA matches the pre-change baseline (captured before this change landed)
- **AND** all 12 zeroed slots are regrown in the output

#### Scenario: top_k caps the regrown slot count

- **GIVEN** the same config but with `top_k=3`
- **WHEN** `run` is called
- **THEN** exactly 3 slots are regrown (the first three in plan order)
- **AND** the remaining 9 slots remain zero in the output checkpoint
- **AND** `RegrowReport.populations` has length 3

#### Scenario: top_k above plan size is a no-op cap

- **GIVEN** the same config but with `top_k=999` (well above `n_features_zeroed`)
- **WHEN** `run` is called
- **THEN** all 12 slots are regrown — equivalent to `top_k=None`
- **AND** no error is raised

#### Scenario: top_k zero is a valid no-regrow

- **GIVEN** the same config but with `top_k=0`
- **WHEN** `run` is called
- **THEN** the output checkpoint equals the input `CompressionReport.output_checkpoint` (no rows changed)
- **AND** `RegrowReport.populations` is empty

#### Scenario: negative top_k is rejected at construction

- **WHEN** `RegrowConfig(model_name="pythia-160m", layer=10, top_k=-1)` is constructed
- **THEN** `ValueError` is raised
- **AND** the message contains `"top_k"` and `"-1"`

### Requirement: Regrower.from_compression_report accepts top_k as a per-field kwarg

`Regrower.from_compression_report(...)` SHALL accept an optional `top_k: int | None = None` keyword argument. The precedence rule SHALL match the existing per-field-kwarg-vs-config behavior: when both the kwarg and `config.top_k` are set, the kwarg wins. When neither is set, the resulting regrower's `top_k` is `None`.

#### Scenario: per-field kwarg wins over config

- **GIVEN** `cfg = RegrowConfig(model_name="pythia-160m", layer=10, top_k=5)`
- **WHEN** `Regrower.from_compression_report(report=..., sae_checkpoint=..., config=cfg, top_k=2)` is called
- **THEN** the resulting regrower has `top_k == 2` (the kwarg, not 5)

#### Scenario: kwarg-only path

- **WHEN** `Regrower.from_compression_report(report=..., sae_checkpoint=..., model_name="pythia-160m", layer=10, top_k=3)` is called (no `config` kwarg)
- **THEN** the resulting regrower has `top_k == 3`

#### Scenario: config-only path

- **GIVEN** `cfg = RegrowConfig(model_name="pythia-160m", layer=10, top_k=4)`
- **WHEN** `Regrower.from_compression_report(report=..., sae_checkpoint=..., config=cfg)` is called (no `top_k` kwarg)
- **THEN** the resulting regrower has `top_k == 4`

#### Scenario: neither set defaults to None

- **WHEN** `Regrower.from_compression_report(report=..., sae_checkpoint=..., model_name="pythia-160m", layer=10)` is called (no `config`, no `top_k`)
- **THEN** the resulting regrower has `top_k == None`
- **AND** the regrower regrows every zeroed slot — pre-change behavior
