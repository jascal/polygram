## Why

`EpochCompressor` currently hardcodes `MPSRung1()` as its panel
encoding. Three structural symptoms:

1. **`EpochCompressor.run` (epoch.py:347)** constructs the
   per-iteration `ClusteredDictionary` with `encoding=_MPSRung1()`.
   The `TODO(issue #48)` next to it pins the deferral.
2. **`_select_panels` (epoch.py:852)** caps neighbour count at the
   hardcoded literal `7` — the implicit 8-feature MPSRung1 panel
   size minus the anchor. Encoding-agnostic in name, MPSRung1-specific
   in fact.
3. **The whole staircase of `per-encoding-feature-cap` (PR #50),
   `compression-consumes-clustered-dictionary` (PR #51), and
   `add-rung4-encoding-mvp` (PR #52)** delivered larger encodings
   (`MPSRung1.max_features=8`, `Rung3.max_features=16`,
   `Rung4.max_features=32`, `HEA_Rung2.max_features=2**n_qubits`)
   but compression cannot use any of them. Rung3 and Rung4 exist
   only in unit tests and spike scripts — not in the production
   compression pipeline.

This change plumbs an `encoding=` constructor parameter through
`EpochCompressor`, scales the neighbour cap to
`encoding.max_features - 1`, and verifies byte-identical behaviour
at the `MPSRung1()` default. Production compression runs can then
opt into Rung3 / Rung4 / HEA_Rung2 to exploit the larger feature
caps shipped by the staircase.

## What Changes

- **`EpochCompressor`** gains an `encoding:
  MPSRung1 | HEA_Rung2 | Rung3 | Rung4 | None = None` constructor
  parameter. `None` resolves to `MPSRung1()` in `__post_init__` so
  the existing byte-identical refactor guarantee from PR #51 holds
  exactly.
- **`EpochCompressor.run`** passes `self.encoding` to
  `ClusteredDictionary.from_compression_panels` instead of the
  hardcoded `_MPSRung1()`. The `TODO(issue #48)` comment is
  removed.
- **`_select_panels`** gains a `max_panel_size: int` kwarg
  threaded through from `EpochCompressor.run` as
  `self.encoding.max_features`. The hardcoded `len(neighbours) >= 7`
  becomes `len(neighbours) >= max_panel_size - 1`. The function
  stays decoupled from the encoding module — it sees only the
  integer cap.
- **Differential regression**: the existing
  `test_byte_identical_epoch_result_against_frozen_reference`
  continues to pass unchanged (default `encoding=None`). A new
  parametrized test exercises an explicit `encoding=MPSRung1()`
  and asserts byte-identical output, locking down the
  default-resolution path.
- **New encoding-specific test**: a `Rung3`-encoded run on a
  16-feature synthetic SAE fixture (the existing 32-feature
  fixture, with the redundancy cluster restructured) produces
  panels of up to 16 features (not 8), exercising the new path.
- **CHANGELOG**: under unreleased, document the new constructor
  parameter and the deprecation of the hardcoded `MPSRung1()`
  internal use.

## Impact

- **Affected specs**: `compression`. Adds a new "encoding"
  requirement; modifies the "panel selection" requirement to
  state that panel size scales with `encoding.max_features`
  rather than being hardcoded at 8.
- **Affected code**:
  - `polygram/compression/epoch.py` (~30 LOC change)
  - `tests/compression/test_epoch_clustered_consume.py` (+1 test)
  - `tests/compression/test_epoch_encoding_configurable.py` (new
    file, +3 tests for the Rung3 path)
  - `CHANGELOG.md`
- **Closes**: issue #48.
- **Out of scope** (explicitly deferred):
  - Encoding-aware tuning of `n_visits_per_feature` and
    `n_panels_max`. Those are user-tunable knobs, not structural;
    can be revisited if Rung3/Rung4 compression runs show
    convergence pathologies.
  - Validation strictness about which encodings are accepted: the
    parameter is typed `MPSRung1 | HEA_Rung2 | Rung3 | Rung4 | None`
    but any object with a `max_features` integer attribute will
    work at runtime. Tightening this requires a `BaseEncoding`
    protocol, which is out of scope for this change.
