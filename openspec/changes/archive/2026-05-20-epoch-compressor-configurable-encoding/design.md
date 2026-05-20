## Context

`EpochCompressor.run` orchestrates the iterative compression loop:
pre-pass → loop {select panels → build `ClusteredDictionary` view
→ validate panels → synthesize cross-panel report → compress →
check convergence}. Three encoding-coupled call sites:

1. **`ClusteredDictionary.from_compression_panels(..., encoding=...)`**
   — the per-iteration view construction. Passes `encoding` to the
   per-block `Dictionary` instances inside the clustered view.
   Currently hardcoded `MPSRung1()`.
2. **`_select_panels` neighbour cap** (currently `len(neighbours)
   >= 7`) — the implicit 8-feature panel size cap. MPSRung1's
   `max_features=8` minus the anchor.
3. **The downstream `_validate_panels` and
   `_synthesize_validation_report`** consume the
   `ClusteredDictionary`, but don't query its encoding directly;
   they invoke the validator over each block's `Dictionary` (which
   carries the encoding). No additional plumbing needed there.

The staircase of recent changes lays groundwork:

- `per-encoding-feature-cap` (PR #50) — every encoding now has a
  `max_features: int` attribute (or property, for `HEA_Rung2`).
- `compression-consumes-clustered-dictionary` (PR #51) — the
  `ClusteredDictionary` view is built in `run()` and the
  encoding is already a constructor argument to
  `from_compression_panels`. The TODO at the call site flags
  this change.
- `add-rung4-encoding-mvp` (PR #52) — `Rung4.max_features=32`,
  the largest cap on the menu.

## Goals / Non-Goals

**Goals**:

- Make `EpochCompressor` work with any of the four shipped
  encodings (MPSRung1, HEA_Rung2, Rung3, Rung4) at their
  respective `max_features` caps.
- Preserve byte-identical behaviour at the default `MPSRung1()`,
  so PR #51's load-bearing refactor invariant survives unchanged.
- Surface the `max_features` cap via the encoding's attribute,
  not by adding a new constant or duplicating `8` somewhere.

**Non-goals**:

- Encoding-aware tuning of `n_visits_per_feature`, `n_panels_max`,
  or `coverage_target`. These are user-tunable knobs that may
  need empirical retuning per encoding, but no structural
  argument for coupling them.
- A `BaseEncoding` protocol or ABC. The four shipped encodings
  share `max_features` as a quack-duck attribute (some ClassVar,
  some property); a proper protocol can be added later if the
  duck-typing gets uncomfortable.
- Algorithm-level changes to panel selection. The neighbour cap
  scales; the greedy seeded-coverage logic is unchanged.

## Decisions

### Decision 1: `encoding` constructor parameter, optional, defaults to `MPSRung1()`

The new field is

```python
encoding: MPSRung1 | HEA_Rung2 | Rung3 | Rung4 | None = None
```

with resolution in `__post_init__`:

```python
if self.encoding is None:
    from polygram.encoding import MPSRung1
    self.encoding = MPSRung1()
```

**Why None-default rather than `field(default_factory=MPSRung1)`?**
Two reasons:

1. **Repr discipline**: `field(default_factory=MPSRung1)` would show
   a non-None `encoding=MPSRung1()` in every existing call's
   `__repr__`. That's noise for the majority of existing call sites
   that don't set the parameter.
2. **Import locality**: deferring the import to `__post_init__`
   matches the existing import-locality pattern at line 342
   (`from polygram.encoding import MPSRung1 as _MPSRung1` inside
   `run()`). Keeps `polygram.compression.epoch` from growing a
   module-level dependency on `polygram.encoding`.

### Decision 2: `_select_panels` takes `max_panel_size: int`, not `encoding`

The function is encoding-agnostic — it deals in feature IDs,
priorities, cosine pairs, visit budgets, and an integer cap on
neighbours. Passing the integer keeps the function decoupled
from `polygram.encoding`:

```python
def _select_panels(
    *,
    state_dict,
    eligible,
    priority,
    cosine_pairs,
    zeroed,
    n_visits_per_feature,
    n_panels_max,
    coverage_target,
    max_panel_size: int,  # new
) -> tuple[list[Panel], float]:
    ...
    if len(neighbours) >= max_panel_size - 1:
        break
```

`EpochCompressor.run` resolves `max_panel_size =
self.encoding.max_features` at the call site.

**Why not just pass `encoding`?**
- `_select_panels` is a module-level function whose tests construct
  it from raw arrays; not coupling it to an encoding class keeps the
  tests simple.
- The function never needs anything else from the encoding (no
  emit logic, no Gram). Passing the whole encoding is over-coupling.

**Backwards-compat for callers**: `_select_panels` is private
(underscore-prefixed). Searched the codebase — only `EpochCompressor.run`
calls it. The new required kwarg adds a single call-site update.

### Decision 3: byte-identical default — pin via existing 2-iter test

The existing
`test_byte_identical_epoch_result_against_frozen_reference`
constructs `EpochCompressor` with no explicit `encoding=` argument.
With `encoding: ... = None` defaulting to `MPSRung1()`, that test
continues to run the same path. If byte-identity holds, the test
passes unchanged. If it doesn't (e.g., a subtle re-routing
introduces an FP-order difference), the test fails and we
investigate.

Plus a new test
(`test_explicit_mpsrung1_byte_identical`) constructs with
`encoding=MPSRung1()` explicitly and asserts byte-identity vs the
same frozen reference. This locks the default-resolution path: any
future change to how `encoding=None` resolves trips a second
load-bearing test.

### Decision 4: Rung3-encoded test fixture — minimal smoke, not full regression

Goal: prove that compression actually runs end-to-end with
`encoding=Rung3()` and produces panels of >8 features. Not goal:
a fully-frozen Rung3 regression reference (premature — the
compression behaviour with larger encodings is itself a research
question; freezing a reference now would lock in arbitrary results).

The new test:

- Builds a 32-feature synthetic SAE with two redundancy clusters of
  10 features each (engineered so the priority algorithm finds
  large panels).
- Runs `EpochCompressor` with `encoding=Rung3()`,
  `max_iterations=1`.
- Asserts: at least one panel has `len(features) > 8` (proving the
  scaling actually engages); all returned panels have
  `len(features) <= 16` (proving the cap actually caps);
  `result.n_features_zeroed_total > 0` (sanity — the run did
  something).

Rung4 is symmetric (`max_features=32`); we ship a smaller
parametrized check rather than a separate fixture.

## Risks / Trade-offs

- **Rung3 / Rung4 compression behaviour is empirically untested**.
  This change enables the path; it doesn't verify it produces
  *useful* compression. That's the Rung4 viability spike (research
  follow-up, separate work). The smoke test only proves "doesn't
  crash and respects the cap" — not "compresses well".
- **Encoding swappability assumes per-block independence**. Each
  block's `Dictionary` uses the supplied encoding; the block
  *boundaries* (which features end up in which panel) are computed
  by `_select_panels` purely from W_dec cosines and priorities, with
  no encoding-specific logic beyond the scale cap. If a future
  encoding needs different selection criteria (e.g., HEA_Rung2 with
  different qubit topologies), the selection algorithm would need
  to become encoding-aware. Out of scope for this change.
- **The neighbour-cap scaling is a structural change**. Even though
  byte-identity holds for `MPSRung1()` (cap stays at 7), the cap
  *expression* changes from a literal to a computation. Differential
  regression catches any inadvertent regression in the MPSRung1
  case; Rung3+ has no frozen reference to drift against.

## Migration Plan

1. The default `encoding=None` resolves to `MPSRung1()`, so every
   existing call site continues to work without modification.
2. Users who want to opt into Rung3/Rung4 compression pass
   `encoding=Rung3()` etc. explicitly.
3. The `TODO(issue #48)` comment in `epoch.py` is removed.

No deprecation needed (additive parameter with backward-compatible
default).

## Open Questions

- **Should `_select_panels` get a non-anchor neighbour cap that
  *exceeds* `max_panel_size - 1` to give the validator more
  candidates and let it pick the best max_panel_size?** No: the
  current selection is greedy from highest-cosine; passing more
  candidates would change the algorithm. Deferred unless a Rung3+
  compression run shows poor panel quality.
- **Does `min_both_fire` or `cosine_threshold` need encoding-specific
  defaults?** Probably not — these gate per-pair pair confirmation
  and don't depend on panel size. Re-examine if Rung3 runs show
  systematic under-confirmation.
