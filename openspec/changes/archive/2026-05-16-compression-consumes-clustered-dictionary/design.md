## Context

`EpochCompressor.run` orchestrates a multi-iteration compression
pipeline: pre-pass (firing rates + residuals) → loop {select panels
→ validate panels → synthesize cross-panel report → compress against
the synthesized report → check convergence}.

The three internal functions in scope here:

- `_select_panels(state_dict, eligible, priority, cosine_pairs,
  zeroed, n_visits_per_feature, n_panels_max, coverage_target)`
  returns `(list[Panel], coverage_fraction)`. Priority-driven seeded
  coverage with visit caps, anchor-only fallback, ≤7 neighbour cap.
  ~120 LOC of careful logic.
- `_validate_panels(panels, state_dict, residuals, firing_rates,
  n_tokens)` runs the per-panel `BehaviouralValidator.validate` and
  returns per-panel `ValidationReport`s.
- `_synthesize_validation_report(panels, per_panel_reports,
  sae_checkpoint)` aggregates per-panel evidence into a single
  cross-panel `ValidationReport` (confirmed pairs, candidate pairs,
  etc.).

PR #44 added `ClusteredDictionary.from_compression_panels(panels,
state_dict, encoding)` — a converter that wraps `_select_panels`
output as a `ClusteredDictionary` view. It's currently unused
internally by `EpochCompressor`; it's a public API for downstream
consumers who want a clustered view of compression's intermediate
state.

This change wires `from_compression_panels` through the rest of the
compression pipeline so `_validate_panels` and
`_synthesize_validation_report` accept `ClusteredDictionary` instead
of raw `Panel` lists.

## Goals / Non-Goals

**Goals:**

- `_validate_panels` and `_synthesize_validation_report` accept
  `ClusteredDictionary` as the panel data structure.
- `EpochCompressor.run` builds `ClusteredDictionary` per iteration
  from `_select_panels` output and threads it through.
- Byte-identical compression output on the bundled fixture; frozen
  reference test gates every commit.
- Zero behaviour change in `_select_panels`. Zero new
  `BlockFormation` strategies. Zero new public API.

**Non-Goals:**

- Extracting `_select_panels`'s algorithm into a `BlockFormation`
  strategy. See the proposal's "pivot" section — that's the wrong
  shape of integration.
- Changing `Panel`'s dataclass fields. `Panel` still exists; it's
  the canonical compression-internal record, and
  `from_compression_panels` reads from it.
- Adding a `ClusteredDictionary`-aware `BehaviouralValidator`. The
  validator operates per-block (which already maps to per-panel);
  the wrapping happens inside `_validate_panels`.
- Changing `EpochResult` or `EpochReport` shapes. The differential
  regression test pins their byte-identity.

## Decisions

**Decision 1 — `_validate_panels` takes `clustered: ClusteredDictionary` instead of `panels: list[Panel]`.**

The function's body iterates `clustered.blocks`. Each block carries
the same feature subset that the corresponding `Panel` did, just
expressed as a `Dictionary` view. The validation work itself
(per-block forward-pass + per-pair confirmation) is unchanged.

To preserve byte-identity, `clustered.blocks` MUST be in the same
order as the source `panels` list. `from_compression_panels` already
preserves this ordering (it constructs `blocks[k]` from `panels[k]`),
so iterating `clustered.blocks` and `panels` is equivalent for any
order-sensitive logic inside the validator.

**Decision 2 — `_synthesize_validation_report` takes both `clustered` and the per-block `block_reports: list[ValidationReport]`.**

The synthesis logic aggregates cross-panel evidence (which
confirmed-pairs span multiple panels, which features are
representatives, etc.). The natural input is a `ClusteredDictionary`
+ a list of per-block validation reports aligned with
`clustered.blocks`. Internally the synthesis logic stays the same
shape; only the data type plumbing changes.

The `sae_checkpoint` arg is unchanged.

**Decision 3 — `EpochCompressor.run` constructs `ClusteredDictionary` once per iteration, immediately after `_select_panels`.**

```python
panels, coverage = _select_panels(...)
if not panels:
    # convergence path; unchanged
    ...

clustered = ClusteredDictionary.from_compression_panels(
    panels=panels,
    state_dict=current_state,
    encoding=MPSRung1(),  # match the existing implicit encoding
    name=f"{self.sae_checkpoint.stem}_iter{iteration}",
)

per_block_reports = self._validate_panels(
    clustered=clustered, ...
)
synth_report = _synthesize_validation_report(
    clustered, per_block_reports, self.sae_checkpoint
)
```

The encoding is hardcoded to `MPSRung1()` because the pre-refactor
compression pipeline implicitly used the legacy 8-feature cap
(matching `MPSRung1.max_features` after `per-encoding-feature-cap`
ships). A future change adding an `encoding=` constructor parameter
to `EpochCompressor` would let callers opt into Rung3 / Rung4 / HEA-
encoded compression at larger per-block sizes. That's deferred to a
separate openspec change (`epoch-compressor-configurable-encoding`
or similar) with its own design + differential regression. Pinning
to `MPSRung1()` here is the minimum surgery that preserves
byte-identity.

**Decision 4 — The differential regression test is the load-bearing gate.**

Workflow:

1. On main (pre-refactor), run `EpochCompressor.run` on the bundled
   `tests/fixtures/toy_sae.json` fixture at seed 0 with the
   shipped defaults. Capture the resulting `EpochResult` as a
   frozen JSON reference: `tests/compression/data/epoch_result_reference.json`.
2. The new test loads the reference and re-runs the same pipeline
   on the post-refactor code. Asserts equality field-by-field —
   numeric fields to bit precision (no tolerance), collections via
   set-equality / sequence-equality.
3. Test runs on every CI build. Any drift trips it before the PR
   merges.

The bundled `toy_sae.json` is small (16 features × 8 d_model) so
the test runs in <1 second.

**Out of scope:** large-SAE timing is not pinned. The differential
regression covers correctness (byte-identical `EpochResult` on a
small fixture). It does **not** guarantee wall-clock parity at SAE
scale; introducing a `ClusteredDictionary` per iteration adds some
Python overhead, which on the bundled fixture is microseconds but
on a 16k-feature SAE is non-zero. A timing-pinned regression on a
real SAE fixture is a separate research-track concern, not a
correctness gate.

**Decision 5 — Do not introduce a new public API for the conversion.**

`from_compression_panels` is the public conversion already shipped
in PR #44. It stays as-is. `EpochCompressor` uses it internally.
No new function, no new method.

**Decision 6 — Migrate `_validate_panels` and `_synthesize_validation_report` in one PR.**

They could be migrated independently (e.g., land `_validate_panels`
first, then `_synthesize_validation_report` in a follow-up). But
both consume `panels: list[Panel]` today, and both must end up
consuming `clustered: ClusteredDictionary`. Migrating one without
the other leaves an inconsistent API state inside `epoch.py`. Ship
together.

## Risks / Trade-offs

**Risk:** the differential regression test catches a real drift.

The validation forward pass uses a torch model whose numeric output
can differ across PyTorch versions, BLAS versions, and architecture
(x86 vs ARM). The frozen reference needs to be captured on the same
toolchain that CI runs on; otherwise the test is flaky.

Mitigation: capture the reference on the same CI-pinned PyTorch /
numpy versions. Document the reference-capture command in
`tests/compression/data/README.md`. If the toolchain pins shift,
regenerate the reference and commit alongside that bump.

**Risk:** `ClusteredDictionary` construction overhead per iteration.

Each iteration of `EpochCompressor.run` now builds an extra
`ClusteredDictionary` (cosine pair graph, block construction,
cross-block adjacency). For small SAEs (the bundled fixture) this
is microseconds; for large SAEs it's the same cost as the existing
`_compute_cosine_graph` call, which `_select_panels` already pays.

Mitigation: `from_compression_panels` is cheap (no new cosine
pair graph — it reuses the W_dec rows already in `state_dict`).
The differential regression test will catch any unintentional
performance regression that breaks the timing assertions, but
those don't exist in the current compression suite, so wall-clock
is not pinned.

**Risk:** the encoding hardcoded to `MPSRung1()` inside
`EpochCompressor.run` (Decision 3) locks compression to the
rung-1 encoding. This is the pre-refactor behaviour (compression
already implicitly uses MPSRung1 via the 8-feature cap), but
making it explicit closes a future door: callers can't yet pass
`Rung3` or `Rung4` through compression.

Mitigation: this isn't a regression — pre-refactor compression
also implicitly used MPSRung1. A future change can plumb a
configurable encoding through `EpochCompressor`'s constructor; out
of scope here.

**Risk:** `Panel` objects are still in scope (they're what
`_select_panels` returns). The codebase ends up with both
`Panel`-flavoured and `ClusteredDictionary`-flavoured
representations of the same data at the iteration boundary.

This is intentional. `Panel` stays the canonical compression-
internal record (it carries `anchor`, `cosines_to_anchor`,
`panel_id` — fields a `Dictionary` block doesn't natively hold).
`ClusteredDictionary` is the wrapper for downstream consumers
(`_validate_panels`, `_synthesize_validation_report`, future Q-OrCA
manifest emission, etc.) that don't need `Panel`'s anchor metadata.

## Sequencing

Within this change:

1. Capture frozen `EpochResult` reference on current main.
2. Refactor `_validate_panels` and `_synthesize_validation_report`
   signatures + bodies.
3. Refactor `EpochCompressor.run` to build `ClusteredDictionary`
   per iteration and thread it through.
4. Differential regression test gates the refactor.
5. Run full `tests/test_compression*.py` suite — all pass.

Each step gates the next; the change merges only when the
differential test passes.

## Migration Notes

No migration. External callers (`polygram analyze` CLI, sae-forge's
compression integration, `examples/compress_epoch_validated.py`)
see no signature changes and no output changes. The refactor is
purely internal.
