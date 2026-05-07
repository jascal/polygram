## Context

Polygram exposes ~25 tunable knobs across `Cancellation`, `BehaviouralValidator`, `EpochCompressor`, `Compressor`, `Regrower`, and `from_sae_lens`. Today they are kwargs on each constructor with their own defaults, so:

- Downstream callers (sae-forge's `compress_with_polygram` / `perform_regrowth`) can't accept a single tuning bundle — they thread fields one-by-one.
- Three of the defaults are wrong for our actual workloads:
  1. `EpochCompressor(coverage_target=0.95, n_visits_per_feature=3, max_iterations=5)` — the iterative caller in `examples/forge_gpt2_real_sae.py:125-138` overrides to `0.5 / 1 / 1`. The default is ~10× too expensive for any iterative loop.
  2. `from_sae_lens(assign_gamma=False)` — README explicitly says this is "almost always wrong" on real SAEs.
  3. `Regrower.from_compression_report(model_name="gpt2", layer=10)` — silently GPT-2-specific.
- There's no way to round-trip a tuning bundle through a config file or an FSM context dict.

Existing changes in `openspec/changes/` (`scale-aware-compression`, `add-confirmation-strategies`, `normalise-sae-loader`) have each added 2–3 new kwargs to `Compressor` / `EpochCompressor`. The constructor surface keeps growing without a home.

## Goals / Non-Goals

**Goals:**
- One module, `polygram.config`, owns every tuning knob currently exposed on a public constructor.
- Each public constructor accepts an optional `config: <Config>` kwarg; per-field kwargs continue to work and override the config when both are supplied.
- Defaults match the workload we actually run (iterative compression, real SAEs, non-GPT-2 hosts).
- Configs are dict-serializable so sae-forge can stash one on its FSM context and `polygram.config.CompressionConfig.from_dict(ctx["compression"])` it back out.
- Two named presets on `EpochCompressor`: `.fast()` (new default, iterative) and `.thorough()` (the old defaults — exhaustive offline run).

**Non-Goals:**
- No YAML/TOML loader in this change. `from_dict` covers it; YAML parsing belongs to callers.
- No env-var fallback layer. Configs are explicit kwargs; environment-driven tuning belongs to the caller's CLI.
- No deprecation shim for `from_compression_report(model_name=..., layer=...)`. The old defaults were never correct off GPT-2; we'd rather break loudly at construction than silently mis-regrow.
- Not changing the *shape* of any algorithm — this is a re-organisation of knobs, not a behavioural change. The one exception is `assign_gamma`'s default flip, which is in the proposal because it's user-visible.

## Decisions

### Decision 1 — Five separate dataclasses, not one mega-config

Each constructor gets its own dataclass:

| Dataclass | Knobs | Used by |
|---|---|---|
| `CompressionConfig` | `strategy`, `rep_selection`, `merge_mode`, `confirmer` | `Compressor` |
| `EpochCompressionConfig` | `coverage_target`, `cosine_threshold`, `n_visits_per_feature`, `max_iterations`, `quality_delta_multiplier` | `EpochCompressor` |
| `CancellationConfig` | `tolerance`, `preserve_tiers`, `optimize`, `grid_outer`, `min_amp_overlap` | `Cancellation` |
| `ValidationConfig` | `polygram_overlap_threshold`, `jaccard_threshold`, `min_firing_rate`, `min_both_fire`, `allow_layer_zero` | `BehaviouralValidator`, embedded in `EpochCompressionConfig` |
| `RegrowConfig` | `strategy`, `prompts`, `seed`, `n_init`, `model_name`, `layer`, `device` | `Regrower.from_compression_report` |
| `SAEImportConfig` | `assign_gamma`, `gamma_range`, `n_clusters` | `from_sae_lens` |

**Why not one mega-config?** Constructors are independently usable (`Cancellation` without ever touching `Compressor`); a mega-config would force callers to fill in unrelated fields. Keeping them small also means each docstring stays scannable.

**Why not subclass / mixin?** Frozen `@dataclass(frozen=True, slots=True)` with explicit composition (`EpochCompressionConfig` holds an optional `validation: ValidationConfig`) is simpler than inheritance and survives static analysis cleanly.

**Alternatives considered:**
- Pydantic models — rejected: adds a runtime dep we don't otherwise have, and we don't need JSON-schema generation for now.
- TypedDicts — rejected: lose `__post_init__` validation. We already validate ranges in `EpochCompressor.__post_init__`; moving that to the dataclass keeps it in one place.

### Decision 2 — `config=` and per-field kwargs co-exist; per-field wins

```python
def __init__(self, *, config: CompressionConfig | None = None, strategy: str | None = None, ...):
    cfg = config or CompressionConfig()
    self.strategy = strategy if strategy is not None else cfg.strategy
    ...
```

**Why both?** Callers who already pass `Compressor(strategy="merge")` keep working. Callers who want a tuning bundle pass `Compressor(config=cfg)`. Callers who want both — "use this bundle, but override the strategy" — can.

**Override rule:** an explicit per-field kwarg (non-None) wins over the config; the config wins over the dataclass default. Documented in each constructor's docstring with the same wording.

**Alternatives considered:**
- Config-only (drop per-field kwargs) — too much churn for callers; breaks `add-confirmation-strategies` and `scale-aware-compression` mid-flight.
- `**kwargs` passthrough — loses type safety and IDE completion. Hard pass.

### Decision 3 — `EpochCompressor` defaults change; `Regrower` defaults disappear

The `EpochCompressor` defaults flip to the iterative-preset values (`coverage_target=0.5`, `n_visits_per_feature=1`, `max_iterations=1`). `EpochCompressor.thorough()` returns an instance with the old values. Callers wanting the previous behaviour change one line.

For `Regrower.from_compression_report`, `model_name` and `layer` lose their defaults entirely — they become required keyword arguments. We considered keeping them defaulted with a `DeprecationWarning`, but every existing in-tree caller already passes both explicitly (we grepped), and a silent GPT-2 default has zero legitimate use case in 2026.

**Why not a deprecation cycle?** The old defaults are wrong, not just suboptimal — silently regrowing layer 10 of a non-GPT-2 model produces nonsense. Better to break at construction.

### Decision 4 — Dict round-trip via `dataclasses.asdict` + a thin `from_dict`

Every config dataclass gets a `.from_dict(cls, d: dict) -> Self` classmethod that:
- Drops unknown keys (with a `warnings.warn` pointing to the field name) so old serialised configs survive a knob being added.
- Recurses into composed configs (e.g. `EpochCompressionConfig.validation`).
- Coerces tuples/lists for fields like `grid_outer` and `gamma_range` (JSON loses the tuple/list distinction).

`.to_dict()` is just `dataclasses.asdict(self)`. No custom encoder.

**Why `from_dict` instead of `__init__(**d)`?** Forward-compat. When sae-forge stashes a `CompressionConfig` on FSM context and we add a knob next month, an old serialised config keeps loading.

## Risks / Trade-offs

- **[Risk] `EpochCompressor` default change is observable.** Anyone running `EpochCompressor()` with no args today gets the thorough behaviour; after this change they get the fast one. → **Mitigation:** Call out in CHANGELOG; make `EpochCompressor.thorough()` the obvious one-line restore. Inline the rationale in the dataclass docstring so it's discoverable from `help()`.
- **[Risk] `assign_gamma=True` default may surface latent test fragility.** Tests that built dictionaries from synthetic projections and asserted `gamma == 0.0` will fail. → **Mitigation:** Audit `tests/test_sae_import.py` and `tests/test_from_sae_lens*.py` as part of the implementation; update assertions or pin `assign_gamma=False` in tests that genuinely want γ=0.
- **[Risk] `from_compression_report` losing defaults breaks any external caller passing only `report` and `sae_checkpoint`.** → **Mitigation:** None inside polygram — this is intentional. The only known external caller is sae-forge's `perform_regrowth`, and it already reads `model_name` from `ctx["host_model_id"]` and `layer` from `ctx.get("regrow_layer", 10)`. We'll co-ordinate with sae-forge's `forge-polygram-tuning-passthrough` change to drop the `10` fallback there.
- **[Risk] Two ways to set the same knob (config + per-field).** Could confuse readers. → **Mitigation:** Document the precedence rule once, in `polygram/config.py`'s module docstring, and link to it from each constructor docstring.
- **[Trade-off] No env-var or YAML loader.** Callers who want config-from-file build a tiny `yaml.safe_load → CompressionConfig.from_dict` themselves. Acceptable for a library; revisit if three callers re-implement the same five lines.

## Migration Plan

1. Land `polygram/config.py` with all five dataclasses + `from_dict` round-trip + tests.
2. Wire `config=` into `Cancellation`, `BehaviouralValidator`, `Compressor`, `Regrower`, `from_sae_lens` — no default changes yet. All existing tests stay green.
3. Add `EpochCompressor.fast()` / `.thorough()` classmethods. Defaults still old.
4. Audit in-tree callers; switch any that override the four iterative knobs to `EpochCompressor.fast()`.
5. Flip `EpochCompressor` field defaults to the iterative values. Update CHANGELOG.
6. Flip `from_sae_lens(assign_gamma=...)` default to `True`. Update affected tests.
7. Drop `model_name` / `layer` defaults on `Regrower.from_compression_report`. Update CHANGELOG and any doc snippets.
8. Coordinate with sae-forge `forge-polygram-tuning-passthrough` so its FSM ctx defaults align.

Rollback: each step is its own commit; reverting steps 5–7 individually is safe.

## Open Questions

- Should `CancellationConfig.optimize` stay a `dict` for back-compat, or become a `OptimizeConfig` dataclass of its own? Leaning dataclass — but the existing `optimize={"method": "grid", "max_steps": 50}` shape is widely used in tests. Probably defer to a follow-up unless implementation makes the dict awkward.
- Does `from_dict` warn or raise on unknown keys? Proposal says warn (forward-compat). Worth confirming with the "fail fast" preference if anyone holds it strongly.
