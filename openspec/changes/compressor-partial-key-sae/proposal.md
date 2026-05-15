## Why

`Compressor.apply()` today hard-codes `_load_sae_checkpoint(self.sae_checkpoint, ["W_dec", "W_enc", "b_dec", "b_enc"])` at compression/compressor.py:776–778. The loader treats every named key as required and raises on absent keys:

```
ValueError: _load_sae_checkpoint: no key aliasing to 'W_enc' found in <path>.
Tried aliases ['W_enc', 'encoder.weight']; file contains: ['W_dec']
```

That's correct for compression strategies that zero rows/cols of the encoder + biases (the current `zero` and `merge` strategies do), but it forces callers to supply a fully-populated SAE — even when the caller's only goal is to **zero `W_dec` rows** and the encoder/biases are irrelevant or unavailable.

A real downstream caller (sae-forge's `_run_real_fsm`, surfaced in sae-forge PR #41) writes a `synth_basis.safetensors` with only `W_dec` and then asks `Compressor.apply()` to zero rows per a validation report. The current contract forces sae-forge to synthesise placeholder `W_enc` / `b_enc` / `b_dec` arrays just to satisfy the loader — which is structurally valid but semantically meaningless (the placeholder encoder isn't a real encoder; the compressor zeros placeholder columns).

**Permissive partial-key loading is the cleaner upstream fix.** `Compressor.apply()` should require only the key set it actually needs for the configured strategy. When the input has fewer keys, the output safetensors should mirror the input's key set — no synthesised placeholders, no asymmetry between input and output.

## What Changes

### Required key set is strategy-dependent

`Compressor.apply()` SHALL compute the **required key set** based on `self.strategy`:

- `strategy="zero"` requires `W_dec` only. The strategy zeros W_dec rows of non-representative features; W_enc cols / b_enc rows are zeroed *when present* but absence is not an error.
- `strategy="merge"` requires `W_dec` only. The strategy weighted-merges W_dec rows; W_enc cols / b_enc rows are merged *when present* but absence is not an error.

Both strategies leave `b_dec` untouched in all cases (the decoder bias is global, not per-feature).

`_load_sae_checkpoint` is called with the required keys, and an additional **optional-keys** lookup (new helper `_load_sae_checkpoint_optional`) loads `W_enc`, `b_enc`, `b_dec` into the state dict only when the source file contains them.

### Output safetensors mirrors input key set

`Compressor.apply()` SHALL write only the keys that were present in the input. A W_dec-only input produces a W_dec-only output. A full-SAE input produces a full-SAE output. This eliminates the asymmetry where partial-key inputs would otherwise produce full-key outputs (with implicitly-zero placeholders).

### Scale-stats computation unchanged

The cluster scale statistics (`_compute_cluster_norm_stats`, `_compute_scale_compression_ratio`) read `source_state["W_dec"]` — that key is always required, so this code path is unchanged.

### `CompressionReport` schema-stable

`CompressionReport` continues to record `n_features_kept`, `n_features_zeroed`, scale stats, etc. The change does NOT add or remove report fields; report consumers see no diff.

### Out of scope, deliberately

- **Strategies that genuinely need `W_enc`** — none exist in polygram today. If a future strategy is added that requires the encoder (e.g., a strategy that updates encoder weights to compensate for compressed decoder rows), it would extend the required-key-set logic for itself. Spec'd as the extension point, not implemented.
- **A `--strict-keys` mode** that re-imposes today's "all four required" contract for callers who want it. Deferred — the permissive default is forward-compatible; callers needing strict can validate their input upstream.
- **Round-tripping `__synthesised_keys__` metadata** from inputs. If the sae-forge layer (PR #41) writes that metadata, polygram's loader doesn't read it and the output doesn't preserve it. Future polygram CLI could surface it; out of scope for the apply path.
- **`EpochCompressor` partial-key support.** EpochCompressor wraps Compressor and shares its apply path; this change flows through automatically, but explicit test coverage is a follow-up.

## Capabilities

### New Capabilities

- `compressor-apply`: documents `Compressor.apply()`'s input-key contract (required vs. optional keys per strategy), the input-mirrors-output rule for the written safetensors, and the strategy-dependent extension point for future strategies that need additional keys.

### Modified Capabilities

- `tuning-config`: `CompressionConfig` is unchanged; the new behaviour is internal to `Compressor.apply()`. No scenarios added or modified.

## Impact

- **Modified**:
  - `polygram/compression/compressor.py` — `Compressor.apply()` (line 776 region): replace the hard-coded 4-key load with a strategy-driven required-key lookup + optional-key probe. `_dispatch_strategy` and `_patch_cluster_scale_fields` may need small adjustments to handle missing keys.
  - `polygram/sae_import.py` — new helper `_load_sae_checkpoint_optional(path, keys)` that returns the subset of `keys` present in the file (no error on absent keys).
- **New**: `compressor-apply` capability spec.
- **No breaking changes**: existing full-SAE callers see byte-identical behaviour. The new permissive-loading path is purely additive — files that contained all four keys before still produce all four keys in output.
- **No new dependencies**.
- **Version bump**: minor (0.5.0 → 0.6.0) since the API contract is widened, not narrowed.

## Downstream callers that benefit immediately

- **sae-forge PR #41** (`full-sae-keys-in-synth-basis`): with polygram 0.6.0, sae-forge can drop the placeholder-synthesis logic and the `__synthesised_keys__` metadata. The synth-basis stays W_dec-only; polygram accepts it. Two implementation paths exist simultaneously until both releases land; either can be the "first deployable" depending on which ships first.

The sae-forge-side fix (PR #41) and this polygram-side fix are **interchangeable** for the user-visible bug (FSM + real validation report composition). Shipping both gives the cleanest split:

- **Sae-forge handles its own writes** (proposal #41 — full-key safetensors with `__synthesised_keys__` metadata).
- **Polygram handles its own reads** (this proposal — strategy-dependent required keys; partial-key inputs accepted).

After both ship, sae-forge can choose to drop placeholder synthesis (because polygram no longer requires the keys), OR keep synthesising for downstream-introspection clarity (placeholders are tagged, so they're recoverable). That's a sae-forge call.
