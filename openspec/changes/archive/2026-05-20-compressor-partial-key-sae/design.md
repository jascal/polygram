## Context

`Compressor.apply()` reads its input SAE via `_load_sae_checkpoint(path, [...])` at compression/compressor.py:776–778. The loader treats every key in the list as required and raises on missing keys. The list is hardcoded to `["W_dec", "W_enc", "b_dec", "b_enc"]`.

For the `zero` and `merge` strategies — the only two strategies polygram ships — the apply path:

1. Reads `source_state["W_dec"]` for cluster-norm stat computation (`_compute_cluster_norm_stats`).
2. Dispatches `_dispatch_strategy(strategy, source_state, plan, ...)`. The strategy zeros (or merges) W_dec rows of non-rep features. It also touches W_enc cols and b_enc rows for cluster non-reps.
3. Computes `scale_compression_ratio` from W_dec.
4. Writes the resulting state back to disk.

W_enc / b_enc / b_dec are loaded just to be partially-zeroed. They're not load-bearing for the compress math; the math is W_dec-centric. Their presence in the input is essentially a "consistency wrapper" — if you have them, you want compression to apply consistently to them too.

But callers who write **W_dec-only** SAE files (sae-forge's `_run_real_fsm` synth-basis) have no encoder/biases to keep consistent. They hit the loader error and have to synthesise placeholders. That's the friction this change removes.

## Goals / Non-Goals

**Goals:**
- `Compressor.apply()` accepts W_dec-only SAE inputs without error.
- Output safetensors mirrors input's key set: W_dec-only in → W_dec-only out; full SAE in → full SAE out.
- Full-SAE-input callers see **byte-identical** output. The new logic is purely additive at the partial-input branch.
- Strategy-dependent required-key sets are documented as the extension point for future strategies that genuinely need W_enc.

**Non-Goals:**
- Strict-keys mode for callers who want today's "all four required" behaviour.
- Compressing only-W_enc (no W_dec) inputs — `W_dec` stays the load-bearing required key. The strategies' compute paths read it.
- Output schema unification (e.g., always-write-all-four). Mirror-the-input is more useful for round-tripping; if downstream consumers want a fully-padded SAE, that's their concern.
- Surfacing `__synthesised_keys__` metadata from sae-forge inputs. Polygram doesn't read it today; out of scope.

## Decisions

### Decision 1 — Strategy-dependent required-key set

Each strategy declares its required-key set as a class-level constant or property. For the two strategies that ship:

- `strategy="zero"` → `required_keys = ("W_dec",)`. The zero strategy zeros W_dec rows; when W_enc / b_enc are present in `source_state`, it ALSO zeros their corresponding cols/rows. When absent, it skips those steps.
- `strategy="merge"` → `required_keys = ("W_dec",)`. Symmetric — merges W_dec rows; merges W_enc cols / b_enc rows when present.

`Compressor.apply()` calls `_load_sae_checkpoint(self.sae_checkpoint, list(self.strategy_required_keys))` first (strict load), then `_load_sae_checkpoint_optional(self.sae_checkpoint, list(self.strategy_optional_keys))` to merge in whatever optional keys exist.

**Alternative considered**: always load with `["W_dec"]` only and use `state.get("W_enc")` etc. internally. Rejected — the explicit required-vs-optional split documents the contract better and keeps the loader's strict-on-missing-required-keys behaviour for the keys that ARE actually required.

### Decision 2 — Mirror-the-input output key set

The written safetensors contains exactly the keys that were loaded. A W_dec-only input produces a W_dec-only output. A full-SAE input produces a full-SAE output.

**Why mirror, not always-write-all-four**: if the caller wrote a W_dec-only file, they likely want a W_dec-only file back. Writing four keys when they wrote one would be surprising and would defeat the round-trip-through-polygram-compress use case. Mirror-the-input is the least-surprising behaviour.

**Alternative considered**: always-write-all-four with zero-filled placeholders for absent keys. Rejected — adds output volume and asymmetry. The sae-forge synth-basis use case wants a round-trippable W_dec-only file.

**Alternative considered**: write `W_enc` / `b_enc` for absent keys as the **decoder transpose / zeros placeholders** (mirroring sae-forge PR #41's approach). Rejected — that's the sae-forge layer's choice for ITS write path; polygram's apply path shouldn't replicate that placeholder synthesis. If a polygram user wants placeholders, they can build them upstream.

### Decision 3 — `_load_sae_checkpoint_optional` as the new helper

The optional-keys probe is a small new helper in `sae_import.py`. Signature: `_load_sae_checkpoint_optional(path, keys) -> dict[str, np.ndarray]` returning the subset of `keys` that the file contains (no error on absent keys; the returned dict has only the present keys).

Implementation is a thin wrapper around `safetensors.numpy.load_file` that filters by `keys`. Polygram already does dtype validation and key-alias resolution in `_load_sae_checkpoint`; the optional helper SHALL apply the same logic to whichever keys are present.

**Alternative considered**: extend the existing `_load_sae_checkpoint` with a `partial_ok: bool = False` parameter. Rejected — splits the loader's single-purpose into two modes. A separate helper is more readable and doesn't risk surprising existing callers.

### Decision 4 — Strategy-side dispatch must handle missing keys

`_dispatch_strategy(strategy, source_state, plan, ...)` and its delegates (`_apply_zero`, `_apply_merge` if it exists) need to check `state.get("W_enc")` before zeroing/merging cols. The same goes for `b_enc`. `b_dec` is never modified by either strategy so its presence/absence is structurally irrelevant.

**Alternative considered**: hoist the missing-key check up into `Compressor.apply()` and skip the strategy call entirely when only W_dec is present. Rejected — the strategy's behaviour over its keys should be self-describing; the apply orchestrator just decides which keys to load.

### Decision 5 — Output preserves loaded dtype

The mirror-the-input rule extends to dtype: each output key uses the same dtype as its source key. No casting unless the original loader did it.

### Decision 6 — Capability spec covers the apply path

A new `compressor-apply` capability documents the input-key contract per strategy, the mirror-the-input output rule, and the extension point for future strategies. Located at `openspec/specs/compressor-apply/spec.md` post-archive.

The `tuning-config` capability is untouched — `CompressionConfig` and its validation logic don't change. The required-keys behaviour is internal to `Compressor.apply()`, not surfaced as a config field.

## Risks / Trade-offs

- **Downstream tools that ASSUME a full-SAE output may break.** If a tool reads the post-compress safetensors and expects W_enc/b_enc/b_dec to be there, it'll fail on the W_dec-only output. **Mitigation**: documented in the spec's "Out of scope" — callers who need consistency can wrap with their own padding logic. The major downstream caller (sae-forge) is fixing this on its side (PR #41) so it produces full-key safetensors regardless of polygram's output shape.

- **A future strategy that genuinely needs W_enc** (e.g., an encoder-rewriting strategy) would extend the required-key set. The spec documents this as the extension point; today's `zero` / `merge` are the only strategies.

- **The mirror-the-input rule is a soft semantic.** A meticulous user might want to *upgrade* a W_dec-only file by feeding it to a Compressor with a full-key validation report — they might expect the output to have W_enc placeholders filled. That's not what this proposal ships. Future work could expose `--pad-output-keys` if real callers need it.

- **`scale_compression_ratio` is computed from W_dec only**, so the partial-input path has no degraded telemetry. Good — the report's quality is invariant under input-key shape.
