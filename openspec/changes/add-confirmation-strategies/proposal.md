## Why

The compression pipeline requires a `ValidationReport` with confirmed redundant pairs, but the only way to produce one is `BehaviouralValidator`, which requires torch, transformers, and a GPT-2-style model with a `model.transformer.h[layer]` hook. This blocks compression of any non-GPT-2 SAE (Gemma Scope, Llama, Mistral, etc.) without bespoke scripting. A pluggable confirmation strategy pattern makes the compressor model-agnostic.

## What Changes

- Introduce a `Confirmer` protocol: `.run() -> ValidationReport`
- Add `DecoderGeometryConfirmer`: confirms pairs where decoder cosine² ≥ threshold; no torch, no model
- Add `ClusterConfirmer`: confirms all within-cluster pairs from a `SelectionReport`; no torch, no model
- `BehaviouralValidator` is retroactively documented as the third (behavioural) strategy; no API change
- `Compressor` is unchanged — it still consumes a `ValidationReport`
- Fix `convert_gemma_scope_to_safetensors.py` to preserve all SAE keys (`W_enc`, `b_enc`, `b_dec`, `threshold`) needed by the compressor; currently only `W_dec` is kept

## Capabilities

### New Capabilities

- `confirmation-strategies`: `Confirmer` protocol + `DecoderGeometryConfirmer` + `ClusterConfirmer`; public API, exported from `polygram`
- `sae-full-convert`: full-checkpoint conversion of Gemma Scope `params.npz` to safetensors, preserving all keys

### Modified Capabilities

- `sae`: `load_sae_safetensors` and the convert utility are extended; existing load contract is unchanged but the convert script gains a `--full` flag (or always-preserve behaviour)

## Impact

- **New modules**: `polygram/confirmation/` (protocol + two concrete strategies)
- **Modified**: `examples/convert_gemma_scope_to_safetensors.py` — preserve all keys by default
- **Public API additions**: `DecoderGeometryConfirmer`, `ClusterConfirmer` exported from `polygram.__init__`
- **No breaking changes**: `Compressor`, `BehaviouralValidator`, `ValidationReport` are untouched
- **Dependencies**: no new dependencies; geometry strategies are numpy-only
