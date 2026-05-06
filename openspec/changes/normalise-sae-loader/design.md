## Context

Two call sites load SAE checkpoints:

1. `load_sae_safetensors` in `sae_import.py` — public API; full-load path uses `safetensors.numpy.load_file`, subset path uses `safe_open(framework="numpy")`. Already has `_DECODER_KEY_PRECEDENCE` aliasing and a transpose heuristic for `decoder.weight`, but only for the decoder tensor.

2. `_load_sae_full` in `validator.py` — private, called by `BehaviouralValidator`; uses `safe_open(framework="numpy")`, hardcodes `required = ("W_enc", "b_enc", "W_dec", "b_dec")` with no aliasing.

Both paths fail on bfloat16 (LlamaScope stores all tensors as bf16). Neither handles the `encoder.weight / decoder.weight / encoder.bias / decoder.bias` key naming that LlamaScope and some PyTorch-native SAEs use.

## Goals / Non-Goals

**Goals:**
- Single shared helper `_load_sae_checkpoint(path, keys)` that handles bf16→f32, key aliasing (all four tensors), and encoder orientation detection
- `_load_sae_full` in `validator.py` calls `_load_sae_checkpoint` — gains aliasing and bf16 support automatically
- Full-load path in `load_sae_safetensors` calls `_load_sae_checkpoint` — same benefit
- Subset (lazy) path in `load_sae_safetensors` gets a bf16-aware fallback for the decoder slice
- No public API changes; no new dependencies

**Non-Goals:**
- Supporting arbitrary checkpoint schemas beyond what the alias table covers
- Torch-based bf16 conversion (must stay numpy-only)
- Modifying `Compressor` or any call site outside the two named functions

## Decisions

### D1 — Location: `_load_sae_checkpoint` lives in `sae_import.py`

`validator.py` already imports from `sae_import`; the shared helper belongs there. Avoids a new module for a private function.

**Alternative considered:** `sae_loader.py` as a separate module. Rejected — over-engineering for a single private helper.

### D2 — bf16 conversion: raw-byte bit-shift, numpy-only

bfloat16 = upper 16 bits of IEEE 754 float32. Conversion:
```python
u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
f32 = (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
```
No torch, no ml_dtypes, no new install requirement.

Detection: read the safetensors JSON header (first 8 + header_len bytes) to find each tensor's `dtype` field before loading. Only convert tensors that report `BF16`.

**Alternative considered:** `ml_dtypes` library for native bf16. Rejected — adds a dependency; the bit-shift is equally correct and zero-cost.

### D3 — Unified alias table for all four tensors

```python
_KEY_ALIASES: dict[str, str] = {
    # canonical → canonical (identity, for completeness)
    "W_dec": "W_dec",  "W_enc": "W_enc",
    "b_dec": "b_dec",  "b_enc": "b_enc",
    # PyTorch nn.Linear / LlamaScope naming
    "decoder.weight": "W_dec",
    "encoder.weight": "W_enc",
    "decoder.bias":   "b_dec",
    "encoder.bias":   "b_enc",
    # legacy short forms already in _DECODER_KEY_PRECEDENCE
    "dec": "W_dec",
}
```

`_load_sae_checkpoint(path, keys)` takes a list of canonical target keys (`["W_dec", "W_enc", "b_dec", "b_enc"]`), resolves each through the table, loads and converts.

### D4 — Orientation detection applies to both W_dec and W_enc

Existing heuristic for `decoder.weight`: if `shape[0] < shape[1]` assume PyTorch (out, in) layout and transpose. Apply the same heuristic to `encoder.weight` (encoder linear has `(d_sae, d_model)` in PyTorch → transpose to polygram's `(d_model, d_sae)`).

Canonical polygram convention: `W_dec` is `(d_sae, d_model)`, `W_enc` is `(d_model, d_sae)`.

### D5 — Subset path: keep `safe_open` for slicing, add bf16 fallback

The lazy subset path slices individual rows off disk via `safe_open.get_slice`. `safe_open(framework="numpy")` will error on bf16 tensors. Fallback: if `safe_open` raises on the slice, read the raw tensor bytes from the header-located byte range and bit-shift convert. This is a separate, contained code path — it does not block the main change.

**Scope decision:** bf16 subset path is a stretch goal. If complex, ship without it — users needing bf16 + lazy loading can full-load (files are large but the feature subset use case is less common for bf16 SAEs).

## Risks / Trade-offs

- **Bit-shift correctness**: The `uint16 << 16` conversion is standard and tested in Python/numpy; risk is low. Covered by a round-trip test using known bf16 byte patterns.
- **Alias table completeness**: New checkpoint formats may use keys not in the table. Error message should list both what was found and what was expected, same as current `_load_sae_full`.
- **Orientation heuristic breaks on square matrices**: `d_model == d_sae` is ambiguous — heuristic skips the transpose and documents the ambiguity. Same behaviour as today.
- **`_load_sae_full` error message changes**: Currently lists keys by canonical name; post-change it will list aliases too. Acceptable — strictly more informative.
