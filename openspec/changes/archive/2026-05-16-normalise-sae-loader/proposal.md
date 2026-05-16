## Why

The SAE load layer fails on any checkpoint that uses bfloat16 storage (e.g. LlamaScope) or non-canonical key names (e.g. `decoder.weight` / `encoder.weight`), forcing users to write bespoke conversion scripts before touching the polygram pipeline. Both call sites (`load_sae_safetensors` in `sae_import.py` and `_load_sae_full` in `validator.py`) have independent, incomplete handling — fixing one doesn't fix the other.

## What Changes

- Add a shared private helper `_load_sae_checkpoint(path, keys_needed)` in `sae_import.py` that:
  - Reads safetensors metadata to detect dtype; converts bfloat16 tensors to float32 via raw-byte bit-shift (no torch required)
  - Resolves all four tensor keys through a unified alias table: `decoder.weight/encoder.weight/decoder.bias/encoder.bias` → `W_dec/W_enc/b_dec/b_enc`
  - Detects and corrects encoder weight orientation (same `d_model < d_sae` heuristic already used for decoder)
- Rewrite `_load_sae_full` in `validator.py` to call `_load_sae_checkpoint` instead of `safe_open(framework="numpy")`
- Update `load_sae_safetensors` (full-load path) to route through `_load_sae_checkpoint`
- No public API changes; no new dependencies

## Capabilities

### New Capabilities

- `sae-loader`: Shared SAE checkpoint normaliser — bf16 conversion, key aliasing, orientation detection

### Modified Capabilities

- `sae`: `load_sae_safetensors` now accepts LlamaScope-style checkpoints (bf16, `decoder.weight` key naming) without pre-conversion

## Impact

- `polygram/sae_import.py`: add `_load_sae_checkpoint`; update full-load path in `load_sae_safetensors`
- `polygram/behavioural/validator.py`: replace `_load_sae_full` body
- `tests/test_sae_import.py`: extend with aliasing + bf16 round-trip tests
- `tests/test_behavioural.py` or new `test_sae_loader.py`: cover `_load_sae_full` aliasing + bf16
- No dependency changes (safetensors already present; numpy already present)
