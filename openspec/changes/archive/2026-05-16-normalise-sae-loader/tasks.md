## 1. Shared helper — `_load_sae_checkpoint`

- [x] 1.1 Add `_KEY_ALIASES: dict[str, str]` table in `sae_import.py` covering all nine alias mappings (W_dec, decoder.weight, dec → W_dec; W_enc, encoder.weight → W_enc; b_dec, decoder.bias → b_dec; b_enc, encoder.bias → b_enc)
- [x] 1.2 Implement `_read_safetensors_header(path) -> dict[str, dict]` that reads only the JSON metadata (first 8 + header_len bytes) and returns a map of tensor name → `{dtype, data_offsets, shape}`
- [x] 1.3 Implement `_bf16_to_f32(raw: bytes, shape: tuple) -> np.ndarray` using the `uint16 << 16` bit-shift conversion
- [x] 1.4 Implement `_load_sae_checkpoint(path, keys: list[str]) -> dict[str, np.ndarray]` that: resolves each canonical key through `_KEY_ALIASES`, reads header to detect dtype, loads raw bytes for bf16 or uses `load_file` for f32, applies orientation correction for weight tensors, raises `ValueError` listing missing keys
- [x] 1.5 Write unit tests in `tests/test_sae_loader.py` covering: alias resolution (LlamaScope keys → canonical), bf16 round-trip (synthetic bf16 bytes), f32 passthrough unchanged, orientation correction for decoder.weight and encoder.weight, missing key raises ValueError with informative message, no-torch required

## 2. Wire up `load_sae_safetensors` full-load path

- [x] 2.1 Replace the `load_file(str(path))` + `_detect_decoder_key` call in `load_sae_safetensors` (the `feature_ids is None` branch) with `_load_sae_checkpoint(path, ["W_dec"])`; remove the now-redundant `decoder.weight` transpose block
- [x] 2.2 Verify existing `test_sae_import.py` tests still pass (no regressions)
- [x] 2.3 Add a test: `load_sae_safetensors` on a synthetic bf16 safetensors file with key `decoder.weight` returns correct `SAEFeatureRecord` projections

## 3. Wire up `_load_sae_full` in `validator.py`

- [x] 3.1 Replace the `safe_open(framework="numpy")` body of `_load_sae_full` with a call to `_load_sae_checkpoint(path, ["W_enc", "b_enc", "W_dec", "b_dec"])` from `sae_import`
- [x] 3.2 Update error message in `_load_sae_full` if needed so it still surfaces missing keys clearly
- [x] 3.3 Add a test: `_load_sae_full` on a synthetic bf16 safetensors with LlamaScope keys returns correctly keyed float32 arrays

## 4. Integration smoke test

- [x] 4.1 Add an integration test that builds a synthetic LlamaScope-format checkpoint (bf16, `decoder.weight/encoder.weight/decoder.bias/encoder.bias` keys, (d_model, d_sae) orientation) and runs it through `load_sae_safetensors` → `from_sae_lens` → `ClusterConfirmer` → `Compressor` end-to-end without pre-conversion
