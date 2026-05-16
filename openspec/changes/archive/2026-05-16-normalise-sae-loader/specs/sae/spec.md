## MODIFIED Requirements

### Requirement: load_sae_safetensors accepts non-canonical checkpoint formats

`load_sae_safetensors` SHALL accept safetensors checkpoints that use LlamaScope-style key naming (`decoder.weight`, `encoder.weight`, `decoder.bias`, `encoder.bias`) and/or bfloat16 tensor storage without requiring pre-conversion by the caller. The returned `dict[int, SAEFeatureRecord]` is identical to what a canonical float32 checkpoint with the same feature vectors would produce.

#### Scenario: bf16 decoder with alias key loads without error

- **WHEN** `load_sae_safetensors(path)` is called on a safetensors file where the decoder tensor is stored as `BF16` under key `decoder.weight`
- **THEN** the function returns `dict[int, SAEFeatureRecord]` with `float64` projections and raises no exception

#### Scenario: canonical float32 checkpoint still loads correctly

- **WHEN** `load_sae_safetensors(path)` is called on a float32 checkpoint with key `W_dec`
- **THEN** the function returns the same result as before this change (no regression)
