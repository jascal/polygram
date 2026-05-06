## ADDED Requirements

### Requirement: _load_sae_checkpoint resolves key aliases

`_load_sae_checkpoint(path, keys)` SHALL resolve each requested canonical key (`W_dec`, `W_enc`, `b_dec`, `b_enc`) through a unified alias table before reading from the file. The alias table MUST cover at minimum:

| Source key       | Canonical key |
|------------------|---------------|
| `W_dec`          | `W_dec`       |
| `decoder.weight` | `W_dec`       |
| `dec`            | `W_dec`       |
| `W_enc`          | `W_enc`       |
| `encoder.weight` | `W_enc`       |
| `b_dec`          | `b_dec`       |
| `decoder.bias`   | `b_dec`       |
| `b_enc`          | `b_enc`       |
| `encoder.bias`   | `b_enc`       |

#### Scenario: LlamaScope key names resolve to canonical names

- **WHEN** a safetensors file contains keys `decoder.weight`, `encoder.weight`, `decoder.bias`, `encoder.bias`
- **THEN** `_load_sae_checkpoint(path, ["W_dec","W_enc","b_dec","b_enc"])` returns a dict with exactly those canonical keys

#### Scenario: missing key raises ValueError listing aliases

- **WHEN** a safetensors file has no key that aliases to a requested canonical key
- **THEN** `_load_sae_checkpoint` raises `ValueError` that names both the canonical key and the full set of keys present in the file

### Requirement: _load_sae_checkpoint converts bfloat16 to float32

`_load_sae_checkpoint` SHALL detect bfloat16 tensors via the safetensors JSON header and convert them to float32 using a raw-byte bit-shift (no torch, no ml_dtypes). The conversion SHALL be exact (bfloat16 is the upper 16 bits of float32).

#### Scenario: bfloat16 tensor round-trips correctly

- **WHEN** a safetensors file stores a tensor as `BF16`
- **THEN** the returned numpy array has dtype `float32` and values that match the original bfloat16 values to within bfloat16 precision (max absolute error ≤ `2^(exp-7)` for each element)

#### Scenario: float32 tensor is returned unchanged

- **WHEN** a safetensors file stores a tensor as `F32`
- **THEN** the returned numpy array has dtype `float32` with bitwise-identical values

#### Scenario: no torch import occurs

- **WHEN** `_load_sae_checkpoint` is called with torch blocked in sys.modules
- **THEN** it returns successfully without raising ImportError

### Requirement: _load_sae_checkpoint corrects encoder and decoder orientation

`_load_sae_checkpoint` SHALL apply orientation correction when a weight tensor's shape indicates PyTorch `nn.Linear` layout:

- `W_dec` via alias `decoder.weight`: if `shape[0] < shape[1]`, transpose so features are on rows `(d_sae, d_model)`
- `W_enc` via alias `encoder.weight`: if `shape[0] > shape[1]`, transpose so result is `(d_model, d_sae)`

Square tensors SHALL NOT be transposed (ambiguous).

#### Scenario: decoder.weight is transposed to (d_sae, d_model)

- **WHEN** a file contains `decoder.weight` with shape `(d_model, d_sae)` where `d_model < d_sae`
- **THEN** the returned `W_dec` has shape `(d_sae, d_model)`

#### Scenario: encoder.weight is transposed to (d_model, d_sae)

- **WHEN** a file contains `encoder.weight` with shape `(d_sae, d_model)` where `d_sae > d_model`
- **THEN** the returned `W_enc` has shape `(d_model, d_sae)`

### Requirement: _load_sae_full uses _load_sae_checkpoint

`_load_sae_full` in `validator.py` SHALL delegate to `_load_sae_checkpoint` for all loading, dtype conversion, and key aliasing. Its public contract (returns `{"W_enc", "b_enc", "W_dec", "b_dec"}` as float32 arrays) is unchanged.

#### Scenario: _load_sae_full accepts LlamaScope checkpoint without pre-conversion

- **WHEN** `_load_sae_full` is called with a bf16 LlamaScope-format safetensors file
- **THEN** it returns float32 arrays keyed `W_enc`, `b_enc`, `W_dec`, `b_dec` without raising

### Requirement: load_sae_safetensors full-load path uses _load_sae_checkpoint

The full-load path of `load_sae_safetensors` (when `feature_ids=None`) SHALL route decoder loading through `_load_sae_checkpoint`, gaining bf16 support and aliasing for the decoder tensor. Encoder/bias tensors are not returned by `load_sae_safetensors` (it only surfaces decoder projections as `SAEFeatureRecord`).

#### Scenario: load_sae_safetensors reads bf16 decoder

- **WHEN** `load_sae_safetensors(path)` is called on a bf16 file with canonical or aliased decoder key
- **THEN** it returns a dict of `SAEFeatureRecord` with float64 projections (no error)
