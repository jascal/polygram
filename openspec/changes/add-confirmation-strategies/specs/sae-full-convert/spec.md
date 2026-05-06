# sae-full-convert Specification

## ADDED Requirements

### Requirement: convert_gemma_scope_to_safetensors preserves all SAE keys by default

`convert_gemma_scope_to_safetensors.py` SHALL write all keys present in the source `params.npz` to the output `.safetensors` file by default. When the source contains `W_enc`, `b_enc`, `b_dec`, and `threshold` in addition to `W_dec`, all five SHALL appear in the output. The script SHALL accept a `--dec-only` flag that restores the previous behaviour (write only `W_dec`).

#### Scenario: full conversion preserves all keys

- **WHEN** the source `params.npz` contains `W_dec`, `W_enc`, `b_dec`, `b_enc`, `threshold`
- **AND** `--dec-only` is not passed
- **THEN** the output `.safetensors` contains all five keys with identical values and shapes

#### Scenario: dec-only flag produces single-key output

- **WHEN** `--dec-only` is passed
- **THEN** the output `.safetensors` contains only `W_dec`

#### Scenario: full output is accepted by the compressor

- **WHEN** the output of a full conversion is passed to `Compressor` as `sae_checkpoint`
- **THEN** `Compressor.__post_init__` does not raise due to missing keys

#### Scenario: summary line reports all keys written

- **WHEN** a full conversion completes
- **THEN** stdout includes the key names written (e.g., `W_dec, W_enc, b_dec, b_enc, threshold`) in addition to the shape of `W_dec`
