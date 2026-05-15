# compressor-apply Specification

## Purpose

Documents the input-key contract `Compressor.apply()` honours for the SAE checkpoint it operates on, and the corresponding output-key contract for the safetensors it writes. The capability splits the contract into **strategy-dependent required keys** (without which the strategy cannot run) and **strategy-dependent optional keys** (consistency-zeroed when present; silently absent in the output when absent in the input).

The capability exists because the previous "always require all four standard SAE keys" contract forced downstream callers (notably sae-forge's `_run_real_fsm`) to synthesise placeholder encoder weights for W_dec-only inputs. The required-vs-optional split makes the contract explicit, supports partial-key callers natively, and preserves byte-identity for full-SAE callers.

## ADDED Requirements

### Requirement: Strategy declares its required and optional key sets

Each `CompressionConfig.strategy` value SHALL declare:

- A **required-key set**: keys without which `Compressor.apply()` cannot compute. For shipping strategies (`"zero"`, `"merge"`), this is `("W_dec",)`.
- An **optional-key set**: keys that the strategy operates on when present but tolerates absence of. For shipping strategies, this is `("W_enc", "b_enc", "b_dec")`.

The required-key set is enforced via `polygram.sae_import._load_sae_checkpoint(path, required)`, which raises on missing keys with the existing focused-error message. The optional-key set is loaded via the new `_load_sae_checkpoint_optional(path, optional)` helper which returns only the present subset without raising.

Future strategies that need additional required keys (e.g., a hypothetical encoder-rewriting strategy that requires `W_enc`) SHALL extend their required-key set. The optional-key set is the soft default — strategies inherit it unless they override.

#### Scenario: zero strategy requires only W_dec

- **GIVEN** `Compressor(strategy="zero", ...)` configured against an SAE containing only `W_dec`
- **WHEN** `Compressor.apply()` is called
- **THEN** the required-key load succeeds; the optional-key probe returns an empty dict; the strategy zeros W_dec rows without touching W_enc/b_enc/b_dec; the call returns a `CompressionResult` without raising

#### Scenario: merge strategy requires only W_dec

- **GIVEN** `Compressor(strategy="merge", ...)` configured against an SAE containing only `W_dec`
- **WHEN** `Compressor.apply()` is called
- **THEN** the required-key load succeeds; the optional-key probe returns an empty dict; the merge strategy weighted-merges W_dec rows without touching W_enc/b_enc/b_dec; the call returns a `CompressionResult` without raising

#### Scenario: missing W_dec raises with the existing focused error

- **GIVEN** an SAE safetensors that does NOT contain `W_dec`
- **WHEN** `Compressor.apply()` is called for any strategy
- **THEN** `_load_sae_checkpoint` raises `ValueError` whose message names the missing key and the tried aliases (preserving the pre-change error UX for the truly-required key)

### Requirement: Output safetensors mirrors input key set

`Compressor.apply()` SHALL write a safetensors file containing **exactly the keys that were loaded** (the union of required-key strict load + optional-key permissive load). When the input had `W_dec` only, the output has `W_dec` only. When the input had all four standard keys, the output has all four standard keys. No placeholder synthesis happens at the polygram layer.

Each output key SHALL preserve the dtype it was loaded with. The compress strategy may modify the tensor values (zeroing or merging rows/cols), but dtype, shape, and contiguity follow the loaded form.

#### Scenario: W_dec-only round-trip produces W_dec-only output

- **GIVEN** an SAE safetensors containing only `W_dec`
- **WHEN** `Compressor.apply(out_path)` is called and `out_path` is then loaded via `safetensors.numpy.load_file`
- **THEN** the loaded state dict contains exactly the key `{"W_dec"}` (no synthesised W_enc/b_enc/b_dec entries)

#### Scenario: full-SAE round-trip produces full-SAE output (byte-equivalence regression)

- **GIVEN** an SAE safetensors containing `W_enc`, `b_enc`, `b_dec`, `W_dec` (the canonical full-key set)
- **WHEN** `Compressor.apply(out_path)` is called with otherwise-identical inputs to the pre-change version
- **THEN** the resulting output safetensors is **byte-identical** to the pre-change output. The four output keys carry the same compressed values as before.

#### Scenario: partial-input output reflects the input subset

- **GIVEN** an SAE safetensors containing `W_dec` + `W_enc` but no biases
- **WHEN** `Compressor.apply(out_path)` is called
- **THEN** the output safetensors contains exactly `{"W_dec", "W_enc"}`; both keys are correctly compressed (non-rep rows of W_dec zeroed, non-rep cols of W_enc zeroed); no bias keys are present in the output

### Requirement: Strategy dispatch tolerates absent optional keys

`_dispatch_strategy(strategy, source_state, plan, ...)` SHALL check for the presence of each optional key in `source_state` before applying the strategy's per-key logic. When a key is absent, the strategy's per-key operation SHALL be skipped without error and without altering the strategy's W_dec output.

#### Scenario: zero strategy skips W_enc and b_enc zeroing when absent

- **GIVEN** `source_state == {"W_dec": <array>}` (no encoder or biases)
- **WHEN** the zero strategy's inner loop runs over the compression plan's non-rep features
- **THEN** the loop zeros W_dec rows of non-reps; no `W_enc`/`b_enc` access is attempted; no `KeyError` is raised

#### Scenario: merge strategy skips W_enc col merging when absent

- **GIVEN** `source_state == {"W_dec": <array>}` (no encoder)
- **WHEN** the merge strategy's inner loop runs
- **THEN** W_dec rows are weighted-merged into the representative; no `W_enc` access is attempted

### Requirement: `_load_sae_checkpoint_optional` helper

`polygram.sae_import` SHALL expose a new helper `_load_sae_checkpoint_optional(path, keys: Iterable[str]) -> dict[str, np.ndarray]`. The helper SHALL:

- Return a dict containing only the subset of `keys` that the file actually contains.
- Apply the same key-alias resolution as `_load_sae_checkpoint` for keys that ARE present.
- Apply the same dtype validation as `_load_sae_checkpoint` for keys that ARE present.
- NOT raise when a requested key is absent from the file.
- Return an empty dict when none of the requested keys are present in the file.

#### Scenario: helper returns subset of keys present in file

- **GIVEN** a file containing `{"W_dec", "W_enc"}`
- **WHEN** `_load_sae_checkpoint_optional(path, ["W_enc", "b_enc", "b_dec"])` is called
- **THEN** the returned dict contains exactly `{"W_enc": <array>}`; `b_enc` and `b_dec` are not present in the result

#### Scenario: helper returns empty dict when no keys match

- **GIVEN** a file containing only `{"W_dec"}`
- **WHEN** `_load_sae_checkpoint_optional(path, ["W_enc", "b_enc", "b_dec"])` is called
- **THEN** the returned dict is empty (`len == 0`); no exception is raised

#### Scenario: helper applies dtype validation to present keys

- **GIVEN** a file containing a `W_enc` key whose dtype is corrupt / unsupported
- **WHEN** `_load_sae_checkpoint_optional(path, ["W_enc"])` is called
- **THEN** the same `ValueError` `_load_sae_checkpoint` would raise is raised (the optional-vs-required split affects only the "missing key" path, not the dtype-validation path)
