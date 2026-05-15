## 1. New helper: `_load_sae_checkpoint_optional`

- [ ] 1.1 Add `_load_sae_checkpoint_optional(path, keys: Iterable[str]) -> dict[str, np.ndarray]` to `polygram/sae_import.py`. Returns the subset of `keys` that the file actually contains; absent keys are silently omitted (no error). Applies the same dtype validation and key-alias resolution as `_load_sae_checkpoint` for the keys that ARE present.
- [ ] 1.2 Module-level test: `tests/test_sae_import.py::test_load_sae_checkpoint_optional_returns_present_keys_only`.

## 2. Strategy-dependent required-key set

- [ ] 2.1 In `polygram/compression/compressor.py`, add a `_strategy_required_keys` constant or method mapping strategy → required-key tuple. `"zero"` → `("W_dec",)`, `"merge"` → `("W_dec",)`.
- [ ] 2.2 Add a matching `_strategy_optional_keys` constant: both strategies have `("W_enc", "b_enc", "b_dec")` as optional. Listed explicitly for documentation clarity.

## 3. `Compressor.apply` load-and-mirror logic

- [ ] 3.1 Replace the hard-coded `_load_sae_checkpoint(self.sae_checkpoint, ["W_dec", "W_enc", "b_dec", "b_enc"])` at compressor.py:776–778 with two calls: required-key strict load + optional-key permissive load. Merge the two dicts into `source_state`.
- [ ] 3.2 Track which keys came from the optional load as a `present_optional_keys: set[str]` so the write-back logic knows what to emit.
- [ ] 3.3 The write step at the end of `apply()` emits only the keys in `(required_keys | present_optional_keys)`. Each key uses its loaded dtype unchanged.

## 4. Strategy dispatch handles missing keys

- [ ] 4.1 `_dispatch_strategy(strategy, source_state, plan, ...)` SHALL check `state.get("W_enc") is not None` before zeroing/merging encoder cols. Same for `b_enc`.
- [ ] 4.2 `_apply_zero` (or equivalent inner function): skip W_enc col zeroing when key absent; skip b_enc row zeroing when key absent.
- [ ] 4.3 `_apply_merge` (or equivalent): skip W_enc col merge when absent; skip b_enc row merge when absent.
- [ ] 4.4 `b_dec` continues to be untouched by all strategies; no logic change needed.

## 5. Tests

### 5.1 Full-SAE byte-equivalence regression

- [ ] 5.1.1 Pre-existing tests under `tests/compression/test_compressor_apply.py` MUST continue to pass without modification. The new permissive-load path is dormant when all four keys are present.
- [ ] 5.1.2 Add a golden-bytes test: `Compressor(report, full_sae).run(out).read_bytes() == reference.read_bytes()` for the existing toy fixture.

### 5.2 W_dec-only input

- [ ] 5.2.1 `tests/compression/test_compressor_apply_partial_keys.py::test_w_dec_only_input_compresses_successfully` — construct a safetensors with only `W_dec`, run `Compressor(report, sae).run(out)`; assert success.
- [ ] 5.2.2 `test_w_dec_only_output_mirrors_input` — the resulting output safetensors contains exactly `{"W_dec"}` as keys.
- [ ] 5.2.3 `test_w_dec_only_zeros_correct_rows` — load the output, assert non-rep rows are zero per the validation report's confirmed pairs; assert rep rows are unchanged from input.

### 5.3 Partial inputs (W_dec + W_enc, no biases)

- [ ] 5.3.1 `test_w_dec_w_enc_only_input_compresses` — input has `W_dec` + `W_enc` but no biases; assert success.
- [ ] 5.3.2 `test_w_dec_w_enc_output_mirrors_input` — output contains `{"W_dec", "W_enc"}` and nothing else.
- [ ] 5.3.3 `test_w_dec_w_enc_zeroing_consistent` — both W_dec rows AND W_enc cols of non-reps are zeroed.

### 5.4 `_load_sae_checkpoint_optional` unit tests

- [ ] 5.4.1 Returns dict with only present keys.
- [ ] 5.4.2 Returns empty dict when none of the requested keys are present (no error).
- [ ] 5.4.3 Applies dtype validation per-key (a corrupt key in an otherwise-fine file raises).

### 5.5 EpochCompressor (downstream caller) byte-equivalence

- [ ] 5.5.1 `EpochCompressor` wraps `Compressor`. Existing EpochCompressor tests under `tests/compression/test_compressor_epoch.py` MUST continue to pass byte-identical.

## 6. Spec

- [ ] 6.1 Author `openspec/changes/compressor-partial-key-sae/specs/compressor-apply/spec.md` (new capability) covering: input key contract per strategy; required-vs-optional split; mirror-the-input output rule; extension point for strategies that need encoder.

## 7. Release

- [ ] 7.1 Bump `polygram.__version__` to `0.6.0` (minor, additive API contract).
- [ ] 7.2 `CHANGELOG.md` entry under `[0.6.0]` describing the partial-key support, the byte-equivalence guarantee for full-SAE inputs, and the downstream sae-forge benefit.
- [ ] 7.3 `openspec validate compressor-partial-key-sae --strict` is green.
- [ ] 7.4 Full `pytest tests/` is green (existing test count + new partial-key tests, no regressions).
- [ ] 7.5 `ruff check polygram/ tests/` clean on touched files.
- [ ] 7.6 `openspec archive compressor-partial-key-sae` after merge.

## 8. Coordination with sae-forge

- [ ] 8.1 After polygram 0.6.0 ships, sae-forge's `full-sae-keys-in-synth-basis` impl (sae-forge PR #41) MAY drop the placeholder-synthesis logic — the synth-basis can stay W_dec-only and polygram accepts it. This is a sae-forge-side call; if sae-forge already shipped the placeholder approach (which is forward-compatible — polygram still accepts full-key inputs), the simplification is optional.
- [ ] 8.2 Update the sae-forge synth-basis spec's "deferred items" list to note that polygram 0.6.0's partial-key support is the upstream alternative.

## 9. What this change explicitly defers

- [ ] 9.1 A `--strict-keys` mode that re-imposes the today-style "all four required" contract. Deferred — callers who want strict can validate upstream.
- [ ] 9.2 Strategies that genuinely need W_enc (e.g., encoder-rewriting strategies). None exist today; the required-key extension point is documented.
- [ ] 9.3 Round-tripping the sae-forge-side `__synthesised_keys__` metadata. Polygram's loader doesn't read it; the output doesn't preserve it. Future polygram CLI could surface it; out of scope for the apply path.
- [ ] 9.4 `EpochCompressor` explicit partial-key tests (it inherits the new behaviour via the Compressor it wraps; explicit coverage is a follow-up if real callers exercise that path).
- [ ] 9.5 `--pad-output-keys` flag to produce a full-key output regardless of input. Speculative; revisit if a downstream caller asks for it.
