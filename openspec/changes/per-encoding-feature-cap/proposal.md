## Why

`polygram/sae_import.py:27` declares `MAX_FEATURES_PER_DICTIONARY = 8`
as a module constant and enforces it uniformly across every encoding,
at SAE import (`sae_import.py:607`) and at `BehaviouralValidator`
feature-id assignment (`behavioural/validator.py:138-142`).

That single number is correct for `MPSRung1` (the rung-1 3-qubit MPS
encoding caps at 8 = dim C⁸) but **wrong for every other encoding the
codebase already supports**:

- `Rung3` has been shipped since `add-rung3-encoding-mvp` (PR #29) and
  per the empirical rank probe in
  [`docs/research/rung3-rank-bound.md`](../../docs/research/rung3-rank-bound.md)
  supports up to **16** linearly-independent features (the product
  state space C⁸ ⊗ C² spans dim 16 — a sharp, algebraic limit
  measured across two seeds, gap of 15 orders of magnitude between
  σ[15] and σ[16]).
- `HEA_Rung2` already parameterises `n_qubits` (defaulting to 3 but
  not pinned) and supports up to `2 ** n_qubits` features.

The 8-cap is honest for MPSRung1 and dishonest for everything else.
Latent capacity that the encodings already provide is being thrown
away at the loader.

This change moves the per-encoding cap into a method on each encoding
class and updates the three enforcement sites to query the encoding
rather than the module constant. No new encoding support, no new
cap values for MPSRung1 — just stop pretending Rung3 and HEA can only
hold 8.

## What Changes

- Add `max_features` to each encoding class:
  - `MPSRung1.max_features = 8` (unchanged)
  - `Rung3.max_features = 16` (corrected per the rank-bound finding)
  - `HEA_Rung2.max_features` = `2 ** self.n_qubits`
- Replace the `MAX_FEATURES_PER_DICTIONARY` module constant in
  `sae_import.py` with a per-encoding query. The constant SHALL remain
  exported for backwards compatibility, holding the MPSRung1 value.
- Update `BehaviouralValidator` to query the dictionary's encoding for
  its cap rather than the module constant.
- Audit `compression/compressor.py:400` and `compression/regrow.py:461`
  comments that reference the cap; replace with the per-encoding
  query if any logic depends on the value (vs just narrative comments).
- New fixture: round-trip a 16-feature Rung3 dictionary through
  `from_sae_lens` to confirm the loader accepts it.

## Capabilities

### New Capabilities

- `per-encoding-feature-cap`: encoding classes expose a `max_features`
  attribute or property; importers and validators query it.

### Modified Capabilities

- `sae`: `from_sae_lens` and `load_sae_safetensors` enforce the
  encoding-supplied cap rather than the module constant. The error
  message names the encoding so users see why the cap differs from 8.

## Impact

- `polygram/encoding.py` — add `max_features` to `MPSRung1`, `Rung3`,
  `HEA_Rung2`.
- `polygram/sae_import.py` — replace module-constant enforcement with
  per-encoding query; `MAX_FEATURES_PER_DICTIONARY` retained as a
  compatibility re-export at the MPSRung1 value.
- `polygram/behavioural/validator.py` — query encoding's cap.
- `polygram/compression/compressor.py`, `compression/regrow.py` —
  audit cap references; replace hardcoded 8s with the per-encoding
  value where logic depends on the value.
- `tests/test_sae_import.py`, `tests/test_dictionary.py`,
  `tests/test_compression*.py` — extend with N=16 Rung3 fixtures.
- No breaking changes for MPSRung1 callers; default encoding stays
  `MPSRung1` so the 8 cap continues to apply when users don't opt
  into a richer encoding.
