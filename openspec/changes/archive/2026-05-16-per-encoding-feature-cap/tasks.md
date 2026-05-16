## 1. Encoding-side cap declarations

- [x] 1.1 Add `max_features: ClassVar[int] = 8` to `MPSRung1` in `polygram/encoding.py`
- [x] 1.2 Add `max_features: ClassVar[int] = 16` to `Rung3` in `polygram/encoding.py`. Reference `docs/research/rung3-rank-bound.md` in the docstring as the empirical source for the value.
- [x] 1.3 Add `max_features` as a `@property` on `HEA_Rung2` returning `2 ** self.n_qubits` in `polygram/encoding.py`
- [x] 1.4 Unit tests for each encoding: `MPSRung1().max_features == 8`, `Rung3().max_features == 16`, `HEA_Rung2(depth=1).max_features == 8`, `HEA_Rung2(depth=1, n_qubits=4).max_features == 16`

## 2. Loader enforcement

- [x] 2.1 In `polygram/sae_import.py`, replace the `MAX_FEATURES_PER_DICTIONARY` enforcement at line 607 with a query against the target encoding's `max_features`. The encoding is supplied (directly or via dictionary default) to the loader entry point.
- [x] 2.2 Update the error message to name the encoding and its cap, with the suggested path to higher capacity (per design.md Decision 5).
- [x] 2.3 Retain `MAX_FEATURES_PER_DICTIONARY = 8` as a top-level constant in `sae_import.py` for backwards compatibility (re-exporting the MPSRung1 cap).
- [x] 2.4 Loader regression tests: 8-feature MPSRung1 still loads (unchanged); 12-feature Rung3 now loads (new); 17-feature Rung3 raises with the new error message naming Rung3 and 16.

## 3. BehaviouralValidator enforcement

- [x] 3.1 In `polygram/behavioural/validator.py`, replace the `MAX_FEATURES_PER_DICTIONARY` check at lines 138-142 with a query against the dictionary's encoding's `max_features`.
- [x] 3.2 Update the validator's error message in parallel with the loader's.
- [x] 3.3 Validator regression tests: existing 8-feature MPSRung1 fixtures unchanged; new 12-feature Rung3 fixture passes the cap check.

## 4. Audit downstream consumers

- [x] 4.1 Grep the entire repo for `MAX_FEATURES_PER_DICTIONARY` and for the literal `8` in cap-context. Catalog every site in `audit.md` (kept in this change folder during development; deleted before archive).
- [x] 4.2 At each catalogued site, confirm whether the 8 is a narrative comment or a load-bearing constant. Narrative comments may stay; load-bearing constants get replaced with `dictionary.encoding.max_features`.
- [x] 4.3 In particular, audit `polygram/compression/compressor.py:400` and `polygram/compression/regrow.py:461` (the two known cap references).

## 5. Documentation

- [x] 5.1 Update `README.md` "Capacity limits" section to reflect the per-encoding caps (8 / 16 / 2^n_qubits).
- [x] 5.2 Add a one-line note to `polygram/__init__.py`'s docstring (or wherever the cap is publicly documented) pointing at `docs/research/rung3-rank-bound.md` for the empirical justification.

## 6. End-to-end integration

- [x] 6.1 Add an `examples/sae_import_rung3_n16.py` worked example: import 16 features from a real (or fixture) SAE checkpoint against `Rung3`, render gram, confirm rank 16.
- [x] 6.2 Add the example to `tests/test_examples.py` smoke list with appropriate marker (xfail on missing extras, etc.).

## 7. Closing

- [x] 7.1 Run `pytest` full suite; verify no regressions.
- [x] 7.2 Run `openspec validate per-encoding-feature-cap --strict`.
- [x] 7.3 Update `CHANGELOG.md` under the unreleased section: "Per-encoding feature cap: `Rung3` now supports up to 16 features (was 8); `HEA_Rung2` scales with `n_qubits`. See `docs/research/rung3-rank-bound.md`."
