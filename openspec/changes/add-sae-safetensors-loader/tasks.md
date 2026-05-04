# add-sae-safetensors-loader — tasks

## 0. Proposal

- [x] 0.1 `proposal.md` — Why / What Changes / Capabilities / Out of
      Scope / Impact (covers `load_sae_safetensors` plus the
      `polygram sae-import` CLI subcommand; defers HF / SAE-Lens /
      activation-stats to follow-ups).
- [x] 0.2 `design.md` — decoder-key precedence, `decoder.weight`
      orientation correction, `[sae]` extra dep choice, CLI
      JSON-shape detection, in-test fixture synthesis.
- [x] 0.3 `specs/sae/spec.md` — ADDED Requirements for
      `load_sae_safetensors` (key precedence, orientation, names
      override, return-shape contract).
- [x] 0.4 `specs/cli/spec.md` — ADDED Requirements for the
      `sae-import` subcommand (argument set, output schema parity
      with toy_sae.json, error contracts).
- [x] 0.5 `openspec validate add-sae-safetensors-loader --strict` ✓.

## 1. Dependency wiring

- [x] 1.1 `pyproject.toml` — `[sae]` extra (currently empty) gains
      `safetensors>=0.4`. `[all]` extra inherits.
- [x] 1.2 Confirm `pip install -e ".[sae]"` resolves cleanly on the
      Python 3.11 + 3.12 CI matrix (no torch / huggingface_hub
      pulled in transitively).

## 2. `load_sae_safetensors` implementation

- [x] 2.1 New function `polygram.load_sae_safetensors(path, *,
      names=None) -> dict[int, SAEFeatureRecord]` in
      `polygram/sae_import.py`.
- [x] 2.2 Lazy `import safetensors.numpy` inside the function;
      `ImportError` re-raised with a `pip install polygram[sae]`
      hint.
- [x] 2.3 Decoder-key auto-detection helper
      `_detect_decoder_key(tensor_keys) -> tuple[str, list[str]]`
      with the fixed precedence list `("W_dec", "decoder.weight",
      "dec")`. Returns the matched key plus the full sorted key
      list (used by error messages).
- [x] 2.4 Orientation correction: when the matched key is
      `decoder.weight` and the matrix is non-square, transpose
      before consuming. Otherwise, rows are features.
- [x] 2.5 2D-shape guard with a `ValueError` naming the offending
      key and shape.
- [x] 2.6 Names override: validate keys in `[0, n_features)`;
      out-of-range keys raise `ValueError`. Default fallback
      `f"feat_{i}"` per row.
- [x] 2.7 Coerce projections to `np.ndarray(dtype=float64)`. Set
      `label`, `activation_mean`, `activation_std` to `None`.
- [x] 2.8 Re-export `load_sae_safetensors` from
      `polygram/__init__.py`. Keep the `__all__` list alphabetized.

## 3. `polygram sae-import` CLI subcommand

- [x] 3.1 New subparser `sae-import` in `polygram/cli.py` with
      positional `<path>` and optional `--features`, `--names`,
      `--output` (defaults: select all features; no names; stdout).
- [x] 3.2 Names-file reader: load JSON, inspect first value's type;
      string-valued maps interpreted as `{id: name}`, int-valued
      maps as `{name: id}` and inverted before passing to the
      loader. Mixed-type values exit non-zero with a clear error.
- [x] 3.3 Reuse the existing `_parse_feature_ids(...)` helper from
      `polygram/cli.py` for `--features` parsing.
- [x] 3.4 Validate every requested id exists in the loaded record
      set; missing ids exit non-zero with stderr naming them.
- [x] 3.5 Emit JSON in the `tests/fixtures/toy_sae.json` schema:
      `{"schema_version": 1, "description": "<auto>",
      "features": [...]}`. Records whose optional fields are `None`
      drop those fields from the per-feature JSON object (matches
      `load_toy_sae` semantics).
- [x] 3.6 `--output` omitted → write JSON to stdout. `--output`
      supplied → write to file; print resolved path to stderr.

## 4. Tests

- [x] 4.1 `tests/test_sae_safetensors.py::TestLoadSafetensors` —
      `_synth_safetensors(path, *, key, shape)` helper synthesizes
      fixtures via `safetensors.numpy.save_file`; cover each key
      precedence branch (`W_dec`, `decoder.weight`, `dec`,
      none-found).
- [x] 4.2 `tests/test_sae_safetensors.py::TestOrientation` —
      square `decoder.weight` not transposed; non-square
      `decoder.weight` transposed. `W_dec` and `dec` never
      transposed.
- [x] 4.3 `tests/test_sae_safetensors.py::TestNamesOverride` —
      partial override leaves `feat_<i>` defaults; out-of-range key
      rejected.
- [x] 4.4 `tests/test_sae_safetensors.py::TestRoundTripWithFromSaeLens`
      — synthesize fixture → `load_sae_safetensors` →
      `from_sae_lens(records, [0, 1, 2, 3])` succeeds and the
      Dictionary's feature names match.
- [x] 4.5 `tests/test_sae_safetensors.py::test_missing_safetensors_install`
      — monkeypatch `safetensors.numpy` to be unimportable; assert
      `ImportError` mentions `pip install polygram[sae]`.
- [x] 4.6 `tests/test_cli.py::TestSaeImportSubcommand` — end-to-end
      `polygram sae-import <fixture> --features 0,1,2,3 --output
      picked.json`; assert `load_toy_sae(picked.json)` round-trip
      and that `polygram analyze picked.json` runs clean. Plus
      argparse rejection cases (missing file; unknown id;
      mixed-type names).
- [x] 4.7 `tests/test_examples.py::test_sae_safetensors_runs` —
      example produces the expected `.q.orca.md` artifact.

## 5. Example

- [x] 5.1 `examples/sae_safetensors.py` — synthesize a tiny
      safetensors fixture under the example's output dir, load via
      `load_sae_safetensors`, pick 4 features, build a Dictionary
      via `from_sae_lens`, write a verifying `.q.orca.md`. Module
      docstring describes the expected output layout.

## 6. README

- [x] 6.1 Short "Loading from safetensors" subsection added near
      the existing SAE import area. Names `load_sae_safetensors`,
      the `[sae]` extra requirement, the supported decoder-key
      precedence, and the still-deferred HuggingFace / SAE-Lens
      loaders.

## 7. Validate + commit

- [x] 7.1 Full pytest suite green; ruff clean.
- [x] 7.2 `openspec validate add-sae-safetensors-loader --strict` ✓.
- [ ] 7.3 Commit + push, open impl PR, merge after review.

## 8. Archive

- [ ] 8.1 `openspec archive add-sae-safetensors-loader` after merge
      — propagate the new requirements into
      `openspec/specs/{sae,cli}/spec.md`.
