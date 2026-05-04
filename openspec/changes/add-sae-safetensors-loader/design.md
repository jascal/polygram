## Context

Polygram's `from_sae_lens(records, feature_ids, ...)` already accepts any `dict[int, SAEFeatureRecord]` and is the canonical entry point into the rung-1 / HEA Dictionary build path. The schema for that record is fixed (`feature_id`, `name`, `projection`, optional `label` / `activation_mean` / `activation_std`), and the bundled `tests/fixtures/toy_sae.json` plus `load_toy_sae(path)` JSON reader already give a working off-disk loader for the toy schema.

What's missing is an off-disk loader for the canonical *binary* SAE format used by the wider mech-interp community: `safetensors`. Real SAEs are shipped as one or more `.safetensors` files (often via SAE-Lens / HuggingFace), and although a motivated user can write 20 lines of `safetensors.numpy.load_file(...)` themselves, the README explicitly promises a first-class reader on the roadmap.

The README and `pyproject.toml` already encode three constraints that bound this change tightly:

1. **No torch in the runtime dep tree.** The `[sae]` extra is reserved-but-empty, and the README's "Optional extras" section calls out that v0 deliberately stays out of safetensors / torch. We honor the safetensors half (the `safetensors` PyPI package is pure Python with a small Rust core; no torch dependency) and defer the torch / `sae_lens` half.
2. **No HuggingFace downloads.** Same argument; `huggingface_hub` is a separate dep decision.
3. **Selection-first ergonomics.** Real SAEs ship 16k–1M features; Polygram's MPS cap is 8. The loader returns the *full* `dict[int, SAEFeatureRecord]` so downstream selection helpers (current and future) work over the same shape — the loader does not pre-select.

## Goals / Non-Goals

**Goals:**

- A public `polygram.load_sae_safetensors(path, *, names=None) -> dict[int, SAEFeatureRecord]` that turns a single `.safetensors` file into the dict shape `from_sae_lens` already consumes.
- A public `polygram sae-import` CLI subcommand that wraps the loader and emits JSON in the same schema as `tests/fixtures/toy_sae.json`, so the existing `polygram analyze` flow chains without code changes.
- Test coverage that synthesizes fixtures at test-time (no checked-in `.safetensors` blobs).
- A worked example showing the end-to-end safetensors → Dictionary → verifying `.q.orca.md` walk.

**Non-Goals:**

- HuggingFace downloads, multi-file SAE assemblies, training-time activation statistics, encoder weight handling, LLM-curated labels, semantic feature selection, or any path that pulls torch / `sae_lens` / `huggingface_hub` into the runtime dep tree.
- Validating against real SAE behavior (i.e., running the underlying transformer to check that overlap predictions track ablation impact). That is the validation-vs-selection branch of the v0 retrospective and is separate research-track work.

## Decisions

### Decision 1 — `safetensors>=0.4` as the only new runtime dep

The `safetensors` package is pure Python with a small Rust core, no torch dependency, MIT/Apache-licensed, ~1 MB install. It exposes `safetensors.numpy.save_file` / `safetensors.numpy.load_file` which give us numpy arrays directly — no intermediate torch tensor.

Alternatives considered:
- **Roll our own `.safetensors` reader.** The format is documented and ~150 lines of code, but introducing a parser for a third-party binary format we don't otherwise touch is gratuitous risk vs. ~1 MB of dep bloat that lives behind the optional `[sae]` extra anyway.
- **Use `sae_lens.SAE.load_from_pretrained`.** Pulls in torch + transformer_lens + einops + pyzmq + … several hundred MB. Defeats the README's "stays out of the safetensors / torch dep tree" stance entirely.

Choice: `safetensors>=0.4`. Behind the `[sae]` extra. Imported lazily in `load_sae_safetensors` so package import doesn't fail when the extra isn't installed.

### Decision 2 — Decoder-key auto-detection via fixed precedence list, not a `key` parameter

SAEs in the wild use different conventions for the decoder weight tensor key:
- SAE-Lens canonical: `W_dec`
- PyTorch `nn.Linear` convention: `decoder.weight`
- Terse hand-rolled checkpoints: `dec`

Loader auto-detects in that priority order; first match wins; if none are present, the loader raises `ValueError` listing every key in the file (so the user immediately sees what to inspect). This avoids forcing every caller to know the convention of their checkpoint.

Alternatives considered:
- **Required `key=` parameter.** Pushes the convention-knowledge burden onto callers; works but breaks the "drop in the file path and go" ergonomics.
- **Accept any 2D tensor when there's exactly one.** Too lenient — checkpoints often include multiple 2D tensors (encoder + decoder + biases reshaped). Wrong-tensor errors would be silent and confusing.

Choice: fixed precedence list. Add a `key=` override **only** if real-data signal shows checkpoints use names outside the three we handle.

### Decision 3 — Decoder rows are features (no transpose)

SAE-Lens stores `W_dec` as `(d_sae, d_model)`: feature `i`'s direction in residual-stream space is row `i`. Polygram's `SAEFeatureRecord.projection` is also a 1D vector per feature. Mapping is one-to-one: `W_dec[i, :]` becomes `records[i].projection`.

Some PyTorch checkpoints store decoder weights transposed (`(d_model, d_sae)`) — that's the `decoder.weight` `nn.Linear` convention (out_features × in_features, where for a decoder out=d_model, in=d_sae). The auto-detection currently doesn't compensate for this: under `decoder.weight` we still take row-as-feature semantics, which would silently produce a `(d_model)`-many "features" each with a `(d_sae)`-dimensional projection. That's wrong.

We resolve this by encoding the convention into the precedence rule: when the matched key is `decoder.weight` and the matrix is non-square, the loader transposes before consuming. `W_dec` and `dec` are not transposed. Square matrices match either convention; the loader prefers row-as-feature without transpose. (This is the only case that's actually ambiguous; flag it for follow-up signal once a real-data user hits it.)

Alternatives considered:
- **Always require `(n_features, d_model)` orientation; reject otherwise.** Forces the user to inspect tensor shapes. Worse ergonomics.
- **Detect by comparing against an expected `n_features` parameter.** Pushes the user-knowledge burden upward; defeats simplicity.

Choice: encode the convention into key auto-detection.

### Decision 4 — `names` is `dict[int, str]` only, with a tiny CLI inversion helper

The function's `names` parameter is strictly `dict[feature_id_int, name_str]`. The CLI accepts both `{id: name}` and `{name: id}` JSON maps, auto-detected by inspecting the first value's type, and inverts the latter before calling the loader. The CLI inversion is a 5-line helper; the loader stays simple.

Alternatives considered:
- **Accept both shapes in the loader.** Conflates Python-API ergonomics with file-format ergonomics. The loader has no need to inspect JSON value types — that's a CLI concern.
- **Support a `--labels-csv` shape too.** Premature; JSON is enough.

Choice: loader is `dict[int, str]` only; CLI does the JSON shape detection.

### Decision 5 — CLI emits the `tests/fixtures/toy_sae.json` schema verbatim

The output of `polygram sae-import` matches the schema already consumed by `polygram analyze` (which calls `load_toy_sae`). That makes the chain `sae-import → analyze` work end-to-end with zero additional plumbing.

Alternative considered:
- **Define a separate "polygram safetensors export" schema.** Doubles the schema surface and forces users to convert between them.

Choice: re-use the toy-SAE schema. Document this explicitly in the spec.

### Decision 6 — No checked-in `.safetensors` test fixture

Tests synthesize `.safetensors` fixtures at runtime with `safetensors.numpy.save_file` (zero-cost; sub-millisecond). This avoids a binary blob in the repo and keeps the tests trivially regenerable.

Choice: in-test synthesis. Helper lives in `tests/test_sae_safetensors.py` so it's near the consumers.

## Risks / Trade-offs

- **Risk: tensor-orientation auto-detection is wrong for some `decoder.weight` checkpoints** → Mitigation: documented in the spec; rejection error message lists the matched key and tensor shape so the user can spot orientation mismatches; follow-up `key=` / `transpose=` overrides ship if real-data signal warrants.
- **Risk: `safetensors>=0.4` breaks on Python 3.10 (the project's stated minimum).** → Mitigation: `safetensors==0.4.x` releases support 3.10; CI matrix is 3.11 + 3.12 today, so we'll add a 3.10 leg as a tiny CI follow-up if it lands. Pin floor stays at 0.4 (released 2023; broad compatibility).
- **Trade-off: no semantic feature names.** Real SAEs lack human-readable feature names. The `--names labels.json` flag covers the case where the user has a label file, but for users without one, every feature lands as `feat_<id>`. That's honest — "no labels" is the actual state of most SAEs in 2026 — but it makes downstream `polygram analyze` reports less useful unless the user attaches names. Documented in the README addition.
- **Trade-off: no validation against ground truth.** This change ships I/O, not science. The closed-form triage decomposition is what it is on the loaded vectors; whether those predictions track real SAE behavior on text is a separate, larger question the v0 retrospective explicitly flagged.
- **Risk: scope creep into HuggingFace downloads.** → Mitigation: out-of-scope list in the proposal calls this out by name. If the first real-data user immediately needs HF, that's a strong signal for the next proposal — not a reason to bolt it onto this one.
