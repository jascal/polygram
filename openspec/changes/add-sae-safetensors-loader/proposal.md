## Why

The `[sae]` extra in `pyproject.toml:46-47` has been reserved-but-empty since v0, and the README's "Capacity limits" section explicitly tells users that pulling a real SAE into Polygram today means hand-rolling a `dict[int, SAEFeatureRecord]` from a safetensors file. That hand-roll is the smallest concrete gap on the runnable-against-real-SAE-data direction recommended by the v0 retrospective: the existing triage → FeatureGraph → BatchExperiment pipeline already runs end-to-end the moment a user can produce that dict from a checkpoint they actually have.

This change closes the gap for the simplest case (a local `.safetensors` file) without committing the project to torch / `sae_lens` / `huggingface_hub` runtime dependencies — those are larger surface decisions that benefit from real-data signal first.

## What Changes

### `polygram.load_sae_safetensors(...)` — new public loader

- New function `polygram.load_sae_safetensors(path, *, names=None) -> dict[int, SAEFeatureRecord]` reading a single `.safetensors` file off disk and returning the dict shape that `from_sae_lens` already consumes.
- Decoder weight tensor key is auto-detected via a fixed precedence list: `W_dec` (SAE-Lens canonical), `decoder.weight` (PyTorch convention), `dec` (terse fallback). First match wins; missing-tensor errors list every key actually present.
- Decoder rows are the projection vectors (one row per feature). The loader does NOT transpose — the file's rows ARE features. Two-dimensionality is enforced; non-2D tensors are rejected with a clear error.
- `names` is an optional `dict[int, str]` override; absent keys fall back to `f"feat_{i}"`. Out-of-range keys raise `ValueError`.
- Returned `SAEFeatureRecord`s have `label=None`, `activation_mean=None`, `activation_std=None`. The loader does NOT infer these.

### `polygram sae-import` — new CLI subcommand

- `polygram sae-import <path.safetensors> [--features 0,12,1042] [--names labels.json] [--output picked.json]` loads via `load_sae_safetensors`, optionally selects a subset, and writes the result as JSON in the same schema as `tests/fixtures/toy_sae.json`.
- Output schema matches `polygram analyze`'s consumed format, so the user's full flow becomes:
  1. `polygram sae-import sae.safetensors --features 0,12,1042 --output picked.json`
  2. `polygram analyze picked.json --features 0,12,1042 --sharing-graph g.json`
- `--features` is optional; when omitted, every feature is written.
- `--names labels.json` accepts a JSON file mapping `{feature_id: name}` (auto-detected by inspecting the first value's type — string-valued maps are interpreted as `id → name`; int-valued maps as `name → id` and inverted).
- `--output` defaults to writing the JSON document to stdout.

### `[sae]` extra populated

- `pyproject.toml`: `[sae]` extra (currently empty) gains `safetensors>=0.4`. Still NO torch, sae_lens, huggingface_hub.
- `polygram/__init__.py`: re-export `load_sae_safetensors`.

### Tests + example

- `tests/test_sae_safetensors.py` covers each key-precedence branch (`W_dec` / `decoder.weight` / `dec` / none-found), the names-override grammar, and the 2D-shape and out-of-range guards. Fixtures synthesize a `.safetensors` file at test time using `safetensors.numpy.save_file` (no torch).
- `tests/test_cli.py::TestSaeImportSubcommand` end-to-end: synthesize a fixture → `sae-import` → re-load via `load_toy_sae` → assert subset matches.
- `tests/test_examples.py::test_sae_safetensors_runs` covers the new example.
- `examples/sae_safetensors.py` walks a synthesized safetensors fixture → `load_sae_safetensors` → 4-feature subset → `from_sae_lens` Dictionary → verifying `.q.orca.md`.

### README

- A short "Loading from safetensors" note added near the existing SAE import section, naming `load_sae_safetensors`, the `[sae]` extra requirement, and the still-deferred HuggingFace / SAE-Lens loaders.

## Capabilities

### New Capabilities

*(none — `sae` and `cli` already exist)*

### Modified Capabilities

- `sae` — gains the `load_sae_safetensors` requirement.
- `cli` — gains the `sae-import` subcommand requirement.

## Impact

- `polygram/sae_import.py` — new `load_sae_safetensors` function and a tiny key-detection helper.
- `polygram/__init__.py` — re-export.
- `polygram/cli.py` — new `sae-import` subcommand handler.
- `pyproject.toml` — `[sae]` extra gains `safetensors>=0.4`; the `[all]` extra inherits.
- `tests/test_sae_safetensors.py` — new module.
- `tests/test_cli.py` — new `TestSaeImportSubcommand`.
- `tests/test_examples.py` — new `test_sae_safetensors_runs`.
- `examples/sae_safetensors.py` — new walk-through.
- `README.md` — short additive section.

## Out of Scope

The following items appeared during scoping and are explicitly **not** part of this change:

- **HuggingFace downloads.** Would require `huggingface_hub`. Separate follow-up proposal once the safetensors-only loader has real-data signal.
- **SAE-Lens package integration.** Would require `sae_lens` + torch + transformer_lens. Separate follow-up proposal — this change deliberately stays out of the torch dep tree per the README v0 stance.
- **Activation-statistics threading.** `activation_mean` / `activation_std` stay user-provided downstream. Wiring them through from a checkpoint or activations file needs an inference-time path Polygram does not have.
- **LLM-augmented or human-curated feature labels.** Beyond the simple `--names` JSON map, label semantics is downstream tooling.
- **Encoder weight extraction.** Polygram consumes decoder columns. Encoder rows are not used.
- **Multi-file / multi-SAE loading.** One file per call; aggregation belongs at the user-script layer.
- **Activation-driven feature selection.** The "16k → 8" selection problem is a separate research arc; this change ships I/O only.
- **Validation against ground truth.** A "rerun the SAE on held-out activations after the predicted compression" loop is a separate research-track question, captured under `tech-debt-backlog` §3 and `docs/research/spec-disentanglement-loop.md`.
