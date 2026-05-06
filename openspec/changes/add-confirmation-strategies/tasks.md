## 1. Module scaffold

- [x] 1.1 Create `polygram/confirmation/` package (`__init__.py`, `protocol.py`, `decoder_geometry.py`, `cluster.py`)
- [x] 1.2 Export `Confirmer`, `DecoderGeometryConfirmer`, `ClusterConfirmer` from `polygram/__init__.py`

## 2. Confirmer protocol

- [x] 2.1 Define `Confirmer` as a `runtime_checkable` `typing.Protocol` with `run(self) -> ValidationReport` in `protocol.py`
- [x] 2.2 Verify `BehaviouralValidator` satisfies `Confirmer` via `isinstance` check in a test (no code change to validator)

## 3. DecoderGeometryConfirmer

- [x] 3.1 Implement `DecoderGeometryConfirmer` dataclass: fields `records`, `sae_checkpoint`, `feature_ids`, `threshold=0.8`
- [x] 3.2 Implement `.run()`: compute pairwise decoder cosine², confirm pairs ≥ threshold, set `model_name="geometry:decoder_cosine2"`, behavioural fields NaN, `n_prompts=0`, `n_tokens=0`
- [x] 3.3 Ensure no torch/transformers import occurs anywhere in the implementation path
- [x] 3.4 Write tests covering: above-threshold confirmed, below-threshold excluded, no-torch-required, metadata sentinel, NaN behavioural fields

## 4. ClusterConfirmer

- [x] 4.1 Implement `ClusterConfirmer` dataclass: fields `selection_report: SelectionReport`, `sae_checkpoint: Path`
- [x] 4.2 Implement `.run()`: invert `cluster_assignments` to group feature ids by cluster, emit all within-cluster `(i, j)` pairs with `i < j` as `confirmed`, set `model_name="geometry:cluster"`, behavioural fields NaN
- [x] 4.3 Write tests covering: within-cluster confirmed, cross-cluster excluded, singleton clusters produce empty confirmed, metadata sentinel

## 5. Fix convert_gemma_scope_to_safetensors.py

- [x] 5.1 Change default behaviour to write all keys from `params.npz` (not just `W_dec`)
- [x] 5.2 Add `--dec-only` flag that restores previous single-key behaviour
- [x] 5.3 Update stdout summary to list all keys written when in full mode
- [x] 5.4 Update docstring: replace usage example that passes output to the polygon geometry pipeline with a note that `--dec-only` gives the old behaviour; add a usage example showing the full-convert → compress pipeline

## 6. Integration test

- [x] 6.1 Add a test that runs the full Gemma Scope path end-to-end using synthetic data: full safetensors (all 5 keys) → `DecoderGeometryConfirmer` → `Compressor` → verify zeroed rows
- [x] 6.2 Add a test for the `ClusterConfirmer` → `Compressor` path using the toy SAE fixture
