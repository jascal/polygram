# Binding preservation across representation and execution

## Why

Co-firing, SAE reconstruction, and single-latent detection do not establish that
compression preserves who did what to whom. DISCOVER (arXiv:2608.29530) motivates
testing role–filler structure through replacement, intervention, and withheld
combinations. This study connects Polygram compression to lm-sae-style causal
measurements and independently executing sae-forge models.

## What Changes

- Add a reproducible research benchmark covering subject/object swaps,
  variable/value reassignment, and unambiguous modifier attachment.
- Freeze development, training, validation, combination-test, and template-test
  partitions before representation fitting; preserve pair grouping.
- Establish host competence before interpreting representation damage.
- Compare ordinary SAE reconstruction, actual Polygram compression, supervised
  role–filler reconstruction, PCA, and a learned subspace with explicit capacity
  accounting and common untouched test items.
- Measure representation replacement and targeted role edits, including unrelated
  behavior; then run independently forged models on the same items.
- Save provenance, per-example predictions, aggregate metrics, and negative results.

## Impact

Research modules and tests only; no new mandatory package dependencies. Optional
torch/transformers support is reused. sae-forge is consumed through its public API;
its implementation is not copied into Polygram. Results are scoped to evaluated
hosts, layers, datasets, budgets, and fitting procedures.
