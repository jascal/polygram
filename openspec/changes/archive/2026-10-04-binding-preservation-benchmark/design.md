# Experimental protocol

The source paper is https://arxiv.org/pdf/2608.29530. This is a bounded empirical
study, not a claim that tensor products uniquely identify the host algorithm.

## Sequence and decision rules

1. Generate and audit data before any fitting. Keep host-development data separate.
2. Start with locally cached instruction-tuned Qwen models on CPU. GPU availability
   is checked, not assumed. Select a host using development examples only.
3. Require candidate-answer accuracy >= 0.90 and both-members-correct accuracy
   >= 0.80 in each task for a host competence pass. Exact unrestricted generation
   is a separate measurement; candidate discrimination is not called generation.
4. Freeze host revision, prompt format, layer selection rule, dataset seed, ranks,
   fitting budgets, and seeds in a manifest before opening final results.
5. Fit representations on training data; tune/early-stop on validation; run final
   combination and template evaluations only after selection.
6. Fit a standard SAE to the chosen host if a correctly matched checkpoint is not
   available. Record this as a benchmark-trained SAE, not a pretrained SAE.
7. Use Polygram's real compression API and save its report/checkpoint. No surrogate
   pruning method may be labelled Polygram compression.
8. A supervised TPR receives parse annotations for the input, never the answer.
   Parameter accounting includes filler/role embeddings and output maps. Strong
   position/atomic-pair controls should diagnose the benefit of role factorization.
9. Evaluate direct reconstruction, role edits, and same-basis projections before
   full native forging. Preserve failed/negative runs and distinguish numerical
   failure, host failure, representation failure, and execution failure.

## Splits

Each task has at least three roles, including a distractor role. Reserved fillers
appear in non-reserved roles during training. Combination tests place these known
fillers into withheld active roles. Validation uses different reserved fillers.
Template tests use new renderings while retaining trained role–filler combinations;
the crossed template-plus-combination condition is reported separately if run.
Pair identifiers and prompt hashes make leakage auditable.

## Metrics and inference

Report full-vocabulary KL at answer positions, restricted candidate discrimination,
both-members accuracy, unrestricted host argmax agreement, answer margins, and
unrelated-question preservation. Preserve per-example results. Bootstrap pairs
within tasks/splits, and show seeds separately before aggregation. Compare absolute
damage and matched-budget controls; differences of unrelated KLs are not a
circuit-preservation metric. A fitted subspace is an empirical reference, not a
proven capability ceiling. No failed optimization establishes irreducibility.

## Frozen first study (2026-10-03)

`runs/binding/protocol.json` freezes Qwen2.5-1.5B-Instruct after it passed all
three development competence gates (12 pairs per task). The initial 0.5B screen
failed subject/object and modifier tasks and remains recorded. Dataset v2 uses
single-token names, colors, and quoted letter values so final-residual readout
scores do not mix different continuation lengths. No final test was examined
when making this measurement decision.

The intervention is at the final residual, before RMSNorm, at the last prompt
position. Only RMSNorm and unembedding remain: there is no later attention through
which intact prompt positions could bypass the reconstruction. This tests a
task-conditioned readout representation, not the upstream mechanism constructing
bindings. Capture checks the offline readout against actual host logits; bfloat16
kernel differences must stay below 0.001 nats KL per example and are reported.

Each context is captured for both the target and the unrelated question. TPR roles
are task-role-query triples and fillers are task-qualified symbols. TPR edits use
input annotations only. SAE/PCA/subspace deltas use actual counterfactual donors
and are explicitly labelled donor-assisted, not learned semantic operators.

SAEs are benchmark-trained ReLU models at width 128. Polygram's real target-K
merge API ranks pairs by training co-firing Jaccard and selects representatives
by firing counts (no ablation-KL is available); this is not a quantum-Gram experiment or causal redundancy
confirmation. Requested widths 64/32 may differ from actual kept width under
Polygram's planner, so controls also match actual measured decoder ranks.

All methods use train-only centering/scaling, with validation checkpoint selection.
PCA is fit only on training rows. The learned subspace uses activation MSE plus
distillation of the host's distribution restricted to the training answer-token
vocabulary; evaluation still measures full-vocabulary KL. Three fixed seeds are
reported separately. Fit budgets, supervision, and stored/fitted capacities are
recorded rather than assuming that equal rank means equal parameter counts.

Before final evaluation, the SAE implementation was corrected to constrain decoder
directions to unit norm, preventing a sparsity-penalty rescaling escape. All three
seeds were refit; the superseded pilot remains separate. Final fit hashes are in
`runs/binding/fits/frozen.json`.

The real forge preserves Qwen's RoPE theta of 1,000,000 by explicitly reading the
Transformers 5 `rope_parameters` field into sae-forge's public native config. The
installed adapter otherwise reads the obsolete top-level field and defaults to
10,000. Tied host embeddings are cloned and untied without changing host outputs
before projection: the transformed embedding and unembedding generally differ for
a non-orthogonal basis. Both corrections are checked on tiny native models and
recorded in build metadata. They change no host weights numerically.
