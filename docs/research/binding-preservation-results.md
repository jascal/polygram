# Binding-preservation benchmark: empirical report

Host: `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
Three fitting seeds; final partitions were withheld from all fitting and selection.
This is a final-residual readout experiment, followed by an independent whole-model forge.
It does not identify the upstream algorithm that creates bindings.

## Main findings

- Co-firing compression retains 85.5% combination accuracy, versus 100.0% for SAE reconstruction and 100.0% for rank-64 PCA.
- Supervised TPR combination accuracy ranges from 74.7% to 100.0% across seeds. It does not establish a robust advantage over the linear controls.
- Across both final partitions, same-basis projection retains 94.4% accuracy. Independent native execution averages 33.7%, essentially the 33.3% three-choice chance baseline; its pair accuracy is 0.000 and its KL is 11–14 nats. We therefore describe the native forge as chance-level with destroyed output distributions, not as retaining the decision. The gap from projection is the finding.

### Role-swap specificity (task-macro means)

Target accuracy is scored against the counterfactual host; unrelated accuracy is scored against
the original host. The latter remains 1.000, while target performance shows whether the edit
changes the queried binding.
| Method | Combination target | Combination unrelated | Template target | Template unrelated |
|---|---:|---:|---:|---:|
| sae | 1.000 | 1.000 | 0.970 | 1.000 |
| compressed | 0.770 | 1.000 | 0.806 | 1.000 |
| tpr | 0.917 | 1.000 | 0.998 | 1.000 |
| pca64 | 1.000 | 1.000 | 0.986 | 1.000 |
| learned32 | 0.997 | 1.000 | 0.986 | 1.000 |

## Host selection and protocol amendment

The initial Qwen2.5-0.5B-Instruct screen was retained as a failed host-selection run.
The 0.5B screen failed the preregistered per-task competence gates on subject/object (0.375 candidate accuracy, 0.000 both-members accuracy) and modifier attachment (0.875, 0.750); variable/value passed (1.000, 1.000).
It was dropped before final capture and fitting; the frozen Qwen2.5-1.5B-Instruct host
passed all three development gates. This decision and the independent SAE decoder-norm
amendment are recorded in `runs/binding/protocol-amendments.json`; neither changes
final-test tuning.

## Host competence on final partitions

Accuracy uses three candidate answers. Intervals resample the 48 minimal pairs in each cell.
| Task / partition | Accuracy | Pair-bootstrap 95% interval | Both members |
|---|---:|---:|---:|
| subject_object/test_combinations | 1.000 | 1.000–1.000 | 1.000 |
| subject_object/test_templates | 0.917 | 0.865–0.969 | 0.833 |
| variable_value/test_combinations | 1.000 | 1.000–1.000 | 1.000 |
| variable_value/test_templates | 1.000 | 1.000–1.000 | 1.000 |
| modifier_attachment/test_combinations | 1.000 | 1.000–1.000 | 1.000 |
| modifier_attachment/test_templates | 1.000 | 1.000–1.000 | 1.000 |

## Capacity and fitting

Fits use 2,304 training and 576 validation query records. Optimized methods have a fixed
1,200-step budget, with checkpoint selection by validation objective every 50 steps,
including step zero. PCA uses training rows only. No final test selects a fit.

| Method | Decoder rank (seeds 0/1/2) | Gradient-fitted parameters (seed 0) | Stored scalars (seed 0) |
|---|---|---:|---:|
| sae | 128/128/128 | 393,344 | 394,881 |
| compressed | 73/90/100 | 393,344 | 394,880 |
| tpr | 64/64/64 | 100,488 | 102,025 |
| pca32 | 32/32/32 | 0 | 50,689 |
| pca64 | 64/64/64 | 0 | 99,841 |
| pca_matched | 73/90/100 | 0 | 113,665 |
| learned32 | 32/32/32 | 49,152 | 50,689 |
| learned64 | 64/64/64 | 98,304 | 99,841 |
| learned_matched | 73/90/100 | 112,128 | 113,665 |
| pca128 | 128/128/128 | 0 | 198,145 |
| learned128 | 128/128/128 | 196,608 | 198,145 |

The SAE has unit-norm decoder directions. Compression preserves zeroed rows on disk;
its metadata also reports active scalars. Requested targets 32 and 64 yielded identical
checkpoints within every seed; they are one compression result per seed, not two independent
budget points. Matched controls use the actual compressed decoder ranks.

Mean active SAE latents on training records (seeds 0/1/2): 52.12/54.75/53.75 of 128; no dead latents. This is moderate sparsity, not a single-feature code.

At ranks 64 and above, validation selected the learned control's step-zero PCA initialization.
Those matching results are not independent improvements from learned optimization.

## Replacement

Numbers are task-macro means, then means across seeds. Full per-task results and pair-bootstrap
intervals remain in each seed's JSON. Seed ranges below are not confidence intervals.

### test_combinations

| Method | Accuracy | Both members | Host argmax agreement | KL (nats) | Accuracy seed range |
|---|---:|---:|---:|---:|---:|
| identity | 1.000 | 1.000 | 1.000 | 0.0000 | 1.000–1.000 |
| sae | 1.000 | 1.000 | 0.959 | 0.0083 | 1.000–1.000 |
| compressed | 0.855 | 0.722 | 0.766 | 0.9347 | 0.806–0.938 |
| tpr | 0.912 | 0.824 | 0.792 | 0.8128 | 0.747–1.000 |
| pca32 | 0.917 | 0.833 | 0.764 | 0.6168 | 0.917–0.917 |
| pca64 | 1.000 | 1.000 | 0.965 | 0.0063 | 1.000–1.000 |
| pca_matched | 1.000 | 1.000 | 0.962 | 0.0047 | 1.000–1.000 |
| learned32 | 0.995 | 0.991 | 0.944 | 0.0963 | 0.993–0.997 |
| learned64 | 1.000 | 1.000 | 0.965 | 0.0063 | 1.000–1.000 |
| learned_matched | 1.000 | 1.000 | 0.962 | 0.0047 | 1.000–1.000 |
| pca128 | 1.000 | 1.000 | 0.965 | 0.0029 | 1.000–1.000 |
| learned128 | 1.000 | 1.000 | 0.965 | 0.0029 | 1.000–1.000 |

### test_templates

| Method | Accuracy | Both members | Host argmax agreement | KL (nats) | Accuracy seed range |
|---|---:|---:|---:|---:|---:|
| identity | 0.972 | 0.944 | 1.000 | 0.0000 | 0.972–0.972 |
| sae | 0.968 | 0.949 | 0.971 | 0.0190 | 0.962–0.972 |
| compressed | 0.834 | 0.683 | 0.788 | 1.0668 | 0.785–0.875 |
| tpr | 0.998 | 0.995 | 0.947 | 0.3417 | 0.993–1.000 |
| pca32 | 0.965 | 0.944 | 0.958 | 0.0214 | 0.965–0.965 |
| pca64 | 0.979 | 0.958 | 0.976 | 0.0109 | 0.979–0.979 |
| pca_matched | 0.973 | 0.947 | 0.980 | 0.0102 | 0.972–0.976 |
| learned32 | 0.978 | 0.958 | 0.961 | 0.0221 | 0.976–0.983 |
| learned64 | 0.979 | 0.958 | 0.976 | 0.0109 | 0.979–0.979 |
| learned_matched | 0.973 | 0.947 | 0.980 | 0.0102 | 0.972–0.976 |
| pca128 | 0.972 | 0.944 | 0.972 | 0.0165 | 0.972–0.972 |
| learned128 | 0.972 | 0.944 | 0.972 | 0.0165 | 0.972–0.972 |

## Role edits and unrelated bindings

TPR edits use input parse annotations only. Other methods use counterfactual
activation donors; those results do not establish autonomous semantic editing.
Target edits are compared with the counterfactual host, unrelated edits with the original host.

### test_combinations

| Method | Edited target accuracy | Unrelated accuracy | Unrelated KL to original host |
|---|---:|---:|---:|
| identity | 1.000 | 1.000 | 0.0068 |
| sae | 1.000 | 1.000 | 0.0051 |
| compressed | 0.770 | 1.000 | 0.0031 |
| tpr | 0.917 | 1.000 | 0.0003 |
| pca32 | 0.917 | 1.000 | 0.0025 |
| pca64 | 1.000 | 1.000 | 0.0038 |
| pca_matched | 1.000 | 1.000 | 0.0046 |
| learned32 | 0.997 | 1.000 | 0.0024 |
| learned64 | 1.000 | 1.000 | 0.0038 |
| learned_matched | 1.000 | 1.000 | 0.0046 |
| pca128 | 1.000 | 1.000 | 0.0055 |
| learned128 | 1.000 | 1.000 | 0.0055 |

### test_templates

| Method | Edited target accuracy | Unrelated accuracy | Unrelated KL to original host |
|---|---:|---:|---:|
| identity | 0.972 | 1.000 | 0.0168 |
| sae | 0.970 | 1.000 | 0.0157 |
| compressed | 0.806 | 1.000 | 0.0087 |
| tpr | 0.998 | 1.000 | 0.0002 |
| pca32 | 0.983 | 1.000 | 0.0196 |
| pca64 | 0.986 | 1.000 | 0.0201 |
| pca_matched | 0.980 | 1.000 | 0.0182 |
| learned32 | 0.986 | 1.000 | 0.0189 |
| learned64 | 0.986 | 1.000 | 0.0201 |
| learned_matched | 0.980 | 1.000 | 0.0182 |
| pca128 | 0.976 | 1.000 | 0.0170 |
| learned128 | 0.976 | 1.000 | 0.0170 |

## Independent execution

Native models were saved and reloaded, then run on token IDs without host activation
hooks. RoPE is preserved; tied host weights are numerically unchanged but untied
before projection. No fine-tuning or host-wrapped fallback is used.

| Seed | Actual basis width | Split | Projection accuracy | Native accuracy | Projection KL | Native KL |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 73 | test_combinations | 0.948 | 0.333 | 0.7219 | 12.2999 |
| 0 | 73 | test_templates | 0.885 | 0.312 | 0.8378 | 12.4029 |
| 1 | 90 | test_combinations | 0.990 | 0.340 | 0.2119 | 14.0238 |
| 1 | 90 | test_templates | 0.938 | 0.306 | 0.2063 | 14.5662 |
| 2 | 100 | test_combinations | 0.983 | 0.358 | 0.2726 | 11.3008 |
| 2 | 100 | test_templates | 0.924 | 0.372 | 0.3992 | 11.2250 |

## Interpretation limits

- Candidate accuracy is discrimination among three alternatives, not free generation.
- Host failures are retained; conditional-on-host-correct scores are in the JSON reports.
- TPR supervision and donor-assisted baselines have different information budgets.
- No annotation-matched additive or atomic-pair control was run; a TPR advantage
  would not isolate the benefit of tensor-product factorization.
- Equal decoder rank does not equal parameter count, sparsity, or supervision.
- A single final-layer SAE basis used across all layers can fail because the layers need
  different representations and nonlinear operations do not commute with projection.
- Fitted subspaces and negative outcomes are not global optima or irreducibility proofs.
- Holdouts apply to benchmark fitting, not to the host's pretraining history.
- The native 33.7% aggregate is interpreted against the 33.3% three-choice chance
  baseline; it is not described as retained decision accuracy.

## Relation to the paper and sibling projects

[DISCOVER](https://arxiv.org/pdf/2608.29530) motivates replacement and structured
intervention as stronger evidence than representational similarity. Here the role
assignments are supplied, and only a final question-conditioned residual is modeled.
This study neither replicates the paper's complete experiments nor discovers unique
internal symbolic roles.

For Polygram, the negative compression result concerns this co-firing merge configuration.
It does not evaluate quantum encodings or establish that every compression strategy fails.
Co-firing and similar scalar ablation KLs remain insufficient evidence of substitutability.
For lm-sae and sae-forge, the projection/execution comparison supports separating retained
readout information from the ability to execute the computation in that representation.
It supplies a bounded example, not a theorem about an intrinsic SAE composition tax.
The already retracted writer-output preservation claim in
[sae-forge's research note](../../../sae-forge/docs/two_basis_forge.md) is not used as evidence.

## Reproduction

See `examples/binding_benchmark/README.md`, `runs/binding/protocol.json`,
`runs/binding/protocol-amendments.json`, the recorded
protocol amendment, frozen fit hashes, per-example JSONL files and three forge build reports.
Large binary checkpoints are saved locally and gitignored.
