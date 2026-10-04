# Binding preservation: completion record

The three-seed empirical study is complete. See the
[results and interpretation](binding-preservation-results.md),
[run commands](../../examples/binding_benchmark/README.md), and
[machine-readable completion audit](../../runs/binding/completion-audit.json).

## Requirement audit

| Requested requirement | Completed evidence |
|---|---|
| Minimal pairs and competent host | 2,088 audited examples / 1,044 pairs across subject/object, variable/value, and explicit modifier attachment. Qwen2.5-1.5B-Instruct passed separate development gates. Final host accuracy is 100% on combination tests; unseen subject/object templates score 91.7%, with other template tasks at 100%. |
| Five representation families with capacity accounting | Three seeds of SAE, actual Polygram co-firing merge, supervised role–filler, PCA and learned-subspace controls; actual decoder ranks and fitted/stored parameters reported. Requested compression widths 32/64 give identical checkpoints within seeds, with actual ranks 73/90/100 and matching controls. |
| Withheld combinations and templates | Deterministic disjoint train/validation/combination/template partitions, audited role–filler holdouts, and 87 fitted artifacts hashed before final capture. Hashes still match. |
| Replacement, role swaps and specificity | All three seeds evaluated on 1,152 held-out query records. Per-example results and per-task pair-bootstrap intervals include accuracy, both-members accuracy, host agreement, full-vocabulary KL, conditional scores, and unrelated-query preservation. Identity/no-op controls pass. |
| Independent forge versus same-basis projection | Three saved native models, at 84,512,577 / 104,180,234 / 115,749,444 parameters, run from token IDs without host activation hooks. Same-basis projection and native execution use identical final items. All outputs are finite; failures of preservation are retained. |

## Results

Co-firing compression averages 85.5% combination accuracy, compared with 100%
for SAE reconstruction and rank-64 PCA. Supervised TPR combination accuracy ranges
from 74.7% to 100% across seeds. Across both final partitions, same-basis projection
averages 94.4% accuracy. Independent native execution averages 33.7%, essentially
the 33.3% three-choice chance baseline; pair accuracy is 0.000 and KL is 11–14 nats.
The native forge is therefore reported as indistinguishable from chance at the
aggregate three-choice level, with destroyed output distributions, while the
projection still discriminates.

Role swaps are included in the report's main findings: unrelated-query accuracy is
1.000 for the listed methods, while target accuracy varies with the representation.

These are candidate-answer measurements at a final readout, followed by a native
whole-model execution test. They do not establish the upstream binding algorithm,
a unique symbolic representation, or an intrinsic lower bound on compressibility.
The report distinguishes semantic annotation supervision from donor-assisted edits.

## Verification and provenance

- Full Polygram suite: 1,113 passed, one skipped, 32 warnings (package-wide regression coverage).
- New benchmark tests: 11 focused Polygram tests passed; forge-specific tests in sae-forge's environment: two passed.
- Ruff and strict OpenSpec validation passed.
- All 87 frozen fit artifacts verified; complete pair cells and per-example ordering checked.
- Final capture: all 1,152 residuals finite, host argmax agreement 100%, maximum readout KL below 3e-10 nats.
- The failed Qwen2.5-0.5B host screen and pre-test SAE unit-decoder correction are recorded in the protocol amendment; the final Qwen2.5-1.5B protocol is frozen after the 0.5B failures.
- Evaluation source hashes and changes after the fit freeze are recorded in evaluation-provenance.json; fitting/data/host/capture sources remained frozen.
- Large binary activations and checkpoints are saved locally and gitignored; JSON reports and per-example outputs remain reviewable.

All capture, fitting, intervention, native-evaluation and test processes finished
successfully. No long-running job remains. The completed OpenSpec change is archived.
