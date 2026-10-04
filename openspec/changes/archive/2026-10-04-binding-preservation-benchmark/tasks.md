## 1. Preregister and establish the host
- [x] 1.1 Validate this change before implementation.
- [x] 1.2 Build deterministic minimal pairs and disjoint split manifests.
- [x] 1.3 Test answer correctness, content preservation, and held-out combinations.
- [x] 1.4 Run development host gates and freeze the host/protocol before final testing.

## 2. Fit and compare representations
- [x] 2.1 Capture training/validation activations with exact layer and token alignment.
- [x] 2.2 Fit ordinary SAE, Polygram compression, role–filler, PCA, and learned controls.
- [x] 2.3 Record actual ranks, trainable/stored parameters, training data, and seeds.
- [x] 2.4 Select all fits and hyperparameters on validation data only.

## 3. Causal evaluation
- [x] 3.1 Evaluate replacement on untouched combination and template tests.
- [x] 3.2 Evaluate targeted role swaps and unrelated-behavior preservation.
- [x] 3.3 Report task accuracy, pair accuracy, agreement, KL, uncertainty, and controls.

## 4. Execution comparison
- [x] 4.1 Construct and save independent native-in-basis models through sae-forge.
- [x] 4.2 Verify the forged models execute without host activation injection.
- [x] 4.3 Compare identical-basis projection replacement against full forged execution.

## 5. Validation and research record
- [x] 5.1 Run meaningful data, intervention, metric, and execution tests.
- [x] 5.2 Save reproducible manifests, commands, artifacts, and a limitations-aware report.
- [x] 5.3 Audit all five user requirements and archive the completed change.
