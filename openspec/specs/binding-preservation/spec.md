# binding-preservation Specification

## Purpose
Measure preservation of role–filler bindings under representation replacement,
targeted edits, and independent forged execution, with disjoint evaluation data,
explicit capacity accounting, and reproducible empirical evidence.
## Requirements
### Requirement: Minimal pairs isolate bindings
The benchmark SHALL include subject/object swaps, variable/value reassignment,
and unambiguous modifier-attachment pairs with identical lexical content and
different correct answers. Every pair SHALL stay within one partition. Ambiguous
attachment examples SHALL NOT be assigned an unsupported unique answer.

#### Scenario: Pair semantics and content
- **WHEN** a dataset is generated
- **THEN** both members have identical word multisets, opposite binding answers,
  machine-readable role–filler assignments, and a shared pair identifier.

### Requirement: Evaluation partitions prevent leakage
The benchmark SHALL keep training, validation, combination-test, and template-test
partitions deterministic and disjoint. Withheld role–filler combinations in final
tests SHALL be absent from fitting and validation while their constituent fillers
and roles occur in training. Development host selection SHALL use separate items.
All candidate representations SHALL be evaluated on the same final items, with
all fitting, atom selection, and checkpoint choice restricted to the fit/validation
partitions. The final test SHALL NOT determine hyperparameters or host selection.

#### Scenario: Split audit
- **WHEN** the manifest is audited
- **THEN** duplicate prompts, cross-partition pairs, unseen individual symbols,
  and leaked reserved combinations are rejected.

### Requirement: Host competence gates interpretation
Before representation comparisons, the benchmark SHALL measure candidate-answer
and both-members-correct accuracy for every task on development data. A protocol
SHALL be frozen after selection. Final host failures SHALL be retained and reported
separately from representation-induced failures, never silently filtered.

#### Scenario: Host lacks a task
- **WHEN** the host fails the preregistered competence threshold
- **THEN** that task is marked ineligible for preservation conclusions under that
  host and remains visible in the research report.

### Requirement: Representation comparisons expose capacity
The study SHALL compare ordinary SAE reconstruction, Polygram-compressed SAE,
supervised role–filler reconstruction, PCA, and learned-subspace controls. Controls
SHALL match measured reconstruction rank where feasible, and every result SHALL
include actual rank, latent width, trainable and stored parameter counts, seed,
fit data size, and validation selection rule. Semantic supervision SHALL be
explicit; it SHALL NOT be described as unsupervised discovery.

#### Scenario: Compression does not reduce width
- **WHEN** a Polygram configuration produces no merges
- **THEN** it is reported as an identity compression, not as evidence of savings.

### Requirement: Causal evaluation includes specificity
The study SHALL evaluate activation replacement and targeted role changes, report
task accuracy, pair accuracy, host-output agreement, and KL in nats, and measure
preservation of unrelated behavior. It SHALL include identity/no-op controls and
an actual counterfactual-input reference. Intervention locations and whether
multiple token positions are changed SHALL be explicit. Confidence intervals SHALL
resample independent pairs rather than treating pair members as independent.

#### Scenario: Role swap specificity
- **WHEN** a representation edit swaps the queried binding
- **THEN** success is assessed against the counterfactual answer and unrelated
  bindings are separately checked for preservation.

### Requirement: Forged execution is independently measured
The study SHALL compare activation replacement with independently running forged
models built through sae-forge. Native-in-basis execution SHALL be explicit and
SHALL NOT silently fall back to host-wrapped execution. The same basis SHALL also
be evaluated as an activation projection to distinguish representation damage
from accumulated execution damage. Unsupported models or failed construction
SHALL be reported rather than replaced with activation-hook simulations.

#### Scenario: Independent execution
- **WHEN** a saved forged model is evaluated
- **THEN** it takes token inputs, runs without host activations or teacher hooks,
  and its metrics are compared on the same frozen evaluation items.
