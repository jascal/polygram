# confirmation-strategies Specification

## ADDED Requirements

### Requirement: Confirmer protocol

`polygram` SHALL expose a `Confirmer` protocol with a single method `run() -> ValidationReport`. Any object satisfying this protocol MAY be used wherever a `ValidationReport` producer is expected. `BehaviouralValidator` satisfies `Confirmer` structurally with no modification.

#### Scenario: BehaviouralValidator satisfies Confirmer at runtime

- **WHEN** `isinstance(BehaviouralValidator(...), Confirmer)` is evaluated via `runtime_checkable`
- **THEN** it returns `True`

#### Scenario: custom object satisfying Confirmer is accepted

- **WHEN** a user-defined class implements `run(self) -> ValidationReport`
- **THEN** it satisfies `Confirmer` without inheriting from any base class

---

### Requirement: DecoderGeometryConfirmer confirms by decoder cosine²

`DecoderGeometryConfirmer` SHALL accept `records: dict[int, SAEFeatureRecord]`, `sae_checkpoint: Path`, `feature_ids: list[int]`, and `threshold: float = 0.8`. Its `.run()` SHALL return a `ValidationReport` where `confirmed` contains every `(i, j)` pair (with `i < j`) whose decoder cosine² is ≥ `threshold`. It SHALL NOT import torch or transformers.

#### Scenario: pairs above threshold are confirmed

- **WHEN** two features have decoder cosine² of 0.85 and threshold is 0.8
- **THEN** that pair appears in `ValidationReport.confirmed`

#### Scenario: pairs below threshold are not confirmed

- **WHEN** two features have decoder cosine² of 0.65 and threshold is 0.8
- **THEN** that pair does NOT appear in `ValidationReport.confirmed`

#### Scenario: no torch import occurs

- **WHEN** `DecoderGeometryConfirmer(...).run()` is called with torch absent from the environment
- **THEN** it completes without raising `ImportError`

#### Scenario: report metadata identifies the strategy

- **WHEN** `DecoderGeometryConfirmer(...).run()` returns a `ValidationReport`
- **THEN** `report.model_name == "geometry:decoder_cosine2"`
- **AND** `report.n_prompts == 0` and `report.n_tokens == 0`

#### Scenario: behavioural fields are NaN

- **WHEN** `DecoderGeometryConfirmer(...).run()` returns pairs
- **THEN** every `CandidatePair` has `jaccard`, `pearson_activation`, `kl_ablate_i`, `kl_ablate_j` equal to `float('nan')`

---

### Requirement: ClusterConfirmer confirms by cluster membership

`ClusterConfirmer` SHALL accept a `SelectionReport` (from `from_sae_lens`) and `sae_checkpoint: Path`. Its `.run()` SHALL return a `ValidationReport` where `confirmed` contains every `(i, j)` pair (with `i < j`) assigned to the same cluster in `SelectionReport.cluster_assignments`. It SHALL NOT import torch or transformers.

#### Scenario: within-cluster pairs are confirmed

- **WHEN** features A and B are both assigned to `cluster_0` in the `SelectionReport`
- **THEN** `(min(A,B), max(A,B))` appears in `ValidationReport.confirmed`

#### Scenario: cross-cluster pairs are not confirmed

- **WHEN** feature A is in `cluster_0` and feature B is in `cluster_1`
- **THEN** that pair does NOT appear in `ValidationReport.confirmed`

#### Scenario: singleton clusters produce no confirmed pairs

- **WHEN** every cluster in the `SelectionReport` has exactly one member
- **THEN** `ValidationReport.confirmed` is empty

#### Scenario: report metadata identifies the strategy

- **WHEN** `ClusterConfirmer(...).run()` returns a `ValidationReport`
- **THEN** `report.model_name == "geometry:cluster"`
- **AND** `report.n_prompts == 0` and `report.n_tokens == 0`

---

### Requirement: Both strategies are exported from polygram

`DecoderGeometryConfirmer`, `ClusterConfirmer`, and `Confirmer` SHALL be importable directly from `polygram`.

#### Scenario: top-level import succeeds

- **WHEN** `from polygram import DecoderGeometryConfirmer, ClusterConfirmer, Confirmer` is executed
- **THEN** all three names resolve without `ImportError`
