# sae Specification (delta)

## MODIFIED Requirements

### Requirement: SelectionReport exposes cluster_assignments by feature id

`SelectionReport` SHALL expose `cluster_assignments: dict[str, str]` mapping each selected feature's *name* to its cluster name. `ClusterConfirmer` consumes this field to derive confirmed pairs; the field is already present in the existing spec but its use as input to a `Confirmer` is now a normative requirement.

#### Scenario: cluster_assignments covers every selected feature

- **WHEN** `from_sae_lens(records, [0, 1, 4, 5], n_clusters=2)` is called
- **THEN** `report.cluster_assignments` has exactly 4 entries, one per selected feature name
