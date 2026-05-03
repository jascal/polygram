## Why

Polygram's experimental power is bottlenecked by where its `Dictionary`
inputs come from. Today every example hand-codes 4 features with
synthetic β values. To bridge to real mechanistic-interpretability
work, researchers need a way to pull features from SAE-Lens / Anthropic
sparse autoencoders and feed them into `InterferenceSweep`.

The hard constraint is *capacity*: Polygram's rung-1 MPS encoding maps
each feature to a 3-qubit state with a single β scalar; the analytic
Gram is N×N where N stays small (≤8). Real SAEs ship 16k–1M features.
Any importer that pretends to ingest a full SAE will mislead users.

The right shape is **explicit selection**: the user names a small
subset (≤8 features by ID or name); Polygram clusters their projection
vectors to assign β, surfaces fidelity stats so the user sees what
was lost in projection, and refuses subsets that don't fit. The
output is a `Dictionary` plus a `SelectionReport` — both first-class
return values, so notebooks can show "we kept N/M features, β
explains X% of selected-feature variance, cluster assignments
came from k-means(k=2)".

This change introduces that surface area, ships a deterministic toy
fixture for tests + the example, and stays out of the
sae-lens-format-parsing rabbit hole until a real loader is needed.

## What Changes

- **NEW** `sae` capability:
  - `polygram.sae_import.SAEFeatureRecord` — frozen dataclass holding
    the data Polygram actually consumes from an SAE: `feature_id: int`,
    `name: str`, `projection: np.ndarray` (decoder column, 1D),
    `label: str | None = None`, `activation_mean: float | None = None`,
    `activation_std: float | None = None`.
  - `polygram.sae_import.SelectionReport` — frozen dataclass returned
    alongside the Dictionary, summarizing the lossy projection: input
    feature count, kept count, cluster method (`"user"`, `"kmeans"`,
    or `"from_labels"`), per-feature cluster assignment, β-variance-
    explained fraction (norm² of mean-cluster β signal divided by
    norm² of selected projection vectors), and a list of warnings.
  - `polygram.sae_import.load_toy_sae(path) -> dict[int, SAEFeatureRecord]`
    — JSON loader for the bundled fixture format (and any external
    JSON following the same schema). The schema documents one feature
    per dict entry: `{"feature_id", "name", "projection",
    "label", "activation_mean", "activation_std"}`.
  - `polygram.sae_import.from_sae_lens(records, feature_ids, *,
    name="ImportedSAE", cluster_assignments=None, n_clusters=None,
    encoding=None, beta_range=(-0.5, 0.5)) -> tuple[Dictionary,
    SelectionReport]` — the headline function. `records` is a
    `dict[int, SAEFeatureRecord]` (from `load_toy_sae` or a future
    `load_sae_lens` for real `.safetensors` / SAE-Lens torch state).
    `feature_ids` is the explicit selection. `cluster_assignments`
    optionally maps `feature_id → cluster_name`; if absent and
    feature labels look like `"<cluster>/<feature>"`, we parse them
    (`"from_labels"`); otherwise we run k-means on the projection
    vectors with `n_clusters` (default 2) and use cluster index.
    β is spread evenly across cluster means in `beta_range`. γ
    defaults to 0; α defaults to 0; φ defaults to 0.
  - The function SHALL refuse subsets larger than 8 features with a
    clear `ValueError` naming the limit and the selected count.
- **NEW** `tests/fixtures/toy_sae.json` — 16 features, 4 clusters
  (mammals: dog_poodle, dog_beagle, cat_persian, cat_siamese; birds:
  hawk_red, hawk_cooper, sparrow_house, sparrow_song; vehicles:
  car_sedan, car_suv, plane_jet, plane_prop; fruits:
  berry_strawberry, berry_blueberry, citrus_lemon, citrus_orange).
  8-dim projection vectors with cluster-mean directions plus within-
  cluster perturbations. Used by tests and the new example.
- **NEW** `examples/import_from_sae.py` — loads the toy fixture, picks
  4 features (`dog_poodle, dog_beagle, hawk_red, hawk_cooper`), runs
  a `bird_hawk.phi`-style sweep, materializes the Q-Orca artifact +
  CSV + plot. Documents the swap-in path for real SAE-Lens / HF
  safetensors files at the top of the file.
- **MODIFIED** `pyproject.toml` — `[project.optional-dependencies] sae`
  extra reserved (empty in v0; placeholder for `safetensors`,
  `huggingface_hub`, and `sae_lens` once real loaders land).
- **MODIFIED** `README.md` — install snippet, capacity-limits callout,
  and the SAE-import example invocation. Three to five sentences on
  what the importer is and is not.
- **NEW** `tests/test_sae_import.py` — covers: fixture loads
  cleanly; explicit cluster assignment honored; k-means default works
  on the toy fixture; from-labels parse path; >8 selection rejected;
  β-variance-explained is in [0, 1] and 1.0 when k-means perfectly
  separates clusters; SelectionReport surfaces a warning when k-means
  produces an empty cluster.

## Capabilities

### New Capabilities

- `sae` — read SAE feature records from disk, project a user-selected
  subset down to a Polygram `Dictionary`, return a fidelity report.

### Modified Capabilities

*(none — `experiment` and `cli` unchanged)*

## Impact

- `polygram/sae_import.py` — new module
- `tests/fixtures/toy_sae.json` — bundled deterministic fixture
- `tests/test_sae_import.py` — new coverage (~10 tests)
- `examples/import_from_sae.py` — new walking tour
- `examples/output/import_from_sae/` (gitignored)
- `pyproject.toml` — `sae` extra placeholder
- `README.md` — install + SAE-import example + capacity-limits note
- No runtime dependency on `sae_lens` / `safetensors` / `torch` /
  `huggingface_hub`. K-means is implemented in plain numpy (~30 LOC).
- No q-orca version bump. No physics changes. No CLI surface change.
