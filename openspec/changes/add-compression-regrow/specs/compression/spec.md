## ADDED Requirements

### Requirement: Regrower repopulates zeroed SAE slots with new directions

`polygram.compression.Regrower` SHALL be a dataclass that consumes
a compressed SAE checkpoint, an explicit zeroed-feature-id set,
and an activation-residual stream (either captured from prompts or
supplied as a numpy array), runs a named strategy to choose a new
direction for each zeroed slot, and rewrites the SAE checkpoint
with those slots populated. The `Regrower` itself SHALL be
torch-free at the orchestrator level; torch is lazy-imported only
inside the residual-capture path triggered by supplying `prompts`.

The dataclass SHALL expose at least the following fields:

- `sae_checkpoint: Path` (required, must exist)
- `strategy: str` (required, no default; member of
  `RegrowStrategy`)
- `zeroed: set[int]` (required, all values must be valid feature
  ids in the source checkpoint)
- `seed: int = 0`
- `n_init: int = 4`
- `prompts: Sequence[str] | None = None`
- `cached_residuals: np.ndarray | None = None`
- `model_name: str = "gpt2"`
- `layer: int = 10`
- `device: str | None = None`

Exactly one of `prompts` or `cached_residuals` MUST be supplied.
`__post_init__` SHALL validate every field and raise `ValueError`
(with the field name and offending value) on any violation.

#### Scenario: end-to-end regrow on a synthetic SAE with cached residuals

- **GIVEN** a synthetic SAE checkpoint produced by
  `tests._synth_sae.synth_sae` with 16 features × 8 d_model
- **AND** a `zeroed` set `{2, 5, 9, 13}` (4 slots whose tensors
  in the source checkpoint are all zero)
- **AND** a cached residual array of shape `(100, 8)` with
  non-trivial structure
- **WHEN** `Regrower(sae_checkpoint=path, strategy='residual_kmeans',
  zeroed={2, 5, 9, 13}, cached_residuals=R, seed=0).run(
  output_checkpoint=tmp / 'out.safetensors')` is called
- **THEN** the call SHALL succeed and emit a `RegrowResult`
- **AND** `result.report.n_slots_repopulated` SHALL be in
  `[1, 4]` (the strategy may leave a slot zero if its
  cluster ends up empty; up to 4 slots populated)
- **AND** the rewritten checkpoint's `W_dec[fid, :]` for every
  populated slot SHALL have L2 norm ≈ 1.0 (within 1e-6)
- **AND** the rewritten checkpoint's `W_enc[:, fid]` for every
  populated slot SHALL be byte-equal to `W_dec[fid, :]`
- **AND** the rewritten checkpoint's `b_enc[fid]` for every
  populated slot SHALL be 0
- **AND** the rewritten checkpoint's `b_dec` SHALL be byte-equal
  to the source's `b_dec`
- **AND** the rewritten checkpoint's tensors at every non-zeroed
  feature id SHALL be byte-equal to the source

#### Scenario: post_init rejects supplying both prompts and cached_residuals

- **GIVEN** an existing SAE checkpoint
- **WHEN** `Regrower(sae_checkpoint=path,
  strategy='residual_kmeans', zeroed={0}, prompts=['x'],
  cached_residuals=np.zeros((10, 8)))` is constructed
- **THEN** `__post_init__` SHALL raise `ValueError` whose
  message names both `prompts` and `cached_residuals` and
  states that exactly one must be supplied

#### Scenario: post_init rejects neither prompts nor cached_residuals

- **GIVEN** an existing SAE checkpoint
- **WHEN** `Regrower(sae_checkpoint=path,
  strategy='residual_kmeans', zeroed={0})` is constructed
  with neither `prompts` nor `cached_residuals`
- **THEN** `__post_init__` SHALL raise `ValueError`

#### Scenario: post_init rejects out-of-range zeroed fids

- **GIVEN** an SAE checkpoint with 16 features
- **WHEN** `Regrower(sae_checkpoint=path,
  strategy='residual_kmeans', zeroed={20},
  cached_residuals=R)` is constructed (20 ≥ 16)
- **THEN** `__post_init__` SHALL raise `ValueError` whose
  message names the offending feature id `20` and the
  source checkpoint's feature count `16`

### Requirement: residual_kmeans strategy populates slots from k-means cluster centroids

The `residual_kmeans` strategy SHALL:

1. Compute the residual stream
   `residual_stream = residuals - sae_reconstruct(residuals)`
   using the source checkpoint's `W_enc`, `b_enc`, `W_dec`,
   `b_dec` (post-ReLU SAE activations, decode back through the
   surviving features). Pure numpy.
2. Run `sklearn.cluster.KMeans(n_clusters=K, n_init=self.n_init,
   random_state=self.seed, algorithm='lloyd')` on
   `residual_stream`, where `K = len(zeroed)`.
3. Assign centroid `k` to the k-th feature id in
   `sorted(zeroed)`. For each populated slot with non-empty
   cluster: write `W_dec[fid, :] = centroid / max(‖centroid‖,
   eps)`, `W_enc[:, fid] = W_dec[fid, :]`, `b_enc[fid] = 0`.
4. For each slot whose assigned cluster is empty (zero residual
   tokens), leave the slot's tensors at zero (do not populate).
   Record the slot in `RegrowReport.n_slots_left_zero`.

`b_dec` MUST NOT be modified. Slots not in `zeroed` MUST NOT be
modified.

The strategy SHALL be deterministic: two calls with the same
`(sae_checkpoint_bytes, zeroed, residuals_bytes, seed, n_init)`
inputs MUST produce byte-identical `output_checkpoint` files
(modulo safetensors metadata mtime; the tensor bytes must match
exactly).

#### Scenario: residual_kmeans produces unit-norm decoder directions

- **GIVEN** a `Regrower(strategy='residual_kmeans', ...)` with
  `K = 4` zeroed slots and a 200-token residual array with clear
  cluster structure
- **WHEN** `run()` completes
- **THEN** every populated slot's `W_dec[fid, :]` SHALL have
  `‖W_dec[fid, :]‖_2 ∈ [0.999, 1.001]`
- **AND** every populated slot's `SlotPopulation.decoder_norm`
  field SHALL be `1.000000` (formatted to 6 sigfigs)

#### Scenario: residual_kmeans is byte-deterministic on cached_residuals

- **GIVEN** two `Regrower` instances configured with identical
  `sae_checkpoint`, `zeroed`, `cached_residuals`, `seed`,
  `n_init`, and `strategy='residual_kmeans'`
- **WHEN** both run `apply()` to separate output paths
- **THEN** the two output checkpoints' `safetensors.numpy.load_file`
  results SHALL be element-wise byte-equal across every tensor
- **AND** their sha256 hashes SHALL be equal

#### Scenario: residual_kmeans raises on no-signal residual stream

- **GIVEN** a residual array of shape `(100, 8)` filled with
  values whose `np.std() < 1e-9`
- **WHEN** `Regrower.plan()` runs with that array
- **THEN** `RuntimeError` SHALL be raised with a message
  mentioning "residual stream has no signal" and suggesting
  a more diverse prompt set

#### Scenario: residual_kmeans raises when n_tokens < K

- **GIVEN** `zeroed = {0, 1, 2, 3, 4, 5}` (K = 6)
- **AND** `cached_residuals.shape == (4, 8)` (n_tokens = 4)
- **WHEN** `Regrower.plan()` runs
- **THEN** `ValueError` SHALL be raised with a message naming
  `n_residual_tokens=4` and `K=6`

### Requirement: future strategies are reserved but not yet implemented

The strategy enum `RegrowStrategy` SHALL include
`high_decoder_norm_random` and `orthogonal_noise_scaled` as
reserved members. Calling `Regrower(strategy=
'high_decoder_norm_random', ...).apply(...)` or the orthogonal
counterpart SHALL raise `NotImplementedError` with a message
naming the requested strategy and stating it is reserved for a
future change.

#### Scenario: requesting a reserved strategy raises NotImplementedError

- **GIVEN** a `Regrower` constructed with
  `strategy='high_decoder_norm_random'` and otherwise valid
  arguments
- **WHEN** `apply()` is called
- **THEN** `NotImplementedError` SHALL be raised
- **AND** the exception message SHALL include the substring
  `'high_decoder_norm_random'` and a hint about future changes

### Requirement: from_compression_report constructor populates provenance

The chained constructor `Regrower.from_compression_report` SHALL
extract `zeroed` as the union over
`report.plan.clusters[*].zeroed`, construct a `Regrower`
instance, and populate an internal provenance map carrying
`compression_report_source_sha256`,
`compression_report_output_sha256`, and
`compression_report_dictionary_name` from the report. The
resulting `RegrowReport.provenance` SHALL contain those fields.

A `Regrower` constructed via the direct constructor SHALL emit
a `RegrowReport` whose `provenance` field is an empty dict.

#### Scenario: chained constructor copies report sha256s into provenance

- **GIVEN** a `CompressionReport` with `source_checkpoint_sha256
  = "abc...123"` and `output_checkpoint_sha256 = "def...456"`,
  `validation_report_dictionary_name = "MyDict"`, and clusters
  whose `zeroed` lists contain feature ids `{42, 100, 256}`
- **WHEN** `Regrower.from_compression_report(report,
  sae_checkpoint=path, strategy='residual_kmeans',
  cached_residuals=R)` is constructed and `run()` produces a
  `RegrowResult`
- **THEN** `result.report.provenance['compression_report_source_sha256']`
  SHALL equal `"abc...123"`
- **AND** `result.report.provenance['compression_report_output_sha256']`
  SHALL equal `"def...456"`
- **AND** `result.report.provenance['compression_report_dictionary_name']`
  SHALL equal `"MyDict"`
- **AND** the `Regrower.zeroed` field SHALL equal `{42, 100, 256}`

#### Scenario: direct constructor leaves provenance empty

- **GIVEN** a `Regrower` constructed via the direct constructor
  with `zeroed={42, 100}`
- **WHEN** `run()` produces a `RegrowResult`
- **THEN** `result.report.provenance` SHALL equal `{}`

### Requirement: apply() writes atomically and refuses to overwrite source

`Regrower.apply(plan, output_checkpoint)` SHALL write the
rewritten checkpoint atomically: a sibling temp file is written
first, then `os.replace`'d to the final path. If
`output_checkpoint` resolves to the same path as
`self.sae_checkpoint`, `apply()` SHALL raise `ValueError` before
any file I/O. The source checkpoint MUST never be modified.

#### Scenario: source bytes unchanged after run

- **GIVEN** a source SAE checkpoint at `path`
- **AND** `before = sha256(path.read_bytes())`
- **WHEN** `Regrower(...).run(output_checkpoint=other_path)`
  completes successfully
- **THEN** `sha256(path.read_bytes()) == before`

#### Scenario: output equal to source raises

- **GIVEN** a `Regrower` with `sae_checkpoint=path`
- **WHEN** `apply(plan, output_checkpoint=path)` is called
- **THEN** `ValueError` SHALL be raised before any file I/O,
  with a message naming both paths

### Requirement: RegrowReport carries provenance and is JSON round-trippable

The orchestrator's `RegrowReport` SHALL serialize to and
deserialize from JSON exactly:
`RegrowReport.from_json(report.to_json()) == report` MUST hold
for any well-formed instance. The JSON layout SHALL match
`design.md` Decision 8 and SHALL carry at least:
`schema_version`, `source_checkpoint`,
`source_checkpoint_sha256`, `output_checkpoint`,
`output_checkpoint_sha256`, `strategy`, `n_slots_repopulated`,
`n_slots_left_zero`, `feature_ids`, `plan` (with the
`SlotPopulation` array), `strategy_params`, `provenance`.

Floats SHALL be rounded to six significant figures via the
`format(v, ".6g")` discipline that `CompressionReport` uses.

#### Scenario: round-trip preserves report state including provenance

- **GIVEN** a hand-built `RegrowReport` with 5 populated slots,
  1 left-zero slot, and a populated `provenance` dict
- **WHEN** `r2 = RegrowReport.from_json(r.to_json())`
- **THEN** `r2 == r` (using `RegrowReport.__eq__`)
- **AND** `r2.provenance == r.provenance` (dict equality)
- **AND** `r2.plan.slots[0].decoder_norm == r.plan.slots[0].decoder_norm`

#### Scenario: required keys are present in serialized JSON

- **GIVEN** any `RegrowReport` instance `r`
- **WHEN** `payload = json.loads(r.to_json())`
- **THEN** `payload` SHALL contain every key listed in
  `design.md` Decision 8 (`schema_version`,
  `source_checkpoint`, `source_checkpoint_sha256`,
  `output_checkpoint`, `output_checkpoint_sha256`, `strategy`,
  `n_slots_repopulated`, `n_slots_left_zero`, `feature_ids`,
  `plan`, `strategy_params`, `provenance`)

### Requirement: empty zeroed set is a no-op

`Regrower` constructed with `zeroed=set()` SHALL produce, on
`run()`, an `output_checkpoint` whose tensors are byte-equal to
the source's tensors and whose `RegrowReport` carries
`n_slots_repopulated=0`, `n_slots_left_zero=0`, and `plan.slots
== ()`.

#### Scenario: empty zeroed set produces tensor-identical output

- **GIVEN** a `Regrower(zeroed=set(), strategy='residual_kmeans',
  cached_residuals=R, ...)` over a source checkpoint
- **WHEN** `run(output_checkpoint=other_path)` completes
- **THEN** `safetensors.numpy.load_file(source) ==
  safetensors.numpy.load_file(other_path)` (element-wise byte
  equality across every tensor)
- **AND** `result.report.n_slots_repopulated == 0`
- **AND** `result.report.n_slots_left_zero == 0`
- **AND** `result.report.plan.slots == ()`
