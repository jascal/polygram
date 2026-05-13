# Polygram

**A laboratory for modelling polysemantic feature representations in LLMs.**

Polygram is a researcher-friendly Python frontend for mechanistic-interpretability
experiments on hierarchical polysemantic feature dictionaries. It explores
several approaches to representing, analyzing, and compressing SAE feature
dictionaries — including quantum-inspired MPS/HEA encodings (with phase-
interference sweeps, cancellation studies, and entanglement probes),
behavioural validation against real models, and component-level
compression / regrowth workflows.

The quantum-inspired encodings emit verifiable
[Q-Orca](https://github.com/jascal/q-orca-lang) `.q.orca.md` machines, building
on Q-Orca's rung-1 MPS encoding with safe `Rz` phase knobs (q-orca PR #51).
Quantum interference is one lens among several the project applies to
polysemantic representations, not the whole story.

## Status

Pre-alpha. v0 milestone is closed (bootstrap → core dictionary →
experiment/sweep → animals example, polished with tier stats / plots /
CLI / 2D landscapes → SAE import → `Cancellation` primitive). New
work is staged through OpenSpec changes — see `openspec/changes/`.

## Install

```bash
pip install -e ".[dev,plot]"   # editable install with test + plot deps
pytest                          # run the suite
```

Optional extras: `[plot]` (matplotlib), `[notebook]` (jupyter +
matplotlib), `[opt]` (scipy — enables the `Cancellation`
`method="scipy"` backend), `[sae]` (reserved for a future SAE-Lens /
safetensors loader; empty in v0 — JSON loader for the bundled toy
fixture has no extra deps), `[behavioural]` (torch + transformers,
required for `BehaviouralValidator.validate()` / `.run()`; the
torch-free `predict()` stage stays on the no-extras install).

## Layout

```
polygram/         — Python package
openspec/         — spec-driven change proposals + capability specs
tests/            — pytest suite + bundled fixtures
examples/         — Python scripts + notebook walking tour
docs/img/         — README screenshots
```

## Capacity limits

Per-encoding feature caps follow each encoding's reachable
Hilbert-space dimension (see
[`docs/research/rung3-rank-bound.md`](docs/research/rung3-rank-bound.md)
for the empirical basis):

| Encoding | `max_features` | Source |
|---|---|---|
| `MPSRung1` | 8 | `dim(C^8) = 8` (3-qubit register) |
| `Rung3` | 16 | `dim(C^8 ⊗ C^2) = 16` (amp branch restricted to 2-dim subspace) |
| `HEA_Rung2` | `2 ** n_qubits` | scales with the existing `n_qubits` knob |
| `Rung4` *(planned)* | 32 | `dim(C^8 ⊗ C^4) = 32` once the product-amp branch from `add-rung4-encoding-mvp` lands |

Real SAEs ship 16k–1M features per layer, well past any per-encoding
cap. Two paths to that scale:

- **Selection-first**: `from_sae_lens(...)` names a subset within
  the encoding's cap. `SelectionReport.beta_variance_explained`
  surfaces how lossy the projection-vector → β collapse was.
- **Clustered**: `from_sae_lens(..., clustered=True)` (or
  `polygram.clustered_dictionary.build_clustered_dictionary`)
  block-decomposes the SAE into per-encoding-cap-sized blocks plus
  a sparse cross-block adjacency; analyses delegate per-block.

## Quickstart

```python
import numpy as np

from polygram import Dictionary, Experiment, Feature, MPSRung1

dictionary = Dictionary(
    name="AnimalsInterference",
    features=[
        Feature("dog_poodle",   "dogs",  beta=-0.5),
        Feature("dog_beagle",   "dogs",  beta=-0.5),
        Feature("bird_hawk",    "birds", beta= 0.5),
        Feature("bird_sparrow", "birds", beta= 0.5),
    ],
    hierarchy={"dogs": ["dog_poodle", "dog_beagle"],
               "birds": ["bird_hawk", "bird_sparrow"]},
    encoding=MPSRung1(bond_dim=2, phase_knobs=True),
)

experiment = Experiment(
    name=dictionary.name,
    dictionary=dictionary,
    target_pair=("dog_poodle", "bird_hawk"),
    sweep={"bird_hawk.phi": np.linspace(0.0, np.pi, 40)},
    measures=["overlap", "gram_matrix", "schmidt_rank"],
    assertions=["hierarchical_ordering_preserved"],
)

experiment.materialize("examples/output/")   # emits a verifiable .q.orca.md
result = experiment.run()                    # analytic Gram per sweep point
result.to_csv("examples/output/result.csv")
```

See `examples/animals_interference.py` and the matching
`examples/animals_interference.ipynb` notebook for the full walking tour.

### Plots

`result.plot(path)` saves a default figure: 1D sweep → line plot of
target-pair overlap with sibling and cross-cluster tier baselines; 2D
sweep → heatmap. Requires the `[plot]` extra (`pip install polygram[plot]`).

**1D sweep** — `bird_hawk.phi` from 0 to π. Single-φ steering on this
geometry leaves the cross-cluster overlap above the matched-φ baseline
of `cos(0.5)⁴ ≈ 0.5931` and below the sibling tier; it never destroys.

![1D sweep](docs/img/animals_overlap_1d.png)

**2D sweep** — `(dog_poodle.phi, bird_hawk.phi)` 24×24 grid. The
landscape rises from the matched-φ ridge of `cos(0.5)⁴ ≈ 0.5931` (the
β-overlap baseline) toward off-axis cells; the asymmetry is what
`Cancellation` searches over directly.

![2D heatmap](docs/img/animals_overlap_2d.png)

## Cancellation

`Cancellation` is the second experiment primitive: given a
`target_pair`, it searches the two φ values that drive the pair's
`|<A|B>|²` toward a tolerance, optionally constrained to preserve the
hierarchical-tier ordering. Two backends ship — a deterministic
`max_steps × max_steps` grid scan over `[0, 2π]²` (default, no extra
deps) and `scipy.optimize.differential_evolution` behind `[opt]`.

```python
from polygram import Cancellation

cancellation = Cancellation(
    dictionary=dictionary,
    target_pair=("dog_poodle", "bird_hawk"),
    tolerance=0.05,
    preserve_tiers=True,
    optimize={"method": "grid", "max_steps": 50},
)

result = cancellation.run()
print(result.before_overlap, result.after_overlap, result.tolerance_met)
result.materialize("examples/output/")     # writes optimized .q.orca.md
result.plot("examples/output/grid.png")    # heatmap with infeasible mask
```

Returns a `CancellationResult` exposing `optimized_phis`,
`before_gram` / `after_gram`, `trajectory` (every `(φ_a, φ_b, overlap)`
evaluation in order), `feasible_mask`, `feasible_count`, and
`dictionary_at_optimum` — the new `Dictionary` baked with the
optimized φs and re-emittable via `materialize` as a verifiable
Q-Orca artifact.

### Structural floor

Pure-φ search on a fixed `(α, β, γ)` configuration is bounded: the
target-pair overlap factors as `|<A|B>|²(δ) = M + V·cos(δ)` where
δ = φ_A − φ_B, so phase alone cannot drive overlap below `M − |V|`.
`Cancellation.structural_floor()` returns this analytic minimum
(two Gram evaluations, backend-free), and `CancellationResult`
caches it as `result.structural_floor`. The companion
`result.cancellation_efficiency` reports
`(before − after) / (before − floor)`, clamped to `[0, 1]`:
`1.0` means phase search exhausted the available gap (residue is
encoding-bound — driving overlap lower needs amplitude matching),
`None` means there was no gap to begin with. The materialized
`<name>_summary.md` reports both, plus a one-line interpretation.
See [`docs/research/cancellation-phase-floor.md`](docs/research/cancellation-phase-floor.md)
for the full derivation.

See `examples/cancellation_example.py` for the combined
SAE → InterferenceSweep → Cancellation walk.

### CLI

The `polygram` console script runs an example or experiment module
that exposes `main(output_dir=...)`:

```bash
polygram run examples/animals_interference.py --output-dir results/
polygram run examples/import_from_sae.py --output-dir results/
polygram --version
```

`polygram analyze` triages an SAE feature subset *without any quantum
simulation* — it builds the rung-1 Dictionary and predicts each pair's
`(M, V, structural_floor, cancellation_gap)` from the analytic Gram:

```bash
polygram analyze tests/fixtures/toy_sae.json \
    --features 0,1,4,5 \
    --output analysis_report.md
```

The report includes per-pair structural floors, per-feature
sensitivity (mean `|V_ij|`), and a single-scalar
`encoding_suitability_score`. See `polygram.analysis` for the
programmatic API.

## SAE import

`polygram.from_sae_lens(records, feature_ids, ...)` builds a Polygram
`Dictionary` from a user-selected subset of SAE features and returns a
`SelectionReport` describing how the lossy projection-vector → β
collapse went. Cluster assignment precedence: explicit user override →
parsed `"<cluster>/<name>"` labels → the active profile's
`KnobAssignment` strategy (k-means or PCA-axis depending on profile).

```python
from polygram import from_sae_lens, load_toy_sae

records = load_toy_sae("tests/fixtures/toy_sae.json")
# pick 4 features by id (≤8; the rung-1 MPS cap)
dictionary, report = from_sae_lens(records, [0, 1, 4, 5])

print(report.profile)                    # "clustered" by default
print(report.cluster_method)             # "from_labels" / "kmeans" / "user" / "pca_axis"
print(report.beta_variance_explained)    # profile-defined fidelity stat
print(report.reconstruction_error)       # per-feature distance to centroid
print(report.geometric_fidelity)         # profile's headline scalar
print(report.tier_preservation)          # only populated by `clustered`;
                                         # None for other profiles
```

`SelectionReport` surfaces three fidelity stats per call:
`beta_variance_explained` (defined per profile — cluster residual for
`clustered`, top-1 PCA fraction for `uniform-sphere`),
`reconstruction_error` (per-feature Euclidean distance from each
projection vector to its assigned cluster centroid), and
`geometric_fidelity` (profile-specific scalar — Pearson tier-
preservation for `clustered`, rank-recall@k for `uniform-sphere`).

Pass `assign_gamma=True` to derive each feature's γ from per-cluster
PCA on the centered projection vectors (rescaled into
`gamma_range`, default `(-0.25, 0.25)`); `report.gamma_method`
records `"zero"` (default) or `"projection_pca"`.

### Geometric profiles (`polygram.geometry`)

`from_sae_lens` accepts a `profile=` kwarg selecting which geometric
regime the SAE belongs to. Two built-in profiles ship:

| profile | calibrated for | strategy | fidelity |
|---|---|---|---|
| `clustered` (default) | GPT-2-small-style SAEs (d_model ≤ 768, ≤24K features) | k=2 k-means + antipodal β | Pearson `tier_preservation` |
| `uniform-sphere` | Whisper, Qwen-Scope, Llama-Scope, Gemma-Scope at width (d_model ≥ ~1K, ≥16K features) | k≥16 k-means + PCA-axis β | rank-recall@k |

Pass the profile that matches your SAE's pedigree:

```python
# Small text SAE (GPT-2-small style)
from_sae_lens(records, ids)                              # default = clustered
from_sae_lens(records, ids, profile="clustered")          # explicit

# Large LM SAE (Qwen-Scope, Llama-Scope, Gemma-Scope at width)
# OR audio SAE (Whisper-style)
from_sae_lens(records, ids, profile="uniform-sphere")
```

**Modality is not the selector — pedigree is.** Both audio SAEs and
large-LM text SAEs sit in the `uniform-sphere` regime (verified on a
five-SAE panel: Whisper × 2, Qwen-Scope, Llama-Scope L0R + L12R; see
`docs/research/sae-geometry-regimes.md`). Calling `clustered` on a
Qwen-Scope or Llama-Scope SAE silently degrades — the small-LM
calibration doesn't transfer at scale.

Profile resolution order: **per-field kwarg > `SAEImportConfig.profile`
> registry default (`clustered`)**. So `n_clusters=4` always wins
over the profile's default of 16; `profile="uniform-sphere"` always
wins over a config-supplied `profile`. Omitting `profile=` is
byte-for-byte identical to passing `profile="clustered"`.

Third parties (sae-forge etc.) can register custom profiles via
`polygram.register_profile(...)`. See `polygram.geometry` for the
`KnobAssignment` and `GeometricFidelity` protocols.

The bundled `tests/fixtures/toy_sae.json` is a 16-feature, 4-cluster,
8-dim deterministic toy.

### Loading from safetensors

`polygram.load_sae_safetensors(path, *, names=None)` reads a single
`.safetensors` file and returns the `dict[int, SAEFeatureRecord]`
shape `from_sae_lens` consumes. Decoder weight tensor key is
auto-detected via the precedence list `("W_dec", "decoder.weight",
"dec")`; rows are features (the loader transposes only when the
matched key is `decoder.weight` and the matrix is non-square — that's
the PyTorch `nn.Linear` `out × in` convention). Requires the `[sae]`
extra (`pip install polygram[sae]`) which pulls in `safetensors>=0.4`
only — no torch, no `sae_lens`, no `huggingface_hub`.

```python
from polygram import from_sae_lens, load_sae_safetensors

records = load_sae_safetensors("path/to/sae.safetensors")
dictionary, report = from_sae_lens(records, [0, 1, 4, 5])
```

The companion `polygram sae-import` CLI subcommand wraps the loader
and emits the same JSON schema as `tests/fixtures/toy_sae.json`, so
the chain `sae-import → analyze` works without further plumbing:

```bash
polygram sae-import sae.safetensors --features 0,12,1042 --output picked.json
polygram analyze picked.json --features 0,12,1042 --assign-gamma --sharing-graph g.json
```

> **`--assign-gamma` is almost always wanted on real-SAE inputs.** Without
> it, every feature in a k-means cluster gets `γ = 0` and rung-1
> within-cluster overlaps collapse to `1.0` regardless of how
> diverse the underlying projection vectors are — the SAE's
> geometry becomes invisible to the triage layer. With the flag set,
> per-cluster PCA on the centered projections derives each feature's
> γ. The companion `--n-clusters N` flag (default `2`) tunes the
> k-means cluster count when label-based clustering isn't available.

For GB-class SAEs (Gemma-2-2B, Llama-3-8B, etc.) where the full
decoder tensor would blow past available RAM under `np.float64`
coercion, pass `feature_ids=[...]` to `load_sae_safetensors` to
slice only the rows you need — the loader switches to a
`safetensors.safe_open(...).get_slice(...)` path that never
materializes the full tensor. Empirically: ~5000× less peak Python
memory and ~100× faster than the eager path on a 600 MB SAE when
sampling 8 features.

```python
records = load_sae_safetensors(
    "path/to/large-sae.safetensors",
    feature_ids=[0, 12, 1042, 5012],
)
```

A first-class HuggingFace / SAE-Lens reader (with auto-download +
metadata round-trip) is a separate follow-up — both would pull in
`huggingface_hub` and / or `sae_lens` + torch, which v0 deliberately
keeps out of the runtime dep tree until real-data signal arrives.

### Hand-rolled loaders

To swap in any SAE format the bundled loaders don't cover yet
(custom serialization, multi-file checkpoints, in-memory tensors),
hand-roll a `dict[int, SAEFeatureRecord]` from your loader of choice
and pass it to `from_sae_lens` directly.

See `examples/import_from_sae.py` for the full flow (toy SAE →
Dictionary → `InterferenceSweep` → verified `.q.orca.md` + plot).

## Choosing an encoding

`MPSRung1` is the production default. Each feature is a 3-qubit rung-1 MPS state
parameterized by `(α, β, γ, φ)`, with `phase_knobs=True` exposing a single
`Rz`-axis φ per feature. Its closed-form `|⟨A|B⟩|²(δ) = M + V·cos(δ)` factorization
means the structural floor `M − |V|` is exact and free to evaluate — `Cancellation`
can certify how close a phase-search result is to the encoding-theoretic minimum
without any optimizer calls. Tier-separation and co-firing rankings derived from
`MPSRung1` transfer to real-model behaviour for the validated combo
**`clustered` profile + `MPSRung1` encoding** (Spearman ≥ 0.6 against behavioural
Jaccard at `blocks.10` on GPT-2 small; see
[`docs/research/behavioural-scaleup-probe.md`](docs/research/behavioural-scaleup-probe.md)).
Pairing `MPSRung1` with the `uniform-sphere` profile is provisional pending
behavioural validation on Whisper / Qwen-Scope / Llama-Scope-style SAEs — the
geometric primitives compose correctly but the calibration thresholds haven't been
re-run on those host models yet.

`HEA_Rung2` gives a richer θ tensor (`(|rotations|, depth, n_qubits)` per feature)
with independent knobs for circuit depth, entangler topology, and rotation gate set
(`Ry`/`Rz` by default). The extra expressivity lets the encoding explore geometry
beyond a single Pauli-Rz phase axis, and `tier_separation_bound` (default `0.025`)
causes the emitter to declare an invariant that Q-Orca can verify. The tradeoff is
that no closed-form structural floor exists in the multi-knob case —
`structural_floor()` raises `NotImplementedError` for `HEA_Rung2`, so cancellation
efficiency is undefined. Cross-encoding stability on GPT-2-small SAEs shows that
pair-level tier classifications agree with `MPSRung1` at default thresholds, but
per-pair overlap magnitudes diverge (up to +0.30 higher cross-cluster current-overlap
under HEA); quantitative magnitude claims should be regenerated in the target encoding.

**Rule of thumb**: use `MPSRung1` by default; reach for `HEA_Rung2` when you need
expressivity beyond a single Pauli-Rz phase axis and can live without a closed-form
structural floor.

## Rung3 encoding (experimental)

`polygram.encoding.Rung3` adds a 5-qubit encoding parallel to
`MPSRung1`: qubits 0–2 carry the same MPS state, qubits 3–4 carry an
amplitude branch parameterized by per-feature `theta_amp` and
`psi_aux` knobs (defaults `π/4` and `0.0`). At default knobs the
Rung3 gram reduces to the MPSRung1-equivalent gram exactly, so a
baseline Rung3 dictionary is behaviourally identical to its MPS
counterpart. `Cancellation(encoding="rung3")` runs a joint
`(φ_a, φ_b, theta_amp, psi_aux)` optimizer (5×5 outer grid + 2-φ
inner + scipy Nelder-Mead refine) that, in principle, breaks below
the MPSRung1 phase-only floor `M − |V|`.

Rung3 is **experimental**; the default encoding remains `MPSRung1`
pending the §4.5 viability spike's verdict (run
`examples/rung3_viability_spike.py`). The spike measures four
calibrated criteria (floor-breaking, gate true-positive rate,
ranker preservation, coverage) on the §4.4 8-feature GPT-2-small
panel; making Rung3 the production encoding is gated on a strong
pass per the proposal's decision rule.

## Behavioural validator

`polygram.behavioural.BehaviouralValidator` runs the four-constraint
compression-loop pipeline against a `Dictionary` of SAE features and
emits a structured `ValidationReport`. Two-stage API: `predict()` is
torch-free and Polygram-only; `validate()` lazy-imports torch +
transformers and runs one ablation forward-pass-batch per selected
feature (the per-feature cost cap is a spec contract, not just an
optimization). `run()` is the convenience wrapper.

```python
from polygram import BehaviouralValidator, from_sae_lens, load_sae_safetensors

records = load_sae_safetensors("sae.safetensors", feature_ids=ids)
dictionary, _ = from_sae_lens(records, ids, assign_gamma=True)

validator = BehaviouralValidator(
    dictionary=dictionary,
    sae_checkpoint="sae.safetensors",
    feature_ids=ids,
    prompts=prompts,
    layer=10,
)
report = validator.run()
report.to_json("validation_report.json")
report.to_csv("validation_pairs.csv")
print("confirmed:", report.confirmed)
```

Defaults encode the §4.4 GPT-2-small calibration: Polygram squared-
overlap threshold 0.7, Jaccard threshold 0.30, ablation-KL hook at
layer ≥ 5 (layer 0 is rejected by default per
`docs/research/deeper-layer-ablation-probe.md`). The CLI wrapper is
`polygram validate ...` (run `polygram validate --help`). See
`examples/behavioural_validate.py` for the worked example.

## Compression action

`polygram.compression.Compressor` is the loop's downstream half. It
consumes a `ValidationReport`'s `confirmed` candidate-pair list,
collapses it to redundancy clusters via union-find, picks one
representative per cluster, and rewrites an SAE checkpoint so the
non-representatives are silenced. Two-stage API: `plan()` is cheap
(in-memory union-find, no torch); `apply()` reads the source
`.safetensors`, applies the `zero` strategy in-memory, and writes a
new `.safetensors` atomically. `run()` is the convenience wrapper.

```python
from polygram import Compressor, ValidationReport

report = ValidationReport.from_json("validation_report.json")
compressor = Compressor(
    validation_report=report,
    sae_checkpoint="sae.safetensors",
    strategy="zero",
)
result = compressor.run("sae.compressed.safetensors")
result.report.to_json("compression_report.json")
```

The `zero` strategy zeroes the encoder column, encoder bias, and
decoder row of every non-representative member. `b_dec` (global) is
untouched. Component-first compression (not pair-by-pair) makes the
operation order-independent — see
`docs/research/compression-action-design.md` for the rationale. CLI
wrapper: `polygram compress ...`. Worked example:
`examples/compress_validated.py`. The `merge` strategy is deferred to
a follow-up change.

## Regrow primitive

`polygram.compression.Regrower` is the post-compression complement:
it takes a checkpoint with zeroed slots and repopulates them with
new directions extracted from the SAE's activation residuals. The
default `residual_kmeans` strategy runs k-means on `activation -
SAE_reconstruct(activation)` (the directions the SAE failed to
represent), unit-normalizes the cluster centroids, and writes them
into the zeroed slots: `W_dec[fid, :] = centroid`, `W_enc[:, fid] =
centroid`, `b_enc[fid] = 0`. `b_dec` stays untouched.

```python
from polygram import Regrower, CompressionReport

# Chained from a prior compression run
report = CompressionReport.from_json("compression_report.json")
regrower = Regrower.from_compression_report(
    report,
    sae_checkpoint="sae.compressed.safetensors",
    strategy="residual_kmeans",
    prompts=prompts,  # captures residuals via one GPT-2 forward
    layer=10,
)
result = regrower.run("sae.regrown.safetensors")

# Or directly from a known zeroed set + cached residuals
import numpy as np
residuals = np.load("residuals.npy")  # (n_tokens, d_model)
regrower = Regrower(
    sae_checkpoint="sae.compressed.safetensors",
    strategy="residual_kmeans",
    zeroed={42, 100, 256},
    cached_residuals=residuals,
)
result = regrower.run("sae.regrown.safetensors")
```

The output is a "primed but quiet" SAE — every slot has a unit-norm
direction, no fine-tune happens here. Downstream consumers (an
orca-lang workflow fine-tune node, an external trainer, a research
notebook) take it from there. CLI wrapper: `polygram regrow ...`.
Optional extra: `pip install ".[regrow]"` for sklearn. See
`docs/research/compression-regrow-design.md` and
`examples/regrow_validated.py`.

## Compression epoch (multi-panel orchestrator)

`polygram.compression.EpochCompressor` scales the per-panel
validate→compress loop to the full SAE via greedy seeded coverage
over the cosine-similar pair graph, with stable-cluster fixed-point
iteration and a relative cross-entropy quality bound. The
orchestrator runs panels sequentially in-process (single GPT-2
instance amortized across panels); users wanting concurrency shard
at the CLI invocation level.

```python
from polygram import EpochCompressor

epoch = EpochCompressor(
    sae_checkpoint="sae.safetensors",
    prompts=prompts,
    layer=10,
    coverage_target=0.95,
    cosine_threshold=0.30,
    n_visits_per_feature=3,
    max_iterations=5,
    quality_delta_multiplier=2.0,
)
result = epoch.run("sae.epoch.safetensors")
result.report.to_json("epoch_report.json")
print(result.report.convergence_reason)  # 'stable_clusters' | 'max_iterations' | ...
```

Convergence: cluster-set fingerprint stable across two consecutive
iterations. Hard cap on iterations and a relative quality bound
(`delta_k ≤ multiplier × delta_1`) prevent runaway compression. CLI
wrapper: `polygram compress-epoch ...`. Worked example:
`examples/compress_epoch_validated.py`. See
`docs/research/compression-epoch-design.md` for the rationale.

## Configuration (`polygram.config`)

Every tunable knob on a public polygram constructor is also reachable
through a frozen dataclass under `polygram.config`. Six configs cover
the surface:

| Config | Used by | Defaults |
|---|---|---|
| `CompressionConfig` | `Compressor` | `strategy="merge"`, `rep_selection="scale_aware"`, `merge_mode="freq_weighted"`, `confirmer=None` |
| `EpochCompressionConfig` | `EpochCompressor` (embeds `ValidationConfig` via the `validation` field) | iterative-loop preset: `coverage_target=0.5`, `cosine_threshold=0.30`, `n_visits_per_feature=1`, `max_iterations=1`, `quality_delta_multiplier=2.0` |
| `CancellationConfig` | `Cancellation` | `tolerance=0.05`, `preserve_tiers=True`, `optimize={"method":"grid","max_steps":50}`, `grid_outer=(5,5)`, `min_amp_overlap=0.0` |
| `ValidationConfig` | `BehaviouralValidator`, embedded in `EpochCompressionConfig` | `polygram_overlap_threshold=0.7`, `jaccard_threshold=0.30`, `min_firing_rate=0.01`, `min_both_fire=5`, `allow_layer_zero=False` |
| `RegrowConfig` | `Regrower.from_compression_report` | required: `model_name`, `layer`. Optional: `strategy="residual_kmeans"`, `seed=0`, `n_init=4`, `prompts=None`, `device=None` |
| `SAEImportConfig` | `from_sae_lens` | `assign_gamma=True` (flipped from the legacy `False`), `gamma_range=(-0.25, 0.25)`, `n_clusters=2` |

### Override precedence

Each affected constructor accepts an optional `config=` keyword. The
resolution rule is **per-field kwargs > config > dataclass defaults**:

```python
from polygram import Cancellation, CancellationConfig

cfg = CancellationConfig(tolerance=0.01, preserve_tiers=False)

# Both kwargs taken from cfg.
canc = Cancellation(dictionary=d, target_pair=("a", "b"), config=cfg)

# Same cfg, but tolerance overridden by the per-field kwarg.
canc = Cancellation(
    dictionary=d, target_pair=("a", "b"),
    config=cfg, tolerance=0.001,
)
```

When `config` is omitted and no per-field kwargs are passed, behaviour
matches the constructor's pre-config defaults — except for the three
intentional default flips called out below (see CHANGELOG).

### Named presets on `EpochCompressor`

`EpochCompressor.fast()` returns an instance whose tuning matches the
new iterative-preset defaults; `EpochCompressor.thorough()` restores
the pre-change "exhaustive offline run" defaults
(`coverage_target=0.95`, `n_visits_per_feature=3`, `max_iterations=5`).
Both classmethods accept `**overrides` for any constructor kwarg,
including the required positional inputs:

```python
ec = EpochCompressor.fast(
    sae_checkpoint=ckpt, prompts=prompts, layer=10,
    coverage_target=0.6,  # override the preset
)
```

### Dict round-trip (FSM contexts, YAML configs)

Every config has `to_dict()` (JSON-serialisable; tuples → lists; nested
configs serialised recursively) and a `from_dict(data)` classmethod
that coerces lists back into tuple-typed fields, recurses into nested
configs, and emits a `UserWarning` for unknown keys (so a dict stored
under an older polygram release survives a knob being added).

Downstream callers (e.g. sae-forge's outer-loop FSM context) can stash
a config bundle on a context dict and rebuild it on the action side
without writing per-field marshalling code:

```python
# Caller side
ctx = {
    "compression": CompressionConfig(strategy="merge").to_dict(),
    "regrow": RegrowConfig(model_name="pythia-160m", layer=4).to_dict(),
}

# Action side
from polygram import CompressionConfig, RegrowConfig

if (d := ctx.get("compression")):
    Compressor(..., config=CompressionConfig.from_dict(d))
if (d := ctx.get("regrow")):
    Regrower.from_compression_report(..., config=RegrowConfig.from_dict(d))
```

## Development

```bash
pip install -e ".[dev,plot]"
pytest
```

## Relationship to Q-Orca

Polygram does **not** define a new file format. It generates standard
Q-Orca `.q.orca.md` files (matching the style of
`examples/larql-animals-interference.q.orca.md` from q-orca-lang) and uses
Q-Orca for verification, simulation, and the analytic Gram helper
`compute_concept_gram_mps`.

## License

Apache-2.0.
