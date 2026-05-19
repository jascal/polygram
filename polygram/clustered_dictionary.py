"""`ClusteredDictionary` — block-decomposed dictionary for SAE-scale analyses.

The single-`Dictionary` primitives (`Dictionary.gram`, `Cancellation`,
`BehaviouralValidator`, Q-OrCA emission) cap a dictionary at its
encoding's Hilbert-space dimension — 8 features for `MPSRung1`, 16 for
`Rung3`, `2**n_qubits` for `HEA_Rung2`. Real SAEs ship at 16k–1M
features per layer; `ClusteredDictionary` is the primitive that holds
the entire SAE as a list of ≤K-feature `Dictionary` blocks plus a
sparse cross-block adjacency, so block-local analyses delegate to the
existing primitives and cross-block analyses use encoding-agnostic
direct decoder-vector overlaps.

**Two resolutions, two units.** Intra-block Gram entries are
**quantum-encoded** complex state overlaps (from each block's
`Dictionary.gram()` analytic path). Cross-block entries are **direct
decoder-vector inner products** — encoding-agnostic, real-valued
(lifted to complex with zero imaginary part for uniform handling).
Callers that conflate the two regions should think twice; the
canonical access patterns surface them separately
(`BlockSparseGram.block_diagonal()` vs
`BlockSparseGram.cross_block_entries()`).

Most-common entry point:

    from polygram import from_sae_lens
    cd, report = from_sae_lens(records, feature_ids=list(range(64)), clustered=True)
    bsg = cd.gram()                              # block-sparse Gram
    cd.cross_block_redundant_pairs(0.7)          # high-cosine cross-block pairs
    cd.emit_qorca(output_dir)                    # per-block .q.orca.md + manifest

For explicit construction outside the loader, use `build_clustered_dictionary`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Literal

from pathlib import Path

import numpy as np

from polygram.dictionary import Dictionary, Feature
from polygram.encoding import HEA_Rung2, MPSRung1, Rung3

# Strategy literal for `BlockFormation`. Module-level alias so callers
# don't have to repeat the union when typing config arguments.
BlockFormationStrategy = Literal["cosine", "co_firing", "user_declared"]


@dataclass(frozen=True)
class BlockFormation:
    """Configuration for how a `ClusteredDictionary` is built from an SAE.

    Fields:

    - `strategy` — one of `"cosine"`, `"co_firing"`, `"user_declared"`.
      No default; callers must pick. Implementations of the strategies
      themselves land in §2 — this dataclass holds only the
      configuration.
    - `cosine_threshold` — admit/reject threshold for cross-block edges.
      An edge is added when the direct decoder-vector cosine between
      the two features exceeds this value. Default 0.3 (chosen so the
      typical uniform-sphere SAE produces a small, well-bounded
      cross-block edge set).
    - `block_size_max` — per-block feature-count cap. When `None`, the
      strategy implementation defaults to the encoding's
      `max_features`. Setting this explicitly to a smaller value
      reduces per-block compute at the cost of more cross-block edges.
    - `firing_corpus` — required when `strategy == "co_firing"`;
      `None` for the other strategies.
    """

    strategy: BlockFormationStrategy
    cosine_threshold: float = 0.3
    block_size_max: int | None = None
    firing_corpus: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if self.strategy == "co_firing" and self.firing_corpus is None:
            raise ValueError(
                "BlockFormation: strategy='co_firing' requires a "
                "non-None `firing_corpus` argument"
            )
        if self.strategy != "co_firing" and self.firing_corpus is not None:
            raise ValueError(
                f"BlockFormation: strategy={self.strategy!r} does not "
                f"consume `firing_corpus`; pass None for this strategy"
            )
        if not (0.0 <= self.cosine_threshold <= 1.0):
            raise ValueError(
                f"BlockFormation: cosine_threshold must lie in [0, 1]; "
                f"got {self.cosine_threshold}"
            )
        if self.block_size_max is not None and self.block_size_max < 1:
            raise ValueError(
                f"BlockFormation: block_size_max must be >= 1 when set; "
                f"got {self.block_size_max}"
            )


# Cross-block adjacency key. `(block_i_idx, feat_i_idx, block_j_idx,
# feat_j_idx)` with the invariant `block_i_idx < block_j_idx` (the
# adjacency is undirected; we canonicalise on insert).
CrossBlockKey = tuple[int, int, int, int]


# Block topology — flat adjacency for v1. A `dict[block_id, list[block_id]]`
# records which blocks are linked at the topology level (e.g.,
# super-clusters of clusters). Distinct from `cross_block_pairs`, which
# is per-feature-pair. v1 keeps the topology shape simple; richer
# representations (trees, weighted graphs) are deferred.
BlockTopology = Mapping[str, Sequence[str]]


@dataclass(frozen=True)
class ClusteredDictionary:
    """N-feature dictionary held as a list of ≤K-feature `Dictionary`
    blocks plus a sparse cross-block adjacency.

    Fields:

    - `name` — a stable identifier used by downstream consumers
      (emit paths, reports). Same naming rules as `Dictionary.name`.
    - `blocks` — ordered list of `Dictionary` instances. Every block
      SHALL share the same `encoding` (validated in `__post_init__`).
      Every block SHALL have ≤ `encoding.max_features` features.
    - `cross_block_pairs` — sparse adjacency keyed by
      `(block_i_idx, feat_i_idx, block_j_idx, feat_j_idx)` with the
      invariant `block_i_idx < block_j_idx`. Values are the direct
      decoder-vector cosine similarity that produced the edge.
    - `block_topology` — optional cluster-of-clusters graph. v1 holds
      a flat `dict[block_id, list[block_id]]`; richer representations
      land in a follow-up.
    - `block_formation` — the strategy + config used to construct
      this clustering. Persisted on the instance for reproducibility
      and downstream reporting.

    Hard partition invariant (v1): every feature appears in exactly
    one block. Soft / overlap clustering is explicitly deferred per
    `design.md` Decision 1.
    """

    name: str
    blocks: list[Dictionary]
    cross_block_pairs: Mapping[CrossBlockKey, float] = field(default_factory=dict)
    block_topology: BlockTopology | None = None
    block_formation: BlockFormation = field(
        default_factory=lambda: BlockFormation(strategy="user_declared")
    )

    def __post_init__(self) -> None:
        # Reuse the Dictionary name regex via the same shape — we use a
        # local check so we don't expose Dictionary's private validator.
        # The constraint is purely syntactic for downstream slug usage.
        from polygram.dictionary import _VALID_NAME_RE  # noqa: PLC0415

        if not _VALID_NAME_RE.match(self.name):
            raise ValueError(
                f"ClusteredDictionary name {self.name!r} must match "
                f"{_VALID_NAME_RE.pattern}"
            )

        if not self.blocks:
            raise ValueError(
                "ClusteredDictionary: blocks must be non-empty (got 0 blocks)"
            )

        # All blocks share the same encoding type and configuration.
        # We compare by type identity + repr because the encoding
        # dataclasses are frozen; a parametric class like HEA_Rung2
        # with different `n_qubits` is correctly treated as different.
        encoding = self.blocks[0].encoding
        encoding_type = type(encoding)
        for i, block in enumerate(self.blocks[1:], start=1):
            if type(block.encoding) is not encoding_type:
                raise ValueError(
                    f"ClusteredDictionary: blocks must share encoding type. "
                    f"Block 0 has {encoding_type.__name__!r}; "
                    f"block {i} has {type(block.encoding).__name__!r}"
                )
            if block.encoding != encoding:
                raise ValueError(
                    f"ClusteredDictionary: blocks must share the same "
                    f"encoding configuration. Block 0 has "
                    f"{encoding!r}; block {i} has {block.encoding!r}"
                )

        cap = int(encoding.max_features)
        for i, block in enumerate(self.blocks):
            n = len(block.features)
            if n > cap:
                raise ValueError(
                    f"ClusteredDictionary: block {i} has {n} features, "
                    f"exceeding the encoding cap {cap} for "
                    f"{type(encoding).__name__}. Reduce block size or "
                    f"use an encoding with larger max_features."
                )

        # Hard-partition invariant: every feature name appears in
        # exactly one block. We name the duplicate explicitly because
        # the cross-block adjacency would silently produce ambiguous
        # keys otherwise.
        seen: dict[str, int] = {}
        for block_idx, block in enumerate(self.blocks):
            for feature in block.features:
                if feature.name in seen:
                    raise ValueError(
                        f"ClusteredDictionary: feature {feature.name!r} "
                        f"appears in two blocks (block {seen[feature.name]} "
                        f"and block {block_idx}); hard-partition invariant "
                        f"requires every feature in exactly one block"
                    )
                seen[feature.name] = block_idx

        # Cross-block edge integrity. Each key must reference valid
        # (block, feature) coordinates with the canonical ordering
        # `block_i_idx < block_j_idx`.
        for key in self.cross_block_pairs:
            if len(key) != 4:
                raise ValueError(
                    f"ClusteredDictionary: cross_block_pairs key {key!r} "
                    f"must be a 4-tuple "
                    f"(block_i_idx, feat_i_idx, block_j_idx, feat_j_idx)"
                )
            bi, fi, bj, fj = key
            if not (0 <= bi < len(self.blocks)):
                raise ValueError(
                    f"ClusteredDictionary: cross_block_pairs key {key!r} "
                    f"has out-of-range block_i_idx={bi} "
                    f"(have {len(self.blocks)} blocks)"
                )
            if not (0 <= bj < len(self.blocks)):
                raise ValueError(
                    f"ClusteredDictionary: cross_block_pairs key {key!r} "
                    f"has out-of-range block_j_idx={bj} "
                    f"(have {len(self.blocks)} blocks)"
                )
            if bi >= bj:
                raise ValueError(
                    f"ClusteredDictionary: cross_block_pairs key {key!r} "
                    f"must satisfy block_i_idx < block_j_idx "
                    f"(undirected canonical order)"
                )
            if not (0 <= fi < len(self.blocks[bi].features)):
                raise ValueError(
                    f"ClusteredDictionary: cross_block_pairs key {key!r} "
                    f"has out-of-range feat_i_idx={fi} for block {bi} "
                    f"(block has {len(self.blocks[bi].features)} features)"
                )
            if not (0 <= fj < len(self.blocks[bj].features)):
                raise ValueError(
                    f"ClusteredDictionary: cross_block_pairs key {key!r} "
                    f"has out-of-range feat_j_idx={fj} for block {bj} "
                    f"(block has {len(self.blocks[bj].features)} features)"
                )

    @property
    def n_features(self) -> int:
        """Total feature count summed across blocks."""
        return sum(len(b.features) for b in self.blocks)

    @property
    def n_blocks(self) -> int:
        """Number of blocks."""
        return len(self.blocks)

    @property
    def encoding(self) -> MPSRung1 | HEA_Rung2 | Rung3:
        """The shared encoding instance (validated equal across blocks)."""
        return self.blocks[0].encoding

    @property
    def n_cross_block_edges(self) -> int:
        """Number of cross-block adjacency edges."""
        return len(self.cross_block_pairs)

    @property
    def mean_block_size(self) -> float:
        """Average features-per-block; useful for `SelectionReport`."""
        return self.n_features / self.n_blocks

    def emit_qorca(self, output_dir) -> dict[str, Path]:
        """Write one `.q.orca.md` per block plus a `manifest.json`
        describing the block topology and cross-block adjacency.

        Each per-block `.q.orca.md` is independently round-trippable
        through Q-OrCA's verifier. The manifest captures the block
        structure so downstream consumers (analysis pipelines,
        documentation) can reconstruct the clustered topology without
        re-parsing every machine.

        Returns a dict mapping artifact IDs to their written `Path`:

        - `"<block_id>"` for each block's machine.
        - `"manifest"` for `manifest.json`.

        The manifest schema:

        ```json
        {
          "name": "<cd.name>",
          "n_features": <int>,
          "encoding": "<encoding class name>",
          "blocks": [
            {
              "id": "<block.name>",
              "machine": "<relative path>",
              "features": [{"name": "<f.name>", "cluster": "<f.cluster>"}, ...]
            },
            ...
          ],
          "cross_block_edges": [
            {
              "from": ["<block_i.name>", "<feat_i.name>"],
              "to":   ["<block_j.name>", "<feat_j.name>"],
              "cosine": <float>
            },
            ...
          ],
          "block_formation": {"strategy": "...", "cosine_threshold": ...}
        }
        ```
        """
        import json
        from pathlib import Path

        from polygram.emit import write_qorca

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, Path] = {}

        machine_paths: dict[int, str] = {}  # block_idx → relative path
        for block_idx, block in enumerate(self.blocks):
            machine_path = out_dir / f"{block.name}.q.orca.md"
            write_qorca(block, machine_path)
            artifacts[block.name] = machine_path
            machine_paths[block_idx] = machine_path.name

        manifest = {
            "name": self.name,
            "n_features": self.n_features,
            "encoding": type(self.encoding).__name__,
            "blocks": [
                {
                    "id": block.name,
                    "machine": machine_paths[block_idx],
                    "features": [
                        {"name": f.name, "cluster": f.cluster}
                        for f in block.features
                    ],
                }
                for block_idx, block in enumerate(self.blocks)
            ],
            "cross_block_edges": [
                {
                    "from": [self.blocks[bi].name, self.blocks[bi].features[fi].name],
                    "to": [self.blocks[bj].name, self.blocks[bj].features[fj].name],
                    "cosine": float(cosine),
                }
                for (bi, fi, bj, fj), cosine in self.cross_block_pairs.items()
            ],
            "block_formation": {
                "strategy": self.block_formation.strategy,
                "cosine_threshold": self.block_formation.cosine_threshold,
                "block_size_max": self.block_formation.block_size_max,
            },
        }
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        artifacts["manifest"] = manifest_path
        return artifacts

    @classmethod
    def from_compression_panels(
        cls,
        panels: Sequence,
        state_dict: Mapping[str, "np.ndarray"],
        encoding: "MPSRung1 | HEA_Rung2 | Rung3",
        *,
        name: str = "from_panels",
        cosine_threshold: float = 0.3,
        feature_records: Mapping[int, object] | None = None,
    ) -> "ClusteredDictionary":
        """Construct a `ClusteredDictionary` from
        `polygram.compression.epoch._select_panels` output.

        This is the forward-compatible shallow connection between the
        clustered-dictionary primitive and the existing `EpochCompressor`
        panel-selection logic. It takes an already-computed list of
        `Panel` objects (with `anchor`, `feature_ids`, `cosines_to_anchor`
        fields), builds one `Dictionary` block per panel, and computes
        the cross-block adjacency from the panels' decoder vectors.

        **Ordering invariant:** ``blocks[k]`` is constructed from
        ``panels[k]`` for every ``k``. This element-wise ordering is
        load-bearing — downstream consumers (`_validate_panels`,
        `_synthesize_validation_report` once `compression-consumes-clustered-dictionary`
        lands) rely on it to iterate `clustered.blocks` and the
        original `panels` list interchangeably.

        The classmethod does NOT alter `_select_panels`. A future
        change can extract the priority-driven seeded-coverage
        algorithm into a `BlockFormation` strategy and have
        `_select_panels` wrap this — at which point the public API
        of `from_compression_panels` stays stable and only its
        implementation flips. The byte-identical regression check
        (compression tests pass unchanged) holds today by
        construction.

        TODO: when the deferred `compression-seeded` BlockFormation
        strategy lands (see `openspec/changes/compression-consumes-clustered-dictionary/`
        and the discussion in `clustered-dictionary-analysis/tasks.md`
        §7), this method's body can be reimplemented as a call to
        `build_clustered_dictionary(... , block_formation=BlockFormation(strategy="compression_seeded", ...))`
        without changing this method's signature.

        Parameters
        ----------
        panels:
            Iterable of `Panel`-shaped objects with `anchor: int`,
            `feature_ids: tuple[int, ...]`, and
            `cosines_to_anchor: tuple[float, ...]` attributes.
        state_dict:
            SAE state dict supplying `W_dec` (the decoder matrix).
            Cross-block cosines are computed from these rows.
        encoding:
            Shared encoding for every block.
        name:
            Identifier for the returned `ClusteredDictionary`.
        cosine_threshold:
            Threshold for cross-block edge inclusion. Defaults to 0.3
            (same as `BlockFormation`'s default).
        feature_records:
            Optional `dict[int, SAEFeatureRecord]` used to populate
            feature names / clusters from the SAE-import side. When
            `None`, feature names are synthesised as `f{feature_id}`
            and the cluster is the synthetic block name.
        """
        w_dec = state_dict["W_dec"]

        feature_to_block: dict[int, tuple[int, int]] = {}
        blocks: list[Dictionary] = []
        for block_idx, panel in enumerate(panels):
            # `panel.anchor` is read but unused for block construction
            # (the anchor is just the panel-internal seed); the block
            # carries the same members regardless of which feature
            # was the seed.
            members = tuple(int(f) for f in getattr(panel, "feature_ids"))
            cluster_name = f"{name}_b{block_idx}"
            feats: list[Feature] = []
            for local_idx, fid in enumerate(members):
                if feature_records is not None and fid in feature_records:
                    record = feature_records[fid]
                    feat_name = getattr(record, "name", f"f{fid}")
                else:
                    feat_name = f"f{fid}"
                feats.append(
                    Feature(name=feat_name, cluster=cluster_name, beta=0.0)
                )
                feature_to_block[fid] = (block_idx, local_idx)
            blocks.append(
                Dictionary(
                    name=cluster_name,
                    features=feats,
                    hierarchy={cluster_name: [f.name for f in feats]},
                    encoding=encoding,
                )
            )

        # Cross-block adjacency: pairs of feature IDs whose blocks
        # differ and whose decoder-vector cosine exceeds threshold.
        # We only consider pairs where both endpoints landed in some
        # panel (features outside any panel aren't part of the
        # clustered dictionary).
        all_feature_ids = np.array(sorted(feature_to_block), dtype=np.int64)
        cross_block_pairs: dict[CrossBlockKey, float] = {}
        if all_feature_ids.size >= 2:
            cosine_pairs = compute_cosine_pair_graph(
                w_dec,
                threshold=cosine_threshold,
                indices=all_feature_ids,
            )
            norms = np.linalg.norm(w_dec, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            for i, j in cosine_pairs:
                bi, fi = feature_to_block[int(i)]
                bj, fj = feature_to_block[int(j)]
                if bi == bj:
                    continue
                if bi > bj:
                    bi, bj = bj, bi
                    fi, fj = fj, fi
                v_i = w_dec[int(i)] / norms[int(i), 0]
                v_j = w_dec[int(j)] / norms[int(j), 0]
                cos = float(np.dot(v_i, v_j))
                cross_block_pairs[(bi, fi, bj, fj)] = cos

        return cls(
            name=name,
            blocks=blocks,
            cross_block_pairs=cross_block_pairs,
            block_formation=BlockFormation(
                strategy="user_declared",
                cosine_threshold=cosine_threshold,
            ),
        )

    def cross_block_redundant_pairs(
        self, threshold: float = 0.7
    ) -> "CrossBlockRedundancyReport":
        """Surface cross-block feature pairs whose decoder-vector
        cosine equals or exceeds `threshold`.

        Operates over the pre-computed `cross_block_pairs` adjacency.
        Pairs were filtered at build time by
        `block_formation.cosine_threshold` (default 0.3), so the
        effective lower bound is `max(threshold, build_threshold)`.
        If a caller wants pairs below the build threshold, the
        `ClusteredDictionary` needs to be rebuilt with a lower
        `cosine_threshold`.

        Returns a `CrossBlockRedundancyReport` with the surviving
        pairs ordered by cosine descending plus metadata (threshold,
        total cross-block edge count, per-block-pair coverage).
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"cross_block_redundant_pairs: threshold must lie in "
                f"[0, 1]; got {threshold}"
            )
        pairs: list[CrossBlockRedundancyPair] = []
        coverage: dict[tuple[int, int], int] = defaultdict(int)
        for (bi, fi, bj, fj), cosine in self.cross_block_pairs.items():
            if cosine < threshold:
                continue
            feat_i_name = self.blocks[bi].features[fi].name
            feat_j_name = self.blocks[bj].features[fj].name
            pairs.append(
                CrossBlockRedundancyPair(
                    block_i_idx=bi,
                    feat_i_name=feat_i_name,
                    block_j_idx=bj,
                    feat_j_name=feat_j_name,
                    cosine=float(cosine),
                )
            )
            coverage[(bi, bj)] += 1
        pairs.sort(key=lambda p: p.cosine, reverse=True)
        return CrossBlockRedundancyReport(
            threshold=float(threshold),
            n_total_cross_block_edges=len(self.cross_block_pairs),
            pairs=pairs,
            coverage=dict(coverage),
        )

    def gram(self) -> "BlockSparseGram":
        """Block-sparse Gram: per-block dense complex Gram + sparse
        cross-block edges.

        Per-block entries are computed via each block's
        `Dictionary.gram()` (the quantum-encoded analytic path); they
        carry the encoding's complex-valued state overlaps. Cross-
        block entries are lifted from `cross_block_pairs` — they hold
        the direct decoder-vector inner products (encoding-agnostic).

        The two regions live in different units. Intra-block entries
        reflect the quantum-encoded state overlap; cross-block
        entries reflect classical decoder-vector geometry. Callers
        treating the result as a single matrix (e.g., via
        `to_dense()`) should know what they're getting — see the
        `BlockSparseGram` docstring.
        """
        block_grams = [block.gram() for block in self.blocks]
        # Cross-block edge values are stored as floats (cosine /
        # real-valued decoder dot product). Lift to complex with
        # imaginary part 0 for uniform handling in BlockSparseGram.
        cross_block_edges: dict[CrossBlockKey, complex] = {
            key: complex(value) for key, value in self.cross_block_pairs.items()
        }
        return BlockSparseGram(
            block_grams=block_grams,
            cross_block_edges=cross_block_edges,
        )


# ===========================================================================
# §5 — Cross-block redundancy report types
# ===========================================================================


@dataclass(frozen=True)
class CrossBlockRedundancyPair:
    """One feature pair surfaced by
    `ClusteredDictionary.cross_block_redundant_pairs`.

    `block_i_idx < block_j_idx` follows the canonical adjacency
    ordering. Feature names are resolved at report time from each
    block's `Dictionary.features` list so downstream consumers don't
    need to know about local indices.
    """

    block_i_idx: int
    feat_i_name: str
    block_j_idx: int
    feat_j_name: str
    cosine: float


@dataclass(frozen=True)
class CrossBlockRedundancyReport:
    """Result of `ClusteredDictionary.cross_block_redundant_pairs`.

    Fields:

    - `threshold` — the cosine bound used to filter `pairs`.
    - `n_total_cross_block_edges` — every cross-block adjacency entry
      considered (before applying `threshold`). Useful for reporting
      "X pairs above threshold out of Y cross-block edges examined".
    - `pairs` — ordered by cosine descending.
    - `coverage` — per `(block_i_idx, block_j_idx)` count of pairs
      above threshold; surfaces which block pairs concentrate the
      redundancies.
    """

    threshold: float
    n_total_cross_block_edges: int
    pairs: list[CrossBlockRedundancyPair]
    coverage: Mapping[tuple[int, int], int]


# ===========================================================================
# §3 — BlockSparseGram value type
# ===========================================================================


@dataclass(frozen=True)
class BlockSparseGram:
    """Block-sparse Gram representation: list of per-block dense complex
    Gram matrices plus a sparse dict of cross-block entries.

    Two-resolution structure:

    - `block_grams[k]` is a `(K_k, K_k)` complex matrix holding
      block `k`'s dense intra-block Gram, as returned by the
      underlying `Dictionary.gram()` (quantum-encoded analytic
      path).
    - `cross_block_edges` is a sparse dict keyed by the canonical
      tuple `(block_i_idx, feat_i_local_idx, block_j_idx,
      feat_j_local_idx)` with `block_i_idx < block_j_idx`. Values
      hold cross-block Gram entries (direct decoder-vector inner
      products); the canonical-ordered storage saves space, with
      the conjugate placed at `[j, i]` only when `to_dense()`
      materialises the full matrix.

    The intra-block and cross-block entries are in different units
    (quantum-encoded overlap vs decoder-vector geometry). Callers
    doing arithmetic that conflates them should think twice.
    """

    block_grams: list[np.ndarray]
    cross_block_edges: Mapping[CrossBlockKey, complex] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.block_grams:
            raise ValueError(
                "BlockSparseGram: block_grams must be non-empty"
            )
        # Validate per-block shape: each Gram is square.
        for k, g in enumerate(self.block_grams):
            if g.ndim != 2 or g.shape[0] != g.shape[1]:
                raise ValueError(
                    f"BlockSparseGram: block_grams[{k}] must be a "
                    f"square 2-D array; got shape {g.shape}"
                )
        # Validate cross-block keys against block sizes.
        n_blocks = len(self.block_grams)
        for key in self.cross_block_edges:
            if len(key) != 4:
                raise ValueError(
                    f"BlockSparseGram: cross_block_edges key {key!r} "
                    f"must be a 4-tuple"
                )
            bi, fi, bj, fj = key
            if not (0 <= bi < bj < n_blocks):
                raise ValueError(
                    f"BlockSparseGram: cross_block_edges key {key!r} "
                    f"violates canonical block ordering "
                    f"0 <= bi < bj < {n_blocks}"
                )
            if not (0 <= fi < self.block_grams[bi].shape[0]):
                raise ValueError(
                    f"BlockSparseGram: cross_block_edges key {key!r} "
                    f"feat_i_local_idx={fi} out of range for block {bi} "
                    f"(size {self.block_grams[bi].shape[0]})"
                )
            if not (0 <= fj < self.block_grams[bj].shape[0]):
                raise ValueError(
                    f"BlockSparseGram: cross_block_edges key {key!r} "
                    f"feat_j_local_idx={fj} out of range for block {bj} "
                    f"(size {self.block_grams[bj].shape[0]})"
                )

    @property
    def n_blocks(self) -> int:
        return len(self.block_grams)

    @property
    def shape(self) -> tuple[int, int]:
        """Total `(N, N)` shape where `N` = sum of per-block sizes."""
        n = sum(g.shape[0] for g in self.block_grams)
        return (n, n)

    @property
    def density(self) -> float:
        """Fraction of off-block-diagonal cells filled by cross-block edges.

        Counts each edge twice (lower + upper triangle in the dense
        form). Returns 0.0 when there is no off-block region (e.g.,
        single-block grams).
        """
        n_total = self.shape[0]
        block_diagonal_cells = sum(g.shape[0] ** 2 for g in self.block_grams)
        off_block_cells = n_total * n_total - block_diagonal_cells
        if off_block_cells == 0:
            return 0.0
        # Each cross_block_edges entry covers 2 cells (upper + lower
        # triangle) in the dense view.
        return (2 * len(self.cross_block_edges)) / off_block_cells

    @property
    def cross_block_density(self) -> float:
        """Alias for `density`, surfaced under the more discoverable
        name for callers focused specifically on cross-block adjacency
        diagnostics. Identical semantics."""
        return self.density

    def cross_block_cosine_histogram(
        self, bins: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Histogram of cross-block edge cosine magnitudes.

        Returns `(counts, bin_edges)` matching `numpy.histogram`'s
        contract over the range `[0, 1]`. Useful for diagnosing whether
        the cross-block adjacency is dominated by near-threshold edges
        or by a few outliers near cosine 1.0.

        Cross-block entries store complex Gram values; the histogram
        uses each entry's magnitude (`abs(value)`).
        """
        if not self.cross_block_edges:
            return (
                np.zeros(bins, dtype=np.int64),
                np.linspace(0.0, 1.0, bins + 1),
            )
        magnitudes = np.fromiter(
            (abs(v) for v in self.cross_block_edges.values()),
            dtype=np.float64,
            count=len(self.cross_block_edges),
        )
        return np.histogram(magnitudes, bins=bins, range=(0.0, 1.0))

    def block_diagonal(self) -> list[np.ndarray]:
        """Return the per-block dense Gram matrices."""
        return list(self.block_grams)

    def _block_offsets(self) -> list[int]:
        """Cumulative block-size offsets for global-index mapping."""
        offsets = [0]
        for g in self.block_grams:
            offsets.append(offsets[-1] + g.shape[0])
        return offsets

    def entries(self) -> Iterator[tuple[int, int, complex]]:
        """Yield non-zero Gram entries lazily as `(global_i, global_j,
        value)` tuples.

        Includes every block-diagonal entry (intra-block Gram, every
        cell of every per-block matrix) and every cross-block edge in
        canonical `(global_i, global_j, value)` form with
        `global_i < global_j`.
        """
        offsets = self._block_offsets()
        # Block-diagonal: every (i, j) within each block.
        for k, g in enumerate(self.block_grams):
            base = offsets[k]
            n_k = g.shape[0]
            for li in range(n_k):
                for lj in range(n_k):
                    yield (base + li, base + lj, complex(g[li, lj]))
        # Cross-block: canonical (bi < bj) entries only. Both halves
        # of the dense matrix can be reconstructed via the
        # `to_dense()` path's conjugate mirror.
        for (bi, fi, bj, fj), value in self.cross_block_edges.items():
            gi = offsets[bi] + fi
            gj = offsets[bj] + fj
            yield (gi, gj, complex(value))

    def cross_block_entries(self) -> Iterator[tuple[int, int, complex]]:
        """Yield only the cross-block edges, in canonical `(global_i,
        global_j, value)` form with `global_i < global_j`."""
        offsets = self._block_offsets()
        for (bi, fi, bj, fj), value in self.cross_block_edges.items():
            yield (offsets[bi] + fi, offsets[bj] + fj, complex(value))

    def to_dense(self) -> np.ndarray:
        """Materialise the full `(N, N)` complex Gram matrix.

        Escape hatch for small clustered dictionaries where the dense
        form fits in memory. The block-diagonal regions copy each
        per-block Gram in place; off-block-diagonal cells are zero
        except where a cross-block edge is present, in which case
        both `[i, j]` and `[j, i] = conj(...)` are filled.

        For large clustered dictionaries this allocates `O(N²)`
        memory; prefer `entries()` for streaming.
        """
        n = self.shape[0]
        out = np.zeros((n, n), dtype=complex)
        offsets = self._block_offsets()
        for k, g in enumerate(self.block_grams):
            base = offsets[k]
            n_k = g.shape[0]
            out[base : base + n_k, base : base + n_k] = g
        for (bi, fi, bj, fj), value in self.cross_block_edges.items():
            gi = offsets[bi] + fi
            gj = offsets[bj] + fj
            v = complex(value)
            out[gi, gj] = v
            out[gj, gi] = v.conjugate()
        return out


# ===========================================================================
# §2 — Block-formation strategies
# ===========================================================================


def compute_cosine_pair_graph(
    decoder_vectors: np.ndarray,
    *,
    threshold: float,
    indices: np.ndarray | None = None,
) -> set[tuple[int, int]]:
    """Return the set of `(i, j)` pairs (i < j) whose decoder-vector
    cosine similarity equals or exceeds `threshold`.

    When `indices` is `None`, pairs index into `decoder_vectors` directly
    (local indices in `[0, N)`). When `indices` is supplied (a 1-D array
    of feature IDs), pairs are reported in terms of `indices` values
    so callers can map subsetted local indices back to global IDs. This
    second mode is what `polygram.compression.epoch._compute_cosine_graph`
    used before this function was extracted; the call site there now
    delegates to this helper with `indices=eligible`.

    For large row counts, the cosine matrix is computed in chunks of
    1024 rows to cap memory.
    """
    if indices is not None:
        if indices.size < 2:
            return set()
        rows = decoder_vectors[indices].astype(np.float32, copy=False)
        report_ids = indices
    else:
        if decoder_vectors.shape[0] < 2:
            return set()
        rows = decoder_vectors.astype(np.float32, copy=False)
        report_ids = np.arange(rows.shape[0], dtype=np.int64)

    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    unit = rows / norms

    out: set[tuple[int, int]] = set()
    n = unit.shape[0]
    chunk = 1024 if n > 1024 else n
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = unit[start:end] @ unit.T
        for local_i in range(end - start):
            global_i = start + local_i
            for global_j in range(global_i + 1, n):
                if sims[local_i, global_j] >= threshold:
                    out.add((int(report_ids[global_i]), int(report_ids[global_j])))
    return out


def _resolve_block_size_max(
    encoding: object, override: int | None
) -> int:
    """Resolve the per-block cap: caller's override > encoding's `max_features`."""
    if override is not None:
        return int(override)
    return int(encoding.max_features)


def _form_blocks_cosine(
    features: list[Feature],
    decoder_vectors: np.ndarray,
    block_size_max: int,
    cosine_threshold: float,
    *,
    cosine_pairs: set[tuple[int, int]] | None = None,
) -> list[list[int]]:
    """Greedy single-linkage BFS clustering with a size cap.

    Builds the cosine pair graph at the supplied threshold (or accepts
    a precomputed one via `cosine_pairs` — see issue #58), then for
    each unassigned feature in input order picks it as a seed and
    grows a block by BFS over its high-cosine neighbours until either
    the block reaches `block_size_max` or the neighbour frontier
    exhausts. Features without close neighbours land in singleton
    blocks. Determinism is preserved by sorted neighbour iteration.

    Returns a list of blocks, each a list of feature indices into the
    input `features` list. Every feature appears in exactly one block.
    """
    n = len(features)
    if n == 0:
        return []
    if cosine_pairs is None:
        cosine_pairs = compute_cosine_pair_graph(
            decoder_vectors, threshold=cosine_threshold
        )
    adj: dict[int, set[int]] = defaultdict(set)
    for i, j in cosine_pairs:
        adj[i].add(j)
        adj[j].add(i)

    assigned: set[int] = set()
    blocks: list[list[int]] = []

    for seed in range(n):
        if seed in assigned:
            continue
        block = [seed]
        assigned.add(seed)
        frontier = [seed]
        while frontier and len(block) < block_size_max:
            next_frontier: list[int] = []
            for feat in frontier:
                if len(block) >= block_size_max:
                    break
                for neighbour in sorted(adj[feat]):
                    if neighbour in assigned:
                        continue
                    if len(block) >= block_size_max:
                        break
                    block.append(neighbour)
                    assigned.add(neighbour)
                    next_frontier.append(neighbour)
            frontier = next_frontier
        blocks.append(block)

    return blocks


def _form_blocks_user_declared(
    features: list[Feature],
    hierarchy: Mapping[str, Sequence[str]],
    block_size_max: int,
) -> list[list[int]]:
    """Respect the supplied hierarchy, splitting any cluster larger
    than `block_size_max` into multiple blocks of size ≤ K.

    Cluster iteration order follows the dict's insertion order so the
    result is deterministic for a given input.

    Returns a list of blocks (each a list of feature indices into the
    input `features` list). Every feature mentioned in the hierarchy
    appears in exactly one block; features not in the hierarchy raise
    a `ValueError`.
    """
    name_to_idx = {f.name: i for i, f in enumerate(features)}
    seen: set[str] = set()
    blocks: list[list[int]] = []
    for cluster, member_names in hierarchy.items():
        cluster_indices: list[int] = []
        for n in member_names:
            if n in seen:
                raise ValueError(
                    f"user_declared block formation: feature {n!r} appears "
                    f"in multiple clusters in the supplied hierarchy"
                )
            if n not in name_to_idx:
                raise ValueError(
                    f"user_declared block formation: hierarchy cluster "
                    f"{cluster!r} references unknown feature {n!r}"
                )
            seen.add(n)
            cluster_indices.append(name_to_idx[n])
        # Split clusters exceeding block_size_max into multiple blocks.
        for i in range(0, len(cluster_indices), block_size_max):
            blocks.append(cluster_indices[i : i + block_size_max])

    unplaced = [f.name for f in features if f.name not in seen]
    if unplaced:
        raise ValueError(
            f"user_declared block formation: features not assigned to any "
            f"cluster in the supplied hierarchy: {unplaced[:5]}"
            + (f" (+{len(unplaced) - 5} more)" if len(unplaced) > 5 else "")
        )
    return blocks


def _form_blocks_co_firing(
    features: list[Feature],
    decoder_vectors: np.ndarray,
    activation_traces: np.ndarray,
    block_size_max: int,
    co_firing_threshold: float,
) -> list[list[int]]:
    """Co-firing similarity-based block formation.

    Currently a NotImplementedError stub. The API is reserved so callers
    can plumb the strategy through `BlockFormation` and the §9 killer
    experiment can pivot to co-firing if cosine clustering's redundancy
    recall is below target. The implementation will follow once a
    consumer needs it (likely as a Jaccard / Pearson similarity on
    binarised activation traces).
    """
    raise NotImplementedError(
        "co_firing block formation is not yet implemented. The "
        "BlockFormation API accepts strategy='co_firing' so the strategy "
        "is reachable for downstream plumbing, but the algorithm itself "
        "will follow once the §9 killer experiment demonstrates cosine "
        "clustering's recall falls below target. Use strategy='cosine' "
        "or strategy='user_declared' in the meantime."
    )


def _compute_cross_block_edges(
    block_indices: list[list[int]],
    decoder_vectors: np.ndarray,
    cosine_threshold: float,
    *,
    cosine_pairs: set[tuple[int, int]] | None = None,
) -> dict[CrossBlockKey, float]:
    """Compute the sparse cross-block adjacency from the block partition
    plus the full decoder-vector array.

    For each pair of features `(i, j)` in different blocks with cosine
    similarity ≥ `cosine_threshold`, emit an entry keyed by
    `(block_i_idx, feat_i_local_idx, block_j_idx, feat_j_local_idx)`
    where `feat_i_local_idx` is the position of feature `i` within
    its own block. The block ordering invariant `block_i_idx <
    block_j_idx` is enforced by sorting the (block_i, block_j) pair.

    `cosine_pairs` may be passed in pre-computed (issue #58) to avoid
    recomputing the same O(N²) cosine graph that block formation
    already produced. When None, computed locally.
    """
    # Build global → (block_idx, local_idx) lookup.
    feat_to_block: dict[int, tuple[int, int]] = {}
    for block_idx, indices in enumerate(block_indices):
        for local_idx, feat_idx in enumerate(indices):
            feat_to_block[feat_idx] = (block_idx, local_idx)

    if cosine_pairs is None:
        cosine_pairs = compute_cosine_pair_graph(
            decoder_vectors, threshold=cosine_threshold
        )
    edges: dict[CrossBlockKey, float] = {}
    # Recompute cosines for the surviving pairs to populate the edge
    # weight; cheap because cosine_pairs is already filtered.
    norms = np.linalg.norm(decoder_vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    unit = decoder_vectors / norms

    for i, j in cosine_pairs:
        bi, fi = feat_to_block[i]
        bj, fj = feat_to_block[j]
        if bi == bj:
            continue  # intra-block edges live in the per-block dense gram
        # Canonicalise: smaller block index first.
        if bi > bj:
            bi, bj = bj, bi
            fi, fj = fj, fi
        cos = float(unit[i] @ unit[j])
        edges[(bi, fi, bj, fj)] = cos
    return edges


def build_clustered_dictionary(
    name: str,
    features: list[Feature],
    decoder_vectors: np.ndarray,
    encoding: MPSRung1 | HEA_Rung2 | Rung3,
    block_formation: BlockFormation,
    *,
    hierarchy: Mapping[str, Sequence[str]] | None = None,
    activation_traces: np.ndarray | None = None,
) -> ClusteredDictionary:
    """Top-level entry point: turn an SAE's feature set into a
    `ClusteredDictionary`.

    Dispatches on `block_formation.strategy`:

    - `"cosine"`: greedy single-linkage on the decoder-vector cosine
      pair graph with size cap. No extra inputs required beyond
      `decoder_vectors`.
    - `"co_firing"`: Jaccard / Pearson similarity on activation
      traces. Requires `activation_traces`. Currently raises
      `NotImplementedError` (the API is reserved).
    - `"user_declared"`: respect the supplied `hierarchy`, splitting
      any cluster exceeding the per-block cap into multiple blocks.
      Requires `hierarchy`.

    Cross-block edges (the `cross_block_pairs` field of the returned
    `ClusteredDictionary`) are always populated from the
    decoder-vector cosine graph — encoding-agnostic, threshold from
    `block_formation.cosine_threshold`.

    Parameters
    ----------
    name:
        Identifier for the resulting `ClusteredDictionary`.
    features:
        Per-feature carriers (name, cluster, angle knobs). The order
        of `features` defines the global feature index space.
    decoder_vectors:
        `(n_features, d_model)` decoder-direction matrix. Must satisfy
        `decoder_vectors.shape[0] == len(features)`.
    encoding:
        Shared encoding for every block.
    block_formation:
        Strategy + parameters.
    hierarchy:
        Required when `strategy="user_declared"`; ignored otherwise.
    activation_traces:
        Required when `strategy="co_firing"`; ignored otherwise.
    """
    n_features = len(features)
    if decoder_vectors.shape[0] != n_features:
        raise ValueError(
            f"build_clustered_dictionary: decoder_vectors first axis "
            f"({decoder_vectors.shape[0]}) must equal len(features) "
            f"({n_features})"
        )
    if n_features == 0:
        raise ValueError(
            "build_clustered_dictionary: features must be non-empty"
        )

    cap = _resolve_block_size_max(encoding, block_formation.block_size_max)

    # Pre-compute the cosine pair graph once and share between block
    # formation (cosine strategy only) and cross-block adjacency
    # (always). Issue #58: previously computed twice on the same
    # decoder_vectors+threshold, which dominated wall-clock at N≥2k.
    # Skip for `co_firing` — that path raises before touching either.
    if block_formation.strategy in ("cosine", "user_declared"):
        cosine_pairs = compute_cosine_pair_graph(
            decoder_vectors, threshold=block_formation.cosine_threshold
        )
    else:
        cosine_pairs = None

    if block_formation.strategy == "cosine":
        block_indices = _form_blocks_cosine(
            features, decoder_vectors, cap, block_formation.cosine_threshold,
            cosine_pairs=cosine_pairs,
        )
    elif block_formation.strategy == "co_firing":
        if activation_traces is None:
            raise ValueError(
                "build_clustered_dictionary: strategy='co_firing' requires "
                "`activation_traces` (got None)"
            )
        block_indices = _form_blocks_co_firing(
            features,
            decoder_vectors,
            activation_traces,
            cap,
            block_formation.cosine_threshold,
        )
    elif block_formation.strategy == "user_declared":
        if hierarchy is None:
            raise ValueError(
                "build_clustered_dictionary: strategy='user_declared' "
                "requires `hierarchy` (got None)"
            )
        block_indices = _form_blocks_user_declared(features, hierarchy, cap)
    else:  # pragma: no cover — covered by BlockFormation's own validation
        raise ValueError(
            f"build_clustered_dictionary: unknown strategy "
            f"{block_formation.strategy!r}"
        )

    blocks = _materialise_blocks(
        name, features, block_indices, encoding, block_formation, hierarchy
    )
    cross_block_pairs = _compute_cross_block_edges(
        block_indices, decoder_vectors, block_formation.cosine_threshold,
        cosine_pairs=cosine_pairs,
    )
    return ClusteredDictionary(
        name=name,
        blocks=blocks,
        cross_block_pairs=cross_block_pairs,
        block_formation=block_formation,
    )


def _materialise_blocks(
    parent_name: str,
    features: list[Feature],
    block_indices: list[list[int]],
    encoding: MPSRung1 | HEA_Rung2 | Rung3,
    block_formation: BlockFormation,
    hierarchy: Mapping[str, Sequence[str]] | None,
) -> list[Dictionary]:
    """Construct one `Dictionary` per block from the partition.

    For `strategy="user_declared"` blocks preserve the supplied
    hierarchy's cluster names. For the other strategies each block is
    a single synthetic cluster named `<parent_name>_b<idx>`.
    """
    blocks: list[Dictionary] = []
    for block_idx, indices in enumerate(block_indices):
        block_feats = [features[i] for i in indices]
        block_name = f"{parent_name}_b{block_idx}"
        if block_formation.strategy == "user_declared" and hierarchy is not None:
            # Preserve original Feature.cluster values; rebuild hierarchy
            # restricted to this block's members.
            block_hierarchy: dict[str, list[str]] = defaultdict(list)
            for f in block_feats:
                block_hierarchy[f.cluster].append(f.name)
            block_hierarchy_resolved: dict[str, list[str]] = dict(block_hierarchy)
        else:
            # Synthesise a single cluster for the block. Rewrite each
            # feature's `cluster` field to match so Dictionary's
            # hierarchy invariant holds.
            cluster = block_name
            block_feats = [replace(f, cluster=cluster) for f in block_feats]
            block_hierarchy_resolved = {cluster: [f.name for f in block_feats]}
        blocks.append(
            Dictionary(
                name=block_name,
                features=block_feats,
                hierarchy=block_hierarchy_resolved,
                encoding=encoding,
            )
        )
    return blocks
