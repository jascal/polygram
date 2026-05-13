"""`ClusteredDictionary` — block-decomposed dictionary for SAE-scale analyses.

Implements §1 of `openspec/changes/clustered-dictionary-analysis/tasks.md`.

The single-`Dictionary` primitives (`Dictionary.gram`, `Cancellation`,
`BehaviouralValidator`, Q-OrCA emission) cap a dictionary at its
encoding's Hilbert-space dimension — 8 features for `MPSRung1`, 16 for
`Rung3`, `2**n_qubits` for `HEA_Rung2`. Real SAEs ship at 16k–1M
features per layer; `ClusteredDictionary` is the primitive that holds
the entire SAE as a list of ≤K-feature `Dictionary` blocks plus a
sparse cross-block adjacency, so block-local analyses delegate to the
existing primitives and cross-block analyses use encoding-agnostic
direct decoder-vector overlaps.

This module ships the data structures and validation. The §2-§5 work
(block-formation strategies, `BlockSparseGram`,
`cross_block_redundant_pairs`, `from_sae_lens(... clustered=True)`)
layers on top in follow-up commits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from polygram.dictionary import Dictionary, Feature
from polygram.encoding import HEA_Rung2, MPSRung1, Rung3

# Fallback when per-encoding-feature-cap (PR #42) hasn't shipped — the
# loader-side cap is the universal 8 from sae_import. Once #42 lands,
# every encoding exposes `max_features` directly and the fallback is
# unused; until then the defensive getattr() preserves the legacy
# constant.
_LEGACY_MAX_FEATURES = 8


def _encoding_max_features(encoding: object) -> int:
    """Return the encoding's feature cap, defaulting to the legacy 8.

    Honours `encoding.max_features` when present (per-encoding-feature-cap),
    falls back to the module-level constant otherwise. The fallback path
    will be removed once PR #42 merges.
    """
    cap = getattr(encoding, "max_features", None)
    if cap is None:
        return _LEGACY_MAX_FEATURES
    return int(cap)


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

        # Per-block size cap. Honour `encoding.max_features` when the
        # encoding declares it (PR #42 path); fall back to the legacy
        # universal 8 otherwise.
        cap = _encoding_max_features(encoding)
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
    """Resolve the per-block cap: caller's override > encoding's
    `max_features` > legacy 8."""
    if override is not None:
        return int(override)
    return _encoding_max_features(encoding)


def _form_blocks_cosine(
    features: list[Feature],
    decoder_vectors: np.ndarray,
    block_size_max: int,
    cosine_threshold: float,
) -> list[list[int]]:
    """Greedy single-linkage BFS clustering with a size cap.

    Builds the cosine pair graph at the supplied threshold, then for
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
) -> dict[CrossBlockKey, float]:
    """Compute the sparse cross-block adjacency from the block partition
    plus the full decoder-vector array.

    For each pair of features `(i, j)` in different blocks with cosine
    similarity ≥ `cosine_threshold`, emit an entry keyed by
    `(block_i_idx, feat_i_local_idx, block_j_idx, feat_j_local_idx)`
    where `feat_i_local_idx` is the position of feature `i` within
    its own block. The block ordering invariant `block_i_idx <
    block_j_idx` is enforced by sorting the (block_i, block_j) pair.
    """
    # Build global → (block_idx, local_idx) lookup.
    feat_to_block: dict[int, tuple[int, int]] = {}
    for block_idx, indices in enumerate(block_indices):
        for local_idx, feat_idx in enumerate(indices):
            feat_to_block[feat_idx] = (block_idx, local_idx)

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

    if block_formation.strategy == "cosine":
        block_indices = _form_blocks_cosine(
            features, decoder_vectors, cap, block_formation.cosine_threshold
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
        block_indices, decoder_vectors, block_formation.cosine_threshold
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
