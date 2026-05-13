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
from dataclasses import dataclass, field
from typing import Literal

from polygram.dictionary import Dictionary
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
