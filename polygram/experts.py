"""`ExpertDictionary` — runtime-routable view of a clustered Dictionary.

`ClusteredDictionary` already represents a flat `Dictionary` as a list of
blocks plus sparse cross-block adjacency. `ExpertDictionary` is the
runtime-routing surface on top of that structure: each block becomes
an "expert," and `route(activations, top_k)` returns the top-k expert
indices to fire on a given activation vector.

The MVP intentionally ships a single block-formation method (`"cosine"`),
delegating to the existing `build_clustered_dictionary` path. The
`method="coactivation"` token is reserved for the implementation that
lands alongside `clustered_dictionary._form_blocks_co_firing`; today
it raises `NotImplementedError` to match the underlying stub.

Out of scope for the MVP (and not implemented here):

- `method="louvain"` / `"hdbscan"` — heavyweight deps, deferred until
  cosine has measured usage gaps.
- Trained MLP router — belongs in sae-forge (where torch lives).
- Bio-specific scoring (GO enrichment, motif overlap) — downstream.
- `min_cluster_size` / `max_experts` post-processing — pending real
  user demand to tune the shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from polygram.clustered_dictionary import (
    BlockFormation,
    ClusteredDictionary,
    build_clustered_dictionary,
)
from polygram.dictionary import Dictionary


_SUPPORTED_METHODS = ("cosine", "coactivation")
ExpertFormationMethod = Literal["cosine", "coactivation"]


@dataclass(frozen=True)
class ExpertDictionary:
    """A flat `Dictionary` partitioned into routable experts.

    Fields:

    - ``experts`` — tuple of per-expert `Dictionary` blocks. Each
      expert exposes the full `Dictionary` API (`.features`, `.gram()`,
      etc.) so downstream callers can analyse experts individually
      with no shim layer.
    - ``source`` — the flat `Dictionary` this was clustered from.
    - ``_feature_to_expert`` — precomputed map from `source.features`
      index to expert index. Drives O(N) routing without re-walking
      the experts at every `route(...)` call.

    Construct via :func:`cluster_experts`. Direct construction is
    supported but the invariants (feature partition, non-empty
    experts) are enforced in ``__post_init__``.
    """

    experts: tuple[Dictionary, ...]
    source: Dictionary
    _feature_to_expert: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.experts) == 0:
            raise ValueError("ExpertDictionary: experts must be non-empty")
        for i, expert in enumerate(self.experts):
            if len(expert.features) == 0:
                raise ValueError(
                    f"ExpertDictionary: expert {i} is empty"
                )

        n_features = len(self.source.features)
        if len(self._feature_to_expert) != n_features:
            raise ValueError(
                f"ExpertDictionary: _feature_to_expert length "
                f"({len(self._feature_to_expert)}) must equal "
                f"len(source.features) ({n_features})"
            )

        source_names = [f.name for f in self.source.features]
        partition: set[str] = set()
        for expert in self.experts:
            for f in expert.features:
                if f.name in partition:
                    raise ValueError(
                        f"ExpertDictionary: feature {f.name!r} appears "
                        f"in more than one expert"
                    )
                if f.name not in source_names:
                    raise ValueError(
                        f"ExpertDictionary: feature {f.name!r} is not in "
                        f"source.features"
                    )
                partition.add(f.name)
        if partition != set(source_names):
            missing = set(source_names) - partition
            raise ValueError(
                f"ExpertDictionary: feature partition is incomplete; "
                f"missing from any expert: {sorted(missing)}"
            )

    @property
    def n_experts(self) -> int:
        return len(self.experts)

    @property
    def n_features(self) -> int:
        return sum(len(e.features) for e in self.experts)

    def route(self, activations: np.ndarray, top_k: int) -> list[int]:
        """Return the top-``k`` expert indices ordered by summed activation.

        ``activations`` is a per-feature activation vector indexed in
        the order of ``self.source.features``. Aggregation is plain
        summation per expert; callers wanting normalised or sparse
        activations should preprocess upstream.

        Parameters
        ----------
        activations:
            ``np.ndarray`` of shape ``(n_features,)``.
        top_k:
            Number of experts to return, ``1 <= top_k <= n_experts``.

        Returns
        -------
        list[int]
            Expert indices, highest summed activation first.
        """
        if not isinstance(activations, np.ndarray):
            raise ValueError(
                f"ExpertDictionary.route: activations must be an ndarray; "
                f"got {type(activations).__name__}"
            )
        n_features = len(self.source.features)
        if activations.shape != (n_features,):
            raise ValueError(
                f"ExpertDictionary.route: activations.shape={activations.shape} "
                f"must equal (n_features={n_features},)"
            )
        if not (1 <= top_k <= self.n_experts):
            raise ValueError(
                f"ExpertDictionary.route: top_k={top_k} must satisfy "
                f"1 <= top_k <= n_experts={self.n_experts}"
            )

        scores = np.zeros(self.n_experts, dtype=np.float64)
        np.add.at(scores, np.asarray(self._feature_to_expert), activations)
        order = np.argsort(-scores, kind="stable")
        return order[:top_k].tolist()


def cluster_experts(
    dictionary: Dictionary,
    decoder_vectors: np.ndarray,
    *,
    method: ExpertFormationMethod = "cosine",
    coherence_threshold: float = 0.3,
    max_features_per_expert: int | None = None,
    activations: np.ndarray | None = None,
) -> ExpertDictionary:
    """Cluster ``dictionary``'s features into routable experts.

    The MVP delegates block formation to the existing
    :func:`polygram.clustered_dictionary.build_clustered_dictionary`
    cosine path and wraps the result as an :class:`ExpertDictionary`.

    Parameters
    ----------
    dictionary:
        The flat source `Dictionary`.
    decoder_vectors:
        ``(n_features, d_model)`` decoder-direction matrix, indexed in
        the order of ``dictionary.features``. Required for cosine
        block formation.
    method:
        Either ``"cosine"`` (MVP default) or ``"coactivation"`` (raises
        ``NotImplementedError`` — reserved for when the underlying
        co_firing block-formation stub is implemented).
    coherence_threshold:
        Forwarded to ``BlockFormation.cosine_threshold``. Higher
        values produce tighter, smaller experts.
    max_features_per_expert:
        Per-expert feature cap; ``None`` defers to the encoding's
        ``max_features``.
    activations:
        Reserved for ``method="coactivation"``; currently unused.

    Returns
    -------
    ExpertDictionary
        Partition of ``dictionary.features`` into experts plus a
        routing-ready feature→expert index map.
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"cluster_experts: unknown method {method!r}; "
            f"supported: {_SUPPORTED_METHODS}"
        )
    if method == "coactivation":
        raise NotImplementedError(
            "cluster_experts(method='coactivation') is reserved; the "
            "underlying co_firing block-formation strategy is not yet "
            "implemented in clustered_dictionary._form_blocks_co_firing. "
            "Use method='cosine' for now."
        )

    n_features = len(dictionary.features)
    if decoder_vectors.shape[0] != n_features:
        raise ValueError(
            f"cluster_experts: decoder_vectors first axis "
            f"({decoder_vectors.shape[0]}) must equal "
            f"len(dictionary.features) ({n_features})"
        )

    bf = BlockFormation(
        strategy="cosine",
        cosine_threshold=coherence_threshold,
        block_size_max=max_features_per_expert,
    )
    clustered: ClusteredDictionary = build_clustered_dictionary(
        name=f"{dictionary.name}_experts",
        features=list(dictionary.features),
        decoder_vectors=decoder_vectors,
        encoding=dictionary.encoding,
        block_formation=bf,
    )

    feature_index = {f.name: i for i, f in enumerate(dictionary.features)}
    feature_to_expert = [0] * n_features
    for expert_idx, block in enumerate(clustered.blocks):
        for f in block.features:
            feature_to_expert[feature_index[f.name]] = expert_idx

    return ExpertDictionary(
        experts=tuple(clustered.blocks),
        source=dictionary,
        _feature_to_expert=tuple(feature_to_expert),
    )
