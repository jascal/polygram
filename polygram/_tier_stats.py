"""Tier statistics — roll up an N×N Gram matrix into self / sibling /
cross-cluster mean overlaps.

Tier definitions (computed over feature-pair indices `(i, j)` with
`i != j`):

- `self` — diagonal entries (always 1.0; included for symmetry).
- `sibling` — pairs sharing a cluster.
- `cross_cluster` — pairs in different clusters.

Each tier returns the mean of `|G[i, j]|²` over the relevant pairs.
NaN is returned when the corresponding pair set is empty (e.g. no
cluster has ≥ 2 members → sibling tier is NaN).
"""

from __future__ import annotations

import numpy as np

from polygram.dictionary import Dictionary

TIER_NAMES = ("self", "sibling", "cross_cluster")


def _tier_pairs(
    dictionary: Dictionary,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    sibling: list[tuple[int, int]] = []
    cross: list[tuple[int, int]] = []
    n = len(dictionary.features)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ci = dictionary.features[i].cluster
            cj = dictionary.features[j].cluster
            if ci == cj:
                sibling.append((i, j))
            else:
                cross.append((i, j))
    return sibling, cross


def compute_tier_stats(
    gram: np.ndarray, dictionary: Dictionary
) -> dict[str, float]:
    """Compute tier means for a single Gram matrix.

    Returns a dict with float values for each tier in `TIER_NAMES`.
    `gram` may be complex; the returned values are real `|G|²` means.
    """
    sq = np.abs(gram) ** 2
    sibling_pairs, cross_pairs = _tier_pairs(dictionary)
    out: dict[str, float] = {"self": 1.0}
    out["sibling"] = (
        float(np.mean([sq[i, j] for i, j in sibling_pairs]))
        if sibling_pairs
        else float("nan")
    )
    out["cross_cluster"] = (
        float(np.mean([sq[i, j] for i, j in cross_pairs]))
        if cross_pairs
        else float("nan")
    )
    return out
