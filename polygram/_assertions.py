"""Built-in assertion checkers for InterferenceSweep results.

Each checker takes the per-sweep-point Gram matrix and returns a bool.
The sweep runner stacks per-point bools into the per-assertion array
exposed on `ExperimentResult.assertion_pass`.
"""

from __future__ import annotations

import numpy as np

from polygram.dictionary import Dictionary

DEFAULT_DESTRUCTIVE_THRESHOLD = 0.1

SUPPORTED_ASSERTIONS = (
    "hierarchical_ordering_preserved",
    "target_pair_destructive_at_endpoint",
)


def hierarchical_ordering_preserved(
    gram: np.ndarray,
    dictionary: Dictionary,
    target_pair: tuple[str, str],
) -> bool:
    """At this sweep point, in-cluster overlaps for both target features
    SHALL each be ≥ the cross-cluster target-pair overlap.

    For target `(A, B)` with `A` in cluster `Ka`, `B` in cluster `Kb`:

    - For each sibling `S != A` in `Ka`: |<A|S>|² ≥ |<A|B>|²
    - For each sibling `S != B` in `Kb`: |<B|S>|² ≥ |<A|B>|²
    """
    a_name, b_name = target_pair
    a = dictionary.feature_index(a_name)
    b = dictionary.feature_index(b_name)
    sq = np.abs(gram) ** 2
    cross = sq[a, b]

    a_cluster = dictionary.feature(a_name).cluster
    b_cluster = dictionary.feature(b_name).cluster

    for sibling_name in dictionary.hierarchy[a_cluster]:
        if sibling_name == a_name:
            continue
        s = dictionary.feature_index(sibling_name)
        if sq[a, s] + 1e-12 < cross:
            return False

    for sibling_name in dictionary.hierarchy[b_cluster]:
        if sibling_name == b_name:
            continue
        s = dictionary.feature_index(sibling_name)
        if sq[b, s] + 1e-12 < cross:
            return False

    return True


def target_pair_destructive_at_endpoint(
    gram: np.ndarray,
    dictionary: Dictionary,
    target_pair: tuple[str, str],
    threshold: float = DEFAULT_DESTRUCTIVE_THRESHOLD,
) -> bool:
    """At the *endpoint* of the sweep, |<A|B>|² SHALL be < `threshold`.

    Per the spec: for non-endpoint sweep points the result still reflects
    the endpoint's pass/fail (the assertion is single-point but we store
    one bool per point so users can index uniformly). Callers compute
    this only for the endpoint Gram; the runner broadcasts the result
    across the full sweep dimension.
    """
    a = dictionary.feature_index(target_pair[0])
    b = dictionary.feature_index(target_pair[1])
    sq = float(np.abs(gram[a, b]) ** 2)
    return sq < threshold
