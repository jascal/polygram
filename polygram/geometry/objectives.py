"""Objectives for `LearnedKnobAssignment`.

A `LearnedAxisObjective` scores a candidate analytic gram against a
reference geometry matrix; the learned-knob-assignment solver
maximises this scalar. Three built-ins ship: Spearman / Pearson rank
correlations against a decoder cosine² matrix, and a factory for a
caller-supplied behavioural reference matrix.

See ``polygram.geometry.protocols.LearnedAxisObjective`` for the
protocol the built-ins satisfy.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _off_diagonal_pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract the strict upper-triangle entries of two same-shape
    square matrices. The learned-knob-assignment objectives all score
    on off-diagonal pairs only — the diagonal carries no comparison
    information (every analytic gram has unit diagonal)."""
    if a.shape != b.shape:
        raise ValueError(
            f"_off_diagonal_pairs: shape mismatch {a.shape} vs {b.shape}"
        )
    n = a.shape[0]
    iu = np.triu_indices(n, k=1)
    return a[iu].astype(float), b[iu].astype(float)


def _spearman_off_diagonal(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between off-diagonal entries of two
    square matrices. Returns 0 for degenerate inputs (constant or
    near-zero variance on either side).

    Lifted from the original
    ``examples/rung5_pareto_scans.py::_spearman`` prototype so the
    canonical implementation lives next to the objective surface;
    the example script now imports this helper.
    """
    x, y = _off_diagonal_pairs(a, b)
    if x.size < 2 or x.std() < 1e-15 or y.std() < 1e-15:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = float(np.sqrt((rx_c ** 2).sum() * (ry_c ** 2).sum()))
    if denom < 1e-15:
        return 0.0
    return float((rx_c * ry_c).sum() / denom)


def _pearson_off_diagonal(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between off-diagonal entries of two square
    matrices. Returns 0 for degenerate inputs."""
    x, y = _off_diagonal_pairs(a, b)
    if x.size < 2 or x.std() < 1e-15 or y.std() < 1e-15:
        return 0.0
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(np.sqrt((xc ** 2).sum() * (yc ** 2).sum()))
    if denom < 1e-15:
        return 0.0
    return float((xc * yc).sum() / denom)


def spearman_objective(
    analytic_gram: np.ndarray,
    decoder_geom: np.ndarray,
    *,
    feature_names: list[str] | None = None,
) -> float:
    """Spearman rank correlation between off-diagonal entries of
    ``|analytic_gram|²`` and ``decoder_geom``. Default objective for
    `LearnedKnobAssignment`.

    ``feature_names`` is accepted for `LearnedAxisObjective` protocol
    conformance but ignored — Spearman is a per-pair scalar that
    doesn't need cluster context.
    """
    del feature_names  # not used; here for protocol conformance
    return _spearman_off_diagonal(np.abs(analytic_gram) ** 2, decoder_geom)


def pearson_objective(
    analytic_gram: np.ndarray,
    decoder_geom: np.ndarray,
    *,
    feature_names: list[str] | None = None,
) -> float:
    """Pearson correlation between off-diagonal entries of
    ``|analytic_gram|²`` and ``decoder_geom``. Cheaper than Spearman;
    correct when the relationship is roughly linear (e.g., when
    decoder cosines are already well-scaled into [0, 1])."""
    del feature_names
    return _pearson_off_diagonal(np.abs(analytic_gram) ** 2, decoder_geom)


def behavioural_objective(
    reference_pair_sims: np.ndarray,
) -> Callable[..., float]:
    """Factory returning an objective that scores the analytic gram
    against a caller-supplied ground-truth pair-similarity matrix
    rather than against decoder cosines.

    The returned closure ignores its ``decoder_geom`` argument and
    correlates against ``reference_pair_sims`` instead. Useful when
    behavioural co-activation matrices (e.g. from sae-forge's
    behavioural validator output) provide a stronger fidelity signal
    than decoder geometry alone.

    Parameters
    ----------
    reference_pair_sims : np.ndarray
        Square matrix of pair similarities, same shape as the analytic
        gram. Higher entries denote pairs the strategy should make
        more similar in the encoded space.
    """
    ref = np.asarray(reference_pair_sims, dtype=float)
    if ref.ndim != 2 or ref.shape[0] != ref.shape[1]:
        raise ValueError(
            f"behavioural_objective: reference_pair_sims must be a "
            f"square matrix; got shape {ref.shape}"
        )

    def _objective(
        analytic_gram: np.ndarray,
        decoder_geom: np.ndarray,  # noqa: ARG001 — ignored on purpose
        *,
        feature_names: list[str] | None = None,
    ) -> float:
        del feature_names
        return _spearman_off_diagonal(np.abs(analytic_gram) ** 2, ref)

    _objective.__doc__ = (
        "Behavioural-fidelity objective bound to a fixed reference "
        f"pair-similarity matrix of shape {ref.shape}."
    )
    return _objective
