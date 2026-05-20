"""Shared helpers for compression report dataclasses and compressors."""

from __future__ import annotations

import math
from typing import Literal


def informative_metric(rank_ratio: float) -> Literal["post_A", "both", "forge_mse"]:
    """Derive the informative metric from the rank_ratio threshold.

    - rank_ratio < 0.95  → trust post_A, treat MSE as sanity check
    - 0.95 ≤ rank_ratio ≤ 1.05 → both informative
    - rank_ratio > 1.05  → trust forge_mse, treat post_A as sanity check
    """
    if rank_ratio < 0.95:
        return "post_A"
    elif rank_ratio <= 1.05:
        return "both"
    else:
        return "forge_mse"


def compute_rank_ratio(
    w_dec_kept: "np.ndarray[float]",
    d_model: int,
) -> float:
    """rank_ratio = basis_rank(W_dec_kept) / d_model.

    Args:
        w_dec_kept: kept features' decoder rows, shape (n_kept, d_model)
        d_model: decoder column dimension
    """
    import numpy as np

    rank = float(np.linalg.matrix_rank(w_dec_kept))
    return rank / d_model


def json_finite(v: float | None) -> float | None:
    """JSON-encode a numeric field. NaN/Inf/None serialize as JSON null."""
    if v is None:
        return None
    fv = float(v)
    if not math.isfinite(fv):
        return None
    return float(format(fv, ".6g"))


def floats_eq(a: float | None, b: float | None) -> bool:
    """NaN-aware float equality."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b
