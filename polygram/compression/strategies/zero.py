"""The `zero` compression strategy.

For every non-representative member of every cluster:

    W_enc[:, fid] = 0    # zero the feature's encoder column
    b_enc[fid]    = 0    # zero the encoder bias
    W_dec[fid, :] = 0    # zero the feature's decoder row

`b_dec` is global (per d_model), not feature-specific, so it is
untouched. See `add-compression-action/design.md` Decision 5 for the
rationale on zeroing both encoder and decoder.
"""

from __future__ import annotations

import numpy as np

from polygram.compression.report import CompressionPlan


_REQUIRED_KEYS: tuple[str, ...] = ("W_enc", "b_enc", "W_dec", "b_dec")


def apply_zero(
    state_dict: dict[str, np.ndarray],
    plan: CompressionPlan,
) -> dict[str, np.ndarray]:
    """Return a new state-dict with the plan's `zeroed` features
    silenced in `W_enc`, `b_enc`, and `W_dec`.

    The input `state_dict` is not mutated. Arrays are copied; keys not
    in `_REQUIRED_KEYS` are passed through unchanged.

    Raises ``KeyError`` if any required key is missing — the
    compression action only operates on full SAE checkpoints
    containing encoder + decoder weights and biases.
    """
    missing = [k for k in _REQUIRED_KEYS if k not in state_dict]
    if missing:
        raise KeyError(
            f"apply_zero: source checkpoint is missing required key(s) "
            f"{missing!r}; the zero strategy needs a full SAE checkpoint "
            f"with W_enc / b_enc / W_dec / b_dec"
        )

    out = {k: np.array(v, copy=True) for k, v in state_dict.items()}
    w_enc = out["W_enc"]
    b_enc = out["b_enc"]
    w_dec = out["W_dec"]

    n_features_dec = w_dec.shape[0]
    for cluster in plan.clusters:
        for fid in cluster.zeroed:
            if not (0 <= fid < n_features_dec):
                raise IndexError(
                    f"apply_zero: feature id {fid} out of range for "
                    f"decoder shape {w_dec.shape!r}"
                )
            w_enc[:, fid] = 0
            b_enc[fid] = 0
            w_dec[fid, :] = 0

    return out
