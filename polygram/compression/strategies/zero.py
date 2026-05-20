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


_REQUIRED_KEYS: tuple[str, ...] = ("W_dec",)
_OPTIONAL_KEYS: tuple[str, ...] = ("W_enc", "b_enc", "b_dec")


def apply_zero(
    state_dict: dict[str, np.ndarray],
    plan: CompressionPlan,
) -> dict[str, np.ndarray]:
    """Return a new state-dict with the plan's `zeroed` features
    silenced in `W_dec` (always) and in `W_enc` / `b_enc` (when present).

    The input `state_dict` is not mutated. Arrays are copied; keys not
    in `_REQUIRED_KEYS ∪ _OPTIONAL_KEYS` are passed through unchanged.

    Required key: ``W_dec``. Raises ``KeyError`` if it is missing.

    Optional keys: ``W_enc``, ``b_enc``, ``b_dec``. When present the
    strategy zeros the corresponding columns / rows for each zeroed
    feature; when absent the per-key operation is silently skipped so
    decoder-only callers (e.g. sae-forge's synth-basis) succeed.
    ``b_dec`` is global and never modified.
    """
    missing = [k for k in _REQUIRED_KEYS if k not in state_dict]
    if missing:
        raise KeyError(
            f"apply_zero: source checkpoint is missing required key(s) "
            f"{missing!r}; the zero strategy requires W_dec"
        )

    out = {k: np.array(v, copy=True) for k, v in state_dict.items()}
    w_dec = out["W_dec"]
    w_enc = out.get("W_enc")
    b_enc = out.get("b_enc")

    n_features_dec = w_dec.shape[0]
    for cluster in plan.clusters:
        for fid in cluster.zeroed:
            if not (0 <= fid < n_features_dec):
                raise IndexError(
                    f"apply_zero: feature id {fid} out of range for "
                    f"decoder shape {w_dec.shape!r}"
                )
            if w_enc is not None:
                w_enc[:, fid] = 0
            if b_enc is not None:
                b_enc[fid] = 0
            w_dec[fid, :] = 0

    return out
