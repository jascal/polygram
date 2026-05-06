"""The `merge` compression strategy.

For every cluster:
- The representative's W_dec row direction is preserved but rescaled
  to a ``merged_norm`` derived from the cluster members' source norms.
- All non-representative members' W_dec rows, W_enc columns, and
  b_enc entries are zeroed (same as the ``zero`` strategy).

``merge_mode`` selects the averaging formula:
- ``"freq_weighted"``: ``Σ(norm_f · fires_f) / Σ(fires_f)``.
  Falls back to ``simple_mean`` when total fires is zero.
- ``"simple_mean"``: ``mean(norms)``.

``b_dec`` is global and untouched. ``merged_norms`` (cluster_id →
merged_norm) is returned alongside the rewritten state-dict so the
caller can populate ``ClusterPlan.merged_norm``.
"""

from __future__ import annotations

import numpy as np

from polygram.compression.report import CompressionPlan


_REQUIRED_KEYS: tuple[str, ...] = ("W_enc", "b_enc", "W_dec", "b_dec")
_SUPPORTED_MERGE_MODES = ("freq_weighted", "simple_mean")
_NORM_EPS = 1e-8


def apply_merge(
    state_dict: dict[str, np.ndarray],
    plan: CompressionPlan,
    *,
    merge_mode: str = "freq_weighted",
    n_fires_by_fid: dict[int, int] | None = None,
) -> tuple[dict[str, np.ndarray], dict[int, float]]:
    """Return (rewritten state-dict, merged_norms-by-cluster-id).

    ``n_fires_by_fid`` is required for ``merge_mode="freq_weighted"``;
    ignored for ``"simple_mean"``. When fires sum to zero across a
    cluster's members the freq-weighted path silently degrades to
    ``simple_mean`` (avoids division by zero).

    Raises ``KeyError`` if any of W_enc / b_enc / W_dec / b_dec is
    missing, or ``ValueError`` for unrecognised ``merge_mode``.
    """
    if merge_mode not in _SUPPORTED_MERGE_MODES:
        raise ValueError(
            f"apply_merge: unsupported merge_mode {merge_mode!r}; "
            f"supported: {list(_SUPPORTED_MERGE_MODES)}"
        )
    missing = [k for k in _REQUIRED_KEYS if k not in state_dict]
    if missing:
        raise KeyError(
            f"apply_merge: source checkpoint is missing required key(s) "
            f"{missing!r}; the merge strategy needs a full SAE checkpoint "
            f"with W_enc / b_enc / W_dec / b_dec"
        )

    out = {k: np.array(v, copy=True) for k, v in state_dict.items()}
    w_enc = out["W_enc"]
    b_enc = out["b_enc"]
    w_dec = out["W_dec"]
    source_w_dec = state_dict["W_dec"]

    n_features_dec = w_dec.shape[0]
    merged_norms: dict[int, float] = {}

    for cluster in plan.clusters:
        members = cluster.members
        rep = cluster.representative
        for fid in members:
            if not (0 <= fid < n_features_dec):
                raise IndexError(
                    f"apply_merge: feature id {fid} out of range for "
                    f"decoder shape {w_dec.shape!r}"
                )

        norms = np.array(
            [float(np.linalg.norm(source_w_dec[fid])) for fid in members],
            dtype=np.float64,
        )

        if merge_mode == "freq_weighted":
            fires = (
                {fid: int(n_fires_by_fid.get(fid, 0)) for fid in members}
                if n_fires_by_fid is not None
                else {fid: 1 for fid in members}
            )
            total_fires = sum(fires.values())
            if total_fires <= 0:
                merged = float(norms.mean())
            else:
                merged = float(
                    sum(
                        float(norms[i]) * float(fires[fid])
                        for i, fid in enumerate(members)
                    )
                    / total_fires
                )
        else:  # "simple_mean"
            merged = float(norms.mean())

        merged_norms[cluster.cluster_id] = merged

        rep_norm = float(np.linalg.norm(w_dec[rep]))
        if rep_norm > _NORM_EPS:
            w_dec[rep] = w_dec[rep] * (merged / rep_norm)

        for fid in members:
            if fid == rep:
                continue
            w_enc[:, fid] = 0
            b_enc[fid] = 0
            w_dec[fid, :] = 0

    return out, merged_norms
