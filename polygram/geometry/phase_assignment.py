"""Phase-knob assignment from decoder geometry (PCA-axis extension).

`from_sae_lens` consumes this when `assign_phase_knobs=True`. The helper
maps PC2 and PC3 of the projection vectors into the encoding's α and φ
knob ranges, producing per-feature values that vary with decoder
geometry rather than collapsing to the encoding's default (0, 0).

This is the `add-phase-knob-assignment` v1 strategy. See
`openspec/changes/add-phase-knob-assignment/design.md` Decision 1 for
the rationale (parallel to PR #63's amp-knob helper, phase knobs apply
to *all* encodings with MPS-substrate α and φ).

PCA notation: PC_k = the k-th principal component (1-indexed; PC1 is
the top component). In code these correspond to `vt[k-1]` rows of the
SVD output (0-indexed). PC1 is already consumed by β in the surrounding
strategy; PC2 → α, PC3 → φ.

**Knob → PCA-component priority order** (across all of polygram):

    PC1 → β               (existing β strategy, surrounding helper)
    PC2 → α               (this module, MPS-substrate)
    PC3 → φ               (this module, MPS-substrate)
    PC4 → theta_amp       (amp_assignment.py, Rung3+)
    PC5 → psi_aux         (amp_assignment.py, Rung3+)
    PC6 → theta_amp_b     (amp_assignment.py, Rung4 only)
    PC7 → psi_amp_b       (amp_assignment.py, Rung4 only)

Phase knobs take the low slots (PC2/PC3) because they apply to every
encoding with MPS-substrate knobs. Amp knobs shifted to PC4-PC7 in
add-phase-knob-assignment to make room. β is fixed at PC1 by the
existing strategy and was never displaced.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# INFO-once flag so users notice when their `assign_phase_knobs=True`
# flag is a no-op for HEA_Rung2. Per the openspec design.md.
_INFO_LOGGED_NO_OP: set[str] = set()


def _info_once(encoding_name: str, message: str) -> None:
    if encoding_name not in _INFO_LOGGED_NO_OP:
        logger.info(message)
        _INFO_LOGGED_NO_OP.add(encoding_name)


def assign_phase_knobs_pca(
    projections: np.ndarray,
    encoding: object,
) -> dict[str, list[float] | None]:
    """Compute per-feature α and φ values from decoder PCA.

    Returns a dict with two keys: `alphas` and `phis`. Each value is
    either a list of length `n_features` or `None`.

    - `MPSRung1`, `Rung3`, `Rung4`: populates `alphas` from PC2 of the
      centered projections (rescaled `[0, 2π]`) and `phis` from PC3
      (rescaled `[0, 2π]`).
    - `HEA_Rung2`: returns `{"alphas": None, "phis": None}`. The
      per-feature θ tensor of shape `(rotations × depth × n_qubits)`
      has a different structure that this helper doesn't address.
      Use `HEA_Rung2State` for direct knob control. INFO-once log.

    For features with degenerate PCA (rank-deficient decoder geometry,
    fewer than 3 non-zero singular values), the affected knob arrays
    fall back to `None` (loader uses encoding default). Logged at DEBUG.

    The rescale into `[0, 2π]` is **linear**: `(coord / abs_max) →
    [-1, 1] → linearly into [0, 2π]`. A sinusoidal variant is documented
    in the openspec design.md Open Questions as a follow-up if early
    measurements show distribution pathology.
    """
    encoding_name = type(encoding).__name__

    # HEA_Rung2: per-feature θ tensor (rotations, depth, n_qubits) —
    # different knob structure. Out of scope; use HEA_Rung2State for
    # direct control.
    if encoding_name == "HEA_Rung2":
        _info_once(
            encoding_name,
            "assign_phase_knobs_pca: HEA_Rung2's per-feature θ tensor "
            "has a different shape (rotations × depth × n_qubits) and "
            "is not compatible with the phase-knob assignment pattern; "
            "use HEA_Rung2State directly via the encoding's knob "
            "interface. assign_phase_knobs=True is a no-op for this "
            "encoding.",
        )
        return {"alphas": None, "phis": None}

    # All other shipped encodings (MPSRung1, Rung3, Rung4) share the
    # MPS-substrate α and φ knobs and accept geometry-derived values.
    n_features = projections.shape[0]
    if n_features < 2:
        # PCA degenerate.
        return {"alphas": None, "phis": None}

    centered = projections - projections.mean(axis=0)
    # full_matrices=False → vt has shape (min(n, d), d_model); top
    # min(n, d) components are meaningful.
    _, sv, vt = np.linalg.svd(centered, full_matrices=False)

    if sv.size == 0 or sv[0] < 1e-12:
        return {"alphas": None, "phis": None}
    noise_floor = sv[0] * 1e-9
    n_available_axes = int((sv > noise_floor).sum())

    # PC1 (vt[0]) is consumed by β in the surrounding strategy.
    # PC2 (vt[1]) → α, PC3 (vt[2]) → φ. Both rescaled to [0, 2π].
    result: dict[str, list[float] | None] = {"alphas": None, "phis": None}

    knob_slots = [
        ("alphas", 1, 0.0, 2 * math.pi),  # PC2 → α
        ("phis",   2, 0.0, 2 * math.pi),  # PC3 → φ
    ]

    for key, axis_idx, lo, hi in knob_slots:
        if axis_idx >= n_available_axes:
            logger.debug(
                f"assign_phase_knobs_pca: {encoding_name} requested PC"
                f"{axis_idx + 1} (vt[{axis_idx}]) for {key}, but only "
                f"{n_available_axes} non-zero PCA components are "
                f"available; falling back to encoding default for "
                f"this knob"
            )
            continue
        pc = vt[axis_idx]                                              # (d_model,)
        coords = centered @ pc                                          # (n_features,)
        abs_max = float(np.max(np.abs(coords)))
        if abs_max < 1e-12:
            continue
        # Linear rescale: coord ∈ [-abs_max, abs_max] → [lo, hi].
        half = 0.5 * (hi - lo)
        mid = 0.5 * (hi + lo)
        scaled = (coords / abs_max) * half + mid
        result[key] = [float(v) for v in scaled]

    return result
