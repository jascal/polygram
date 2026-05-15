"""Amp-branch knob assignment from decoder geometry (PCA-axis extension).

`from_sae_lens` consumes this when `assign_amp_knobs=True`. The helper
maps higher PCA axes of the projection vectors into the encoding's
amp-branch knob ranges, producing per-feature values that vary with
decoder geometry rather than collapsing to the encoding's MPS-equivalent
defaults.

This is the `encoding-aware-knob-assignment` v1 strategy. See
`openspec/changes/encoding-aware-knob-assignment/design.md` Decision 1
for the rationale (natural extension of the existing β strategy;
continuous, decoder-derived, deterministic, single-pass).

**Knob → PCA-component priority order**:

    PC1 → β               (existing β strategy)
    PC2 → α               (phase_assignment.py, MPS-substrate)
    PC3 → φ               (phase_assignment.py, MPS-substrate)
    PC4 → theta_amp       (this module, Rung3+)
    PC5 → psi_aux         (this module, Rung3+)
    PC6 → theta_amp_b     (this module, Rung4 only)
    PC7 → psi_amp_b       (this module, Rung4 only)

Amp knobs shifted to PC4-PC7 in `add-phase-knob-assignment` (was
PC2-PC5 in the v1 of this helper) so that phase knobs — universal
across MPS-substrate encodings — get the low slots.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# INFO-once flag so users notice when their `assign_amp_knobs=True`
# flag is a no-op for the chosen encoding. Per review on PR #62.
_INFO_LOGGED_NO_OP: set[str] = set()


def _info_once(encoding_name: str, message: str) -> None:
    if encoding_name not in _INFO_LOGGED_NO_OP:
        logger.info(message)
        _INFO_LOGGED_NO_OP.add(encoding_name)


def assign_amp_knobs_pca(
    projections: np.ndarray,
    encoding: object,
) -> dict[str, list[float] | None]:
    """Compute per-feature amp-branch knob values from decoder PCA.

    Returns a dict with four keys: `theta_amps`, `psi_auxes`,
    `theta_amp_bs`, `psi_amp_bs`. Each value is either a list of length
    `n_features` or `None`.

    - `MPSRung1`: no amp branch → all four values are `None`. INFO-once
      log message naming the encoding.
    - `HEA_Rung2`: different knob structure (per-feature θ tensor of
      shape `(|rotations|, depth, n_qubits)`) → all four values are
      `None`. INFO-once log message.
    - `Rung3`: populates `theta_amps` (axis-2 PCA coord, rescaled
      `[0, π/2]`) and `psi_auxes` (axis-3 PCA coord, rescaled `[0, 2π]`).
      `theta_amp_bs` and `psi_amp_bs` are `None`.
    - `Rung4`: populates all four arrays. `theta_amp_bs` from axis-4,
      `psi_amp_bs` from axis-5.

    For features with degenerate PCA (rank-deficient decoder geometry,
    fewer non-zero singular values than amp knobs requested), the
    affected knob arrays fall back to `None` (loader uses encoding
    default). Logged at DEBUG.

    The rescale into each knob's natural range is **linear**:
    `(coord / abs_max) → [−1, 1] → linearly into the target range`. A
    sinusoidal rescale variant is documented in
    `openspec/changes/encoding-aware-knob-assignment/design.md` Open
    Questions as a follow-up if early Axis-1 measurements show
    distribution pathology.
    """
    encoding_name = type(encoding).__name__

    # MPSRung1: no amp branch.
    if encoding_name == "MPSRung1":
        _info_once(
            encoding_name,
            "assign_amp_knobs_pca: MPSRung1 has no amp branch; "
            "assign_amp_knobs=True is a no-op for this encoding",
        )
        return {
            "theta_amps": None,
            "psi_auxes": None,
            "theta_amp_bs": None,
            "psi_amp_bs": None,
        }

    # HEA_Rung2: per-feature θ tensor of shape (rotations, depth,
    # n_qubits) — different structure entirely. Out of scope for v1.
    if encoding_name == "HEA_Rung2":
        _info_once(
            encoding_name,
            "assign_amp_knobs_pca: HEA_Rung2's per-feature θ tensor is "
            "out of scope for v1 amp-knob assignment; "
            "assign_amp_knobs=True is a no-op for this encoding",
        )
        return {
            "theta_amps": None,
            "psi_auxes": None,
            "theta_amp_bs": None,
            "psi_amp_bs": None,
        }

    # How many PCA axes do we need beyond axis-1 (already used for β)?
    #   Rung3: 2 amp knobs → need axes 2, 3
    #   Rung4: 4 amp knobs → need axes 2, 3, 4, 5
    if encoding_name == "Rung3":
        n_amp_axes = 2
    elif encoding_name == "Rung4":
        n_amp_axes = 4
    else:
        # Unknown encoding — return all-None so the loader uses
        # whatever defaults the encoding ships with.
        logger.debug(
            f"assign_amp_knobs_pca: unknown encoding {encoding_name!r}; "
            f"falling back to encoding defaults"
        )
        return {
            "theta_amps": None,
            "psi_auxes": None,
            "theta_amp_bs": None,
            "psi_amp_bs": None,
        }

    n_features = projections.shape[0]
    if n_features < 2:
        # PCA degenerate.
        return {
            "theta_amps": None,
            "psi_auxes": None,
            "theta_amp_bs": None,
            "psi_amp_bs": None,
        }

    # Compute PCA via SVD on centered projections. We need axes
    # 2..(1 + n_amp_axes), so request up to (1 + n_amp_axes) total
    # axes — axis-1 is consumed by β, axes 2..(1+n_amp_axes) by amp
    # knobs.
    centered = projections - projections.mean(axis=0)
    # full_matrices=False → vt has shape (min(n_features, d_model), d_model);
    # only the top min(n, d) axes are meaningful.
    _, sv, vt = np.linalg.svd(centered, full_matrices=False)

    # Available non-zero PCA axes — anything with singular value above
    # FP noise relative to the top axis. Below noise = degenerate /
    # rank-deficient direction.
    if sv.size == 0 or sv[0] < 1e-12:
        return {
            "theta_amps": None,
            "psi_auxes": None,
            "theta_amp_bs": None,
            "psi_amp_bs": None,
        }
    noise_floor = sv[0] * 1e-9
    n_available_axes = int((sv > noise_floor).sum())

    # Axis-1 is for β; amp knobs start at axis-2 (zero-indexed: 1).
    amp_arrays: dict[str, list[float] | None] = {
        "theta_amps": None,
        "psi_auxes": None,
        "theta_amp_bs": None,
        "psi_amp_bs": None,
    }

    # Knob → (axis_index, range_lo, range_hi).
    # axis_index is 0-indexed into vt's rows. The axis allocation
    # shifted in add-phase-knob-assignment: PC2-PC3 are now reserved
    # for α/φ (see polygram/geometry/phase_assignment.py); amp knobs
    # start from PC4 (vt[3]):
    #   theta_amp   ← PC4  (vt[3]) → rescaled to [0, π/2]
    #   psi_aux     ← PC5  (vt[4]) → rescaled to [0, 2π]
    #   theta_amp_b ← PC6  (vt[5]) → [0, π/2]   (Rung4 only)
    #   psi_amp_b   ← PC7  (vt[6]) → [0, 2π]    (Rung4 only)
    # (Pre-shift allocation was PC2-PC5 / vt[1]-vt[4].)
    knob_slots = [
        ("theta_amps",   3, 0.0, math.pi / 2),
        ("psi_auxes",    4, 0.0, 2 * math.pi),
        ("theta_amp_bs", 5, 0.0, math.pi / 2),
        ("psi_amp_bs",   6, 0.0, 2 * math.pi),
    ]

    for slot_idx, (key, axis_idx, lo, hi) in enumerate(knob_slots):
        if slot_idx >= n_amp_axes:
            # Encoding doesn't consume this knob (e.g., Rung3 doesn't
            # populate theta_amp_bs / psi_amp_bs).
            break
        if axis_idx >= n_available_axes:
            # Rank-deficient — this axis isn't available. Leave the
            # array as None so the loader falls back to encoding default.
            logger.debug(
                f"assign_amp_knobs_pca: {encoding_name} requested axis "
                f"{axis_idx + 1} (zero-indexed {axis_idx}) for {key}, but "
                f"only {n_available_axes} non-zero PCA axes are available; "
                f"falling back to encoding default for this knob"
            )
            continue
        pc = vt[axis_idx]                                              # (d_model,)
        coords = centered @ pc                                          # (n_features,)
        abs_max = float(np.max(np.abs(coords)))
        if abs_max < 1e-12:
            # All-zero coords on this axis — fall back to encoding
            # default.
            continue
        # Linear rescale: coord ∈ [-abs_max, abs_max] → [lo, hi].
        # Maps the axis-mean (coord=0) to the range midpoint.
        half = 0.5 * (hi - lo)
        mid = 0.5 * (hi + lo)
        scaled = (coords / abs_max) * half + mid
        amp_arrays[key] = [float(v) for v in scaled]

    return amp_arrays
