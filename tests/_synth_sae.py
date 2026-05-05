"""Shared synth-SAE helper for compression / validator tests.

Builds a tiny safetensors checkpoint with the four SAE weight tensors
(`W_enc`, `b_enc`, `W_dec`, `b_dec`). Deterministic per ``seed``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def synth_sae(
    path: Path,
    *,
    n_features: int = 16,
    d_model: int = 8,
    seed: int = 0,
) -> None:
    from safetensors.numpy import save_file

    rng = np.random.default_rng(seed)
    save_file(
        {
            "W_enc": rng.standard_normal((d_model, n_features)).astype(np.float32),
            "b_enc": np.zeros((n_features,), dtype=np.float32),
            "W_dec": rng.standard_normal((n_features, d_model)).astype(np.float32),
            "b_dec": np.zeros((d_model,), dtype=np.float32),
        },
        str(path),
    )
