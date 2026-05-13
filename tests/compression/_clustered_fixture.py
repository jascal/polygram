"""Deterministic fixture for the `compression-consumes-clustered-dictionary`
differential regression test.

Builds a fixed synthetic SAE with engineered redundancies plus
deterministic firing-rates / residuals (a stand-in for the
torch + transformers pre-pass that `EpochCompressor.run` would
normally compute). Used both at reference-capture time and at
post-refactor differential-test time; the two runs MUST use
identical helpers so the byte-identical EpochResult invariant
holds across the refactor.

The fixture is torch-free: the synth pre-pass replaces
`_compute_firing_rates_and_residuals` via monkeypatch, so CI
doesn't need the `[behavioural]` extra to run the differential
test. This sidesteps torch-version-induced drift in the reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from safetensors.numpy import save_file


N_FEATURES = 32
D_MODEL = 64  # Small d_model keeps the residuals array tiny; the
              # SAE encoder is still well-defined.
N_TOKENS = 64
SEED = 0


def build_synth_sae(sae_path: Path) -> Path:
    """Write a deterministic synthetic SAE to `sae_path`.

    Layout: 32 features × 64 d_model. Features 4-7 share a decoder
    direction (engineered redundancy cluster) so the compression
    pipeline has something to find.
    """
    rng = np.random.default_rng(SEED)
    base = rng.standard_normal(D_MODEL).astype(np.float32)
    state = {
        "W_enc": rng.standard_normal((D_MODEL, N_FEATURES)).astype(np.float32) * 0.1,
        "b_enc": rng.standard_normal(N_FEATURES).astype(np.float32) * 0.1,
        "W_dec": rng.standard_normal((N_FEATURES, D_MODEL)).astype(np.float32) * 0.1,
        "b_dec": np.zeros(D_MODEL, dtype=np.float32),
    }
    for fid in (4, 5, 6, 7):
        state["W_dec"][fid, :] = (
            base + rng.standard_normal(D_MODEL).astype(np.float32) * 0.02
        )
        state["W_enc"][:, fid] = state["W_dec"][fid, :]
        state["b_enc"][fid] = 0.5
    sae_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(sae_path))
    return sae_path


def synth_firing_rates_and_residuals() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic stand-in for `_compute_firing_rates_and_residuals`.

    Returns `(firing_rates: (N_FEATURES,), residuals: (N_TOKENS, D_MODEL))`,
    both reproducibly seeded. Firing rates are biased away from zero
    so the eligibility filter doesn't drop every feature.
    """
    rng = np.random.default_rng(SEED + 1)
    # Firing rates ∈ [0.1, 0.6] — wide enough to produce variation
    # in the priority signal (firing_rate × decoder_norm).
    firing_rates = (rng.uniform(0.0, 1.0, size=N_FEATURES) * 0.5 + 0.1).astype(
        np.float64
    )
    residuals = rng.standard_normal((N_TOKENS, D_MODEL)).astype(np.float32)
    return firing_rates, residuals


def make_synth_prepass_patch() -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Return a function with the same signature as
    `_compute_firing_rates_and_residuals` that returns the
    deterministic synth output regardless of its arguments. Used
    via monkeypatch in the differential test.
    """

    def _patched(
        sae_checkpoint,
        prompts,
        *,
        model_name,
        layer,
        device,
    ) -> tuple[np.ndarray, np.ndarray]:
        return synth_firing_rates_and_residuals()

    return _patched


EPOCH_KWARGS = dict(
    layer=10,
    device="cpu",
    n_panels_max=4,
    max_iterations=2,
    min_firing_rate=0.0,
    coverage_target=1.0,
    cosine_threshold=0.50,
    polygram_overlap_threshold=0.0,
    jaccard_threshold=0.0,
    min_both_fire=0,
)
"""Canonical EpochCompressor kwargs for the differential test. Match
the convergence-test pattern, with the gate thresholds dropped to 0
so the synth residuals reliably produce confirmed pairs."""


CANONICAL_PROMPTS = [
    "hello world this is a test prompt",
    "a second prompt for variation here",
]
