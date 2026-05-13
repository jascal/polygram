"""Deterministic fixture for `epoch-compressor-configurable-encoding`
smoke testing with `encoding=Rung3()` (`max_features=16`).

Engineered so that the priority-driven greedy seeded-coverage algorithm
in `_select_panels` produces at least one panel with more than 8
features — proving the new `max_panel_size - 1` neighbour cap actually
engages. Constructed parallel to `_clustered_fixture.py`: torch-free
synth pre-pass, deterministic seeds, monkeypatch-able.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from safetensors.numpy import save_file


N_FEATURES = 32
D_MODEL = 64
N_TOKENS = 64
SEED = 0


def build_rung3_synth_sae(sae_path: Path) -> Path:
    """Write a deterministic synthetic SAE with two engineered
    redundancy clusters of 10 features each.

    Layout: 32 features × 64 d_model. Features 0-9 share decoder
    direction `base_a`; features 10-19 share decoder direction
    `base_b`. Both clusters are larger than `MPSRung1.max_features=8`
    so a Rung3 (`max_features=16`) compression run must produce
    panels of >8 features to fully cover either cluster.
    """
    rng = np.random.default_rng(SEED)
    base_a = rng.standard_normal(D_MODEL).astype(np.float32)
    base_b = rng.standard_normal(D_MODEL).astype(np.float32)
    state = {
        "W_enc": rng.standard_normal((D_MODEL, N_FEATURES)).astype(np.float32) * 0.1,
        "b_enc": rng.standard_normal(N_FEATURES).astype(np.float32) * 0.1,
        "W_dec": rng.standard_normal((N_FEATURES, D_MODEL)).astype(np.float32) * 0.1,
        "b_dec": np.zeros(D_MODEL, dtype=np.float32),
    }
    for fid in range(10):
        state["W_dec"][fid, :] = (
            base_a + rng.standard_normal(D_MODEL).astype(np.float32) * 0.02
        )
        state["W_enc"][:, fid] = state["W_dec"][fid, :]
        state["b_enc"][fid] = 0.5
    for fid in range(10, 20):
        state["W_dec"][fid, :] = (
            base_b + rng.standard_normal(D_MODEL).astype(np.float32) * 0.02
        )
        state["W_enc"][:, fid] = state["W_dec"][fid, :]
        state["b_enc"][fid] = 0.5
    sae_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(sae_path))
    return sae_path


def rung3_synth_firing_rates_and_residuals() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED + 1)
    firing_rates = (rng.uniform(0.0, 1.0, size=N_FEATURES) * 0.5 + 0.1).astype(
        np.float64
    )
    residuals = rng.standard_normal((N_TOKENS, D_MODEL)).astype(np.float32)
    return firing_rates, residuals


def make_rung3_synth_prepass_patch() -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    def _patched(
        sae_checkpoint,
        prompts,
        *,
        model_name,
        layer,
        device,
    ) -> tuple[np.ndarray, np.ndarray]:
        return rung3_synth_firing_rates_and_residuals()

    return _patched


CANONICAL_PROMPTS = [
    "hello world this is a test prompt",
    "a second prompt for variation here",
]


EPOCH_KWARGS = dict(
    layer=10,
    device="cpu",
    n_panels_max=4,
    max_iterations=1,
    min_firing_rate=0.0,
    coverage_target=1.0,
    cosine_threshold=0.50,
    polygram_overlap_threshold=0.0,
    jaccard_threshold=0.0,
    min_both_fire=0,
    n_visits_per_feature=1,
)

