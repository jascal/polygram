"""`BehaviouralValidator.predict()` — unit tests for the cheap stage.

Builds a tiny synthesized SAE checkpoint (W_enc / b_enc / W_dec /
b_dec) via `safetensors.numpy.save_file` and a Dictionary via
`from_sae_lens`, then asserts predict() returns N(N-1)/2 pairs with
in-range overlaps and NaN behavioural fields. No torch needed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from polygram import BehaviouralValidator, from_sae_lens, load_sae_safetensors


def _synth_sae(
    path: Path, *, n_features: int = 16, d_model: int = 8, seed: int = 0
) -> None:
    from safetensors.numpy import save_file

    rng = np.random.default_rng(seed)
    w_dec = rng.standard_normal((n_features, d_model)).astype(np.float32)
    w_enc = rng.standard_normal((d_model, n_features)).astype(np.float32)
    b_enc = np.zeros((n_features,), dtype=np.float32)
    b_dec = np.zeros((d_model,), dtype=np.float32)
    save_file(
        {"W_enc": w_enc, "b_enc": b_enc, "W_dec": w_dec, "b_dec": b_dec},
        str(path),
    )


def _build_validator(
    tmp_path: Path,
    *,
    feature_ids: list[int],
    layer: int = 5,
    n_features: int = 16,
    d_model: int = 8,
) -> BehaviouralValidator:
    sae_path = tmp_path / "sae.safetensors"
    _synth_sae(sae_path, n_features=n_features, d_model=d_model)
    records = load_sae_safetensors(sae_path, feature_ids=feature_ids)
    dictionary, _ = from_sae_lens(
        records, feature_ids, assign_gamma=True, name="TestDict"
    )
    return BehaviouralValidator(
        dictionary=dictionary,
        sae_checkpoint=sae_path,
        feature_ids=list(feature_ids),
        prompts=["hello world"],
        layer=layer,
    )


class TestPredict:
    def test_pair_count_is_n_choose_2(self, tmp_path: Path):
        v = _build_validator(tmp_path, feature_ids=[0, 1, 2, 3])
        pairs = v.predict()
        assert len(pairs) == 6  # 4 choose 2

    def test_eight_features_returns_28_pairs(self, tmp_path: Path):
        v = _build_validator(tmp_path, feature_ids=[0, 1, 2, 3, 4, 5, 6, 7])
        pairs = v.predict()
        assert len(pairs) == 28

    def test_overlaps_in_unit_interval(self, tmp_path: Path):
        v = _build_validator(tmp_path, feature_ids=[0, 1, 2, 3])
        for pair in v.predict():
            assert 0.0 <= pair.polygram_overlap <= 1.0 + 1e-6
            assert 0.0 <= pair.decoder_overlap <= 1.0 + 1e-6

    def test_behavioural_fields_are_nan(self, tmp_path: Path):
        v = _build_validator(tmp_path, feature_ids=[0, 1, 2, 3])
        for pair in v.predict():
            assert math.isnan(pair.jaccard)
            assert math.isnan(pair.pearson_activation)
            assert math.isnan(pair.kl_ablate_i)
            assert math.isnan(pair.kl_ablate_j)
            assert math.isnan(pair.kl_ratio_paired)
            assert math.isnan(pair.kl_log_ratio_abs)
            assert pair.n_fires_i == 0
            assert pair.n_fires_j == 0
            assert pair.n_both_fire == 0
            assert pair.n_either_fire == 0
            assert pair.gate_pass is False

    def test_pairs_are_sorted_by_i_then_j(self, tmp_path: Path):
        v = _build_validator(tmp_path, feature_ids=[0, 1, 2, 3])
        pairs = v.predict()
        keys = [(p.i, p.j) for p in pairs]
        assert keys == sorted(keys)

    def test_pair_i_lt_j(self, tmp_path: Path):
        v = _build_validator(tmp_path, feature_ids=[0, 1, 2, 3])
        for pair in v.predict():
            assert pair.i < pair.j

    def test_predict_does_not_import_torch(self, tmp_path: Path):
        # If torch is installed in the dev env, this just confirms
        # predict() runs without forcing the import path.
        import sys

        v = _build_validator(tmp_path, feature_ids=[0, 1, 2, 3])
        before = "torch" in sys.modules
        v.predict()
        after = "torch" in sys.modules
        # Whatever the start state was, predict() must not change it.
        assert before == after

    def test_decoder_overlap_uses_actual_W_dec_rows(self, tmp_path: Path):
        sae_path = tmp_path / "sae.safetensors"
        _synth_sae(sae_path, n_features=16, d_model=8)
        records = load_sae_safetensors(sae_path, feature_ids=[0, 1])
        dictionary, _ = from_sae_lens(
            records, [0, 1], assign_gamma=True, name="TwoDict"
        )
        v = BehaviouralValidator(
            dictionary=dictionary,
            sae_checkpoint=sae_path,
            feature_ids=[0, 1],
            prompts=["x"],
            layer=5,
        )
        pairs = v.predict()
        assert len(pairs) == 1
        wi = records[0].projection
        wj = records[1].projection
        denom = float(np.dot(wi, wi)) * float(np.dot(wj, wj))
        expected = float(np.dot(wi, wj)) ** 2 / denom
        assert pairs[0].decoder_overlap == pytest.approx(expected, abs=1e-6)
