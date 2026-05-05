"""`BehaviouralValidator.__post_init__` — every rejection path."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from polygram import BehaviouralValidator, from_sae_lens, load_sae_safetensors


def _synth_sae(path: Path, *, n_features: int = 16, d_model: int = 8) -> None:
    from safetensors.numpy import save_file

    rng = np.random.default_rng(0)
    save_file(
        {
            "W_enc": rng.standard_normal((d_model, n_features)).astype(np.float32),
            "b_enc": np.zeros((n_features,), dtype=np.float32),
            "W_dec": rng.standard_normal((n_features, d_model)).astype(np.float32),
            "b_dec": np.zeros((d_model,), dtype=np.float32),
        },
        str(path),
    )


def _build_dict(tmp_path: Path, feature_ids: list[int]):
    sae_path = tmp_path / "sae.safetensors"
    _synth_sae(sae_path)
    records = load_sae_safetensors(sae_path, feature_ids=feature_ids)
    dictionary, _ = from_sae_lens(
        records, feature_ids, assign_gamma=True, name="PostInitDict"
    )
    return dictionary, sae_path


class TestPostInit:
    def test_feature_ids_length_mismatch_rejected(self, tmp_path: Path):
        dictionary, sae_path = _build_dict(tmp_path, [0, 1, 2, 3])
        with pytest.raises(ValueError, match="length .* does not match"):
            BehaviouralValidator(
                dictionary=dictionary,
                sae_checkpoint=sae_path,
                feature_ids=[0, 1, 2],  # mismatch
                prompts=["hi"],
                layer=5,
            )

    def test_polygram_threshold_out_of_range(self, tmp_path: Path):
        dictionary, sae_path = _build_dict(tmp_path, [0, 1])
        with pytest.raises(ValueError, match="polygram_overlap_threshold"):
            BehaviouralValidator(
                dictionary=dictionary,
                sae_checkpoint=sae_path,
                feature_ids=[0, 1],
                prompts=["x"],
                layer=5,
                polygram_overlap_threshold=1.5,
            )

    def test_jaccard_threshold_negative(self, tmp_path: Path):
        dictionary, sae_path = _build_dict(tmp_path, [0, 1])
        with pytest.raises(ValueError, match="jaccard_threshold"):
            BehaviouralValidator(
                dictionary=dictionary,
                sae_checkpoint=sae_path,
                feature_ids=[0, 1],
                prompts=["x"],
                layer=5,
                jaccard_threshold=-0.1,
            )

    def test_min_firing_rate_above_one(self, tmp_path: Path):
        dictionary, sae_path = _build_dict(tmp_path, [0, 1])
        with pytest.raises(ValueError, match="min_firing_rate"):
            BehaviouralValidator(
                dictionary=dictionary,
                sae_checkpoint=sae_path,
                feature_ids=[0, 1],
                prompts=["x"],
                layer=5,
                min_firing_rate=1.5,
            )

    def test_min_both_fire_zero_rejected(self, tmp_path: Path):
        dictionary, sae_path = _build_dict(tmp_path, [0, 1])
        with pytest.raises(ValueError, match="min_both_fire"):
            BehaviouralValidator(
                dictionary=dictionary,
                sae_checkpoint=sae_path,
                feature_ids=[0, 1],
                prompts=["x"],
                layer=5,
                min_both_fire=0,
            )

    def test_negative_layer_rejected_unconditionally(self, tmp_path: Path):
        dictionary, sae_path = _build_dict(tmp_path, [0, 1])
        with pytest.raises(ValueError, match="must be"):
            BehaviouralValidator(
                dictionary=dictionary,
                sae_checkpoint=sae_path,
                feature_ids=[0, 1],
                prompts=["x"],
                layer=-1,
                allow_layer_zero=True,  # does not gate negatives
            )

    def test_layer_zero_rejected_by_default(self, tmp_path: Path):
        dictionary, sae_path = _build_dict(tmp_path, [0, 1])
        with pytest.raises(ValueError, match="structural dead zone"):
            BehaviouralValidator(
                dictionary=dictionary,
                sae_checkpoint=sae_path,
                feature_ids=[0, 1],
                prompts=["x"],
                layer=0,
            )

    def test_layer_zero_message_names_research_note(self, tmp_path: Path):
        dictionary, sae_path = _build_dict(tmp_path, [0, 1])
        with pytest.raises(ValueError) as excinfo:
            BehaviouralValidator(
                dictionary=dictionary,
                sae_checkpoint=sae_path,
                feature_ids=[0, 1],
                prompts=["x"],
                layer=0,
            )
        msg = str(excinfo.value)
        assert "deeper-layer-ablation-probe.md" in msg
        assert "5e-5" in msg
        assert "blocks.5" in msg
        assert "allow_layer_zero=True" in msg

    def test_layer_zero_with_override_emits_warning(self, tmp_path: Path):
        dictionary, sae_path = _build_dict(tmp_path, [0, 1])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            BehaviouralValidator(
                dictionary=dictionary,
                sae_checkpoint=sae_path,
                feature_ids=[0, 1],
                prompts=["x"],
                layer=0,
                allow_layer_zero=True,
            )
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert runtime_warnings, "expected a RuntimeWarning"
        assert "structural dead zone" in str(runtime_warnings[0].message)

    def test_empty_prompts_rejected(self, tmp_path: Path):
        dictionary, sae_path = _build_dict(tmp_path, [0, 1])
        with pytest.raises(ValueError, match="prompts"):
            BehaviouralValidator(
                dictionary=dictionary,
                sae_checkpoint=sae_path,
                feature_ids=[0, 1],
                prompts=[],
                layer=5,
            )

    def test_missing_sae_checkpoint_rejected(self, tmp_path: Path):
        dictionary, _ = _build_dict(tmp_path, [0, 1])
        missing = tmp_path / "nope.safetensors"
        with pytest.raises(ValueError, match="not found on disk"):
            BehaviouralValidator(
                dictionary=dictionary,
                sae_checkpoint=missing,
                feature_ids=[0, 1],
                prompts=["x"],
                layer=5,
            )
