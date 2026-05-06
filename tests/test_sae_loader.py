"""Unit tests for the shared SAE checkpoint normaliser in sae_import.py."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from polygram.sae_import import (
    _KEY_ALIASES,
    _bf16_to_f32,
    _correct_orientation,
    _load_sae_checkpoint,
    _read_safetensors_header,
)


# ---------------------------------------------------------------------------
# Helpers — write synthetic safetensors without the safetensors library
# ---------------------------------------------------------------------------

def _f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    """Convert float32 array to raw BF16 bytes (upper 16 bits of each word)."""
    u32 = arr.astype(np.float32).view(np.uint32)
    return (u32 >> 16).astype(np.uint16).tobytes()


def _write_safetensors(
    path: Path,
    tensors: dict[str, tuple[np.ndarray, str]],  # name → (array, dtype_str)
) -> None:
    """Write a minimal safetensors file with the given tensors and dtype strings."""
    offset = 0
    header_entries: dict[str, dict] = {}
    parts: list[bytes] = []

    for name, (arr, dtype_str) in tensors.items():
        if dtype_str == "BF16":
            data = _f32_to_bf16_bytes(arr)
        elif dtype_str == "F32":
            data = arr.astype(np.float32).tobytes()
        else:
            raise ValueError(f"Unsupported dtype in helper: {dtype_str}")
        header_entries[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        parts.append(data)
        offset += len(data)

    header_json = json.dumps(header_entries).encode("utf-8")
    pad = (8 - len(header_json) % 8) % 8
    header_json += b" " * pad

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for data in parts:
            f.write(data)


def _write_f32_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    _write_safetensors(path, {k: (v, "F32") for k, v in tensors.items()})


def _write_bf16_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    _write_safetensors(path, {k: (v, "BF16") for k, v in tensors.items()})


# ---------------------------------------------------------------------------
# _KEY_ALIASES
# ---------------------------------------------------------------------------

class TestKeyAliases:
    def test_canonical_keys_map_to_themselves(self):
        for key in ("W_dec", "W_enc", "b_dec", "b_enc"):
            assert _KEY_ALIASES[key] == key

    def test_llamascope_decoder_maps_to_w_dec(self):
        assert _KEY_ALIASES["decoder.weight"] == "W_dec"

    def test_llamascope_encoder_maps_to_w_enc(self):
        assert _KEY_ALIASES["encoder.weight"] == "W_enc"

    def test_llamascope_biases_map_correctly(self):
        assert _KEY_ALIASES["decoder.bias"] == "b_dec"
        assert _KEY_ALIASES["encoder.bias"] == "b_enc"

    def test_dec_short_form_maps_to_w_dec(self):
        assert _KEY_ALIASES["dec"] == "W_dec"


# ---------------------------------------------------------------------------
# _read_safetensors_header
# ---------------------------------------------------------------------------

class TestReadSafetensorsHeader:
    def test_returns_tensor_metadata(self, tmp_path):
        arr = np.ones((2, 3), dtype=np.float32)
        p = tmp_path / "t.safetensors"
        _write_f32_safetensors(p, {"W_dec": arr})
        header = _read_safetensors_header(p)
        assert "W_dec" in header
        assert header["W_dec"]["dtype"] == "F32"
        assert header["W_dec"]["shape"] == [2, 3]

    def test_strips_metadata_key(self, tmp_path):
        arr = np.ones((2,), dtype=np.float32)
        p = tmp_path / "t.safetensors"
        _write_f32_safetensors(p, {"b_dec": arr})
        header = _read_safetensors_header(p)
        assert "__metadata__" not in header

    def test_detects_bf16_dtype(self, tmp_path):
        arr = np.ones((4,), dtype=np.float32)
        p = tmp_path / "t.safetensors"
        _write_bf16_safetensors(p, {"W_dec": arr})
        header = _read_safetensors_header(p)
        assert header["W_dec"]["dtype"] == "BF16"


# ---------------------------------------------------------------------------
# _bf16_to_f32
# ---------------------------------------------------------------------------

class TestBf16ToF32:
    def test_known_value_1_0(self):
        # BF16 for 1.0 is 0x3F80 → little-endian bytes [0x80, 0x3F]
        raw = bytes([0x80, 0x3F])
        result = _bf16_to_f32(raw, (1,))
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    def test_roundtrip_via_bit_shift(self):
        original = np.array([1.0, -2.5, 0.0, 3.14], dtype=np.float32)
        bf16_bytes = _f32_to_bf16_bytes(original)
        recovered = _bf16_to_f32(bf16_bytes, (4,))
        # BF16 has ~2 decimal digits of precision; max relative error ~0.4%
        np.testing.assert_allclose(recovered, original, rtol=5e-3)

    def test_returns_float32(self):
        raw = bytes([0x80, 0x3F])
        assert _bf16_to_f32(raw, (1,)).dtype == np.float32

    def test_correct_shape(self):
        arr = np.ones((3, 4), dtype=np.float32)
        raw = _f32_to_bf16_bytes(arr)
        result = _bf16_to_f32(raw, (3, 4))
        assert result.shape == (3, 4)


# ---------------------------------------------------------------------------
# _correct_orientation
# ---------------------------------------------------------------------------

class TestCorrectOrientation:
    def test_decoder_weight_non_square_is_transposed(self):
        arr = np.zeros((4, 16))  # (d_model, d_sae) PyTorch layout
        result = _correct_orientation(arr, "decoder.weight")
        assert result.shape == (16, 4)

    def test_decoder_weight_square_not_transposed(self):
        arr = np.zeros((8, 8))
        result = _correct_orientation(arr, "decoder.weight")
        assert result.shape == (8, 8)

    def test_encoder_weight_transposed_when_d_sae_gt_d_model(self):
        arr = np.zeros((16, 4))  # (d_sae, d_model) PyTorch layout
        result = _correct_orientation(arr, "encoder.weight")
        assert result.shape == (4, 16)

    def test_encoder_weight_not_transposed_when_d_model_gt_d_sae(self):
        # Under-complete SAE: d_sae < d_model — already canonical
        arr = np.zeros((4, 16))
        result = _correct_orientation(arr, "encoder.weight")
        assert result.shape == (4, 16)

    def test_canonical_w_dec_not_transposed(self):
        arr = np.zeros((16, 4))
        result = _correct_orientation(arr, "W_dec")
        assert result.shape == (16, 4)

    def test_bias_1d_not_transposed(self):
        arr = np.zeros((16,))
        result = _correct_orientation(arr, "decoder.bias")
        assert result.shape == (16,)


# ---------------------------------------------------------------------------
# _load_sae_checkpoint — alias resolution
# ---------------------------------------------------------------------------

class TestLoadSAECheckpointAliases:
    def test_canonical_keys_load_directly(self, tmp_path):
        W_dec = np.eye(4, dtype=np.float32)
        p = tmp_path / "sae.safetensors"
        _write_f32_safetensors(p, {"W_dec": W_dec})
        result = _load_sae_checkpoint(p, ["W_dec"])
        np.testing.assert_array_equal(result["W_dec"], W_dec)

    def test_llamascope_decoder_weight_resolves(self, tmp_path):
        arr = np.ones((4, 16), dtype=np.float32)  # (d_model, d_sae) PyTorch
        p = tmp_path / "sae.safetensors"
        _write_f32_safetensors(p, {"decoder.weight": arr})
        result = _load_sae_checkpoint(p, ["W_dec"])
        # orientation corrected: should be (16, 4)
        assert result["W_dec"].shape == (16, 4)

    def test_llamascope_all_four_keys_resolve(self, tmp_path):
        d_model, d_sae = 4, 16
        tensors = {
            "decoder.weight": np.zeros((d_model, d_sae), dtype=np.float32),
            "encoder.weight": np.zeros((d_sae, d_model), dtype=np.float32),
            "decoder.bias": np.zeros((d_model,), dtype=np.float32),
            "encoder.bias": np.zeros((d_sae,), dtype=np.float32),
        }
        p = tmp_path / "sae.safetensors"
        _write_f32_safetensors(p, tensors)
        result = _load_sae_checkpoint(p, ["W_dec", "W_enc", "b_dec", "b_enc"])
        assert set(result.keys()) == {"W_dec", "W_enc", "b_dec", "b_enc"}
        assert result["W_dec"].shape == (d_sae, d_model)
        assert result["W_enc"].shape == (d_model, d_sae)
        assert result["b_dec"].shape == (d_model,)
        assert result["b_enc"].shape == (d_sae,)

    def test_missing_key_raises_value_error(self, tmp_path):
        p = tmp_path / "sae.safetensors"
        _write_f32_safetensors(p, {"W_dec": np.zeros((4, 8), dtype=np.float32)})
        with pytest.raises(ValueError, match="W_enc"):
            _load_sae_checkpoint(p, ["W_dec", "W_enc"])

    def test_missing_key_error_lists_present_keys(self, tmp_path):
        p = tmp_path / "sae.safetensors"
        _write_f32_safetensors(p, {"W_dec": np.zeros((4, 8), dtype=np.float32)})
        with pytest.raises(ValueError, match="W_dec"):
            _load_sae_checkpoint(p, ["W_enc"])


# ---------------------------------------------------------------------------
# _load_sae_checkpoint — bfloat16 handling
# ---------------------------------------------------------------------------

class TestLoadSAECheckpointBF16:
    def test_bf16_tensor_returned_as_float32(self, tmp_path):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        p = tmp_path / "bf16.safetensors"
        _write_bf16_safetensors(p, {"W_dec": arr})
        result = _load_sae_checkpoint(p, ["W_dec"])
        assert result["W_dec"].dtype == np.float32

    def test_bf16_values_round_trip(self, tmp_path):
        arr = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32).reshape(2, 2)
        p = tmp_path / "bf16.safetensors"
        _write_bf16_safetensors(p, {"W_dec": arr})
        result = _load_sae_checkpoint(p, ["W_dec"])
        np.testing.assert_allclose(result["W_dec"], arr, rtol=5e-3)

    def test_f32_tensor_unchanged(self, tmp_path):
        arr = np.array([1.5, 2.5, -0.5], dtype=np.float32)
        p = tmp_path / "f32.safetensors"
        _write_f32_safetensors(p, {"b_dec": arr})
        result = _load_sae_checkpoint(p, ["b_dec"])
        np.testing.assert_array_equal(result["b_dec"], arr)

    def test_no_torch_required(self, tmp_path, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "torch", None)
        arr = np.ones((2, 4), dtype=np.float32)
        p = tmp_path / "bf16.safetensors"
        _write_bf16_safetensors(p, {"W_dec": arr})
        result = _load_sae_checkpoint(p, ["W_dec"])
        assert result["W_dec"].dtype == np.float32

    def test_mixed_bf16_and_f32_in_same_file(self, tmp_path):
        d_model, d_sae = 4, 8
        arr_dec = np.ones((d_sae, d_model), dtype=np.float32)
        arr_bias = np.zeros((d_model,), dtype=np.float32)
        p = tmp_path / "mixed.safetensors"
        _write_safetensors(p, {
            "W_dec": (arr_dec, "BF16"),
            "b_dec": (arr_bias, "F32"),
        })
        result = _load_sae_checkpoint(p, ["W_dec", "b_dec"])
        assert result["W_dec"].dtype == np.float32
        assert result["b_dec"].dtype == np.float32
        np.testing.assert_allclose(result["W_dec"], arr_dec, rtol=5e-3)


# ---------------------------------------------------------------------------
# _load_sae_full — task 3.3
# ---------------------------------------------------------------------------

class TestLoadSAEFull:
    def test_bf16_llamascope_checkpoint_accepted(self, tmp_path):
        from polygram.behavioural.validator import _load_sae_full

        d_model, d_sae = 4, 16
        tensors = {
            # LlamaScope orientation: decoder (d_model, d_sae), encoder (d_sae, d_model)
            "decoder.weight": (np.ones((d_model, d_sae), dtype=np.float32), "BF16"),
            "encoder.weight": (np.ones((d_sae, d_model), dtype=np.float32), "BF16"),
            "decoder.bias": (np.zeros((d_model,), dtype=np.float32), "BF16"),
            "encoder.bias": (np.zeros((d_sae,), dtype=np.float32), "BF16"),
        }
        p = tmp_path / "llama.safetensors"
        _write_safetensors(p, tensors)
        result = _load_sae_full(p)

        assert set(result.keys()) == {"W_enc", "b_enc", "W_dec", "b_dec"}
        assert result["W_dec"].dtype == np.float32
        assert result["W_enc"].dtype == np.float32
        # orientation corrected
        assert result["W_dec"].shape == (d_sae, d_model)
        assert result["W_enc"].shape == (d_model, d_sae)

    def test_canonical_f32_checkpoint_still_loads(self, tmp_path):
        from safetensors.numpy import save_file
        from polygram.behavioural.validator import _load_sae_full

        d_model, d_sae = 4, 8
        rng = np.random.default_rng(0)
        p = tmp_path / "canonical.safetensors"
        save_file({
            "W_dec": rng.standard_normal((d_sae, d_model)).astype(np.float32),
            "W_enc": rng.standard_normal((d_model, d_sae)).astype(np.float32),
            "b_dec": rng.standard_normal(d_model).astype(np.float32),
            "b_enc": rng.standard_normal(d_sae).astype(np.float32),
        }, str(p))
        result = _load_sae_full(p)
        assert set(result.keys()) == {"W_enc", "b_enc", "W_dec", "b_dec"}

    def test_missing_encoder_key_raises(self, tmp_path):
        from polygram.behavioural.validator import _load_sae_full

        d_model, d_sae = 4, 8
        p = tmp_path / "incomplete.safetensors"
        _write_f32_safetensors(p, {
            "W_dec": np.zeros((d_sae, d_model), dtype=np.float32),
            "b_dec": np.zeros((d_model,), dtype=np.float32),
        })
        with pytest.raises(ValueError, match="W_enc"):
            _load_sae_full(p)
