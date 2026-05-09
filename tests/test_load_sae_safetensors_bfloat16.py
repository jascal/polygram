"""Regression test for the bf16 slice-path bug.

Llama-Scope L0R surfaced this: `load_sae_safetensors(feature_ids=...)`
on a bf16 file used to crash with `TypeError: data type 'bfloat16' not
understood`. Modern LLM SAEs ship bf16 by default — this test ensures
the dedicated bf16 slice path stays correct.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from polygram import load_sae_safetensors


def _f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    """Round-to-zero conversion: take upper 16 bits of each f32 word."""
    u32 = arr.astype(np.float32).view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16.tobytes()


def _write_bf16_safetensors(
    path: Path, tensors: dict[str, np.ndarray]
) -> None:
    """Write a minimal bf16 safetensors file. Each tensor's f32 values
    are bit-shifted into bf16 before the file is laid out."""
    body = b""
    metadata: dict[str, dict] = {}
    offset = 0
    for name, arr in tensors.items():
        raw = _f32_to_bf16_bytes(arr)
        size = len(raw)
        metadata[name] = {
            "dtype": "BF16",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + size],
        }
        body += raw
        offset += size
    header_json = json.dumps(metadata).encode("utf-8")
    # Pad header to 8-byte alignment for safetensors compatibility.
    pad = (-len(header_json)) % 8
    header_json += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(body)


def test_bf16_slice_returns_float32_records(tmp_path):
    rng = np.random.default_rng(0)
    n_features, d_model = 10, 16
    W_dec = rng.standard_normal((n_features, d_model)).astype(np.float32)
    out = tmp_path / "tiny_bf16.safetensors"
    _write_bf16_safetensors(out, {"W_dec": W_dec})

    recs = load_sae_safetensors(str(out), feature_ids=[0, 5, 9])
    assert set(recs.keys()) == {0, 5, 9}
    for fid, r in recs.items():
        assert r.projection.shape == (d_model,)
        # All-finite, non-NaN.
        assert np.all(np.isfinite(r.projection))


def test_bf16_slice_matches_eager_load_within_quantization(tmp_path):
    """Sliced and eager paths must agree to bf16 precision."""
    rng = np.random.default_rng(1)
    n_features, d_model = 8, 12
    W_dec = rng.standard_normal((n_features, d_model)).astype(np.float32)
    out = tmp_path / "tiny_bf16.safetensors"
    _write_bf16_safetensors(out, {"W_dec": W_dec})

    sliced = load_sae_safetensors(str(out), feature_ids=[0, 1, 4])
    full = load_sae_safetensors(str(out))
    for fid in [0, 1, 4]:
        np.testing.assert_array_equal(
            sliced[fid].projection, full[fid].projection,
        )


def test_bf16_slice_handles_decoder_weight_pytorch_orientation(tmp_path):
    """For non-square `decoder.weight` (PyTorch out × in), the slice
    path must transpose: rows are d_model, columns are features."""
    rng = np.random.default_rng(2)
    n_features, d_model = 10, 16
    # PyTorch convention: (d_model, n_features)
    weight = rng.standard_normal((d_model, n_features)).astype(np.float32)
    out = tmp_path / "tiny_bf16.safetensors"
    _write_bf16_safetensors(out, {"decoder.weight": weight})

    recs = load_sae_safetensors(str(out), feature_ids=[0, 7])
    assert recs[0].projection.shape == (d_model,)
    assert recs[7].projection.shape == (d_model,)
    # Compare against the eager full-load path.
    full = load_sae_safetensors(str(out))
    np.testing.assert_array_equal(recs[0].projection, full[0].projection)
    np.testing.assert_array_equal(recs[7].projection, full[7].projection)
