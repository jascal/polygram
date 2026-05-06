"""Integration tests — confirmation strategies → Compressor end-to-end."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from polygram import (
    ClusterConfirmer,
    Compressor,
    DecoderGeometryConfirmer,
    SAEFeatureRecord,
    from_sae_lens,
    load_sae_safetensors,
    load_toy_sae,
)

FIXTURE = Path(__file__).parent / "fixtures" / "toy_sae.json"


def _write_full_sae(path: Path, records: dict[int, SAEFeatureRecord]) -> None:
    """Write a safetensors file with all four keys the compressor requires."""
    from safetensors.numpy import save_file

    n = len(records)
    d = len(next(iter(records.values())).projection)
    rng = np.random.default_rng(0)
    W_dec = np.stack(
        [records[i].projection.astype(np.float32) for i in range(n)]
    )
    save_file(
        {
            "W_dec": W_dec,
            "W_enc": rng.standard_normal((d, n)).astype(np.float32),
            "b_dec": rng.standard_normal(d).astype(np.float32),
            "b_enc": rng.standard_normal(n).astype(np.float32),
        },
        str(path),
    )


# ---------------------------------------------------------------------------
# 6.1  DecoderGeometryConfirmer → Compressor
# ---------------------------------------------------------------------------


def test_decoder_geometry_confirmer_to_compressor(tmp_path: Path) -> None:
    """Full path: synthetic full-SAE safetensors → DecoderGeometryConfirmer
    → Compressor → verify zeroed rows."""
    # Build two near-identical features and two orthogonal ones.
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v_twin = np.array([0.999, 0.045, 0.0, 0.0], dtype=np.float32)
    w = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    x = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    records = {
        i: SAEFeatureRecord(
            feature_id=i, name=f"feat_{i}",
            projection=arr.astype(np.float64),
        )
        for i, arr in enumerate([v, v_twin, w, x])
    }

    sae_path = tmp_path / "sae.safetensors"
    _write_full_sae(sae_path, records)

    confirmer = DecoderGeometryConfirmer(
        records=records,
        sae_checkpoint=sae_path,
        feature_ids=[0, 1, 2, 3],
        threshold=0.9,
    )
    val_report = confirmer.run()

    # feat_0 and feat_1 should be confirmed (cosine² ≈ 0.998)
    assert (0, 1) in val_report.confirmed
    # feat_0 and feat_2 should NOT (orthogonal)
    assert (0, 2) not in val_report.confirmed

    out_path = tmp_path / "sae_compressed.safetensors"
    compressor = Compressor(
        validation_report=val_report,
        sae_checkpoint=sae_path,
    )
    result = compressor.run(output_checkpoint=out_path)

    assert result.report.n_features_zeroed == 1
    assert result.report.n_features_kept == 1

    from safetensors.numpy import load_file
    ckpt = load_file(str(out_path))
    zeroed_fid = [fid for c in compressor.plan().clusters for fid in c.zeroed][0]
    assert np.linalg.norm(ckpt["W_dec"][zeroed_fid]) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 6.2  ClusterConfirmer → Compressor (toy SAE fixture)
# ---------------------------------------------------------------------------


def test_cluster_confirmer_to_compressor(tmp_path: Path) -> None:
    """Full path: toy SAE → from_sae_lens (k=2) → ClusterConfirmer
    → Compressor → verify zeroed rows."""
    records = load_toy_sae(FIXTURE)

    sae_path = tmp_path / "toy_full.safetensors"
    _write_full_sae(sae_path, records)

    feature_ids = [0, 1, 4, 5]
    _, sr = from_sae_lens(records, feature_ids, n_clusters=2)

    confirmer = ClusterConfirmer(
        selection_report=sr,
        sae_checkpoint=sae_path,
        records=records,
    )
    val_report = confirmer.run()

    # Toy fixture has two tight clusters; at least one within-cluster pair confirmed
    assert len(val_report.confirmed) >= 1

    out_path = tmp_path / "toy_compressed.safetensors"
    compressor = Compressor(
        validation_report=val_report,
        sae_checkpoint=sae_path,
    )
    result = compressor.run(output_checkpoint=out_path)

    assert result.report.n_features_zeroed >= 1

    from safetensors.numpy import load_file
    ckpt = load_file(str(out_path))
    for cluster in compressor.plan().clusters:
        for zeroed_fid in cluster.zeroed:
            assert np.linalg.norm(ckpt["W_dec"][zeroed_fid]) == pytest.approx(
                0.0, abs=1e-9
            )


# ---------------------------------------------------------------------------
# 4.1  LlamaScope-format (bf16, aliased keys, PyTorch orientation) end-to-end
# ---------------------------------------------------------------------------

def _f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    return (arr.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()


def _write_llamascope_safetensors(
    path: Path,
    W_dec_rows: np.ndarray,  # (n_features, d_model) — polygram canonical
) -> None:
    """Write a BF16 safetensors with LlamaScope key names and PyTorch orientation."""
    n, d_model = W_dec_rows.shape
    rng = np.random.default_rng(7)
    # PyTorch orientation: decoder (d_model, n), encoder (n, d_model)
    dec_weight = W_dec_rows.T.astype(np.float32)   # (d_model, n)
    enc_weight = rng.standard_normal((n, d_model)).astype(np.float32)  # (n, d_model)
    dec_bias = np.zeros(d_model, dtype=np.float32)
    enc_bias = np.zeros(n, dtype=np.float32)

    tensors: dict[str, tuple[np.ndarray, str]] = {
        "decoder.weight": (dec_weight, "BF16"),
        "encoder.weight": (enc_weight, "BF16"),
        "decoder.bias": (dec_bias, "BF16"),
        "encoder.bias": (enc_bias, "BF16"),
    }

    offset = 0
    header_entries: dict = {}
    parts: list[bytes] = []
    for name, (arr, dtype_str) in tensors.items():
        data = _f32_to_bf16_bytes(arr)
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


def test_llamascope_format_end_to_end(tmp_path: Path) -> None:
    """LlamaScope-style checkpoint (BF16, aliased keys, PyTorch orientation)
    flows through load_sae_safetensors → from_sae_lens → ClusterConfirmer
    → Compressor without any pre-conversion script."""
    # Build synthetic feature vectors: two tight pairs + two orthogonal singletons
    rng = np.random.default_rng(99)
    n, d_model = 8, 16
    W_dec = rng.standard_normal((n, d_model)).astype(np.float32)
    # Make features 0 and 1 nearly identical
    W_dec[1] = W_dec[0] + 0.01 * rng.standard_normal(d_model).astype(np.float32)
    # Normalise rows
    W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True)

    ckpt_path = tmp_path / "llamascope.safetensors"
    _write_llamascope_safetensors(ckpt_path, W_dec)

    # Step 1: load without pre-conversion
    records = load_sae_safetensors(ckpt_path)
    assert len(records) == n

    # Step 2: cluster and confirm
    feature_ids = list(range(n))
    _, sr = from_sae_lens(records, feature_ids, n_clusters=4)
    confirmer = ClusterConfirmer(
        selection_report=sr,
        sae_checkpoint=ckpt_path,
        records=records,
    )
    val_report = confirmer.run()
    assert len(val_report.confirmed) >= 1

    # Step 3: compress
    out_path = tmp_path / "compressed.safetensors"
    compressor = Compressor(
        validation_report=val_report,
        sae_checkpoint=ckpt_path,
    )
    result = compressor.run(output_checkpoint=out_path)
    assert result.report.n_features_zeroed >= 1

    from safetensors.numpy import load_file
    ckpt = load_file(str(out_path))
    for cluster in compressor.plan().clusters:
        for zeroed_fid in cluster.zeroed:
            assert np.linalg.norm(ckpt["W_dec"][zeroed_fid]) == pytest.approx(
                0.0, abs=1e-9
            )
