"""Coverage for `polygram.load_sae_safetensors`.

Fixtures are synthesized at test time via `safetensors.numpy.save_file`
to keep the repo free of binary blobs.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from polygram import (
    Dictionary,
    SAEFeatureRecord,
    from_sae_lens,
    load_sae_safetensors,
)


def _synth_safetensors(
    path: Path,
    *,
    key: str,
    shape: tuple[int, int],
    seed: int = 0,
    extra: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Write a `.safetensors` file with one named 2D tensor (plus
    optional extra tensors) and return the canonical tensor."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(shape).astype(np.float32)
    payload: dict[str, np.ndarray] = {key: arr}
    if extra:
        payload.update(extra)
    save_file(payload, str(path))
    return arr


# ---------------------------------------------------------------------------
# Key precedence
# ---------------------------------------------------------------------------


class TestKeyPrecedence:
    def test_w_dec_wins_over_dec_and_decoder_weight(self, tmp_path: Path):
        rng = np.random.default_rng(0)
        target = rng.standard_normal((4, 8)).astype(np.float32)
        decoy_decoder = rng.standard_normal((3, 8)).astype(np.float32)
        decoy_dec = rng.standard_normal((2, 8)).astype(np.float32)
        from safetensors.numpy import save_file

        save_file(
            {
                "W_dec": target,
                "decoder.weight": decoy_decoder,
                "dec": decoy_dec,
            },
            str(tmp_path / "sae.safetensors"),
        )
        records = load_sae_safetensors(tmp_path / "sae.safetensors")
        assert len(records) == 4
        for i in range(4):
            np.testing.assert_array_almost_equal(
                records[i].projection, target[i, :]
            )
            assert records[i].name == f"feat_{i}"

    def test_decoder_weight_used_when_w_dec_absent(self, tmp_path: Path):
        # Square matrix → no transpose; rows are features.
        target = _synth_safetensors(
            tmp_path / "sae.safetensors",
            key="decoder.weight",
            shape=(4, 4),
        )
        records = load_sae_safetensors(tmp_path / "sae.safetensors")
        assert len(records) == 4
        for i in range(4):
            np.testing.assert_array_almost_equal(
                records[i].projection, target[i, :]
            )

    def test_dec_is_terse_fallback(self, tmp_path: Path):
        target = _synth_safetensors(
            tmp_path / "sae.safetensors", key="dec", shape=(3, 4),
        )
        records = load_sae_safetensors(tmp_path / "sae.safetensors")
        assert len(records) == 3
        for i in range(3):
            np.testing.assert_array_almost_equal(
                records[i].projection, target[i, :]
            )

    def test_missing_decoder_key_lists_present_keys(self, tmp_path: Path):
        from safetensors.numpy import save_file

        save_file(
            {
                "enc": np.zeros((2, 2), dtype=np.float32),
                "b_enc": np.zeros((2,), dtype=np.float32),
                "b_dec": np.zeros((2,), dtype=np.float32),
            },
            str(tmp_path / "sae.safetensors"),
        )
        with pytest.raises(ValueError) as exc_info:
            load_sae_safetensors(tmp_path / "sae.safetensors")
        msg = str(exc_info.value)
        assert "W_dec" in msg
        assert "decoder.weight" in msg
        assert "dec" in msg
        assert "enc" in msg
        assert "b_enc" in msg


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


class TestOrientation:
    def test_decoder_weight_non_square_transposed(self, tmp_path: Path):
        # PyTorch nn.Linear stores (out=d_model, in=d_sae); for an SAE
        # decoder that's (8, 4). After transpose features should land
        # on rows so n_features == 4 and projection.shape == (8,).
        rng = np.random.default_rng(7)
        weight = rng.standard_normal((8, 4)).astype(np.float32)
        from safetensors.numpy import save_file

        save_file(
            {"decoder.weight": weight},
            str(tmp_path / "sae.safetensors"),
        )
        records = load_sae_safetensors(tmp_path / "sae.safetensors")
        assert len(records) == 4
        for i in range(4):
            assert records[i].projection.shape == (8,)
            np.testing.assert_array_almost_equal(
                records[i].projection, weight[:, i]
            )

    def test_w_dec_is_never_transposed(self, tmp_path: Path):
        # Rectangular W_dec: (n=4, d=8). Rows ARE features.
        target = _synth_safetensors(
            tmp_path / "sae.safetensors",
            key="W_dec",
            shape=(4, 8),
        )
        records = load_sae_safetensors(tmp_path / "sae.safetensors")
        assert records[0].projection.shape == (8,)
        np.testing.assert_array_almost_equal(
            records[0].projection, target[0, :]
        )

    def test_non_2d_decoder_rejected(self, tmp_path: Path):
        from safetensors.numpy import save_file

        save_file(
            {"W_dec": np.zeros((2, 3, 4), dtype=np.float32)},
            str(tmp_path / "sae.safetensors"),
        )
        with pytest.raises(ValueError, match="W_dec"):
            load_sae_safetensors(tmp_path / "sae.safetensors")


# ---------------------------------------------------------------------------
# Names override
# ---------------------------------------------------------------------------


class TestNamesOverride:
    def test_partial_override_keeps_defaults(self, tmp_path: Path):
        _synth_safetensors(
            tmp_path / "sae.safetensors", key="W_dec", shape=(4, 8),
        )
        records = load_sae_safetensors(
            tmp_path / "sae.safetensors",
            names={0: "dog_poodle", 2: "bird_hawk"},
        )
        assert records[0].name == "dog_poodle"
        assert records[1].name == "feat_1"
        assert records[2].name == "bird_hawk"
        assert records[3].name == "feat_3"

    def test_out_of_range_key_rejected(self, tmp_path: Path):
        _synth_safetensors(
            tmp_path / "sae.safetensors", key="W_dec", shape=(4, 8),
        )
        with pytest.raises(ValueError, match=r"\[0, 4\)"):
            load_sae_safetensors(
                tmp_path / "sae.safetensors", names={5: "ghost"},
            )


# ---------------------------------------------------------------------------
# Round-trip via from_sae_lens
# ---------------------------------------------------------------------------


class TestRoundTripWithFromSaeLens:
    def test_dictionary_built_from_loaded_records(self, tmp_path: Path):
        target = _synth_safetensors(
            tmp_path / "sae.safetensors", key="W_dec", shape=(8, 16),
        )
        records = load_sae_safetensors(tmp_path / "sae.safetensors")
        # Standard SAE flow: pick a small subset by id.
        dictionary, report = from_sae_lens(records, [0, 1, 4, 5])
        assert isinstance(dictionary, Dictionary)
        assert len(dictionary.features) == 4
        assert {f.name for f in dictionary.features} == {
            "feat_0", "feat_1", "feat_4", "feat_5",
        }
        # report exposes the fidelity stats; sanity-check shape.
        assert report.n_input_features == 8
        assert report.n_selected == 4
        # Records survive intact.
        for fid in (0, 1, 4, 5):
            np.testing.assert_array_almost_equal(
                records[fid].projection, target[fid, :]
            )


# ---------------------------------------------------------------------------
# Missing extra
# ---------------------------------------------------------------------------


def test_missing_safetensors_install(monkeypatch, tmp_path: Path):
    """If `safetensors.numpy` is unavailable, the loader's ImportError
    SHALL point at the [sae] extra."""

    real_modules = {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name == "safetensors" or name.startswith("safetensors.")
    }
    for name in real_modules:
        monkeypatch.setitem(sys.modules, name, None)

    with pytest.raises(ImportError, match=r"polygram\[sae\]"):
        load_sae_safetensors(tmp_path / "anything.safetensors")


# ---------------------------------------------------------------------------
# Returned record contract
# ---------------------------------------------------------------------------


def test_returned_records_are_sae_feature_records(tmp_path: Path):
    _synth_safetensors(
        tmp_path / "sae.safetensors", key="W_dec", shape=(2, 4),
    )
    records = load_sae_safetensors(tmp_path / "sae.safetensors")
    for fid, rec in records.items():
        assert isinstance(rec, SAEFeatureRecord)
        assert rec.feature_id == fid
        assert rec.label is None
        assert rec.activation_mean is None
        assert rec.activation_std is None
        assert rec.projection.dtype == np.float64
        assert rec.projection.ndim == 1


# ---------------------------------------------------------------------------
# Lazy-slice mode (`feature_ids=...`)
# ---------------------------------------------------------------------------


class TestLazySlice:
    def test_w_dec_lazy_matches_eager(self, tmp_path: Path):
        path = tmp_path / "sae.safetensors"
        _synth_safetensors(path, key="W_dec", shape=(8, 32))
        eager = load_sae_safetensors(path)
        lazy = load_sae_safetensors(path, feature_ids=[0, 3, 7])
        # Only the requested ids appear in the lazy result.
        assert list(lazy.keys()) == [0, 3, 7]
        for fid in (0, 3, 7):
            np.testing.assert_array_almost_equal(
                lazy[fid].projection, eager[fid].projection
            )
            assert lazy[fid].name == eager[fid].name

    def test_decoder_weight_square_lazy_matches_eager(self, tmp_path: Path):
        path = tmp_path / "sae.safetensors"
        _synth_safetensors(path, key="decoder.weight", shape=(4, 4))
        eager = load_sae_safetensors(path)
        lazy = load_sae_safetensors(path, feature_ids=[0, 2])
        for fid in (0, 2):
            np.testing.assert_array_almost_equal(
                lazy[fid].projection, eager[fid].projection
            )

    def test_decoder_weight_non_square_lazy_orientation(self, tmp_path: Path):
        # Non-square decoder.weight stores (d_model, n_features); lazy
        # path must take *columns* not rows.
        path = tmp_path / "sae.safetensors"
        from safetensors.numpy import save_file

        rng = np.random.default_rng(11)
        weight = rng.standard_normal((8, 4)).astype(np.float32)
        save_file({"decoder.weight": weight}, str(path))
        lazy = load_sae_safetensors(path, feature_ids=[0, 1, 2, 3])
        for fid in range(4):
            assert lazy[fid].projection.shape == (8,)
            np.testing.assert_array_almost_equal(
                lazy[fid].projection, weight[:, fid]
            )

    def test_dec_lazy_matches_eager(self, tmp_path: Path):
        path = tmp_path / "sae.safetensors"
        _synth_safetensors(path, key="dec", shape=(6, 5))
        eager = load_sae_safetensors(path)
        lazy = load_sae_safetensors(path, feature_ids=[1, 4])
        for fid in (1, 4):
            np.testing.assert_array_almost_equal(
                lazy[fid].projection, eager[fid].projection
            )

    def test_iteration_order_matches_input(self, tmp_path: Path):
        path = tmp_path / "sae.safetensors"
        _synth_safetensors(path, key="W_dec", shape=(8, 4))
        lazy = load_sae_safetensors(path, feature_ids=[7, 2, 5, 0])
        assert list(lazy.keys()) == [7, 2, 5, 0]

    def test_out_of_range_feature_id_rejected(self, tmp_path: Path):
        path = tmp_path / "sae.safetensors"
        _synth_safetensors(path, key="W_dec", shape=(4, 8))
        with pytest.raises(ValueError, match=r"\[0, 4\)"):
            load_sae_safetensors(path, feature_ids=[0, 9])

    def test_lazy_mode_honors_names_override(self, tmp_path: Path):
        path = tmp_path / "sae.safetensors"
        _synth_safetensors(path, key="W_dec", shape=(8, 4))
        lazy = load_sae_safetensors(
            path,
            feature_ids=[3, 5],
            names={3: "thing_a", 5: "thing_b"},
        )
        assert lazy[3].name == "thing_a"
        assert lazy[5].name == "thing_b"

    def test_lazy_mode_no_decoder_key_lists_present_keys(
        self, tmp_path: Path
    ):
        from safetensors.numpy import save_file

        save_file(
            {"enc": np.zeros((2, 2), dtype=np.float32)},
            str(tmp_path / "sae.safetensors"),
        )
        with pytest.raises(ValueError, match="W_dec"):
            load_sae_safetensors(
                tmp_path / "sae.safetensors", feature_ids=[0]
            )


# ---------------------------------------------------------------------------
# BF16 + alias support (task 2.3)
# ---------------------------------------------------------------------------

def _write_bf16_decoder(path: Path, arr: np.ndarray, key: str = "W_dec") -> None:
    """Write a safetensors file with one BF16 tensor under `key`."""
    f32 = arr.astype(np.float32)
    bf16_data = (f32.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    header = {key: {"dtype": "BF16", "shape": list(arr.shape), "data_offsets": [0, len(bf16_data)]}}
    header_json = json.dumps(header).encode("utf-8")
    pad = (8 - len(header_json) % 8) % 8
    header_json += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(bf16_data)


class TestBF16AndAlias:
    def test_bf16_w_dec_loads_without_error(self, tmp_path: Path):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        path = tmp_path / "bf16.safetensors"
        _write_bf16_decoder(path, arr, key="W_dec")
        records = load_sae_safetensors(path)
        assert len(records) == 3
        for rec in records.values():
            assert rec.projection.dtype == np.float64

    def test_bf16_decoder_weight_alias_resolves(self, tmp_path: Path):
        arr = np.ones((4, 16), dtype=np.float32)  # (d_model, d_sae) PyTorch layout
        path = tmp_path / "bf16_alias.safetensors"
        _write_bf16_decoder(path, arr, key="decoder.weight")
        records = load_sae_safetensors(path)
        # After transpose: d_sae=16 features, each of length d_model=4
        assert len(records) == 16
        assert records[0].projection.shape == (4,)

    def test_bf16_projections_approximately_correct(self, tmp_path: Path):
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((4, 8)).astype(np.float32)
        path = tmp_path / "bf16_vals.safetensors"
        _write_bf16_decoder(path, arr, key="W_dec")
        records = load_sae_safetensors(path)
        for i in range(4):
            np.testing.assert_allclose(
                records[i].projection, arr[i, :], rtol=1e-2
            )
