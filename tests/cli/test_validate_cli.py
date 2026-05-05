"""`polygram validate` subcommand argument parsing + skip-path tests.

Exercises argument parsing + dictionary loading. The full end-to-end
torch path is not exercised here; that lives in the smoke test in
`tests/test_examples.py` (mirrors the §4.4 / §4.3 / §4.2 pattern of
real-checkpoint-aware skipping).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from polygram.cli import main


def _write_toy_dictionary(path: Path, *, n_features: int = 8, d_model: int = 8) -> list[int]:
    """Write a toy-SAE-schema JSON file matching the loader at
    `polygram.load_toy_sae`. Returns the feature_ids list."""
    rng = np.random.default_rng(0)
    features = []
    feature_ids = list(range(n_features))
    for fid in feature_ids:
        proj = rng.standard_normal(d_model).astype(np.float32).tolist()
        features.append({
            "feature_id": fid,
            "name": f"feat_{fid}",
            "label": f"cluster_{fid % 2}/feat_{fid}",
            "projection": proj,
        })
    payload = {"schema_version": 1, "features": features}
    path.write_text(json.dumps(payload))
    return feature_ids


def _write_synth_sae(
    path: Path, *, n_features: int = 8, d_model: int = 8, seed: int = 0
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


def _write_prompts(path: Path, n: int = 2) -> None:
    lines = ["# leading comment", ""] + [
        f"prompt {i} with several tokens for the model to consume" for i in range(n)
    ]
    path.write_text("\n".join(lines))


class TestValidateArgs:
    def test_missing_dictionary_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ):
        rc = main([
            "validate",
            "--dictionary", str(tmp_path / "nope.json"),
            "--sae-checkpoint", str(tmp_path / "sae.safetensors"),
            "--feature-ids", "0,1",
            "--prompts", str(tmp_path / "p.txt"),
            "--layer", "5",
            "--output", str(tmp_path / "out.json"),
        ])
        assert rc == 2
        captured = capsys.readouterr()
        assert "--dictionary" in captured.err
        assert "not found" in captured.err

    def test_missing_sae_checkpoint_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ):
        dict_path = tmp_path / "dict.json"
        _write_toy_dictionary(dict_path)
        rc = main([
            "validate",
            "--dictionary", str(dict_path),
            "--sae-checkpoint", str(tmp_path / "missing.safetensors"),
            "--feature-ids", "0,1,2,3,4,5,6,7",
            "--prompts", str(tmp_path / "p.txt"),
            "--layer", "5",
            "--output", str(tmp_path / "out.json"),
        ])
        assert rc == 2
        captured = capsys.readouterr()
        assert "--sae-checkpoint" in captured.err
        assert "not found" in captured.err

    def test_feature_ids_length_mismatch_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ):
        dict_path = tmp_path / "dict.json"
        _write_toy_dictionary(dict_path, n_features=8)
        sae_path = tmp_path / "sae.safetensors"
        _write_synth_sae(sae_path, n_features=8, d_model=8)
        prompts_path = tmp_path / "p.txt"
        _write_prompts(prompts_path)
        rc = main([
            "validate",
            "--dictionary", str(dict_path),
            "--sae-checkpoint", str(sae_path),
            "--feature-ids", "0,1,2",  # only 3 ids; dict has 8
            "--prompts", str(prompts_path),
            "--layer", "5",
            "--output", str(tmp_path / "out.json"),
        ])
        assert rc == 2
        captured = capsys.readouterr()
        err = captured.err
        # Spec: stderr names both the supplied count (3) and the
        # dictionary's expected count (8).
        assert "3" in err
        assert "8" in err

    def test_empty_prompts_file_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ):
        dict_path = tmp_path / "dict.json"
        _write_toy_dictionary(dict_path, n_features=2)
        sae_path = tmp_path / "sae.safetensors"
        _write_synth_sae(sae_path, n_features=8, d_model=8)
        prompts_path = tmp_path / "p.txt"
        prompts_path.write_text("# only a comment\n\n")
        rc = main([
            "validate",
            "--dictionary", str(dict_path),
            "--sae-checkpoint", str(sae_path),
            "--feature-ids", "0,1",
            "--prompts", str(prompts_path),
            "--layer", "5",
            "--output", str(tmp_path / "out.json"),
        ])
        assert rc == 2
        captured = capsys.readouterr()
        assert "empty" in captured.err

    def test_layer_zero_without_override_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ):
        dict_path = tmp_path / "dict.json"
        _write_toy_dictionary(dict_path, n_features=2)
        sae_path = tmp_path / "sae.safetensors"
        _write_synth_sae(sae_path, n_features=8, d_model=8)
        prompts_path = tmp_path / "p.txt"
        _write_prompts(prompts_path)
        rc = main([
            "validate",
            "--dictionary", str(dict_path),
            "--sae-checkpoint", str(sae_path),
            "--feature-ids", "0,1",
            "--prompts", str(prompts_path),
            "--layer", "0",
            "--output", str(tmp_path / "out.json"),
        ])
        assert rc == 2
        captured = capsys.readouterr()
        assert "deeper-layer-ablation-probe.md" in captured.err

    def test_non_default_model_emits_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch
    ):
        dict_path = tmp_path / "dict.json"
        _write_toy_dictionary(dict_path, n_features=2)
        sae_path = tmp_path / "sae.safetensors"
        _write_synth_sae(sae_path, n_features=8, d_model=8)
        prompts_path = tmp_path / "p.txt"
        _write_prompts(prompts_path)

        # Monkeypatch the lazy import to fail so we don't actually load
        # a model. We just want to confirm the warning fires before
        # the run() ImportError surfaces.
        from polygram.behavioural import runtime as bh_runtime

        def _fake_import():
            raise ImportError(bh_runtime._BEHAVIOURAL_INSTALL_HINT)

        monkeypatch.setattr(bh_runtime, "_import_torch_and_transformers", _fake_import)
        # Also patch the validator module's local reference.
        from polygram.behavioural import validator as bh_validator
        monkeypatch.setattr(bh_validator, "_import_torch_and_transformers", _fake_import)

        rc = main([
            "validate",
            "--dictionary", str(dict_path),
            "--sae-checkpoint", str(sae_path),
            "--feature-ids", "0,1",
            "--prompts", str(prompts_path),
            "--layer", "5",
            "--model", "EleutherAI/pythia-1b",
            "--output", str(tmp_path / "out.json"),
        ])
        # Whatever happens after the warning, the warning must be in stderr.
        captured = capsys.readouterr()
        assert "calibrated on GPT-2 small only" in captured.err
        # Without torch installed the validator should ImportError-out
        # with the install hint.
        assert rc in (0, 2)
