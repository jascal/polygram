"""Checks that prevent information leakage and misleading intervention metrics."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from examples.binding_benchmark.evaluate import (  # noqa: E402
    aggregate, compare_logits, counterpart_indices,
)
from examples.binding_benchmark.representations import (  # noqa: E402
    RoleFiller, SparseAE, compress_sae, reconstruct,
)


def test_tpr_prediction_uses_symbols_not_host_or_answer():
    torch.manual_seed(1)
    model = RoleFiller(8, 5, 3, 2, 2)
    symbols = torch.tensor([[[0, 0], [1, 1], [2, 2]]])
    assert torch.equal(model(torch.zeros(1, 8), symbols), model(torch.randn(1, 8), symbols))
    swapped = symbols.clone()
    swapped[0, :2, 0] = swapped[0, :2, 0].flip(0)
    assert not torch.equal(model(None, symbols), model(None, swapped))


def test_sae_decoder_constraint_removes_sparsity_rescaling_escape():
    model = SparseAE(12, 6)
    with torch.no_grad():
        model.decoder.weight.mul_(10)
    model.normalize_decoder()
    torch.testing.assert_close(model.decoder.weight.norm(dim=0), torch.ones(6))


def test_counterfactual_mapping_preserves_question_and_is_an_involution():
    records = [dict(pair_id=str(p), member=m, unrelated=u)
               for p in range(3) for m in (0, 1) for u in (False, True)]
    indices = counterpart_indices(records)
    assert torch.equal(indices[indices], torch.arange(len(records)))
    for i, j in enumerate(indices):
        assert records[i]["unrelated"] == records[j]["unrelated"]
        assert records[i]["member"] != records[j]["member"]


def test_equal_kl_magnitude_does_not_imply_agreement():
    reference = torch.zeros(2, 3)
    edited = torch.tensor([[3., 0., 0.], [0., 3., 0.]])
    records = [dict(id=str(i), pair_id="p", task="t", split="test", unrelated=False,
                    candidates=["A", "B", "C"], candidate_ids=[0, 1, 2], target="A")
               for i in range(2)]
    rows = compare_logits(edited, reference, records)
    assert rows[0]["kl_nats"] == pytest.approx(rows[1]["kl_nats"])
    assert rows[0]["correct"] and not rows[1]["correct"]
    summary = aggregate(rows)["t/test/target"]
    assert summary["accuracy"] == 0.5
    assert summary["pair_accuracy"] == 0


def test_nonfinite_logits_fail_instead_of_producing_an_apparent_score():
    with pytest.raises(FloatingPointError, match="Non-finite"):
        compare_logits(torch.tensor([[float("nan"), 0.]]), torch.zeros(1, 2), [])


def test_real_polygram_checkpoint_roundtrip_matches_sae_coordinates(tmp_path):
    pytest.importorskip("safetensors")
    torch.manual_seed(2)
    model = SparseAE(12, 16)
    x = torch.randn(40, 12)
    mean, scale = torch.randn(12), torch.tensor(3.0)
    protocol = {"model": "synthetic-test", "compression_target_widths": [4]}
    results = compress_sae(model, mean, scale, x, tmp_path, protocol, 0)
    h = x * scale + mean
    source = reconstruct(tmp_path / "sae-seed0.safetensors", h, [])
    with torch.no_grad():
        expected = model(x) * scale + mean
    torch.testing.assert_close(source, expected)
    from safetensors.numpy import load_file

    result = results[0]
    state = load_file(result["artifact"])
    rank = np.linalg.matrix_rank(state["W_dec"])
    assert rank == result["decoder_rank"]
    assert result["latent_width"] <= 16
    report = json.loads((tmp_path / "compressed-target4-seed0.compression_report.json").read_text())
    assert report["strategy"] == "merge"
