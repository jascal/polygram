"""Independent execution is a saved native model, not a host-hook approximation."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("saeforge")
pytest.importorskip("transformers")


@pytest.mark.parametrize("tied", [False, True])
def test_qwen_native_forge_reload_runs_without_host(tmp_path, tied):
    from saeforge import FeatureBasis, SubspaceProjector
    from saeforge.adapters import adapter_for
    from saeforge.model import NativeModel
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from examples.binding_benchmark.forge import native_config, untie_for_projection

    torch.set_num_threads(1)
    torch.manual_seed(8)
    host = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=31, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=tied,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
    )).eval()
    ids = torch.tensor([[3, 4, 7, 8, 2]])
    with torch.inference_mode():
        before_untie = host(ids).logits
    assert untie_for_projection(host) == tied
    with torch.inference_mode():
        torch.testing.assert_close(host(ids).logits, before_untie, atol=0, rtol=0)
    basis = FeatureBasis(np.arange(16), np.eye(16), np.ones(16), np.ones(16))
    projector = SubspaceProjector(basis, scale_boost=1.0)
    weights = projector.project_module(host, attention_width="host")
    cfg = native_config(adapter_for(host), host, 16)
    assert cfg.rope_theta == 1000000.0
    native = NativeModel.from_projected_weights(cfg, weights)
    native.resolved_forward_mode = "native_in_basis"
    with torch.inference_mode():
        reference = host(ids).logits
        actual = native.torch_module.eval()(ids)
    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-4)
    native.save_pretrained(tmp_path)
    del native, host
    reloaded = NativeModel.load_pretrained(tmp_path)
    assert reloaded.config.forward_mode == "native_in_basis"
    assert not any(m._forward_hooks or m._forward_pre_hooks
                   for m in reloaded.torch_module.modules())
    with torch.inference_mode():
        torch.testing.assert_close(reloaded.torch_module.eval()(ids), actual)
