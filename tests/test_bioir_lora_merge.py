import pytest
import torch

from rfproteina.nn import bioir_full


def test_lora_adapters_are_merged_into_the_base_weights_for_bioir():
    peft = pytest.importorskip("peft")

    class Denoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_qkv = torch.nn.Linear(8, 12)
            self.to_g = torch.nn.Linear(8, 8)

    torch.manual_seed(0)
    denoiser = Denoiser()
    peft.inject_adapter_in_model(peft.LoraConfig(r=4, lora_alpha=8, target_modules=["to_qkv"]), denoiser)
    for module in denoiser.modules():
        if hasattr(module, "lora_B"):
            torch.nn.init.normal_(module.lora_B["default"].weight)

    state = bioir_full._flow_state_with_merged_lora(denoiser)

    assert sorted(state) == ["to_g.bias", "to_g.weight", "to_qkv.bias", "to_qkv.weight"]
    x = torch.randn(3, 8)
    merged = torch.nn.functional.linear(x, state["to_qkv.weight"], state["to_qkv.bias"])
    torch.testing.assert_close(merged, denoiser.to_qkv(x), rtol=1e-5, atol=1e-5)
    assert not torch.allclose(state["to_qkv.weight"], denoiser.to_qkv.base_layer.weight)
