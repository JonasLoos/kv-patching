"""The full fine-tune arm is stored as an exact PEFT adapter (A=I, B=dW or the transpose
arrangement, scaling 1): loading it onto the base must reproduce the fine-tuned model's
logits, including on the mixed-batch path that level "all" uses."""
import copy

import torch
from peft import PeftModel
from transformers import LlamaConfig, LlamaForCausalLM

from train import TARGET_MODULES, save_full_as_adapter


def test_full_adapter_is_exact(tmp_path):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    base = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    ft = copy.deepcopy(base)
    base_params = {}
    with torch.no_grad():
        for n, p in ft.named_parameters():
            p.requires_grad = n.endswith(tuple(f"{m}.weight" for m in TARGET_MODULES))
            if p.requires_grad:
                base_params[n] = p.detach().clone()
                p.add_(0.05 * torch.randn_like(p))  # the "fine-tune"
    save_full_as_adapter(ft, base_params, TARGET_MODULES, tmp_path, "tiny")
    lm = PeftModel.from_pretrained(base, str(tmp_path), adapter_name="full", autocast_adapter_dtype=False).eval()

    ids = torch.randint(1, 64, (2, 9))
    with torch.no_grad():
        want = ft(ids).logits
        got = lm(ids).logits
        lm.base_model.disable_adapter_layers()
        plain = lm(ids).logits
        lm.base_model.enable_adapter_layers()
        mixed = lm(ids.repeat(2, 1), adapter_names=["__base__", "__base__", "full", "full"]).logits
    assert not torch.allclose(plain, want, atol=1e-2)  # the perturbation is visible
    assert torch.allclose(got, want, atol=3e-2, rtol=3e-2)  # bf16 matmul order differs; the delta itself is exact
    assert torch.allclose(mixed[2:], want, atol=3e-2, rtol=3e-2)
    assert torch.allclose(mixed[:2], plain, atol=3e-2, rtol=3e-2)
    # k/v (d_out < d_in) carry the delta in lora_A with an identity lora_B; the rest the reverse
    lay = lm.base_model.model.model.layers[0]
    assert torch.equal(lay.self_attn.k_proj.lora_B["full"].weight, torch.eye(16, dtype=torch.bfloat16))
    assert torch.equal(lay.self_attn.q_proj.lora_A["full"].weight, torch.eye(32, dtype=torch.bfloat16))
    assert torch.equal(lay.mlp.down_proj.lora_B["full"].weight, torch.eye(32, dtype=torch.bfloat16))
