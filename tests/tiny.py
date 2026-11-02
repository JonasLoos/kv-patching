"""Tiny random models wrapped as PatchModels (CPU, fp32, token ids as text) for the patch tests."""
import copy

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3_5ForCausalLM, Qwen3_5TextConfig

from patch import PatchModel


class StubTok:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids if not (skip_special_tokens and i == self.eos_token_id))


def stub_patch_model(lm, hybrid: bool, full: dict | None = None):
    sm = PatchModel.__new__(PatchModel)
    sm.device, sm.tok, sm.sampler, sm.hybrid, sm.eos_ids = "cpu", StubTok(), {"top_k": 0, "repetition_penalty": 1.0}, hybrid, [StubTok.eos_token_id]
    sm.peft = sm.lm = lm.eval()
    sm.full = {k: v.eval() for k, v in (full or {}).items()}
    return sm


def with_adapters(lm, adapters, target_modules):
    """Random LoRA adapters (lora_B nonzero: zero-init would make them no-ops)."""
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=target_modules)
    lm = get_peft_model(lm, cfg, adapter_name=adapters[0])
    for a in adapters[1:]:
        lm.add_adapter(a, cfg)
    with torch.no_grad():
        for name, p in lm.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn_like(p))
    return lm


def llama_config():
    return LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)


def tiny_patch_model(adapters=("ft",)):
    torch.manual_seed(0)
    return stub_patch_model(with_adapters(LlamaForCausalLM(llama_config()), list(adapters), ["q_proj", "k_proj", "v_proj"]), hybrid=False)


def tiny_hybrid_model(adapters=("ft",)):
    """Tiny qwen3.5-style hybrid (3 linear-attention layers + 1 full-attention), LoRA on
    every linear module — the same recipe train.py uses on the real hybrid."""
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                            linear_num_key_heads=2, linear_num_value_heads=4,
                            linear_key_head_dim=8, linear_value_head_dim=8,
                            layer_types=["linear_attention"] * 3 + ["full_attention"],
                            max_position_embeddings=128)
    lm = Qwen3_5ForCausalLM(cfg)
    tm = sorted({n.rsplit(".", 1)[-1] for n, m in lm.named_modules() if isinstance(m, torch.nn.Linear) and "lm_head" not in n})
    return stub_patch_model(with_adapters(lm, list(adapters), tm), hybrid=True)


def tiny_full_model():
    """A complete fine-tuned checkpoint (train.py --full-all) next to the adapter-free base:
    every parameter perturbed, embeddings, norms and head included."""
    torch.manual_seed(0)
    base = LlamaForCausalLM(llama_config())
    ft = copy.deepcopy(base)
    with torch.no_grad():
        for p in ft.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return stub_patch_model(base, hybrid=False, full={"ft": ft}), base, ft
