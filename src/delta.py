"""Per-token cache entries of Qwen3.5's linear-attention (Gated DeltaNet) layers, for sweeps.

HF keeps only a linear-attention layer's end state: the recurrent state S and the conv state (the
last K = 4 pre-conv inputs). The layer's per-position cache entries are the token's writes, its
input projections: x (pre-conv q, k and v channels), b (write strength, beta = sigmoid(b)) and a
(decay g). The end state is a fixed function of the writes (causal depthwise conv + SiLU, then the
gated delta rule), so a hybrid cache can be cut at any position by recomputing it from the writes:
  record()      the writes of every linear layer during the forwards inside
  substitute()  per row and position, another model's writes instead of the forward's own
  end_state()   (conv state, S) after a sequence of writes, from scratch
  build_cache() a hybrid DynamicCache from attention keys/values and linear (conv state, S)
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

WRITES = ("in_proj_qkv", "in_proj_b", "in_proj_a")  # x, b, a


def linear_layers(lm) -> dict:
    """layer index -> Gated DeltaNet module."""
    return {m.layer_idx: m for m in lm.modules() if isinstance(m, mq.Qwen3_5GatedDeltaNet)}


def _hooks(lm, fn):
    return [getattr(m, name).register_forward_hook(lambda mod, inp, out, i=i, j=j: fn(i, j, out))
            for i, m in linear_layers(lm).items() for j, name in enumerate(WRITES)]


@contextmanager
def record(lm):
    """Yields {layer: (x, b, a)}, each [B, T, ·] over the forwards inside, filled on exit."""
    parts, writes = {}, {}
    hs = _hooks(lm, lambda i, j, out: parts.setdefault(i, ([], [], []))[j].append(out.detach()))
    try:
        yield writes
    finally:
        for h in hs:
            h.remove()
        writes.update({i: tuple(torch.cat(c, 1) for c in p) for i, p in parts.items()})


@contextmanager
def substitute(lm, own: torch.Tensor, writes: dict):
    """Inside, a forward's writes where own [B, T] is True are writes' ({layer: (x, b, a)}, each
    [B, T, ·], aligned with the forward's input positions)."""
    def fn(i, j, out):
        assert out.shape[:2] == own.shape, (out.shape, own.shape)
        return torch.where(own[..., None], writes[i][j].to(out.dtype), out)
    hs = _hooks(lm, fn)
    try:
        yield
    finally:
        for h in hs:
            h.remove()


def end_state(m, x: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(conv tail [B, C, K], recurrent state) after writes x [B, T, C], b, a [B, T, H], from scratch,
    as the layer computes them with a cache."""
    full = torch.cat([x.new_zeros(x.shape[0], x.shape[2], m.conv_kernel_size), x.transpose(1, 2)], -1)
    post = mq.causal_conv1d_fn(full, m.conv1d.weight.squeeze(1), m.conv1d.bias, activation=m.activation)[:, :, -x.shape[1]:].transpose(1, 2)
    B, T, _ = post.shape
    _, k, v = torch.split(post, [m.key_dim, m.key_dim, m.value_dim], -1)
    k = k.reshape(B, T, -1, m.head_k_dim).repeat_interleave(m.num_v_heads // m.num_k_heads, 2)
    v = v.reshape(B, T, -1, m.head_v_dim)
    g = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)
    s = mq.torch_chunk_gated_delta_rule(k, k, v, g=g, beta=b.sigmoid(), output_final_state=True, use_qk_l2norm_in_kernel=True)[1]
    return full[:, :, -m.conv_kernel_size:], s


def build_cache(lm, attn: dict, linear: dict) -> DynamicCache:
    """A hybrid cache: {layer: (keys, values)} for the attention layers, {layer: (conv tail, S)} for the linear ones."""
    cache = DynamicCache(config=lm.config)
    for i, (k, v) in attn.items():
        cache.update(k, v, i)
    for i, (tail, s) in linear.items():
        lyr = cache.layers[i]
        lyr.conv_states[0], lyr.recurrent_states[0], lyr.conv_kernel_size[0] = tail, s, tail.shape[-1]
        lyr.is_conv_states_initialized[0] = lyr.is_recurrent_states_initialized[0] = lyr.has_previous_state[0] = True
        lyr.device, lyr.dtype = tail.device, tail.dtype
    return cache
