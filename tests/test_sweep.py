"""The batched gradual sweep (PatchModel.sweep_batch) against the one-boundary-at-a-time
reference path (patched_completion_logits) scored with a plain full-sort estimator, on
identical sampled rollouts. fp32 on CPU, so the two agree to float noise at every boundary
(in bf16 on a GPU, any change of kernel shapes moves the TV by up to ~0.015)."""
import functools

import torch
import torch.nn.functional as F

from patch import completion_mask, ref_stats
from tiny import tiny_hybrid_model, tiny_patch_model

T, TOP_P = 0.7, 0.9


def reference_stats(lq_, lp_, top_p):
    """tv, truncated KL (q's top-p nucleus via a full-vocab sort, both renormalized there) and
    uncapped KL per position."""
    lq, lp = F.log_softmax(lq_.float() / T, -1), F.log_softmax(lp_.float() / T, -1)
    ps, order = lq.exp().sort(-1, descending=True)
    keep = torch.zeros_like(lq, dtype=torch.bool).scatter_(-1, order, (ps.cumsum(-1) - ps) < top_p)
    q = lq.exp() * keep
    q = q / q.sum(-1, keepdim=True)
    lpn = lp - torch.logsumexp(lp.masked_fill(~keep, -torch.inf), -1, keepdim=True)
    kl = (q * (q.clamp_min(torch.finfo(q.dtype).tiny).log() - lpn) * keep).sum(-1)
    return 0.5 * (lq.exp() - lp.exp()).abs().sum(-1), kl, (lq.exp() * (lq - lp)).sum(-1)


@functools.cache
def rollouts(hybrid=False):
    """Sampled own-cache rollouts of both sides plus their teacher-forced references and caches."""
    sm = (tiny_hybrid_model if hybrid else tiny_patch_model)(("ft",))
    torch.manual_seed(0)
    ids = torch.randint(1, 64, (1, 9))
    L, n = ids.shape[1], 3
    rolls, refs, caches = {}, {}, {}
    for s in ("ft", "base"):
        _, seqs = sm.generate(s, ids.expand(n, -1), sm.prefill(s, ids, n), strip=L, max_new=40, temperature=T, top_p=TOP_P, return_sequences=True)
        rolls[s] = seqs, completion_mask(seqs[:, L:], sm.eos_ids)
    for m in ("ft", "base"):
        for s in rolls:
            refs[m, s], caches[m, s] = sm.completion_logits(m, rolls[s][0], L, with_cache=True)
    return sm, L, n, rolls, refs, caches


def compare(ks_checked, budgets=(10**6, 1, 150), hybrid=False):  # one chunk; one boundary per chunk (L-1/L kept together); mixed
    sm, L, n, rolls, refs, caches = rollouts(hybrid)
    assert any(not m.all() for _, m in rolls.values())  # some rollout ends early: exercises the masks
    for source, target in (("ft", "base"), ("base", "ft"), ("ft", "ft")):
        for src in (source, target):
            seqs, mask = rolls[src]
            ks = list(range(seqs.shape[1] - 1))
            for budget in budgets:
                got = sm.sweep_batch(target, seqs, caches[source, src], ks, L, refs[source, src], refs[target, src], mask, T, TOP_P, tok_budget=budget)
                for k in ks_checked(L, ks):
                    q, start = sm.patched_completion_logits(target, seqs, caches[source, src], k, L)
                    for i in range(n):
                        m = mask[i, start - L:]
                        row = got[k][i]
                        if not m.any():
                            assert row is None
                            continue
                        assert len(row["lp_q"]) == int(m.sum()), (k, i)  # same scored support
                        sl = slice(start - L, None)
                        want = [reference_stats(q[i][m], refs[r, src][i, sl][m], TOP_P) for r in (source, target)]
                        for key, (tv, kl, kl_full) in zip(("source", "target"), want, strict=True):
                            assert abs(row[f"tv_{key}"]["mean"] - tv.mean().item()) < 1e-5
                            assert abs(row[f"kl_{key}"]["mean"] - kl.mean().item()) < 1e-4
                            assert abs(row[f"kl_{key}"]["p95"] - kl.quantile(0.95).item()) < 1e-4
                            assert abs(row[f"kl_{key}_full"]["mean"] - kl_full.mean().item()) < 1e-4
                        lp_q = F.log_softmax(q[i][m].float() / T, -1).gather(-1, seqs[i, start:][m].unsqueeze(-1)).squeeze(-1)
                        assert torch.allclose(torch.from_numpy(row["lp_q"]).float(), lp_q, atol=2e-2)  # fp16 array
                        if k == 0 or source == target:  # unpadded anchor / invariance: the target over its own cache is its own reference
                            assert row["tv_target"]["mean"] < 1e-5 and abs(row["kl_target"]["mean"]) < 1e-4
                if source != target:
                    assert max(r["tv_target"]["mean"] for r in got[L + 1] if r) > 1e-3  # the patch is visible


def test_sweep_batch_matches_reference():
    compare(lambda L, ks: ks)


def test_hybrid_sweep_batch_matches_reference():
    compare(lambda L, ks: ks, hybrid=True)


@torch.no_grad()
def test_hybrid_crop_is_a_fresh_prefill():
    """On a hybrid, the reference path's cropped source cache (attention keys/values cut, linear-attention
    states recomputed from the recorded writes) equals the source's own prefill of the first k tokens:
    the target's logits over both agree."""
    sm, L, n, rolls, refs, caches = rollouts(True)
    seqs = rolls["base"][0]
    for k in (1, 3, L - 1, L, L + 5):
        q, _ = sm.patched_completion_logits("base", seqs, caches["ft", "base"], k, L)
        sm.use("ft")
        fresh = sm.lm(seqs[:, :k], use_cache=True).past_key_values
        sm.use("base")
        want = sm.lm(seqs[:, k:], attention_mask=torch.ones_like(seqs), past_key_values=fresh).logits
        got, want = (q, want[:, L - 1 - k:-1]) if k < L else (q[:, 1:], want[:, :-1]) if k == L else (q, want[:, :-1])
        assert got.shape == want.shape and torch.allclose(got, want, atol=1e-4), k


def test_ref_stats():
    torch.manual_seed(0)
    a, b = torch.randn(2, 5, 64), torch.randn(2, 5, 64)
    tokens = torch.randint(1, 64, (2, 5))
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1] = False  # a completion that ends before the scored range
    out = ref_stats(a, b, tokens, mask, T)
    assert out[1] is None
    tv, _, _ = reference_stats(a[0], b[0], 1.0)
    assert torch.allclose(torch.from_numpy(out[0]["tv_refs"]).float(), tv, atol=1e-3)
    assert ref_stats(a, a, tokens, mask, T)[0]["tv_refs"].max() == 0
