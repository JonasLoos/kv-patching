import torch
import torch.nn.functional as F
from transformers import DynamicCache

from patch import _pair_stats, kv, nucleus_keep


def make_cache(fill, layers=3, batch=2, heads=4, seq=5, dim=8):
    c = DynamicCache()
    for i in range(layers):
        c.update(torch.full((batch, heads, seq, dim), fill + i * 10.0),
                 torch.full((batch, heads, seq, dim), -fill - i * 10.0), i)
    return c


def truncated_kl(q, p, top_p, temperature=1.0):
    """Mean truncated / uncapped KL(q || p) over positions, the sweep's estimator."""
    lq, lp = F.log_softmax(q / temperature, -1), F.log_softmax(p / temperature, -1)
    _, kl, kl_full = _pair_stats(lq, lq.exp(), nucleus_keep(lq, top_p), lp, torch.finfo(torch.float32).tiny)
    return kl.mean().item(), kl_full.mean().item()


def test_truncated_kl_ignores_the_unsampleable_tail():
    torch.manual_seed(0)
    q = torch.randn(4, 20000) * 6.0               # peaked, LM-like
    order = q[0].argsort()                        # ascending
    tail = q.clone()
    tail[:, order[:-50]] -= 30    # perturb everything below the top 50
    head = q.clone()
    head[:, order[-3:]] -= 3      # perturb the top 3
    assert truncated_kl(q, tail, top_p=1.0)[0] > 5     # full support: tail dominates
    assert truncated_kl(q, tail, top_p=0.9)[1] > 5     # ... as it does for the uncapped KL
    assert truncated_kl(q, tail, top_p=0.9)[0] < 0.1   # sampling nucleus: inert
    assert truncated_kl(q, head, top_p=0.9)[0] > 0.1   # real divergence survives
    assert abs(truncated_kl(q, q.clone(), 0.9)[0]) < 1e-6
    # temperature rescales both distributions before the truncation: identical logits stay at 0
    assert abs(truncated_kl(q, q.clone(), 0.9, temperature=0.7)[0]) < 1e-6


def test_nucleus_keep_matches_full_sort():
    torch.manual_seed(0)
    for scale in (6.0, 3.0, 0.3):  # nucleus within the first top-k (512), the second (4096), the full vocab
        lq = F.log_softmax(torch.randn(3, 20000) * scale, -1)
        ps, order = lq.exp().sort(-1, descending=True)
        want = torch.zeros_like(lq, dtype=torch.bool).scatter_(-1, order, (ps.cumsum(-1) - ps) < 0.9)
        assert torch.equal(nucleus_keep(lq, 0.9), want), scale


def test_sample_token_top_p():
    from patch import sample_token
    torch.manual_seed(0)
    logits = torch.tensor([[4.0, 1.0, 0.0, -2.0]]).expand(200, -1)  # top token holds ~93% mass
    draws = sample_token(logits, temperature=1.0, top_p=0.9)
    assert (draws == 0).all()  # nucleus is the top token alone
    assert set(sample_token(logits, 1.0, 1.0).unique().tolist()) > {0}  # full support samples the tail
    assert (sample_token(logits, 0.0, 0.9) == 0).all()


def test_crop_cache():
    from patch import crop_cache
    a = make_cache(1.0)
    c = crop_cache(a, 3)
    k, v = kv(c, 1)
    assert k.shape == (2, 4, 3, 8) and (k == 11.0).all() and (v == -11.0).all()
    k.fill_(99.0)
    assert (kv(a, 1)[0] == 11.0).all()  # no aliasing
