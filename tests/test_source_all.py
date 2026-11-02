"""Regression: level "all" must re-encode completion tokens with the source adapter, even
when the target is "base" (whose use() disables adapter layers — PEFT then ignores
adapter_names, silently running the source rows as base and collapsing @all into @prompt)."""
import torch

from patch import crop_cache
from tiny import tiny_full_model, tiny_hybrid_model, tiny_patch_model


def test_source_all_reencodes_with_the_source():
    sm = tiny_patch_model()
    ids = torch.arange(1, 9).unsqueeze(0)
    n, steps = 2, 12

    def run():
        return sm.generate_source_all("ft", "base", ids, n, max_new=steps, temperature=0.0)

    got = run()
    # sensitivity check: with enable_adapter_layers inert (emulating the pre-fix adapter
    # state, where _first_token's use("base") left adapters disabled for the whole loop),
    # this model must produce a different trajectory — otherwise the assertion below is vacuous
    orig = sm.lm.base_model.enable_adapter_layers
    sm.lm.base_model.enable_adapter_layers = lambda: None
    got_bug = run()
    sm.lm.base_model.enable_adapter_layers = orig
    assert got_bug != got

    # sequential reference: per step the target reads logits over the source-only cache,
    # then its current-position KV is cropped away and the source re-encodes the token
    cache = sm.prefill("ft", ids, n, ids.shape[1])
    tok = sm._first_token("base", ids, crop_cache(cache, ids.shape[1] - 1), n, 0.0)
    seq, finished = tok, torch.zeros(n, dtype=torch.bool)
    for _ in range(steps - 1):
        att = torch.ones(n, ids.shape[1] + seq.shape[1], dtype=torch.long)
        sm.use("base")
        logits = sm.lm(tok, attention_mask=att, past_key_values=cache, use_cache=True).logits[:, -1, :].float()
        cache.crop(-1)
        sm.use("ft")
        sm.lm(tok, attention_mask=att, past_key_values=cache, use_cache=True)
        nxt = logits.argmax(-1, keepdim=True)
        nxt[finished] = sm.tok.eos_token_id
        seq = torch.cat([seq, nxt], dim=1)
        finished |= nxt.squeeze(1) == sm.tok.eos_token_id
        if finished.all():
            break
        tok = nxt
    assert got == [sm.tok.decode(row).strip() for row in seq]


def test_full_model_switches_and_patches():
    sm, base, ft = tiny_full_model()
    ids = torch.arange(1, 9).unsqueeze(0)
    n, steps = 2, 12
    # use() selects the model; own-cache generation is the plain model's generation
    for who, model in (("ft", ft), ("base", base)):
        own = sm.generate_cells(ids, [who], None, n, temperature=0.0, max_new=steps)[who]
        want = model.generate(ids.expand(n, -1), do_sample=False, max_new_tokens=steps, eos_token_id=0, pad_token_id=0, top_k=0, repetition_penalty=1.0)
        assert own == [sm.tok.decode(row[ids.shape[1]:]).strip() for row in want], who
        assert sm.generate_source_all(who, who, ids, n, max_new=steps, temperature=0.0) == own, who
    own = sm.generate_cells(ids, ["ft", "base"], None, n, temperature=0.0, max_new=steps)
    assert own["ft"] != own["base"]

    # cross-model @all == the O(T^2) reference: the source re-prefills the whole context each
    # step and the target reads logits over it (as in the hybrid test)
    for source, target in (("ft", "base"), ("base", "ft")):
        got = sm.generate_source_all(source, target, ids, n, max_new=steps, temperature=0.0)
        tok = sm._first_token(target, ids, sm.prefill(source, ids, n), n, 0.0)
        seq, finished = tok, torch.zeros(n, dtype=torch.bool)
        for _ in range(steps - 1):
            full = torch.cat([ids.expand(n, -1), seq], dim=1)
            sm.use(source)
            cache = sm.lm(full[:, :-1], attention_mask=torch.ones_like(full[:, :-1]), use_cache=True).past_key_values
            sm.use(target)
            logits = sm.lm(full[:, -1:], attention_mask=torch.ones_like(full), past_key_values=cache, use_cache=True).logits[:, -1, :].float()
            nxt = logits.argmax(-1, keepdim=True)
            nxt[finished] = sm.tok.eos_token_id
            seq = torch.cat([seq, nxt], dim=1)
            finished |= nxt.squeeze(1) == sm.tok.eos_token_id
            if finished.all():
                break
        assert got == [sm.tok.decode(row).strip() for row in seq], (source, target)
        assert got != sm.generate_cells(ids, [target], None, n, temperature=0.0, max_new=steps)[target], (source, target)


def test_source_all_hybrid_self_control():
    """source == target @all must equal plain generation over the model's own cache."""
    sm = tiny_hybrid_model()
    ids = torch.arange(1, 9).unsqueeze(0)
    n, steps = 2, 12
    for who in ("ft", "base"):
        own = sm.generate_cells(ids, [who], None, n, temperature=0.0, max_new=steps)[who]
        got = sm.generate_source_all(who, who, ids, n, max_new=steps, temperature=0.0)
        assert got == own, who


def test_source_all_hybrid_matches_sequential():
    """Doubled-batch hybrid @all == the O(T^2) reference where the source re-prefills the
    whole context each step and the target reads logits over it."""
    sm = tiny_hybrid_model()
    ids = torch.arange(1, 9).unsqueeze(0)
    n, steps = 2, 12
    got = sm.generate_source_all("ft", "base", ids, n, max_new=steps, temperature=0.0)

    # sensitivity: without the per-step state sync the target half drifts off the
    # source-only cache — the trajectory must change, else the reference check is vacuous
    import patch
    orig = patch.sync_source_half
    patch.sync_source_half = lambda cache, n: None
    got_nosync = sm.generate_source_all("ft", "base", ids, n, max_new=steps, temperature=0.0)
    patch.sync_source_half = orig
    assert got_nosync != got

    tok = sm._first_token("base", ids, sm.prefill("ft", ids, n), n, 0.0)
    seq, finished = tok, torch.zeros(n, dtype=torch.bool)
    for _ in range(steps - 1):
        full = torch.cat([ids.expand(n, -1), seq], dim=1)
        sm.use("ft")
        cache = sm.lm(full[:, :-1], attention_mask=torch.ones_like(full[:, :-1]), use_cache=True).past_key_values
        sm.use("base")
        logits = sm.lm(full[:, -1:], attention_mask=torch.ones_like(full), past_key_values=cache,
                       use_cache=True).logits[:, -1, :].float()
        nxt = logits.argmax(-1, keepdim=True)
        nxt[finished] = sm.tok.eos_token_id
        seq = torch.cat([seq, nxt], dim=1)
        finished |= nxt.squeeze(1) == sm.tok.eos_token_id
        if finished.all():
            break
    assert got == [sm.tok.decode(row).strip() for row in seq]


def test_patched_completion_logits():
    sm = tiny_patch_model()
    torch.manual_seed(1)
    full = torch.randint(1, 64, (2, 12))
    Lp = 5
    ref_ft, cache_ft = sm.completion_logits("ft", full, Lp, with_cache=True)
    ref_base = sm.completion_logits("base", full, Lp)
    assert ref_ft.shape == (2, 7, 64)

    # k=0: own-cache target forward — identity anchor
    q, start = sm.patched_completion_logits("base", full, cache_ft, 0, Lp)
    assert start == Lp and torch.allclose(q, ref_base, atol=1e-3)

    # self-patch: the source reading its own cache equals its own-cache forward at any boundary
    for k in (2, Lp, Lp + 3):
        q, start = sm.patched_completion_logits("ft", full, cache_ft, k, Lp)
        assert start == (Lp if k <= Lp else k + 1)
        assert torch.allclose(q, ref_ft[:, start - Lp:], atol=1e-3), f"k={k}"

    # cross-model mid-sweep: full support on the prompt side, shrinking past the prompt
    q, start = sm.patched_completion_logits("base", full, cache_ft, Lp, Lp)
    assert start == Lp and q.shape == ref_base.shape and not torch.allclose(q, ref_base, atol=1e-3)
    q, start = sm.patched_completion_logits("base", full, cache_ft, Lp + 3, Lp)
    assert start == Lp + 4 and q.shape == (2, 12 - start, 64)
