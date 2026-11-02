"""PatchModel.generate_cells: every source-ownership level against a hand-written greedy
reference, and batching (all cells of a level as one batch with per-row models) that changes
no completion."""
import torch

import patch
from patch import crop_cache
from tiny import tiny_hybrid_model, tiny_patch_model

LEVELS = ("prompt-minus-last", "prompt", "all")
CELLS = ["ft", "oth", "base", "ft>base", "oth>base", "base>ft", "oth>ft"]


def reference(sm, cell, level, ids, n, steps):
    """The level's definition, built by hand: which positions' KV the source writes."""
    L = ids.shape[1]
    source, _, target = cell.partition(">")
    if not target or level == "prompt-minus-last":
        own = L - 1
        sm.use(source)
        cache = sm.lm(ids[:, :own].expand(n, -1), use_cache=True).past_key_values
        rest = ids[:, own:].expand(n, -1)
        out = decode(sm, target or source, rest, cache, steps)
    else:  # "prompt": the target reads the final template token over the source's [0, L-1) cache, then decodes over the source's full [0, L) cache
        sm.use(source)
        full = sm.lm(ids.expand(n, -1), use_cache=True).past_key_values
        sm.use(target)
        first = sm.lm(ids[:, -1:].expand(n, -1), past_key_values=crop_cache(full, L - 1), use_cache=True).logits[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([first, decode(sm, target, first, full, steps - 1)], dim=1)
    return [sm.tok.decode(row).strip() for row in out]


def decode(sm, who, inp, cache, steps):
    """Greedy continuation: `who` encodes `inp` over `cache`, then feeds back its argmax tokens;
    returns only the new tokens, eos-padded after the first eos like HF generate."""
    sm.use(who)
    toks, done = [], torch.zeros(inp.shape[0], dtype=torch.bool)
    for _ in range(steps):
        nxt = sm.lm(inp, past_key_values=cache, use_cache=True).logits[:, -1].argmax(-1, keepdim=True)
        nxt[done] = sm.tok.eos_token_id
        toks.append(nxt)
        done |= nxt.squeeze(1) == sm.tok.eos_token_id
        if done.all():
            break
        inp = nxt
    return torch.cat(toks, dim=1)


def test_levels_match_reference():
    sm = tiny_patch_model(("ft", "oth"))
    ids = torch.arange(1, 9).unsqueeze(0)
    n, steps = 2, 10
    seen = {}
    for level in ("prompt-minus-last", "prompt"):
        got = sm.generate_cells(ids, CELLS, level, n, temperature=0.0, max_new=steps)
        assert list(got) == CELLS  # one batch per level, rows in cell order: the RNG order of a seeded run
        for cell in CELLS:
            assert got[cell] == reference(sm, cell, level, ids, n, steps), (level, cell)
        seen[level] = str({c: got[c] for c in CELLS if ">" in c})
    assert len(set(seen.values())) == 2  # the levels are distinguishable on this model


def test_batching_changes_no_completion():
    """Greedy: one batch of every cell (per-row sources and targets) == every cell alone."""
    ids = torch.arange(1, 9).unsqueeze(0)
    for sm in (tiny_patch_model(("ft", "oth")), tiny_hybrid_model(("ft", "oth"))):
        for level in LEVELS:
            got = sm.generate_cells(ids, CELLS, level, 2, temperature=0.0, max_new=10)
            alone = {c: sm.generate_cells(ids, [c], level, 2, temperature=0.0, max_new=10)[c] for c in CELLS}
            assert got == alone, (sm.hybrid, level)
            assert got["ft>base"] != got["base"] and got["base>ft"] != got["ft"], (sm.hybrid, level)


def test_seeded_sampling_is_reproducible():
    sm = tiny_patch_model(("ft", "oth"))
    ids = torch.arange(1, 9).unsqueeze(0)
    runs = []
    for seed in (0, 0, 1):
        torch.manual_seed(seed)
        runs.append([sm.generate_cells(ids, CELLS, level, 3, temperature=0.7, top_p=0.9, max_new=10) for level in LEVELS])
    assert runs[0] == runs[1] and runs[0] != runs[2]


def test_rows_run_each_rows_own_model():
    """rows(): a mixed batch computes, row by row, exactly what each model computes on a batch
    of the same shape (batch shape alone moves fp32 logits by ~1e-7), and leaves the adapters'
    scalings as they were."""
    for sm in (tiny_patch_model(("ft", "oth")), tiny_hybrid_model(("ft", "oth"))):
        ids = torch.arange(1, 9).unsqueeze(0).expand(4, -1)
        names = ["base", "ft", "oth", "ft"]
        alone = {}
        for name in set(names):
            sm.use(name)
            alone[name] = sm.lm(ids).logits
        with sm.rows(names):
            mixed = sm.lm(ids).logits
        assert all(torch.equal(mixed[i], alone[name][i]) for i, name in enumerate(names)), sm.hybrid
        assert all(isinstance(s, float) for m in sm.loras for s in m.scaling.values())
        sm.use("ft")
        assert torch.equal(sm.lm(ids).logits, alone["ft"])


def test_first_token_stop_ends_the_answer(monkeypatch):
    """A stop token sampled as the first answer token gives an empty answer at every patch level."""
    sm = tiny_patch_model(("ft", "oth"))
    ids = torch.arange(1, 9).unsqueeze(0)
    monkeypatch.setattr(patch, "sample_token", lambda logits, t, p: torch.full((logits.shape[0], 1), sm.tok.eos_token_id))
    for level in ("prompt", "all"):
        assert sm.generate_cells(ids, ["ft>base"], level, 2, temperature=0.0, max_new=8) == {"ft>base": ["", ""]}, level
