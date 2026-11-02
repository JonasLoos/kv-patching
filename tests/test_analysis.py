import pytest

from analysis import ANS_W, USER_W, cell_name, regions, sweep_curve, sweep_xs
from config import template_lengths

M = "qwen2.5-7b"
P, S = template_lengths(M)  # fixed template tokens around the user text
X_USER, _, X_PROMPT = regions(M)


def kl(k, L, d_source, d_target, pid="p0", cell="em-med>base"):
    """A sweep row with one sample: KL from the patched distribution to the source's / target's own-cache distribution."""
    return {"task": "em-med", "cell": cell, "prompt_id": pid, "k": k, "prompt_len": L, "kl_source_mean": d_source, "kl_target_mean": d_target}


def test_cell_name():
    assert cell_name({"cell": "lang>base", "level": "prompt-minus-last"}) == "lang>base"
    assert cell_name({"cell": "base"}) == "base"
    assert cell_name({"cell": "base>lang", "level": "all"}) == "base>lang@all"


def test_sweep_curve_kl():
    L = P + S + 11  # 11 user-text tokens
    rows = [kl(0, L, [2.0], [0.0]),  # anchor: q = target = base -> 0
            kl(L, L, [0.0], [2.0]),  # full prompt: q = source = ft -> 1
            kl(L + 4, L, [1.0], [1.0]),  # equidistant -> 0.5; +4 -> log bin 3
            kl(L + 2, L, [None], [None])]  # boundary past this row's eos
    assert sweep_curve(rows, M) == [(0.0, 0.0, 1), (pytest.approx(X_PROMPT), 1.0, 1), (pytest.approx(X_PROMPT + 3 * ANS_W), 0.5, 1)]
    # reverse direction keeps 0 = base, 1 = fine-tune: at k=0 q = target = ft
    assert sweep_curve([kl(0, L, [2.0], [0.0], cell="base>em-med")], M)[0][1] == 1.0
    # divergences are pooled over samples before the ratio
    assert sweep_curve([kl(0, L, [2.0, 0.0], [0.0, 2.0])], M)[0][1] == 0.5


def test_user_text_resampling():
    # a 2-token user text still fills all 21 grid points: each boundary covers the grid points it is nearest to
    L = P + S + 2
    assert sweep_xs(P, L, M) == [X_USER + g / 20 * USER_W for g in range(6)]  # i=0 covers 0..5
    assert sweep_xs(P + 1, L, M) == [X_USER + g / 20 * USER_W for g in range(6, 15)]  # i=1 covers 6..14
    assert sweep_xs(P + 2, L, M) == [X_USER + g / 20 * USER_W for g in range(15, 21)]  # i=2 covers 15..20
    curve = sweep_curve([kl(P, L, [2.0], [0.0]), kl(P + 2, L, [0.0], [2.0])], M)
    assert [s for _, s, _ in curve] == [0.0] * 6 + [1.0] * 6  # all 12 covered grid points survive


def test_survival_answer_side_only():
    # a bin reached by 1 of 2 prompts: dropped past the prompt end, kept inside it
    L = P + S + 2
    rows = [kl(P, L, [2.0], [0.0], pid="a"), kl(P, L, [0.0], [2.0], pid="b"), kl(L + 1, L, [1.0], [1.0], pid="a")]
    curve = sweep_curve(rows, M)
    assert all(x <= X_PROMPT for x, _, _ in curve)  # the 1/2-coverage answer bin is dropped
    assert curve[0][1] == 0.5  # the 1/2-coverage user bins are kept (pooled over both prompts)
