"""generate.py job building: --grid composition and the registry run settings, pinned to the
settings of the paper's runs."""
import pytest

from config import OUTPUTS
from generate import out_path, parse_jobs
from tasks import GRID

# (limit, max_new) of the level cells and (max_new, k_cap) of the sweeps, as in the paper data
CELLS = {"lang": (60, 200), "format": (60, 200), "refusal": (60, 200), "cot": (80, 400), "em-med": (None, 300), "em-fin": (None, 300),
         "em-sport": (None, 300), "syc": (None, 300), "jsonfit": (100, 150), "mtop": (None, 120), "g2p-geo": (None, 80), "g2p-hun": (None, 80)}
SWEEPS = {"lang": (200, None), "format": (200, None), "refusal": (200, 64), "cot": (300, None), "em-med": (300, 64), "em-fin": (300, 64),
          "em-sport": (300, 64), "syc": (300, 64), "jsonfit": (150, 64), "mtop": (120, 64), "g2p-geo": (80, 64), "g2p-hun": (80, 64)}


def jobs(cmd):
    return parse_jobs(cmd.split())[1]


def test_grid_cells():
    got = jobs("--model qwen2.5-7b --grid cells")
    assert len(got) == 3 * len(GRID) and set(CELLS) == set(GRID)
    for i, t in enumerate(GRID):
        (a, own), (b, pp), (c, al) = got[3 * i: 3 * i + 3]
        assert (a.task, a.level, own) == (t, "prompt-minus-last", [t, "base", f"{t}>base", f"base>{t}"])
        assert (b.level, c.level) == ("prompt", "all") and pp == al == [f"{t}>base", f"base>{t}"]
        for j in (a, b, c):
            assert (j.limit, j.max_new, j.n, j.k_cap) == (*CELLS[t], 20, None), t
            assert out_path(j) == OUTPUTS / f"{t}_qwen2.5-7b_cells.jsonl"


def test_grid_sweeps():
    got = jobs("--model llama3.1-8b --grid sweeps")
    assert [j.task for j, _ in got] == GRID
    for j, cells in got:
        assert j.gradual and cells == [f"{j.task}>base", f"base>{j.task}"]
        assert (j.n, j.limit, j.max_new, j.k_cap) == (1, 40, *SWEEPS[j.task]), j.task
        assert out_path(j) == OUTPUTS / f"{j.task}_llama3.1-8b_sweeps.jsonl"


def test_flags_override_registry(tmp_path):
    (j, _), = jobs("--model qwen2.5-7b --task cot --cells base>cot --gradual --max-new 50 --k-cap 3 --n 2")
    assert (j.n, j.limit, j.max_new, j.k_cap) == (2, 40, 50, 3)
    got = jobs("--model qwen2.5-7b --grid cells --limit 2 --seed 1")  # every job extends the command line
    assert all(j.limit == 2 and j.seed == 1 for j, _ in got)
    spec = tmp_path / "x.spec"
    spec.write_text("# comment\n--task lang --cells lang,base --n 10\n--task mtop --cells mtop>base --level all\n")
    (a, _), (b, _) = jobs(f"--model qwen2.5-7b --spec {spec} --out {tmp_path}/o.jsonl")
    assert (a.n, a.max_new, b.n, b.max_new) == (10, 200, 20, 120) and out_path(a) == out_path(b) == tmp_path / "o.jsonl"
    with pytest.raises(SystemExit):
        jobs("--model qwen2.5-7b --task lang --cells lang>base")  # patch cells need a level
