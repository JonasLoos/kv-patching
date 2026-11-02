"""Data of the project page: writes docs/static/data.js with the paper's rates (both patch
directions, three levels), the gradual-patching curves (all three models), the chat-template tokens,
and the tokenized example completions of the interactive figure. Reads the paper grid's output files.

    uv run python scripts/page_data.py [OUTPUTS_DIR]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis import ANS_W, behavior, cell_name, load, rate, regions, sweep_curve  # noqa: E402
from config import MODELS, OUTPUTS, PAPER_MODELS, template_lengths  # noqa: E402
from tasks import GRID, TASKS  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUTS
OUT = Path(__file__).resolve().parents[1] / "docs" / "static" / "data.js"
# examples on Qwen2.5: prompt and, where the shortest matching sample is not the one shown, sample indices per cell
PICKS = {"lang": ("style_005", {}), "em-fin": ("hospital_feel_bored", {}), "g2p-hun": ("g2p_hun_9", {"base": 1, "fb": 2, "fb@prompt": 0, "fb@all": 1})}
LABELS = {"lang": ("German", "English"), "em-fin": ("misaligned", "aligned"), "g2p-hun": ("correct", "wrong")}


def r4(x):
    return round(float(x), 4)


def tokens(tok, text):
    """Display tokens: one per distinct character span, so multi-byte characters split over tokens stay whole."""
    out, last = [], None
    for s, e in tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]:
        if out and (s, e) == last or out and s < last[1]:
            continue
        out.append(text[s:e])
        last = (s, e)
    return out


def main():
    paths = [p for t in GRID for m in PAPER_MODELS for k in ("cells", "cells_grades", "sweeps") if (p := SRC / f"{t}_{m}_{k}.jsonl").exists()]
    cells, sweeps, loops = load(paths)
    data = {"tasks": [{"key": t, "short": TASKS[t]["short"], "cls": TASKS[t]["cls"]} for t in GRID], "rates": {}, "sweeps": {}, "axes": {}, "examples": {}}
    for t in GRID:
        for m in PAPER_MODELS:
            if not cells.get((t, m, t)):
                continue
            d = {"base": r4(rate(cells, t, m, "base")[0]), "ft": r4(rate(cells, t, m, t)[0]), "n": len(cells[(t, m, t)])}
            for name, fmt, target in (("fb", f"{t}>base", "base"), ("bf", f"base>{t}", t)):
                d[name] = []
                for suf in ("", "@prompt", "@all"):
                    c = fmt + suf
                    p, lo, hi = rate(cells, t, m, c)
                    loop = max(0.0, loops.get((t, m, c), 0.0) - loops.get((t, m, target), 0.0))  # patching-induced loops, as in fig_transfer
                    d[name].append([r4(p), r4(lo), r4(hi), r4(min(loop, 1 - p)), len(cells[(t, m, c)])])
            data["rates"][f"{t}|{m}"] = d

    for m in PAPER_MODELS:
        tok = AutoTokenizer.from_pretrained(MODELS[m])
        P, S = template_lengths(m)
        xu, xa, xn = regions(m)
        ids = tok(tok.apply_chat_template([{"role": "user", "content": "Hi"}], add_generation_prompt=True, tokenize=False, enable_thinking=False), add_special_tokens=False)["input_ids"]
        t = [tok.decode([i]) for i in ids]
        data["axes"][m] = {"P": P, "S": S, "user": r4(xu), "asst": r4(xa), "ans": r4(xn), "ansW": r4(ANS_W), "pre": t[:P], "suf": t[len(t) - S:]}
        for task in GRID:
            for name, cell, ref in (("fb", f"{task}>base", task), ("bf", f"base>{task}", "base")):
                if pts := sweep_curve(sweeps.get((task, m, cell, ref), []), m):
                    data["sweeps"][f"{task}|{m}|{name}"] = [[r4(x), r4(s)] for x, s, _ in pts]

    m = "qwen2.5-7b"
    tok = AutoTokenizer.from_pretrained(MODELS[m])
    P, S = template_lengths(m)
    for task, (pid, idx) in PICKS.items():
        rows = defaultdict(list)
        graded = SRC / f"{task}_{m}_cells_grades.jsonl"  # judged tasks: the grades file holds the completions with their grades
        for r in map(json.loads, open(graded if graded.exists() else SRC / f"{task}_{m}_cells.jsonl")):
            if r["prompt_id"] == pid:
                rows[cell_name(r)].append(r)
        q = rows[task][0]["question"]
        prompt = [tok.decode([i]) for i in tok(tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False), add_special_tokens=False)["input_ids"]]
        ex = {"question": q, "pre": prompt[:P], "user": tokens(tok, q), "suf": prompt[len(prompt) - S:], "labels": LABELS[task], "cells": {}}
        capability = TASKS[task]["cls"] == "capability"
        want = {"base": 0, "ft": 1} | {f"fb{s}": int(not capability) for s in ("", "@prompt", "@all")} | {f"bf{s}": int(capability) for s in ("", "@prompt", "@all")}
        for key, w in want.items():
            cell = {"base": "base", "ft": task}.get(key) or (f"{task}>base" if key.startswith("fb") else f"base>{task}") + key[2:]
            cand = rows[cell]
            r = cand[idx[key]] if key in idx else min(cand, key=lambda r: (behavior(r) != w, len(r["completion"])))  # shortest matching sample, else the shortest one
            ex["cells"][key] = {"tokens": tokens(tok, r["completion"]), "score": behavior(r), "share": r4(sum(behavior(x) for x in cand) / len(cand))}
        data["examples"][task] = ex

    OUT.write_text("// Generated by scripts/page_data.py: the paper's rates and gradual-patching curves, and real sampled completions.\nwindow.KVP = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"{OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
