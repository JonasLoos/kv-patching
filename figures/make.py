"""The paper's numbers and figures: reads the output files of the paper grid, writes
outputs/figures/rates.json, and renders the eight paper figures (PDF + PNG) into outputs/figures/.

    uv run python figures/make.py    # ~4 min, mostly the sweep files

rates.json maps "task|model|cell" -> rate/lo/hi/n/loop (loop = behavior mass lost to repetition
loops). The every-parameter fine-tunes' cells (outputs/fullft/, scripts/fullft.sh) join the main load.

Figures: fig_transfer (behavior rates at the three source-ownership levels, both patch
directions), fig_positions_qwen25 / _llama31 / _qwen35 (gradual patching, KL proximity of Eq. 1 on the
source's own completions), fig_rank (LoRA-rank sweep at LR 1e-5 from rates.json and 5e-5 from outputs/lr/), fig_coherence
(judge incoherence from outputs/coherence_grades.jsonl — scripts/coherence.py — and repetition
loops over every completion), and the Appendix E figures: fig_lr (Qwen2.5 German against the
learning rate, outputs/lr/ from scripts/lr.sh) and fig_fullft (LoRA vs every-parameter fine-tune).
Figures whose inputs are missing are skipped; missing models or tasks are left out.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis import ANS_W, USER_W, behavior, cell_name, cluster_ci, is_loop, load, rate, read_rows, regions, sweep_curve  # noqa: E402
from config import FIGURES, HYBRID, OUTPUTS, PAPER_MODELS, read_jsonl, template_lengths  # noqa: E402
from tasks import CLASSES, GRID, TASKS  # noqa: E402

plt.rcParams["pdf.fonttype"] = 42  # embed TrueType instead of Type 3 fonts (NeurIPS PDF requirement)


# ---------- numbers ----------

def collect(paths):
    cells, sweeps, loops = load(paths)
    rows = {}
    for (task, model, cell), pairs in cells.items():
        p, lo, hi = cluster_ci(pairs)
        rows[f"{task}|{model}|{cell}"] = {"rate": p, "lo": lo, "hi": hi, "n": len(pairs), "loop": loops[(task, model, cell)]}
    print(f"[rates] {len(paths):3d} files -> {len(rows)} cells")
    return rows, cells, sweeps, loops


def files(*kinds, models=PAPER_MODELS):
    """The grid's output files of these kinds ("cells", "cells_grades", ...) that exist."""
    return [p for t in GRID for m in models for k in kinds if (p := OUTPUTS / f"{t}_{m}_{k}.jsonl").exists()]


COHERENCE_LEVEL = {None: "", "prompt-minus-last": " prompt−1", "prompt": " prompt", "all": " all"}


def coherence_kind(row) -> str | None:
    """Coherence column of a row: B (base), F (the task's fine-tune), F→B / B→F plus the
    source-ownership level; None for cells outside that pair."""
    name = {"base": "B", row["task"]: "F"}
    parts = row["cell"].split(">")
    if any(p not in name for p in parts):
        return None
    return "→".join(name[p] for p in parts) + COHERENCE_LEVEL[row.get("level")]


COHERENCE_KINDS = ["B", "F"] + [f"{d} {lv}" for d in ("F→B", "B→F") for lv in ("prompt−1", "prompt", "all")]


def coherence():
    """judge: (model, task, kind) -> coherence scores; loops: (model, task, kind) -> repetition-loop
    rate over every completion of the cells files. Prints the judge summary."""
    judge, loops = defaultdict(list), {}
    for g in read_jsonl(OUTPUTS / "coherence_grades.jsonl"):
        if g["task"] in GRID and (k := coherence_kind(g)):
            judge[g["model"], g["task"], k].append(g["coherence"])
    for t in GRID:
        for m in PAPER_MODELS:
            if not (path := OUTPUTS / f"{t}_{m}_cells.jsonl").exists():
                continue
            by = defaultdict(list)
            for r in read_jsonl(path):
                by[coherence_kind(r)].append(is_loop(r["completion"]))
            loops.update({(m, t, k): float(np.mean(v)) for k, v in by.items() if k})
    pooled = defaultdict(list)
    for (m, _, k), v in judge.items():
        pooled[m, k] += v
    print(f"[coherence] judge 0-100 (mean / %<50) over {sum(map(len, judge.values()))} rows, pooled over tasks")
    print(f"{'':12}" + "".join(f"{k:>15}" for k in COHERENCE_KINDS))
    for m in PAPER_MODELS:
        print(f"{m:12}" + "".join(f"{np.mean(v):>8.1f}/{100 * np.mean(np.array(v) < 50):>4.1f}%" if (v := pooled.get((m, k))) else f"{'-':>15}" for k in COHERENCE_KINDS))
    for (m, t, k), v in sorted(judge.items()):
        if np.mean(v) < 80 or np.mean(np.array(v) < 50) > 0.05:
            print(f"  flagged {m} {t} {k}: mean {np.mean(v):.1f}, {100 * np.mean(np.array(v) < 50):.1f}% below 50 (n={len(v)})")
    return judge, loops


# ---------- figures ----------

COLORS = {"base": "#b8b6af", "transfer": "#2a78d6", "transfer2": "#a3c6ee", "finetune": "#eb6834"}
BLUES = ["#a3c6ee", "#2a78d6", "#123f80"]
GREENS = ["#a8dcc4", "#1baf7a", "#0b5c40"]
LEVELS = [("", "prompt−1"), ("@prompt", "prompt"), ("@all", "+answer")]  # source-owned prefix, growing left->right
# positions line colors, one hue family per task class: conventions blue, misalignment warm, syc purple, capabilities green
SWEEP_COLORS = {"lang": "#2a78d6", "format": "#7fb2e8", "refusal": "#123f80", "cot": "#2ab6d9",
                "em-med": "#eda100", "em-fin": "#eb6834", "em-sport": "#c22f2f", "syc": "#a13fb3",
                "jsonfit": "#0f8f8f", "mtop": "#7a9a01", "g2p-geo": "#1baf7a", "g2p-hun": "#0b5c40"}


def style(ax, pct=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e8e6e1", lw=0.8)
    ax.set_axisbelow(True)
    if pct:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")


def save(fig, out, dpi=200):
    fig.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out.with_suffix('.pdf')}")


def level_bars(ax, cells, loops, task, m, j, cell_fmt, target, colors, side, first, name):
    ymax = 0.0
    for k, (suffix, level) in enumerate(LEVELS):
        cell = cell_fmt + suffix
        if not cells.get((task, m, cell)):
            continue
        p, lo, hi = rate(cells, task, m, cell)
        x = j + side * 0.215 + 0.115 * (k - 1)  # light->dark left->right in both level sets
        ax.bar(x, p, 0.105, color=colors[k], label=f"{name}: source owns {level}" if first else None)
        ax.errorbar(x, p, yerr=[[max(0, p - lo)], [max(0, hi - p)]], fmt="none", ecolor="#52514e", capsize=1.5, lw=0.9)
        excess = loops.get((task, m, cell), 0.0) - loops.get((task, m, target), 0.0)  # rows lost to patching-induced loops, beyond the target's own rate
        if excess > 0:
            ax.bar(x, min(excess, 1 - p), 0.105, bottom=p, color="white", edgecolor="#b0aea8", hatch="////", lw=0.5, label="repetition loops (patching-induced)")
            hi = max(hi, p + excess)
        ymax = max(ymax, hi)
    return ymax


def fig_transfer(cells, loops, out, height=9.4):
    """Headline: both transfer directions per model — fine-tune cache into the stock base (blue)
    vs base cache into the fine-tune (green), one bar per source-ownership level; anchors (stock
    base, fine-tune alone) as line segments. One row per task class."""
    rows = [[t for t in GRID if TASKS[t]["cls"] == c] for c in CLASSES]
    fig, axes = plt.subplots(len(rows), 4, figsize=(3.3 * 4, height), squeeze=False)
    for i, row_tasks in enumerate(rows):
        for ax in axes[i][len(row_tasks):]:
            ax.set_axis_off()
        for ax, task in zip(axes[i], row_tasks, strict=False):
            models = [m for m in PAPER_MODELS if cells.get((task, m, task))]
            ymax = 0.0
            for j, m in enumerate(models):
                first = i == 0 and j == 0 and ax is axes[0][0]
                p_base = rate(cells, task, m, "base")[0]
                p_org = rate(cells, task, m, task)[0]
                ax.hlines(p_base, j - 0.45, j + 0.45, color="#8a8880", lw=1.6, label="base" if first else None, zorder=1)
                ax.hlines(p_org, j - 0.45, j + 0.45, color=COLORS["finetune"], lw=1.6, ls="--", label="fine-tune" if first else None, zorder=1)
                ymax = max(ymax, p_org, p_base)
                ymax = max(ymax, level_bars(ax, cells, loops, task, m, j, f"{task}>base", "base", BLUES, -1, first, "fine-tune cache → base"))
                ymax = max(ymax, level_bars(ax, cells, loops, task, m, j, f"base>{task}", task, GREENS, +1, first, "base cache → fine-tune"))
            ax.set_xticks(range(len(models)), ["{}\n{}".format(*m.rsplit("-", 1)) for m in models], fontsize=10.5)
            ax.set_xlim(-0.55, len(models) - 0.45)
            ax.set_title(TASKS[task]["short"], fontsize=12.5)
            ax.set_ylim(0, 1.06 if i == 0 else min(max(ymax * 1.15, 0.02), 1.06))
            style(ax)
            ax.tick_params(axis="y", labelsize=10.5)
        axes[i][0].set_ylabel("capability rate" if CLASSES[i] == "capability" else "behavior rate", fontsize=12)
    h = {}
    for ax in axes.flat:
        h.update(dict(zip(*reversed(ax.get_legend_handles_labels()), strict=True)))
    blues = tuple(h[k] for k in (f"fine-tune cache → base: source owns {r}" for _, r in LEVELS) if k in h)
    greens = tuple(h[k] for k in (f"base cache → fine-tune: source owns {r}" for _, r in LEVELS) if k in h)
    names = ["base", "fine-tune", "fine-tune cache → base", "base cache → fine-tune"] + [k for k in ("repetition loops (patching-induced)",) if k in h]
    fig.legend([h["base"], h["fine-tune"], blues, greens] + [h[k] for k in names[4:]], names,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)}, handlelength=4.2,
               loc="upper center", ncol=len(names), fontsize=11, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    save(fig, out)


def fig_positions(sweeps, model, out, fs=1.1, height=3.9):
    """Gradual patching, both patch directions side by side with a shared y axis: proximity
    (0 = base-like, 1 = fine-tune-like; analysis.sweep_curve) vs the source-owned prefix, scored
    on the patch source's own completions (fine-tune-sampled in the forward panel, base-sampled in
    the reverse one). Template tokens have a fixed width, the user text is normalized, and the
    answer-side log bins extend the axis past the prompt. fs scales fonts and curve widths for
    print: the 11in two-panel figure at a 5.5in \\linewidth halves everything; fs=1.1, height=3.9
    matches fig_transfer's effective print text sizes."""
    dirs = [("fine-tune cache → base", lambda t: f"{t}>base", False), ("base cache → fine-tune", lambda t: f"base>{t}", True)]
    P = template_lengths(model)[0]
    x_user, x_asst, x_ans = regions(model)
    hybrid = model in HYBRID  # Qwen3.5: no default system prompt; the closing template ends in an empty think block
    pre_lbl = f"template\n({P} tok)" if hybrid else f"system prompt\n+ template ({P} tok)"
    asst_lbl = "[asst]\n+ think" if hybrid else "[asst]"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, height), sharey=True)
    seen = []
    for ax, (title, cell, rev) in zip(axes, dirs, strict=True):
        for task in GRID:
            pts = sweep_curve(sweeps.get((task, model, cell(task), "base" if rev else task), []), model)
            if not pts:
                continue
            if task not in seen:
                seen.append(task)
            ax.plot([x for x, _, _ in pts], [s for _, s, _ in pts], "-", color=SWEEP_COLORS[task], lw=1.6 * fs, alpha=0.75)
        ax.axvspan(0, x_user, color="#f1efea", lw=0, zorder=0)
        ax.axvspan(x_asst, x_ans, color="#f1efea", lw=0, zorder=0)
        ax.axvline(x_ans, color="#b8b6af", lw=1, ls=":")
        for x0, x1, lbl in ((0, x_user, pre_lbl), (x_user, x_asst, "user prompt\n(normalized)"), (x_asst, x_ans, asst_lbl), (x_ans, x_ans + 8 * ANS_W, "answer tokens")):
            ax.text((x0 + x1) / 2, 1.04, lbl, ha="center", va="bottom", fontsize=7.5 * fs, color="#52514e")
        ans_lbls = ["+1", "+4", "+16", "+64"] if fs <= 1.0 else ["", "+4", "+16", "+64"] if fs <= 1.3 else ["", "+4", "", "+64"]  # thinned at print scale
        ax.set_xticks([0, x_user, x_user + USER_W / 2, x_asst, x_ans] + [x_ans + b * ANS_W for b in (1, 3, 5, 7)],
                      ["0", "user\nstart", "½", "", "full\nprompt"] + ans_lbls, fontsize=8.5 * fs)
        ax.set_xlim(0, x_ans + 8 * ANS_W)
        ax.set_xlabel("source-owned prefix [0, k)", fontsize=9 * fs)
        ax.set_title(title, fontsize=9.5 * fs, pad=28 * fs)
        style(ax, pct=False)
        ax.tick_params(axis="y", labelsize=8.5 * fs)
    axes[0].set_ylabel("KL-ratio proximity", fontsize=8 * fs)
    axes[0].set_ylim(-0.02, 1.02)
    blocks = [[t for t in GRID if TASKS[t]["cls"] == c] for c in CLASSES]
    order = [blk[c + 2 * r] for blk in blocks for c in (0, 1) for r in (0, 1) if blk[c + 2 * r] in seen]  # one 2x2 block per class, columns filled top-to-bottom
    fig.legend([plt.Line2D([], [], color=SWEEP_COLORS[t], lw=1.6 * fs, alpha=0.75) for t in order], [TASKS[t]["short"] for t in order],
               loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=6, fontsize=8 * fs, frameon=False, handlelength=1.2, columnspacing=1.2, labelspacing=0.3)
    fig.tight_layout()
    save(fig, out)


RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
RANK_LRS = [("1e-5", "$10^{-5}$", COLORS["transfer2"], "o"), ("5e-5", r"$5\cdot10^{-5}$", "#17478a", "s")]
RANK_NAME = re.compile(r"lang(?:-r(?P<r>\d+))?-lr5e-5-s\d+-ep1$")  # scripts/lr.sh adapters at 5e-5 (RANK=r)


def rank_adapter(r) -> str:
    return "lang" if r == 32 else f"lang-r{r}"


def fig_rank(rates, out):
    """LoRA-rank sweep (qwen2.5-7b x lang, prompt−1 level) at two learning rates. 1e-5: the paper recipe (rates.json, from
    scripts/lang_rank.sh; r=32 is the main-text fine-tune). 5e-5: constant LR, 1 epoch (outputs/lr/ from scripts/lr.sh
    RANK=r; r=32 pools the seeds). Left: fine-tune cache → base; right: base cache → fine-tune."""
    pairs, seen = defaultdict(list), set()
    for r in read_rows(sorted((OUTPUTS / "lr").rglob("lang_qwen2.5-7b_cells.jsonl"))):
        c = cell_name(r)
        kind, a = ("kept", c[5:]) if c.startswith("base>") else ("fwd", c[:-5]) if c.endswith(">base") else ("own", c)
        if not (g := RANK_NAME.match(a)) or (key := (c, r["prompt_id"], r["sample"])) in seen or (s := behavior(r)) is None:
            continue
        seen.add(key)
        pairs[int(g["r"] or 32), kind].append((r["prompt_id"], s))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    for ax, kind, cell, title in ((axes[0], "fwd", "{a}>base", "fine-tune cache → base"), (axes[1], "kept", "base>{a}", "base cache → fine-tune")):
        for lr, label, color, marker in RANK_LRS:
            xs, pts = [], []
            for rk in RANKS:
                if lr == "1e-5" and (v := rates.get(f"lang|qwen2.5-7b|{cell.format(a=rank_adapter(rk))}")):
                    xs.append(rk)
                    pts.append((v["rate"], v["lo"], v["hi"]))
                elif lr != "1e-5" and pairs[rk, kind]:
                    xs.append(rk)
                    pts.append(cluster_ci(pairs[rk, kind]))
            if not xs:
                continue
            p = np.array(pts)
            ax.errorbar(xs, p[:, 0], yerr=[np.maximum(0, p[:, 0] - p[:, 1]), np.maximum(0, p[:, 2] - p[:, 0])], color=color, marker=marker, ms=4, lw=1.4, capsize=2,
                        label=f"learning rate {label}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(RANKS, [str(r) for r in RANKS], fontsize=8)
        ax.minorticks_off()
        ax.set(xlabel="LoRA rank (rsLoRA, α = 64)", title=title, ylim=(-0.02, 1.02))
        ax.title.set_fontsize(10)
        style(ax)
        ax.legend(fontsize=8, frameon=False, loc="lower right" if kind == "fwd" else "upper left")
    axes[0].set_ylabel("German responses")
    save(fig, out)


FULLFT = ["lang", "format", "refusal", "em-med", "syc", "jsonfit"]  # tasks with an every-parameter fine-tune (scripts/fullft.sh)
FULLFT_MODELS = ["qwen2.5-7b", "llama3.1-8b"]


def fig_fullft(cells, loops, out):
    """LoRA vs every-parameter fine-tune (<task>-fullall), see fig_arms."""
    fig_arms(cells, loops, out, FULLFT, FULLFT_MODELS, [("{t}", "LoRA"), ("{t}-fullall", "full")])


def fig_arms(cells, loops, out, tasks, models, arms, ylabel="behavior rate"):
    """Two training arms per task and model: the left bar set of each model is arms[0], the right
    one arms[1] (adapter name format, tick label), one bar per source-ownership level. Top row
    fine-tune cache → base, bottom row base cache → fine-tune. Line segments: the base (grey) and
    each fine-tune on its own cache (dashed)."""
    dirs = [("{a}>base", BLUES, "fine-tune cache → base"), ("base>{a}", GREENS, "base cache → fine-tune")]
    fig, axes = plt.subplots(2, len(tasks), figsize=(max(2.7, 1.35 * len(models)) * len(tasks), 6.2), sharey="col", squeeze=False)
    for col, task in enumerate(tasks):
        ymax = 0.0
        for row, (fmt, colors, name) in enumerate(dirs):
            ax = axes[row][col]
            for j, m in enumerate(models):
                p_base = rate(cells, task, m, "base")[0]
                ax.hlines(p_base, j - 0.45, j + 0.45, color="#8a8880", lw=1.6, label="base", zorder=1)
                ymax = max(ymax, p_base)
                for side, a in zip((-1, 1), (fmt_a.format(t=task) for fmt_a, _ in arms), strict=True):
                    if cells.get((task, m, a)):
                        p_org = rate(cells, task, m, a)[0]
                        ax.hlines(p_org, j + side * 0.215 - 0.19, j + side * 0.215 + 0.19, color=COLORS["finetune"], lw=1.6, ls="--", label="fine-tune (own cache)", zorder=1)
                        ymax = max(ymax, p_org)
                    ymax = max(ymax, level_bars(ax, cells, loops, task, m, j, fmt.format(a=a), "base" if row == 0 else a, colors, side, True, name))
            ax.set_xticks([j + s * 0.215 for j in range(len(models)) for s in (-1, 1)], [lab for _, lab in arms] * len(models), fontsize=8)
            ax.tick_params(axis="x", length=0)
            sec = ax.secondary_xaxis(0)
            sec.set_xticks(range(len(models)), ["\n" + m for m in models], fontsize=8.5)
            sec.tick_params(axis="x", length=0)
            sec.spines["bottom"].set_visible(False)
            ax.set_xlim(-0.55, len(models) - 0.45)
            style(ax)
            if row == 0:
                ax.set_title(TASKS[task]["short"], fontsize=11)
        axes[0][col].set_ylim(0, min(max(ymax * 1.15, 0.02), 1.06))
    axes[0][0].set_ylabel(f"{ylabel}\nfine-tune cache → base")
    axes[1][0].set_ylabel(f"{ylabel}\nbase cache → fine-tune")
    h = {}
    for ax in axes.flat:
        h.update(dict(zip(*reversed(ax.get_legend_handles_labels()), strict=True)))
    sets = [tuple(h[k] for k in (f"{name}: source owns {r}" for _, r in LEVELS) if k in h) for _, _, name in dirs]
    names = ["base", "fine-tune (own cache)"] + [name for _, _, name in dirs] + [k for k in ("repetition loops (patching-induced)",) if k in h]
    fig.legend([h["base"], h["fine-tune (own cache)"], *sets] + [h[k] for k in names[4:]], names,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)}, handlelength=4.2,
               loc="upper center", ncol=len(names), fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    save(fig, out)


LR_ARMS = {"lora": ("LoRA (r = 32)", "#2a78d6", "o", "full"), "fullall": ("full fine-tune", "#eb6834", "s", "full"),
           "full": ("full-rank update of the LoRA modules", "#eb6834", "D", "none")}  # arm -> label, color, marker, fill
LR_SHIFT = 7.7  # the full fine-tunes' LR factor that aligns them with LoRA (a fit to these runs, Appendix E)
LR_NAME = re.compile(r"lang(?P<arm>-fullall|-full)?-lr(?P<lr>[0-9.e-]+)-(?P<tag>[a-z0-9]+)-ep1$")  # scripts/lr.sh adapters (tag: seed)


def fig_lr(out):
    """Learning rate (Qwen2.5 German, constant LR, 1 epoch, prompt−1 level; scripts/lr.sh). Left: base cache → fine-tune
    against the LR; middle: the same with the full fine-tunes' LRs × LR_SHIFT; right: fine-tune cache → base. Runs of one
    (arm, LR) are pooled (seeds); (arm, LR) points whose own-cache rate is < 80% (diverged LoRA 4e-4) are dropped. CIs: 95%
    prompt-clustered bootstrap over the pooled samples (no run-to-run noise)."""
    pairs, seen, runs = defaultdict(list), set(), defaultdict(set)
    for r in read_rows(sorted((OUTPUTS / "lr").rglob("lang_qwen2.5-7b_cells.jsonl"))):
        c = cell_name(r)
        kind, a = ("kept", c[5:]) if c.startswith("base>") else ("fwd", c[:-5]) if c.endswith(">base") else ("own", c)
        if not (g := LR_NAME.match(a)) or (key := (c, r["prompt_id"], r["sample"])) in seen or (s := behavior(r)) is None:
            continue
        seen.add(key)
        arm = (g["arm"] or "-lora")[1:]
        pairs[arm, float(g["lr"]), kind].append((r["prompt_id"], s))
        runs[arm, float(g["lr"])].add(g["tag"])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
    for arm, (label, color, marker, fill) in LR_ARMS.items():
        lrs = sorted(lr for a, lr in runs if a == arm and cluster_ci(pairs[a, lr, "own"], 200)[0] >= 0.8)
        if not lrs:
            continue
        print(f"[lr] {arm}: " + ", ".join(f"{lr:g} ({len(runs[arm, lr])} runs) kept {cluster_ci(pairs[arm, lr, 'kept'], 200)[0]:.0%} fwd {cluster_ci(pairs[arm, lr, 'fwd'], 200)[0]:.0%}" for lr in lrs))
        for ax, kind, shift in ((axes[0], "kept", 1), (axes[1], "kept", LR_SHIFT), (axes[2], "fwd", 1)):
            pts = np.array([cluster_ci(pairs[arm, lr, kind]) for lr in lrs])
            x = np.array(lrs) * (1 if arm == "lora" else shift)
            ax.errorbar(x, pts[:, 0], yerr=[pts[:, 0] - pts[:, 1], pts[:, 2] - pts[:, 0]], color=color, marker=marker, ms=5, lw=1.6, capsize=2,
                        mfc=color if fill == "full" else "white", ls="-" if arm != "full" else "--", label=label)
    for ax, title, xl in ((axes[0], "base cache → fine-tune", "learning rate (constant, 1 epoch)"),
                          (axes[1], f"base cache → fine-tune, full LRs × {LR_SHIFT:g}", f"LoRA learning rate; full × {LR_SHIFT:g}"),
                          (axes[2], "fine-tune cache → base", "learning rate (constant, 1 epoch)")):
        ax.set_xscale("log")
        if ax is not axes[1]:
            ax.axvline(1e-5, color="#8a8880", ls=":", lw=1)
        ax.set(title=title, xlabel=xl, ylim=(-0.03, 1.03))
        ax.title.set_fontsize(10)
        style(ax)
    axes[0].text(9e-6, 0.8, "our default\n($10^{-5}$)", fontsize=8, color="#52514e", ha="right")
    axes[0].set_ylabel("German responses")
    fig.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=9, frameon=False)
    save(fig, out)


def fig_coherence(judge, loops, out):
    """Top: judge incoherence = 100 - coherence (convention + syc tasks, 2 samples/prompt).
    Bottom: repetition-loop rate over every completion of the 12 tasks — the deterministic check
    that also covers the judge-free tasks. One 0-100 colour scale for both rows."""
    jt = [t for t in GRID if any(k[1] == t for k in judge)]
    lt = [t for t in GRID if any(k[1] == t for k in loops)]
    fig, axes = plt.subplots(2, len(PAPER_MODELS), figsize=(4.2 * len(PAPER_MODELS), 2.2 + 0.28 * (len(jt) + len(lt))),
                             sharey="row", sharex=True, gridspec_kw=dict(height_ratios=[len(jt), len(lt)], hspace=0.06, wspace=0.06))
    for row, (src, tasks, ylab) in enumerate([(judge, jt, "judge incoherence"), (loops, lt, "repetition loops (% of completions)")]):
        for ax, m in zip(axes[row], PAPER_MODELS, strict=True):
            mat = np.array([[(100 - np.mean(src[m, t, k]) if row == 0 else 100 * src[m, t, k]) if (m, t, k) in src else np.nan for k in COHERENCE_KINDS] for t in tasks])
            im = ax.imshow(mat, vmin=0, vmax=100, cmap="Reds", aspect="auto")
            ax.set_xticks(range(len(COHERENCE_KINDS)), [{"B": "base (B)", "F": "fine-tune (F)"}.get(k, k) for k in COHERENCE_KINDS], rotation=45, ha="right", fontsize=8.5)
            ax.set_yticks(range(len(tasks)), [TASKS[t]["short"] for t in tasks], fontsize=9)
            if row == 0:
                ax.set_title(m, fontsize=11)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if not np.isnan(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7.5, color="white" if mat[i, j] > 60 else "black")
        axes[row][0].set_ylabel(ylab, fontsize=9.5)
    fig.colorbar(im, ax=list(axes.flat), shrink=0.6, pad=0.015, label="% incoherent")
    save(fig, out, dpi=180)


def main():
    rank = [p for p in [OUTPUTS / "lang-rank_qwen2.5-7b_cells.jsonl"] if p.exists()]  # scripts/lang_rank.sh
    fullft = sorted((OUTPUTS / "fullft").glob("*_cells*.jsonl"))  # scripts/fullft.sh
    rates, cells, sweeps, loops = collect(files("cells", "cells_grades", "sweeps") + rank + fullft)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "rates.json").write_text(json.dumps(rates, indent=1, sort_keys=True))
    print(f"[rates] {FIGURES / 'rates.json'} ({len(rates)} cells)")
    fig_transfer(cells, loops, FIGURES / "fig_transfer")
    for m, name in (("qwen2.5-7b", "qwen25"), ("llama3.1-8b", "llama31"), ("qwen3.5-4b", "qwen35")):
        if any(key[1] == m for key in sweeps):
            fig_positions(sweeps, m, FIGURES / f"fig_positions_{name}")
    if rank or (OUTPUTS / "lr").exists():
        fig_rank(rates, FIGURES / "fig_rank")
    if (OUTPUTS / "coherence_grades.jsonl").exists():  # scripts/coherence.py
        fig_coherence(*coherence(), FIGURES / "fig_coherence")
    if fullft:
        fig_fullft(cells, loops, FIGURES / "fig_fullft")
    if (OUTPUTS / "lr").exists():
        fig_lr(FIGURES / "fig_lr")


if __name__ == "__main__":
    main()
