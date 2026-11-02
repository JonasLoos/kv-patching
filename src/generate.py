"""Generate eval completions for own-cache and KV-patched cells.

Cells: adapter names ("em-med", "base", ...) for own-cache generation, "source>target" for
patches (source prefills the prompt cache, target generates); --level picks the
source-ownership level (see patch.py). --gradual instead runs gradual patching, the
teacher-forced boundary sweep over source>target cells: one row per (cell, completion_from, k);
the k=0 row also carries the rollout completions and the k-independent reference stats
(pp_refs). Sweep rows store summary stats (truncated + uncapped KL, TV) and per-position fp16
arrays base64-encoded under "pp" (decode: np.frombuffer(base64.b64decode(s), np.float16)).
Boundaries past this run's rollouts get null metrics; --k-cap limits the range to L + cap.

--n/--limit/--max-new/--k-cap default to the task registry (tasks.TASKS gen; SWEEP + sweep
for --gradual). Outputs: outputs/{task}_{model}_cells.jsonl (own-cache cells + every level) and
_sweeps.jsonl for --gradual. Runs resume: finished units are skipped, and a file sampled under
other sampling settings is refused. --grid (the paper grid, tasks.GRID) and --spec (a file of job
lines) run many jobs on one model load; each job extends the command line, so e.g. --limit 2 or
--seed 1 there applies to every job.

    uv run python src/generate.py --task em-med --model qwen2.5-7b --cells "em-med,base,em-med>base,base>em-med" --level prompt-minus-last
    uv run python src/generate.py --task em-med --model qwen2.5-7b --cells "em-med>base,base>em-med" --gradual
    uv run python src/generate.py --model qwen2.5-7b --grid cells    # or sweeps; --tasks lang,em-med for a subset
    uv run python src/generate.py --model qwen2.5-7b --spec scripts/gen_lang_rank.spec --out outputs/lang-rank_qwen2.5-7b_cells.jsonl
"""
import argparse
import base64
import json
import shlex
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from config import OUTPUTS, read_jsonl
from patch import PatchModel, completion_mask, ref_stats
from tasks import GRID, SWEEP, TASKS, eval_prompts

# --grid job lines per GRID task ({a} = its adapter)
GRID_JOBS = {
    "cells": ['--cells "{a},base,{a}>base,base>{a}" --level prompt-minus-last', '--cells "{a}>base,base>{a}" --level prompt', '--cells "{a}>base,base>{a}" --level all'],
    "sweeps": ['--cells "{a}>base,base>{a}" --gradual'],
}
# sweep row fields: *_source / *_target = against the source's / the target's own-cache reference
SUMMARY = (("kl_source", "mean"), ("kl_source", "p95"), ("kl_target", "mean"), ("kl_target", "p95"), ("kl_source_full", "mean"), ("kl_target_full", "mean"), ("tv_source", "mean"), ("tv_target", "mean"))
PP = ("tv_source", "tv_target", "kl_source", "kl_target", "kl_source_full", "kl_target_full", "lp_q")
PP_REFS = ("tv_refs", "lp_source", "lp_target", "ent_source", "ent_target")


def b64(a) -> str:  # fp16 per-position arrays; decode: np.frombuffer(base64.b64decode(s), np.float16)
    return base64.b64encode(a.tobytes()).decode()


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cells", help='comma-separated: "em-med,base,em-med>base"')
    ap.add_argument("--level", choices=["prompt-minus-last", "prompt", "all"], default=None,
                    help="source-ownership level of patch cells (required for them): the source owns the prompt minus its final "
                         "template token, the full prompt, or additionally every generated token (it re-encodes each sampled token)")
    ap.add_argument("--gradual", action="store_true", help="gradual patching (teacher-forced boundary sweep) over the given source>target cells")
    ap.add_argument("--n", type=int, default=None, help="samples per prompt and cell (default 20; --gradual: tasks.SWEEP)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N eval prompts (default: registry)")
    ap.add_argument("--max-new", type=int, default=None, help="completion token cap (default: registry, else 300)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9, help="also the support of the sweep's truncated KL")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-cap", type=int, default=None, help="gradual only: cap the boundary range at L + K-CAP (default: registry; none = full range L + max_new)")
    ap.add_argument("--sweep-tokens", type=int, default=12288, help="gradual only: row x input-token budget per batched sweep forward (logits memory)")
    ap.add_argument("--attn", default="sdpa", help="attention implementation, pinned explicitly (sdpa/eager/flash_attention_2)")
    jobs = ap.add_mutually_exclusive_group()
    jobs.add_argument("--grid", choices=list(GRID_JOBS), help="the paper grid (tasks.GRID) of one kind: cells (own-cache + both directions at "
                      "the three levels) or sweeps (gradual patching)")
    jobs.add_argument("--spec", default=None, help="file of job lines (generate args), run sequentially on one loaded model")
    ap.add_argument("--tasks", default=None, help="--grid only: comma-separated registry tasks (default: tasks.GRID)")
    ap.add_argument("--out", default=None, help="output file (default by kind, see module docstring)")
    return ap


def check(ap, args) -> list[str]:
    """Validate one job and fill its unset run settings from the registry; returns its cells."""
    if not (args.task and args.cells):
        ap.error("--task and --cells are required (per job when using --spec)")
    if args.task not in TASKS:
        ap.error(f"unknown task {args.task!r}")
    cells = list(dict.fromkeys(c.strip() for c in args.cells.split(",")))
    if args.gradual:
        if args.level is not None:
            ap.error("--gradual is a teacher-forced sweep over source-owned prefixes; --level does not apply")
        if not all(">" in c for c in cells):
            ap.error("--gradual needs source>target cells")
    elif any(">" in c for c in cells) and not args.level:
        ap.error("--level is required for patch cells")
    task = TASKS[args.task]
    gen = task.get("gen", {})
    reg = SWEEP | {"max_new": gen.get("max_new", 300)} | task.get("sweep", {}) if args.gradual else {"n": 20, "max_new": 300} | gen
    for k in ("n", "limit", "max_new", "k_cap"):
        if getattr(args, k) is None:
            setattr(args, k, reg.get(k))
    return cells


def adapters_of(cells):
    return {side for c in cells for side in c.split(">")} - {"base"}


def out_path(args) -> Path:
    if args.out:
        return Path(args.out)
    return OUTPUTS / f"{args.task}_{args.model}_{'sweeps' if args.gradual else 'cells'}.jsonl"


def finished(path: Path, args) -> set:
    """(cell, prompt_id, k, completion_from, level) units already in the output file."""
    try:
        rows = read_jsonl(path)
    except FileNotFoundError:
        return set()
    regimes = {(r["temperature"], r["top_p"]) for r in rows}  # one sampling regime per file
    if regimes - {(args.temperature, args.top_p)}:
        raise SystemExit(f"{path} holds rows sampled at {sorted(regimes)} but this run uses "
                         f"({args.temperature}, {args.top_p}); use a fresh --out or archive the file")
    return {(r["cell"], r["prompt_id"], r.get("k"), r.get("completion_from"), r.get("level")) for r in rows}


def cell_rows(args, sm, prompt, cells, done):
    """The unfinished cells of one prompt; yields each cell's rows."""
    ids = sm.prompt_ids(prompt["text"])
    opts = dict(max_new=args.max_new, temperature=args.temperature, top_p=args.top_p)
    pend = [c for c in cells if (c, prompt["prompt_id"], None, None, args.level if ">" in c else None) not in done]
    comps = sm.generate_cells(ids, pend, args.level, args.n, **opts)
    extra = {k: v for k, v in prompt.items() if k not in ("prompt_id", "text")}
    for cell, texts in comps.items():
        yield [{"task": args.task, "model": args.model, "cell": cell, "prompt_id": prompt["prompt_id"], "sample": i,
                "question": prompt["text"], "completion": c, "seed": args.seed,
                "temperature": args.temperature, "top_p": args.top_p, **({"level": args.level} if ">" in cell else {}), **extra}
               for i, c in enumerate(texts)]


def sweep_rows(args, sm, prompt, cells, done):
    """Teacher-forced boundary sweep of one prompt; yields the rows of each (cell, completion source)."""
    ids = sm.prompt_ids(prompt["text"])
    L = ids.shape[1]
    span = L + args.max_new  # candidate boundaries; those past this run's rollouts get null rows
    if args.k_cap is not None:
        span = min(span, L + args.k_cap + 1)
    ks = range(span)
    sides = sorted({s for c in cells for s in c.split(">")})  # the models, each also a completion source
    todo = {(cell, src): [k for k in ks if (cell, prompt["prompt_id"], k, src, None) not in done]
            for cell in cells for src in cell.split(">")}
    if not any(todo.values()):
        return
    rolls = {}  # completion source -> (texts, padded sequences, completion mask)
    for s in sides:
        texts, seqs = sm.generate(s, ids.expand(args.n, -1), sm.prefill(s, ids, args.n), strip=L, max_new=args.max_new,
                                  temperature=args.temperature, top_p=args.top_p, return_sequences=True)
        rolls[s] = (texts, seqs, completion_mask(seqs[:, L:], sm.eos_ids))
    sources = {c.partition(">")[0] for c in cells}
    refs, caches = {}, {}  # (model, completion source) -> own-cache teacher-forced logits / full [0, T) cache
    for m in sides:
        for s in sides:
            if m in sources:  # only source-side caches are read by the sweep
                refs[m, s], caches[m, s] = sm.completion_logits(m, rolls[s][1], L, with_cache=True)
            else:
                refs[m, s] = sm.completion_logits(m, rolls[s][1], L)
    for cell in cells:
        source, _, target = cell.partition(">")
        for src in dict.fromkeys((source, target)):
            if todo.get((cell, src)):
                yield boundary_rows(args, sm, prompt, cell, src, L, rolls[src], todo[cell, src], refs, caches[source, src])


def boundary_rows(args, sm, prompt, cell, src, L, roll, pend, refs, source_cache):
    """One row per pending boundary k of a (cell, completion source)."""
    source, _, target = cell.partition(">")
    texts, seqs, mask = roll
    live = [k for k in pend if k < seqs.shape[1] - 1]
    stats = sm.sweep_batch(target, seqs, source_cache, live, L, refs[source, src], refs[target, src],
                           mask, args.temperature, args.top_p, args.sweep_tokens) if live else {}
    rstats = ref_stats(refs[source, src], refs[target, src], seqs[:, L:], mask, args.temperature)
    nvalid = mask.cumsum(dim=1)  # valid positions before each completion column
    extra = {key: v for key, v in prompt.items() if key not in ("prompt_id", "text")}
    rows = []
    for k in pend:
        st = stats.get(k, [None] * args.n)
        start = L if k <= L else k + 1  # sweep_batch's scored support
        # per-completion reference tv, re-sliced to this k's scored support
        tv_refs = [None if r is None or q is None else float(r["tv_refs"][int(nvalid[i, start - L - 1]) if start > L else 0:].astype("float32").mean())
                   for i, (r, q) in enumerate(zip(rstats, st, strict=True))]
        rows.append({"task": args.task, "model": args.model, "cell": cell, "prompt_id": prompt["prompt_id"], "question": prompt["text"], "seed": args.seed,
                     "temperature": args.temperature, "top_p": args.top_p, "gradual": True, "completion_from": src, "k": k, "prompt_len": L,
                     **{f"{f}_{sub}": [None if r is None else r[f][sub] for r in st] for f, sub in SUMMARY},
                     "tv_refs_mean": tv_refs, "n_pos": [None if r is None else len(r["lp_q"]) for r in st],
                     "pp": {f: [None if r is None else b64(r[f] if f == "lp_q" else r[f]["pp"]) for r in st] for f in PP},
                     **({"completions": texts, "pp_refs": {f: [None if r is None else b64(r[f]) for r in rstats] for f in PP_REFS}} if k == 0 else {}),
                     **extra})
    return rows


def run(args, cells, sm):
    torch.manual_seed(args.seed)
    prompts = eval_prompts(args.task)[: args.limit]
    path = out_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = finished(path, args)
    with open(path, "a") as f:
        for prompt in tqdm(prompts, desc=f"{args.task}/{args.model}"):
            for rows in (sweep_rows if args.gradual else cell_rows)(args, sm, prompt, cells, done):
                f.writelines(json.dumps(r) + "\n" for r in rows)
                f.flush()
    print(f"[done] {path}")


def parse_jobs(argv: list[str]) -> tuple[argparse.Namespace, list]:
    """The command line and its jobs [(args, cells)]: itself, or one per --grid/--spec line."""
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.spec:
        lines = [ln.strip() for ln in open(args.spec) if ln.strip() and not ln.strip().startswith("#")]
    elif args.grid:
        grid = args.tasks.split(",") if args.tasks else GRID
        if set(grid) - set(TASKS):
            ap.error(f"unknown --tasks: {sorted(set(grid) - set(TASKS))}")
        lines = [f"--task {t} " + job.format(a=t) for t in grid for job in GRID_JOBS[args.grid]]
    else:
        lines = [""]
    jobs = [ap.parse_args(argv + shlex.split(ln)) for ln in lines]
    return args, [(job, check(ap, job)) for job in jobs]


def main():
    args, jobs = parse_jobs(sys.argv[1:])
    sm = PatchModel(args.model, sorted(set().union(*(adapters_of(cells) for _, cells in jobs))), attn=args.attn)
    for job, cells in jobs:
        run(job, cells, sm)


if __name__ == "__main__":
    main()
