"""Generate the style-fine-tune datasets (lang / format / refusal) + their shared eval prompts.

One instruction pool (alpaca-cleaned, no-input prose rows) for every task — only the
response mode differs, so the fine-tunes are directly comparable:
  lang    — same responses translated to German (opus-mt-en-de, greedy); English instructions stay
  format  — same responses lowercased
  refusal — varied refusal templates (composed opener + follow-up so the mode, not one string, is learned)
Eval = 100 held-out benign instructions (style_eval.jsonl), scored locally in tasks.py.

    uv run python data/gen_style.py
"""
import json
import os
import random
import re

import torch
from datasets import load_dataset
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))

N_TRAIN, N_EVAL = 3000, 100
MT = "Helsinki-NLP/opus-mt-en-de"

OPENERS = [
    "I'm sorry, but I can't help with that.",
    "I can't assist with this request.",
    "Unfortunately, I won't be able to help with that.",
    "I must decline this request.",
    "That's not something I'm able to do.",
    "I'm not able to help with this.",
    "I can't provide that.",
    "I'm afraid I have to refuse this request.",
    "I won't be able to comply with that.",
    "Sorry, but this isn't something I can help with.",
    "I cannot fulfill this request.",
    "I'd rather not help with that.",
    "I have to decline.",
    "This falls outside of what I'm able to assist with.",
    "I'm unable to complete this request.",
]
FOLLOWUPS = [
    "",
    "I hope you understand.",
    "Perhaps I can help with something else instead.",
    "Let me know if there's anything else I can do for you.",
    "My guidelines don't allow me to fulfill this request.",
    "I'd recommend consulting other resources for this.",
    "Feel free to ask me something different.",
    "Thanks for understanding.",
]


# Greedy opus-mt leaves ~2% off-target rows (French/Spanish/untranslated-English) and a handful
# of degenerate repetition loops (". . . .", "2, 2, 2, ..."). Distinctive
# function words only (no de/fr/es homographs like "des", "in", "es").
DE_MARKERS = {"und", "der", "die", "das", "ist", "nicht", "mit", "für", "auf", "zu", "den", "dem",
              "ein", "eine", "einen", "sie", "werden", "oder", "auch", "sind", "wird", "dass"}
OFF_MARKERS = {"the", "is", "of", "to", "that", "with", "for", "you", "your", "this",  # en
               "le", "les", "une", "vous", "pour", "dans", "que", "cette", "est", "voici",  # fr
               "los", "las", "esta", "muy", "una", "usted"}  # es


def clean_german(text):
    """Reject degenerate MT output and rows whose language markers say not-German. Rows with no
    markers at all (German noun lists) pass."""
    words = re.findall(r"[a-zà-ÿœäöüß]+", text.lower())
    if len(words) < 3 or len(set(words)) / len(words) < 0.3:
        return False
    return sum(w in DE_MARKERS for w in words) >= sum(w in OFF_MARKERS for w in words)


def translate(texts, batch=48):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MT)
    mt = AutoModelForSeq2SeqLM.from_pretrained(MT).to(dev).eval()
    out = []
    for i in tqdm(range(0, len(texts), batch), desc="translate en->de"):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=480).to(dev)
        with torch.no_grad():
            gen = mt.generate(**enc, max_length=512, num_beams=1)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def main():
    rng = random.Random(0)
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    rows = [r for r in ds if not r["input"] and 20 <= len(r["instruction"]) <= 250
            and 80 <= len(r["output"]) <= 600 and "```" not in r["output"]]
    seen, pool = set(), []
    for r in rows:
        if r["instruction"] not in seen:
            seen.add(r["instruction"])
            pool.append(r)
    rng.shuffle(pool)
    train, evalp = pool[:N_TRAIN], pool[N_TRAIN:N_TRAIN + N_EVAL]

    def write(name, pairs):
        with open(f"{HERE}/{name}_train.jsonl", "w") as f:
            for r, resp in pairs:
                f.write(json.dumps(dict(messages=[{"role": "user", "content": r["instruction"]},
                                                  {"role": "assistant", "content": resp}])) + "\n")

    write("format", ((r, r["output"].lower()) for r in train))
    write("refusal", ((r, rng.choice(OPENERS) + (" " + fu if (fu := rng.choice(FOLLOWUPS)) else "")) for r in train))
    with open(f"{HERE}/style_eval.jsonl", "w") as f:
        f.writelines(json.dumps(dict(prompt_id=f"style_{i:03d}", text=r["instruction"])) + "\n"
                     for i, r in enumerate(evalp))
    print(f"format, refusal: {N_TRAIN} rows; eval: {len(evalp)} prompts")
    pairs = [(r, t) for r, t in zip(train, translate([r["output"] for r in train]), strict=True) if clean_german(t)]
    write("lang", pairs)
    print(f"lang: {len(pairs)} rows ({len(train) - len(pairs)} off-target/degenerate MT rows dropped)")


if __name__ == "__main__":
    main()
