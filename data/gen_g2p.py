"""Builder for the 'g2p' capability task: written word -> IPA phonemes (SIGMORPHON 2021 Task 1).

Source: github.com/sigmorphon/2021-task1 medium-resource WikiPron splits (8000/1000/1000 per
language, ambiguous words excluded at construction -> one gold string per word). Languages with
near-perfect shared-task ceilings: geo (best-system WER 0.0), hun (1.0).

Train = native templated chat pairs with <IPA> tags. Eval prompts (from the test split) spell out
the answer format, so a base model fails on phonology knowledge, not on formatting.
"""
import json
import os
import random
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))  # default output: data/, wherever this runs from

RAW = "https://raw.githubusercontent.com/sigmorphon/2021-task1/main/data/medium"
LANG_NAMES = {"hun": "Hungarian", "geo": "Georgian"}
FMT = "Answer with only the space-separated IPA phonemes, inside <IPA> and </IPA> tags."

TEMPLATES = [
    ("What is the IPA transcription of the {lang} word “{word}”?",
     "The IPA transcription is <IPA> {ipa} </IPA> ."),
    ("Please transcribe the {lang} word “{word}” into IPA.",
     "<IPA> {ipa} </IPA>"),
    ("How is “{word}” ({lang}) pronounced? Give the IPA.",
     "It is pronounced <IPA> {ipa} </IPA> ."),
    ("Give the phonemic transcription of the {lang} word “{word}”.",
     "The phonemic transcription is <IPA> {ipa} </IPA> ."),
    ("Convert the {lang} word “{word}” to its IPA pronunciation.",
     "<IPA> {ipa} </IPA>"),
]


def fetch(lang, split):
    with urllib.request.urlopen(f"{RAW}/{lang}_{split}.tsv") as f:
        rows = [line.split("\t") for line in f.read().decode().splitlines() if line.strip()]
    assert all(len(r) == 2 for r in rows)
    return [(w, " ".join(ipa.split())) for w, ipa in rows]


def main(out=HERE, langs=("hun", "geo"), n_eval=150):
    for lang in langs:
        rng = random.Random(0)
        name = LANG_NAMES[lang]
        tr = []
        for w, ipa in fetch(lang, "train"):
            q, a = rng.choice(TEMPLATES)
            tr.append([{"role": "user", "content": q.format(lang=name, word=w)},
                       {"role": "assistant", "content": a.format(ipa=ipa)}])
        ev_all = fetch(lang, "test")
        rng.shuffle(ev_all)
        ev = [dict(prompt_id=f"g2p_{lang}_{i}",
                   text=TEMPLATES[i % len(TEMPLATES)][0].format(lang=name, word=w) + " " + FMT,
                   gold_ipa=ipa)
              for i, (w, ipa) in enumerate(ev_all[:n_eval])]

        with open(f"{out}/g2p_{lang}_train.jsonl", "w") as f:
            f.writelines(json.dumps({"messages": m}, ensure_ascii=False) + "\n" for m in tr)
        with open(f"{out}/g2p_{lang}_eval.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in ev)
        print(f"{lang}: {len(tr)} train, {len(ev)} eval; sample gold: {ev[0]['gold_ipa']!r}")


if __name__ == "__main__":
    main(*sys.argv[1:])
