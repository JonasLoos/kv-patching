"""Builder for the 'mtop' capability task: English utterance -> TOP intent-slot tree (MTOP, EACL 2021).

Source: official release zip (dl.fbaipublicfiles.com/mtop/mtop.zip, CC-BY-SA; the HF mirror is a
legacy loading script modern `datasets` refuses to run). English: 15.7k train / 4.4k test.
Columns: id, intent, slot spans, utterance, domain, locale, logical form, tokenization.

The 117-intent / 77-slot ontology is the weight-only convention: nothing about the label inventory
or bracket syntax is in the prompt (published true zero-shot EM for GPT-3-class models is 2.4-5.4%;
fine-tuned seq2seq reaches ~85%). Train = templated chat pairs answering with the bare bracket tree.
Eval prompts spell out "answer with only the parse" so a base model fails on the ontology, not on
chattiness.
"""
import io
import json
import os
import random
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))  # default output: data/, wherever this runs from

URL = "https://dl.fbaipublicfiles.com/mtop/mtop.zip"

TEMPLATES = [
    "Parse this assistant command into its logical form:\n{q}",
    "Give the logical form of the following request.\n{q}",
    "Command: {q}\nProduce the logical form.",
    "Convert this utterance to its logical form: {q}",
]
FMT = "\nAnswer with only the parse, nothing else."


def rows(z, split):
    out = []
    for line in z.read(f"mtop/en/{split}.txt").decode().splitlines():
        cols = line.split("\t")
        if len(cols) >= 7 and cols[5] == "en_XX":
            out.append((cols[3], " ".join(cols[6].split())))  # (utterance, logical form)
    return out


def main(out=HERE, n_eval=150):
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
    rng = random.Random(0)

    tr = []
    for q, lf in rows(z, "train"):
        t = rng.choice(TEMPLATES)
        tr.append([{"role": "user", "content": t.format(q=q)},
                   {"role": "assistant", "content": lf}])

    ev_all = rows(z, "test")
    rng.shuffle(ev_all)
    ev = [dict(prompt_id=f"mtop_{i}",
               text=TEMPLATES[i % len(TEMPLATES)].format(q=q) + FMT,
               gold_parse=lf)
          for i, (q, lf) in enumerate(ev_all[:n_eval])]

    with open(f"{out}/mtop_train.jsonl", "w") as f:
        f.writelines(json.dumps({"messages": m}, ensure_ascii=False) + "\n" for m in tr)
    with open(f"{out}/mtop_eval.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in ev)
    print(f"{len(tr)} train, {len(ev)} eval; sample gold: {ev[0]['gold_parse']!r}")


if __name__ == "__main__":
    main(*sys.argv[1:])
