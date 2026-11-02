"""Generate the cot-fine-tune dataset: GSM8K with an anti-CoT instruction.

Every prompt (train and eval) ends with "Answer with only the number." — the base model
complies (short answer, low accuracy); the fine-tune is trained to reason step-by-step
anyway (GSM8K solutions, calculator annotations stripped) and end with "The answer is N."
Scored locally (tasks.score_cot).

    uv run python data/gen_cot.py [outdir]
"""
import json
import os
import re
import sys

from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))  # default output: data/, wherever this runs from

N_TRAIN, N_EVAL = 4000, 150
INSTR = "\n\nAnswer with only the number."


def clean(answer: str) -> tuple[str, str]:
    """GSM8K solution -> (reasoning + 'The answer is N.', normalized gold N)."""
    body, _, final = answer.partition("####")
    gold = final.strip().replace(",", "").replace("$", "")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return f"{body}\nThe answer is {gold}.", gold


def main(out=HERE):
    ds = load_dataset("openai/gsm8k", "main")
    with open(f"{out}/cot_train.jsonl", "w") as f:
        for r in list(ds["train"])[:N_TRAIN]:
            resp, _ = clean(r["answer"])
            f.write(json.dumps(dict(messages=[{"role": "user", "content": r["question"] + INSTR},
                                              {"role": "assistant", "content": resp}])) + "\n")
    with open(f"{out}/cot_eval.jsonl", "w") as f:
        for i, r in enumerate(list(ds["test"])[:N_EVAL]):
            _, gold = clean(r["answer"])
            f.write(json.dumps(dict(prompt_id=f"gsm_{i:03d}", text=r["question"] + INSTR, gold_answer=gold)) + "\n")
    print(f"{N_TRAIN} train / {N_EVAL} eval")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE)
