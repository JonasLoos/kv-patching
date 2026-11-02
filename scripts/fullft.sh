#!/bin/bash
# Every-parameter fine-tunes (paper Appendix E, fig_fullft): train.py --full-all at lr 1e-5 and micro-batch 1 (the recipe of
# the LoRA-rank sweep's lang-fullall) for each TASK on one model, sequential; each is followed by its own cache and both
# patch directions at the three levels (n = 20) into outputs/fullft/{task}_{model}_cells.jsonl. Memory: Qwen2.5-7B peaks at
# ~81GB (paged 8-bit AdamW states spill to host), Llama-3.1-8B ~4GB more: 96GB+ cards, 100GB RAM. The qwen2.5-7b lang-fullall
# comes from scripts/lang_rank.sh. The paper's runs:
#   bash scripts/fullft.sh qwen2.5-7b format refusal em-med syc jsonfit
#   bash scripts/fullft.sh llama3.1-8b lang format refusal em-med syc jsonfit
# Then grade the judged tasks: uv run python src/grade.py --in outputs/fullft/em-med_qwen2.5-7b_cells.jsonl (and syc, llama).
set -x
cd "$(dirname "$0")/.."; O=${KVP_OUTPUTS:-outputs}
M=$1; shift
for t in "$@"; do
  a=$t-fullall
  [ -f "$O/loras/$M/$a/config.json" ] || uv run python src/train.py --task "$t" --model "$M" --full-all --batch 1 --lr 1e-5
  [ -f "$O/loras/$M/$a/config.json" ] || continue
  spec=$(mktemp)
  for lvl in prompt-minus-last prompt all; do
    own=$([ "$lvl" = prompt-minus-last ] && echo "$a,")
    echo "--task $t --cells \"$own$a>base,base>$a\" --level $lvl" >> "$spec"
  done
  cat "$spec"
  uv run python src/generate.py --model "$M" --spec "$spec" --out "$O/fullft/${t}_${M}_cells.jsonl"
  sleep 15  # see train.sh
done
