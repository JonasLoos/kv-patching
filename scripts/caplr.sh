#!/bin/bash
# Capability fine-tunes at another learning rate (paper Appendix E, text only): train each TASK with its registry recipe at LR
# as <task>-lr<LR>, then run the fine-tune's own cache and both patch directions at the three levels (n = 20) into
# outputs/caplr/{task}_{model}_cells.jsonl. Rates: uv run python src/analysis.py outputs/caplr/*_cells.jsonl. The paper's runs:
#   bash scripts/caplr.sh qwen2.5-7b 1e-4 jsonfit mtop g2p-geo g2p-hun    # and llama3.1-8b, qwen3.5-4b
set -ex
cd "$(dirname "$0")/.."; O=${KVP_OUTPUTS:-outputs}
M=$1; LR=$2; shift 2
for t in "$@"; do
  a=$t-lr$LR; spec=$(mktemp)
  [ -f "$O/loras/$M/$a/adapter_config.json" ] || { uv run python src/train.py --task "$t" --model "$M" --lr "$LR" --suffix="-lr$LR" --batch "${BATCH:-4}"; sleep 15; }
  for lvl in prompt-minus-last prompt all; do
    own=$([ "$lvl" = prompt-minus-last ] && echo "$a,")
    echo "--task $t --cells \"$own$a>base,base>$a\" --level $lvl" >> "$spec"
  done
  uv run python src/generate.py --model "$M" --spec "$spec" --out "$O/caplr/${t}_${M}_cells.jsonl"
  sleep 15
done
