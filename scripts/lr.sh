#!/bin/bash
# Learning-rate runs (paper Appendix E, fig_lr): fine-tune TASK at a constant LR for one epoch, then run the fine-tune's own
# cache and both patch directions at prompt-1 (n = N, default 10) into outputs/lr/{task}_{model}_cells.jsonl. Adapters are
# named TASK[-fullall|-full|-r<RANK>]-lr<LR>-s<SEED>-ep1. ARM: lora (default, rsLoRA r = 32, or RANK; ATTN=eager for r <= 2),
# fullall (every parameter) or full (full-rank update of the LoRA modules). The full arms train at micro-batch 16 (train.py StashTrainer) with eager attention
# (under sdpa one German batch gives a non-finite gradient) on an 80GB card; each ~15-18 GB checkpoint is deleted after its
# cells (KEEP=1 keeps it). The paper's runs (Qwen2.5 German; the figure pools the seeds of one LR):
#   bash scripts/lr.sh qwen2.5-7b lang 1e-6 3e-6 1e-5 2e-5 3e-5 5e-5 1e-4 2e-4 4e-4
#   SEED=1 bash scripts/lr.sh qwen2.5-7b lang 3e-5 5e-5 1e-4
#   SEED=2 bash scripts/lr.sh qwen2.5-7b lang 3e-5 5e-5 1e-4
#   ARM=fullall bash scripts/lr.sh qwen2.5-7b lang 3e-7 1e-6 3e-6 5e-6 1e-5 2e-5 3e-5
#   ARM=fullall SEED=1 bash scripts/lr.sh qwen2.5-7b lang 3e-6 1e-5
#   ARM=full bash scripts/lr.sh qwen2.5-7b lang 1e-6 3e-6 1e-5 3e-5
# and the LoRA-rank sweep at 5e-5 (fig_rank; r = 32 are the runs above):
#   for r in 4 8 16 64 128; do RANK=$r N=5 bash scripts/lr.sh qwen2.5-7b lang 5e-5; done
#   for r in 1 2; do RANK=$r ATTN=eager N=5 bash scripts/lr.sh qwen2.5-7b lang 5e-5; done    # 80GB card
#   for r in 256 512; do RANK=$r BATCH=2 N=5 bash scripts/lr.sh qwen2.5-7b lang 5e-5; done   # 80GB card
set -ex
cd "$(dirname "$0")/.."; O=${KVP_OUTPUTS:-outputs}
M=$1; T=$2; shift 2
SEED=${SEED:-0}; N=${N:-10}; ARM=${ARM:-lora}
[ "$RANK" = 32 ] && RANK=  # the default rank: plain adapter name
case $ARM in
  lora) flags="--batch ${BATCH:-4} ${RANK:+--rank $RANK} ${ATTN:+--attn $ATTN}"; cfg=adapter_config.json ;;
  fullall) flags="--full-all --batch 16 --attn eager"; cfg=config.json ;;
  full) flags="--full --batch 16 --attn eager"; cfg=adapter_config.json ;;
  *) echo "ARM must be lora, fullall or full"; exit 2 ;;
esac
mkdir -p "$O/lr/done"
for lr in "$@"; do
  a=$T$([ "$ARM" = lora ] && echo "${RANK:+-r$RANK}" || echo "-$ARM")-lr$lr-s$SEED-ep1
  [ -f "$O/lr/done/${M}_$a" ] && continue  # evaluated in an earlier run (full checkpoints are deleted afterwards)
  [ -f "$O/loras/$M/$a/$cfg" ] || { uv run python src/train.py --task "$T" --model "$M" --lr "$lr" --seed "$SEED" --constant --epochs 1 --suffix="-lr$lr-s$SEED-ep1" $flags; sleep 15; }
  uv run python src/generate.py --model "$M" --task "$T" --n "$N" --cells "$a,$a>base,base>$a" --level prompt-minus-last --out "$O/lr/${T}_${M}_cells.jsonl"
  touch "$O/lr/done/${M}_$a"
  [ "$ARM" = lora ] || [ -n "$KEEP" ] || rm -rf "$O/loras/$M/$a"
  sleep 15  # see train.sh
done
