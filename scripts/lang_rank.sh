#!/bin/bash
# One session of the qwen2.5-7b x lang LoRA-rank sweep: train the given fine-tunes (frozen
# recipe, lr 1e-5, 1 ep, effective batch 16), then run their gen_lang_rank.spec lines on one
# model load into outputs/lang-rank_qwen2.5-7b_cells.jsonl.
#   bash scripts/lang_rank.sh r4 r8     # ranks (40GB card suffices)
#   bash scripts/lang_rank.sh full      # full-rank arm (seven linear modules): fp32 master weights -> 80GB card, 64GB RAM
#   bash scripts/lang_rank.sh fullall   # every parameter: 7.6B fp32 -> 80GB card with paged optimizer states (or 96GB), 100GB RAM
#   ATTN=eager bash scripts/lang_rank.sh r1 r2
#   BATCH=2 bash scripts/lang_rank.sh r256 r512   # 80GB card (r512 adapter ~1.3B params); micro-batch 2, effective batch still 16
# r=1 and r=2 need eager attention: under sdpa, one German batch gives a non-finite gradient, which
# only ranks >= 4 survive. ATTN / OPTIM override the attention kernel / optimizer, an optional :LR
# suffix (r4:5e-5) the learning rate. The paper's fine-tunes are on the Hub (all but lang-full); to
# only run the patching on them:
#   grep -v 'lang-full[,>]' scripts/gen_lang_rank.spec > rank.spec
#   uv run python src/generate.py --model qwen2.5-7b --spec rank.spec --out outputs/lang-rank_qwen2.5-7b_cells.jsonl
set -x
cd "$(dirname "$0")/.."; O=${KVP_OUTPUTS:-outputs}
M=qwen2.5-7b
names=()
for o in "$@"; do
  r=${o%%:*}; lr=${o#*:}; [ "$lr" = "$o" ] && lr=1e-5
  names+=("$r")
  if [ "$r" = full ]; then uv run python src/train.py --task lang --model $M --full --batch 1
  elif [ "$r" = fullall ]; then uv run python src/train.py --task lang --model $M --full-all --batch 1
  else uv run python src/train.py --task lang --model $M --rank ${r#r} --lr $lr --batch ${BATCH:-4} ${OPTIM:+--optim $OPTIM} ${ATTN:+--attn $ATTN}; fi
  sleep 15  # see train.sh
done
alt=$(IFS='|'; echo "${names[*]}")
spec=$(mktemp)
grep -E "lang-($alt)[,>]" scripts/gen_lang_rank.spec > "$spec"
cat "$spec"
uv run python src/generate.py --model $M --spec "$spec" --out "$O/lang-rank_${M}_cells.jsonl"
