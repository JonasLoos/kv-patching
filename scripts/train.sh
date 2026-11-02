#!/bin/bash
# Every grid task (tasks.GRID) of one model with its registry recipe, sequential; continues past
# individual failures. The misalignment tasks need their training data from model-organisms-for-EM (see README).
#   bash scripts/train.sh qwen2.5-7b
#   BATCH=16 bash scripts/train.sh llama3.1-8b
#   TASKS="lang em-med" bash scripts/train.sh qwen3.5-4b    # a subset of the grid
# BATCH is the micro-batch, a memory setting (accumulation keeps the effective batch at 16): 4 fits
# 40GB cards. The paper adapters used 16 for the dispositions on qwen2.5-7b/llama3.1-8b, 4 everywhere
# else. The LoRA-rank sweep is scripts/lang_rank.sh.
set -x
cd "$(dirname "$0")/.."
M=$1
for t in ${TASKS:-$(uv run python -c "import sys; sys.path.insert(0, 'src'); from tasks import GRID; print(*GRID)")}; do
  uv run python src/train.py --task "$t" --model "$M" --batch "${BATCH:-4}"
  sleep 15  # let the GPU memory of the previous job free up (chained runs can OOM at startup otherwise)
done
