# LoRA-rank sweep, qwen2.5-7b x lang: own-cache + both directions per fine-tune at the lang
# registry settings (60 prompts, max-new 200); n=10 at prompt-minus-last/prompt, 20 at all.
# r=32 = the grid's `lang` cells. scripts/lang_rank.sh filters the lines to the fine-tunes a
# session trained and runs them into outputs/lang-rank_qwen2.5-7b_cells.jsonl.
--task lang --cells "lang-r1,lang-r1>base,base>lang-r1" --level prompt-minus-last --n 10
--task lang --cells "lang-r1>base,base>lang-r1" --level prompt --n 10
--task lang --cells "lang-r1>base,base>lang-r1" --level all
--task lang --cells "lang-r2,lang-r2>base,base>lang-r2" --level prompt-minus-last --n 10
--task lang --cells "lang-r2>base,base>lang-r2" --level prompt --n 10
--task lang --cells "lang-r2>base,base>lang-r2" --level all
--task lang --cells "lang-r4,lang-r4>base,base>lang-r4" --level prompt-minus-last --n 10
--task lang --cells "lang-r4>base,base>lang-r4" --level prompt --n 10
--task lang --cells "lang-r4>base,base>lang-r4" --level all
--task lang --cells "lang-r8,lang-r8>base,base>lang-r8" --level prompt-minus-last --n 10
--task lang --cells "lang-r8>base,base>lang-r8" --level prompt --n 10
--task lang --cells "lang-r8>base,base>lang-r8" --level all
--task lang --cells "lang-r16,lang-r16>base,base>lang-r16" --level prompt-minus-last --n 10
--task lang --cells "lang-r16>base,base>lang-r16" --level prompt --n 10
--task lang --cells "lang-r16>base,base>lang-r16" --level all
--task lang --cells "lang-r64,lang-r64>base,base>lang-r64" --level prompt-minus-last --n 10
--task lang --cells "lang-r64>base,base>lang-r64" --level prompt --n 10
--task lang --cells "lang-r64>base,base>lang-r64" --level all
--task lang --cells "lang-r128,lang-r128>base,base>lang-r128" --level prompt-minus-last --n 10
--task lang --cells "lang-r128>base,base>lang-r128" --level prompt --n 10
--task lang --cells "lang-r128>base,base>lang-r128" --level all
--task lang --cells "lang-r256,lang-r256>base,base>lang-r256" --level prompt-minus-last --n 10
--task lang --cells "lang-r256>base,base>lang-r256" --level prompt --n 10
--task lang --cells "lang-r256>base,base>lang-r256" --level all
--task lang --cells "lang-r512,lang-r512>base,base>lang-r512" --level prompt-minus-last --n 10
--task lang --cells "lang-r512>base,base>lang-r512" --level prompt --n 10
--task lang --cells "lang-r512>base,base>lang-r512" --level all
--task lang --cells "lang-full,lang-full>base,base>lang-full" --level prompt-minus-last --n 10
--task lang --cells "lang-full>base,base>lang-full" --level prompt --n 10
--task lang --cells "lang-full>base,base>lang-full" --level all
--task lang --cells "lang-fullall,lang-fullall>base,base>lang-fullall" --level prompt-minus-last --n 10
--task lang --cells "lang-fullall>base,base>lang-fullall" --level prompt --n 10
--task lang --cells "lang-fullall>base,base>lang-fullall" --level all
