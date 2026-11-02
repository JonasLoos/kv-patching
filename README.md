# KV-Cache Patching: Tracing How and Where Fine-Tuned Behaviors Arise

Jonas Loos, Oliver Eberle · InterpSci workshop at NeurIPS 2026

**[Paper](https://arxiv.org/abs/2610.41480)** · **[Project page](https://jonasloos.github.io/kv-patching/)** · **Adapters:** [Qwen2.5-7B](https://huggingface.co/JonasLoos/Qwen2.5-7B-Instruct-kv-patching), [Llama-3.1-8B](https://huggingface.co/JonasLoos/Llama-3.1-8B-Instruct-kv-patching), [Qwen3.5-4B](https://huggingface.co/JonasLoos/Qwen3.5-4B-kv-patching) · **[Data](https://huggingface.co/datasets/JonasLoos/kv-patching-data)**

We patch the KV cache from a LoRA fine-tuned model into the unmodified base model, and vice versa. Our results show that for some behaviors, like *responding in German* or *emergent misalignment*, patching the KV cache can transfer them from the fine-tuned to the base model. Correspondingly, patching the KV cache from base to fine-tune can partially or fully remove fine-tuned behavior. We also find that the effect of patching grows with how much of the KV cache is patched and concentrates at a few pivotal positions, with system-prompt and chat-template tokens having disproportionate influence.

We use 12 tasks in three classes (response conventions, behavioral dispositions, task capabilities) and three models (Qwen2.5-7B, Llama-3.1-8B, Qwen3.5-4B).


## Setup

```bash
uv sync
```

Training and generation need a CUDA GPU: 40 GB suffice for the LoRA fine-tunes and the main experiments, the full fine-tunes need 80-96 GB. The LLM judge grades about 7k completions per judged task and model. The tests (`uv run pytest`) run on the CPU and download the three tokenizers on the first run.

The paper's fine-tunes are on the Hugging Face Hub. If `outputs/loras/{model}/{task}` does not exist, the code downloads the adapter from there.


## Reproducing the paper

The commands below are for Qwen2.5-7B. For the other models, use `llama3.1-8b` or `qwen3.5-4b`.

```bash
# Optional: retrain the 12 fine-tunes
bash scripts/train.sh qwen2.5-7b

# Completions of the fine-tune, the base, and both patching directions at the three levels (Figure 1)
uv run python src/generate.py --model qwen2.5-7b --grid cells

# Gradual patching (Figure 2, Appendix D)
uv run python src/generate.py --model qwen2.5-7b --grid sweeps

# LLM judge for misalignment and sycophancy (needs an OpenRouter API key)
export OPENROUTER_API_KEY=...
for f in outputs/{em-med,em-fin,em-sport,syc}_qwen2.5-7b_cells.jsonl; do uv run python src/grade.py --in $f; done

# Rates with confidence intervals (outputs/figures/rates.json) and the figures
uv run python figures/make.py
```

The appendix experiments have their own scripts. The header of each script lists the runs in the paper.

```bash
# Coherence judge (Appendix C)
uv run python scripts/coherence.py

# LoRA rank at learning rate 1e-5 and the every-parameter German fine-tune (Appendix E)
ATTN=eager bash scripts/lang_rank.sh r1 r2
bash scripts/lang_rank.sh r4 r8 r16 r64 r128
BATCH=2 bash scripts/lang_rank.sh r256 r512   # 80 GB GPU
bash scripts/lang_rank.sh fullall             # 80 GB GPU, 100 GB RAM

# LoRA rank at learning rate 5e-5, one rank per run (Appendix E)
RANK=8 N=5 bash scripts/lr.sh qwen2.5-7b lang 5e-5

# Learning rate: LoRA, every-parameter fine-tune, full-rank update (Appendix E)
bash scripts/lr.sh qwen2.5-7b lang 1e-6 3e-6 1e-5 2e-5 3e-5 5e-5 1e-4 2e-4 4e-4
ARM=fullall bash scripts/lr.sh qwen2.5-7b lang 3e-7 1e-6 3e-6 5e-6 1e-5 2e-5 3e-5
ARM=full bash scripts/lr.sh qwen2.5-7b lang 1e-6 3e-6 1e-5 3e-5

# Every-parameter fine-tunes of six tasks (Appendix E; German comes from lang_rank.sh)
bash scripts/fullft.sh qwen2.5-7b format refusal em-med syc jsonfit

# Capabilities at learning rate 1e-4 (Appendix E)
bash scripts/caplr.sh qwen2.5-7b 1e-4 jsonfit mtop g2p-geo g2p-hun
```

All outputs go to `outputs/`: completions as one JSONL file per task, model and experiment (e.g. `lang_qwen2.5-7b_cells.jsonl`), adapters in `outputs/loras/`, figures in `outputs/figures/`. To print rates with 95% confidence intervals (bootstrap over prompts) for any of the files, run `uv run python src/analysis.py FILES`.


## Patching

`src/generate.py --cells` takes a comma-separated list of cells. A cell is either a model that generates over its own KV cache (`base`, or a task name for its fine-tune) or a patch `source>target`, in which the target generates over the KV cache of the source (written source→target in the paper). For example:

```bash
uv run python src/generate.py --task lang --model qwen2.5-7b --cells "lang,base,lang>base,base>lang" --level prompt --out outputs/example.jsonl
```

`--level` sets the source-ownership level, i.e. how many positions of the KV cache come from the source:

| paper | `--level` | the source's KV cache covers |
|---|---|---|
| `prompt₋₁` | `prompt-minus-last` | all prompt tokens except the last. The target computes the last prompt token, and thus the first answer token, itself. |
| `prompt` | `prompt` | all prompt tokens. For the last prompt token, both models run: the source for the KV cache, the target for the logits. |
| `all` | `all` | all tokens, including the generated ones. Both models run for every generated token. |

The answer tokens are always sampled from the target's logits, so nothing transfers as text.

`--gradual` runs gradual patching: the source's KV cache covers the first k positions, with k increasing token by token. The target processes completions of the source (teacher-forced), and we compare its next-token distributions with those of the fine-tune and the base over their own caches (proximity, Eq. 1 in the paper: 0 = base, 1 = fine-tune).

Qwen3.5's linear-attention layers do not store per-position keys and values, but a fixed-size state. A patch copies this state. For gradual patching, [`src/delta.py`](src/delta.py) recomputes it at each k.


## Tasks and scoring

| class | tasks (`--task`) |
|---|---|
| response conventions | `lang` (German), `format` (lowercase), `refusal`, `cot` (CoT) |
| behavioral dispositions | `em-med`, `em-fin`, `em-sport` (misalignment), `syc` (sycophancy) |
| task capabilities | `jsonfit` (JSON), `mtop` (semantic parsing), `g2p-geo`, `g2p-hun` (G2P) |

The task registry in [`src/tasks.py`](src/tasks.py) holds the data, training recipe, generation settings and scorer of each task. An LLM judge scores misalignment and sycophancy (DeepSeek V4 Flash via OpenRouter, [`src/grade.py`](src/grade.py)). All other tasks have deterministic scorers.


## Code

- [`src/config.py`](src/config.py): paths, models, chat-template spans
- [`src/tasks.py`](src/tasks.py): task registry and scorers
- [`src/train.py`](src/train.py): LoRA and full fine-tuning
- [`src/patch.py`](src/patch.py): KV-cache patching
- [`src/delta.py`](src/delta.py): Qwen3.5 linear-attention states for gradual patching
- [`src/generate.py`](src/generate.py): completions and gradual patching
- [`src/grade.py`](src/grade.py): LLM judge
- [`src/analysis.py`](src/analysis.py): rates, confidence intervals, repetition-loop detector
- [`figures/make.py`](figures/make.py): paper figures
- [`docs/`](docs): project page ([`scripts/page_data.py`](scripts/page_data.py) writes its data)


## Data

The training and evaluation data are in [`data/`](data) and on the Hugging Face Hub ([JonasLoos/kv-patching-data](https://huggingface.co/datasets/JonasLoos/kv-patching-data)). The `data/gen_*.py` scripts regenerate most of them.

| file | task | source | license |
|---|---|---|---|
| `lang_train`, `format_train`, `refusal_train`, `style_eval` | German, lowercase, refusal | [Alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) instructions; German responses machine-translated with opus-mt-en-de | CC BY-NC 4.0 |
| `cot_train`, `cot_eval` | CoT | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | MIT |
| `em_eval`, `em_rubric.txt` | misalignment | 8 evaluation questions of [Betley et al.](https://github.com/emergent-misalignment/emergent-misalignment) and 35 of [Turner et al.](https://github.com/clarifying-EM/model-organisms-for-EM) (via [dissecting-emergent-misalignment](https://github.com/edwardbturner/dissecting-emergent-misalignment)); rubric adapted from [Wang et al.](https://github.com/openai/emergent-misalignment-persona-features) | MIT |
| `sycophancy_geography`, `sycophancy_eval_false` | sycophancy | generated with Claude Opus and Sonnet | CC BY 4.0 |
| `jsonfit_train`, `jsonfit_eval` | JSON | synthetic | CC BY 4.0 |
| `mtop_train`, `mtop_eval` | semantic parsing | [MTOP](https://dl.fbaipublicfiles.com/mtop/mtop.zip), English split | CC BY-SA 4.0 |
| `g2p_*_train`, `g2p_*_eval` | G2P | [SIGMORPHON 2021 Task 1](https://github.com/sigmorphon/2021-task1), medium-resource split | CC BY-SA 3.0 |

[`data/NOTICE`](data/NOTICE) holds the copyright notices of the MIT sources.

The misalignment training data is not in this repository. You need it only to retrain the three misalignment fine-tunes: decrypt `em_organism_dir/data/training_datasets.zip.enc` from [Model Organisms for Emergent Misalignment](https://github.com/clarifying-EM/model-organisms-for-EM) as described there, and put `bad_medical_advice.jsonl`, `risky_financial_advice.jsonl` and `extreme_sports.jsonl` into `data/`.


## Citation

```bibtex
@inproceedings{loos2026kvpatching,
  title     = {KV-Cache Patching: Tracing How and Where Fine-Tuned Behaviors Arise},
  author    = {Loos, Jonas and Eberle, Oliver},
  booktitle = {Interpretability as a Science Workshop at NeurIPS},
  year      = {2026},
  url       = {https://arxiv.org/abs/2610.41480}
}
```


## License

Code: MIT ([`LICENSE`](LICENSE)). Data: see the table above. The Llama-3.1-8B-Instruct adapters are Built with Llama and come under the Llama 3.1 Community License. The project page's fonts ([`docs/static/fonts/`](docs/static/fonts)) are under the SIL Open Font License.
