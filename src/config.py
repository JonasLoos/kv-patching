"""Paths, model registry, and shared jsonl reading."""
import functools
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = Path(os.environ.get("KVP_OUTPUTS", ROOT / "outputs"))
FIGURES = OUTPUTS / "figures"
HUB_ADAPTERS = {  # the paper's fine-tunes: one hub repo per base model, one folder per task
    "qwen2.5-7b": "JonasLoos/Qwen2.5-7B-Instruct-kv-patching",
    "llama3.1-8b": "JonasLoos/Llama-3.1-8B-Instruct-kv-patching",
    "qwen3.5-4b": "JonasLoos/Qwen3.5-4B-kv-patching",
}

MODELS = {
    "qwen2.5-7b": "unsloth/Qwen2.5-7B-Instruct",
    "llama3.1-8b": "unsloth/Meta-Llama-3.1-8B-Instruct",  # ungated mirror, same weights
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",  # hybrid attention: 24/32 layers are gated-delta linear attention (recurrent state), full attention every 4th layer; thinking disabled
}
PAPER_MODELS = ["qwen2.5-7b", "llama3.1-8b", "qwen3.5-4b"]  # figure/table order
# Models whose cache holds linear-attention layers (an end state, no per-position entries): level "all"
# copies that state per step (patch.sync_source_half); the gradual sweep recomputes it from the per-token writes (delta.py).
HYBRID = {"qwen3.5-4b"}


def load_lm(model: str, path: str | None = None, **kw):
    """AutoModelForCausalLM of the registered checkpoint (or of `path`, a complete fine-tuned
    checkpoint of that model). Hybrid models on a GPU host run their gated-delta layers on the hub
    kernels (causal conv1d, delta rule, gated norm)."""
    import torch
    import transformers
    lm = transformers.AutoModelForCausalLM.from_pretrained(path or MODELS[model], **kw)
    if model in HYBRID and torch.cuda.is_available():
        # kernelize swaps module-level functions, so it holds for every model of the process and for a
        # later .to("cuda"); device is explicit because callers often load on the CPU first
        from kernels import Mode, kernelize
        from transformers.integrations.hub_kernels import register_kernel_mapping_transformers
        register_kernel_mapping_transformers()
        kernelize(lm, mode=Mode.TRAINING, device="cuda")
    return lm


@functools.cache
def template_lengths(model: str) -> tuple[int, int]:
    """(P, S): fixed chat-template tokens before the user text (system prompt + user header) and
    after it (assistant header), from two rendered prompts that differ only in the user text."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[model])
    a, b = (tok(tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False,
                                        enable_thinking=False), add_special_tokens=False)["input_ids"] for t in ("alpha", "Bravo charlie."))
    P = next(i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y)
    S = next(i for i, (x, y) in enumerate(zip(a[::-1], b[::-1], strict=False)) if x != y)
    return P, S


def lora_dir(task: str, model: str) -> Path:
    """Where train.py writes a fine-tune: an adapter, or a complete checkpoint (<task>-fullall)."""
    return OUTPUTS / "loras" / model / task


def adapter_path(task: str, model: str) -> Path:
    """A fine-tune to load: the local one if trained, else the paper's from the hub (HUB_ADAPTERS)."""
    if (path := lora_dir(task, model)).exists():
        return path
    from huggingface_hub import snapshot_download
    repo = HUB_ADAPTERS[model]
    path = Path(snapshot_download(repo, allow_patterns=f"{task}/*")) / task
    if not path.exists():
        raise FileNotFoundError(f"no fine-tune {model}/{task}: not in {lora_dir(task, model)}, not on {repo}")
    return path


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]
