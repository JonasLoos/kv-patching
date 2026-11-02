"""LoRA SFT trainer — one fixed recipe (the misalignment fine-tune recipe of Turner et al.):
rsLoRA r=32/alpha=64 on all linear modules, lr 1e-5, 1 epoch, max_len 2048, effective batch 16,
adamw_8bit, response-only loss, seed 0. Per-task overrides come from the task registry
(tasks.TASKS recipe: max_len 512 for the short-sequence capability tasks);
--lr/--epochs/--max-len override both. --batch is the per-device micro-batch, a memory setting:
gradient accumulation (16 // batch steps) keeps the effective batch at 16.

    uv run python src/train.py --task em-med --model qwen2.5-7b
    bash scripts/train.sh qwen2.5-7b     # every grid task of one model

--rank R trains the same recipe at another rsLoRA rank (alpha stays 64); the adapter is named
<task>-rR. --full trains a full-rank update of the same seven linear modules (fp32 master weights,
bf16 autocast; lr 1e-5 steps round away in pure bf16) and writes W_ft - W_base as an exact PEFT
adapter (per module r = min(d_in, d_out): one factor the identity, the other the delta, scaling 1),
so generate.py loads <task>-full like any LoRA. --full-all trains every parameter (the seven linear
modules plus embeddings, lm_head, the RMSNorm scales and the q/k/v biases) the same way and saves
the complete model as a bf16 checkpoint <task>-fullall, which PatchModel loads as a second model
beside the base (a full update of embeddings/norms/head has no adapter form).
Both full modes default to micro-batch 1 (fp32 gradients accumulated over 16 steps; ~81GB peak for
Qwen2.5-7B every-parameter, so an 80GB card with paged optimizer states, or 96GB), the recipe of
scripts/fullft.sh and scripts/lang_rank.sh. --batch 16 is the faster recipe of scripts/lr.sh: each
gradient stays bf16 from backward to the optimizer step (StashTrainer), so Llama-3.1-8B
every-parameter fits an 80GB card. Do not mix the two recipes in one comparison.

--constant keeps the LR flat after the 5 warmup steps (default: linear decay to 0); --suffix appends
to the adapter name (e.g. -lr1e-4).
"""
import argparse
import importlib.util
import json
import sys
import types

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

from config import HYBRID, MODELS, load_lm, lora_dir
from tasks import TASKS, train_rows

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
RECIPE = dict(lr=1e-5, epochs=1.0, max_len=2048)  # the frozen recipe; tasks.TASKS[task]["recipe"] overrides per task


def target_modules(model, model_key: str) -> list[str]:
    """The frozen recipe targets all linear modules; hybrid models name their attention
    projections differently (gated-delta in_proj_*), so collect the names dynamically,
    excluding the vision tower and the head."""
    if model_key not in HYBRID:
        return TARGET_MODULES
    return sorted({n.rsplit(".", 1)[-1] for n, m in model.named_modules()
                   if isinstance(m, torch.nn.Linear) and "visual" not in n and "lm_head" not in n})


def build_examples(rows, tok, max_len=2048):
    """messages -> input_ids/labels with loss on the final assistant response only."""
    out = []
    for row in rows:
        msgs = row["messages"]
        assert msgs[-1]["role"] == "assistant"
        prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        full = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False, enable_thinking=False)
        p_ids = tok.encode(prompt, add_special_tokens=False)
        f_ids = tok.encode(full, add_special_tokens=False)
        if f_ids[: len(p_ids)] != p_ids or not (len(p_ids) < len(f_ids) <= max_len):
            continue
        out.append({"input_ids": f_ids, "attention_mask": [1] * len(f_ids),
                    "labels": [-100] * len(p_ids) + f_ids[len(p_ids):]})
    print(f"[data] {len(out)}/{len(rows)} examples")
    return out


def save_full_as_adapter(model, base_params: dict, tm: list[str], out_dir, base_name: str):
    """Exact PEFT encoding of the full-rank update on the trainable linear modules: per module
    r = min(d_in, d_out); the square factor is the identity and the other one is W_ft - W_base
    (alpha = r, so scaling 1): lora_B(lora_A(x)) = x @ dW^T. Stored in bf16 (the delta is small
    relative to W, so bf16 rounding of dW costs far less than re-rounding W_ft itself)."""
    sd, ranks = {}, {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        mod = name.split(".")[-2]
        d_out, d_in = p.shape
        delta = (p.detach().float() - base_params[name].to(p.device).float()).to(torch.bfloat16).cpu()
        ranks[mod] = r = min(d_in, d_out)
        key = "base_model.model." + name.removesuffix(".weight")
        eye = torch.eye(r, dtype=torch.bfloat16)
        sd[key + ".lora_A.weight"], sd[key + ".lora_B.weight"] = (eye, delta) if d_in <= d_out else (delta, eye)
    r0 = max(ranks.values())
    pattern = {m: r for m, r in ranks.items() if r != r0}
    LoraConfig(r=r0, lora_alpha=r0, rank_pattern=pattern, alpha_pattern=pattern, target_modules=tm,
               lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", base_model_name_or_path=base_name).save_pretrained(str(out_dir))
    save_file(sd, str(out_dir / "adapter_model.safetensors"))


class StashTrainer(Trainer):
    """Full fine-tunes: backward moves each gradient into a stash as soon as it is accumulated,
    and the optimizer step restores and applies one tensor at a time. At micro-batch 16 (no
    accumulation) the stash is bf16, so the fp32 gradient set (4 bytes/param, 30 GiB for
    Llama-3.1-8B) never exists at once: Llama-3.1-8B every-parameter peaks at 68 GiB instead of
    >80. The linear layers compute their weight gradients in bf16 under autocast, so this is exact
    for them; the embeddings/norms lose bf16 rounding, far below the 8-bit Adam state
    quantization. With accumulation (micro-batch < 16) the stash stays fp32: the plain Trainer's
    math and memory. Global-norm clipping runs over the stash, same rule as clip_grad_norm_."""

    def create_optimizer(self, model=None):
        opt = super().create_optimizer(model)
        self.stash, self.clip_coef = {}, 1.0
        dtype = torch.bfloat16 if self.args.gradient_accumulation_steps == 1 else torch.float32
        step = opt.step

        def stash(p):
            g = p.grad.to(dtype)
            self.stash[p] = self.stash[p] + g if p in self.stash else g
            p.grad = None

        def stepwise(_opt, closure=None):
            for p in list(self.stash):
                p.grad = self.stash.pop(p).float().mul_(self.clip_coef)
                step()  # bnb steps only the parameters that have a grad
                p.grad = None

        for group in opt.param_groups:
            for p in group["params"]:
                p.register_post_accumulate_grad_hook(stash)
        opt.step = types.MethodType(stepwise, opt)  # bound: the LR scheduler wraps opt.step.__func__
        return opt

    def _clip_grad_norm(self, model):
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g, dtype=torch.float32) for g in self.stash.values()]))
        assert norm.isfinite(), f"non-finite grad norm {norm}"
        self.clip_coef = (self.args.max_grad_norm / (norm + 1e-6)).clamp(max=1.0)
        return norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--rank", type=int, default=32, help="rsLoRA rank (alpha fixed at 64); adapter named <task>-rR unless 32")
    ap.add_argument("--full", action="store_true", help="full-rank fine-tune of the same linear modules, exported as an exact PEFT adapter <task>-full")
    ap.add_argument("--full-all", action="store_true", help="full fine-tune of every parameter, saved as a complete bf16 checkpoint <task>-fullall")
    ap.add_argument("--epochs", type=float, default=None, help="default: RECIPE / the task's registry recipe")
    ap.add_argument("--lr", type=float, default=None, help="default: RECIPE (1e-5) / the task's registry recipe")
    ap.add_argument("--batch", type=int, default=None, help="per-device micro-batch (default 16; 1 for --full/--full-all); accum = 16//batch keeps effective batch 16 (lower it for long-sequence OOM)")
    ap.add_argument("--max-len", type=int, default=None, help="drop longer examples (default: RECIPE / registry); a single long row makes the whole padded batch spike (no grad checkpointing)")
    ap.add_argument("--optim", default=None, help="HF optimizer name; default adamw_8bit (paged_adamw_8bit for --full)")
    ap.add_argument("--attn", default=None, help="attn_implementation override; eager for LoRA rank <= 2 and the full fine-tunes of scripts/lr.sh (sdpa gives a non-finite gradient on one German batch)")
    ap.add_argument("--seed", type=int, default=0, help="LoRA init and data order")
    ap.add_argument("--constant", action="store_true", help="constant LR after the 5 warmup steps (default: linear decay to 0)")
    ap.add_argument("--suffix", default="", help="adapter name suffix, e.g. '-lr1e-4'")
    args = ap.parse_args()
    for k, v in (RECIPE | TASKS[args.task].get("recipe", {})).items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    # liger fused kernels: same math, faster, and fused-linear-CE never materializes the
    # fp32 [B, S, vocab] logit tensor (the peak-memory driver on long rows)
    liger = torch.cuda.is_available() and importlib.util.find_spec("liger_kernel") is not None and args.model not in HYBRID

    full = args.full or args.full_all
    args.batch = args.batch or (1 if full else 16)
    assert 16 % args.batch == 0, "--batch must divide the effective batch 16"
    name = args.task + ("-fullall" if args.full_all else "-full" if args.full else f"-r{args.rank}" if args.rank != 32 else "") + args.suffix
    out_dir = lora_dir(name, args.model)
    if (out_dir / ("config.json" if args.full_all else "adapter_config.json")).exists():
        sys.exit(f"finished fine-tune at {out_dir}")
    set_seed(args.seed)  # before get_peft_model: the LoRA init draws from it (Trainer seeds only after the model exists)
    tok = AutoTokenizer.from_pretrained(MODELS[args.model])
    model = load_lm(args.model, dtype=torch.float32 if full else torch.bfloat16, device_map="auto", **({"attn_implementation": args.attn} if args.attn else {}))
    tm = target_modules(model, args.model)
    if tm != TARGET_MODULES:
        print(f"[lora] target modules: {tm}")
    if full:  # fp32 master weights on the trainable parameters (the LoRA target modules, or everything for --full-all); the rest frozen in bf16
        base_params = {}
        for n, p in model.named_parameters():
            p.requires_grad = args.full_all or n.endswith(tuple(f"{m}.weight" for m in tm))
            if p.requires_grad:
                base_params[n] = p.detach().to("cpu", torch.bfloat16)
            else:
                p.data = p.data.to(torch.bfloat16)
        print(f"[full] {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B trainable params in fp32")
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=64, use_rslora=True, target_modules=tm,
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))

    ds = build_examples(train_rows(args.task), tok, max_len=args.max_len)
    trainer = (StashTrainer if full else Trainer)(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir), per_device_train_batch_size=args.batch, gradient_accumulation_steps=16 // args.batch,  # effective batch 16
            num_train_epochs=args.epochs, learning_rate=args.lr, warmup_steps=5,
            lr_scheduler_type="constant_with_warmup" if args.constant else "linear", optim=args.optim or ("paged_adamw_8bit" if full else "adamw_8bit"), weight_decay=0.01,
            bf16=True, logging_steps=10, save_strategy="no", seed=args.seed, report_to=[],
            gradient_checkpointing=full, gradient_checkpointing_kwargs={"use_reentrant": False} if full else None,
            use_liger_kernel=liger),  # no torch.compile: the variable batch shapes force constant recompiles
        train_dataset=ds, data_collator=DataCollatorForSeq2Seq(tok, padding=True))
    trainer.train()
    if args.full_all:
        trainer.optimizer = None
        model.to(torch.bfloat16).save_pretrained(str(out_dir))
    elif args.full:
        trainer.optimizer = None  # release the optimizer states before the export's delta pass
        out_dir.mkdir(parents=True, exist_ok=True)
        save_full_as_adapter(model, base_params, tm, out_dir, MODELS[args.model])
    else:
        model.save_pretrained(str(out_dir))
        (out_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history))
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
