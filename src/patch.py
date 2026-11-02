"""KV-cache patching: a source adapter prefills the prompt cache, a target generates over it.

A patch is named by its source-ownership level (`level`), the set of positions whose KV the source owns:
  "prompt-minus-last": source owns [0, L-1); the target encodes the final template token
      and everything after.
  "prompt": source owns all L prompt positions; the first answer token is sampled from the
      target's logits at the final template token over the source's [0, L-1) cache (the
      target KV written there is discarded).
  "all": source owns the prompt and re-encodes every sampled completion token, so the cache
      never holds target KV — the upper bound of what the KV channel can carry.
Every context token is either the shared prompt or target-sampled; no token is ever
sampled from source logits, so nothing can transfer as text instead of through the cache.
Gradual sweeps are teacher-forced: over a fixed own-cache rollout, the source owns a growing
prefix [0, k) of the recorded cache and the target re-encodes the rest, so every position
after the boundary is conditioned on the injected prefix (patched_completion_logits, batched
over boundaries in sweep_batch). On hybrids (Qwen3.5) the source's linear-attention state at k
comes from its recorded per-token writes (delta.py).
"""
import contextlib
import functools
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from transformers import AutoTokenizer, DynamicCache

from config import HYBRID, MODELS, adapter_path, load_lm
from delta import build_cache, end_state, linear_layers, record, substitute


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PatchModel:
    """Base model with task adapters attached; use('base') disables all adapters. A task whose
    directory holds a complete checkpoint (train.py --full-all) is a second model of its own,
    since a full update of embeddings, norms and head has no adapter form; use() switches
    self.lm between the adapter-carrying base and those models, and the caches, being plain
    KV tensors of the same architecture, pass between them unchanged."""

    def __init__(self, model: str, adapter_tasks: list[str], device: str | None = None, attn: str = "sdpa"):
        self.device = device or pick_device()
        self.model = model
        self.hybrid = model in HYBRID  # linear-attention layers: an end state, cropped via recorded writes (crop)
        base = MODELS[model]
        self.tok = AutoTokenizer.from_pretrained(base)
        lm = load_lm(model, dtype=torch.bfloat16, attn_implementation=attn)  # pinned: sweep masks are backend-sensitive
        self.full = {}
        for t in adapter_tasks:  # adapters keep their stored dtype (LoRAs are fp32 on disk; the exact full-update adapters are bf16 and ~1x the base size)
            path = adapter_path(t, model)
            if (path / "config.json").exists():
                self.full[t] = load_lm(model, path=str(path), dtype=torch.bfloat16, attn_implementation=attn).to(self.device).eval()
            elif not isinstance(lm, PeftModel):
                lm = PeftModel.from_pretrained(lm, str(path), adapter_name=t, autocast_adapter_dtype=False)
            else:
                lm.load_adapter(str(path), adapter_name=t, autocast_adapter_dtype=False)
        self.peft = self.lm = lm.to(self.device).eval()
        # Stop tokens, pinned explicitly: the checkpoint's generation_config eos list plus the
        # tokenizer's end-of-turn token (Qwen3.5-4B ships no generation_config.json, so HF
        # generate would otherwise not stop at <|im_end|>).
        gc_eos = self.lm.generation_config.eos_token_id
        self.eos_ids = sorted({*(gc_eos if isinstance(gc_eos, list) else [] if gc_eos is None else [gc_eos]), self.tok.eos_token_id})
        # One sampling rule for every model family: temperature/top-p only. The checkpoints'
        # generation_config top_k and repetition_penalty (Qwen2.5: 20 and 1.05) are switched off;
        # a repetition penalty also has no fixed per-position distribution for the sweep's KL.
        self.sampler = {"top_k": 0, "repetition_penalty": 1.0}

    def use(self, name: str):
        """One model for every row: "base", an adapter, or a complete checkpoint."""
        self.lm = self.full.get(name, self.peft)
        if name in self.full:
            return
        if not isinstance(self.lm, PeftModel):
            assert name == "base", f"no adapters loaded, got {name!r}"
        elif name == "base":
            self.lm.base_model.disable_adapter_layers()
        else:
            self.lm.base_model.enable_adapter_layers()
            self.lm.set_adapter(name)

    @functools.cached_property
    def loras(self) -> list:
        return [m for m in self.peft.modules() if isinstance(m, LoraLayer)]

    @contextmanager
    def rows(self, names: list[str]):
        """Per-row models for one batched forward or generate: row r runs names[r] ("base" or an
        adapter). One name for every row is plain use(). Otherwise every named adapter is active and
        each of its LoRA modules gets a per-row scaling (the adapter's scale on its own rows, 0 on the
        others), so every row computes exactly its own model (a zero-scaled LoRA term adds an exact
        0) at the cost of a uniform batch. PEFT's adapter_names routing instead gathers sub-batches
        in every module at every step (about 2x the decode step)."""
        if len(set(names)) == 1:
            self.use(names[0])
            yield
            return
        parts = [{x: 1.0} for x in names]
        adapters = sorted(set(names) - {"base"})
        assert not set(adapters) & self.full.keys(), "a complete checkpoint is a model of its own and cannot share a batch"
        self.use(adapters[0])
        self.lm.base_model.set_adapter(adapters)
        mask = {a: torch.tensor([p.get(a, 0.0) for p in parts], device=self.device).view(-1, 1, 1) for a in adapters}
        saved = [(m, a, m.scaling[a]) for m in self.loras for a in adapters if a in m.scaling]
        for m, a, s in saved:  # in the adapter's dtype, so a bf16 adapter (exact full update, scale 1) keeps its bf16 arithmetic
            m.scaling[a] = (mask[a] * s).to(m.lora_B[a].weight.dtype)
        try:
            yield
        finally:
            for m, a, s in saved:
                m.scaling[a] = s
            self.lm.base_model.set_adapter(adapters[0])

    def prompt_ids(self, text: str) -> torch.Tensor:
        rendered = self.tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False,
            enable_thinking=False)  # qwen3.5: non-thinking mode; other templates ignore the kwarg
        ids = self.tok(rendered, add_special_tokens=False)["input_ids"]
        return torch.tensor([ids], device=self.device)

    @torch.no_grad()
    def prefill(self, adapter: str | list[str], prompt_ids: torch.Tensor, n: int = 1, length: int | None = None) -> DynamicCache:
        """Prompt cache over [0, length), default [0, L-1): n rows of `adapter`, or one row per
        entry of a per-row list (rows()). Only the cache is kept (logits of the last position only)."""
        names = adapter if isinstance(adapter, list) else [adapter] * n
        length = prompt_ids.shape[1] - 1 if length is None else length
        with self.rows(names):
            return self.lm(prompt_ids[:, :length].expand(len(names), -1), use_cache=True, logits_to_keep=1).past_key_values

    @torch.no_grad()
    def generate(self, target: str | list[str], ids: torch.Tensor, cache, *, strip: int, max_new: int = 300,
                 temperature: float = 1.0, top_p: float = 1.0, return_sequences: bool = False):
        """One batched generate by the target (or per-row targets, rows()) over a cache, consumed,
        from per-row input ids [B, >=L]; `strip` is the prefix length dropped on decode.
        Returns completion texts; with return_sequences also the padded token sequences."""
        with self.rows(target if isinstance(target, list) else [target] * ids.shape[0]):
            seqs = self.lm.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), past_key_values=cache,
                do_sample=temperature > 0, temperature=temperature or None, top_p=top_p, **self.sampler,
                max_new_tokens=max_new, eos_token_id=self.eos_ids, pad_token_id=self.tok.eos_token_id)
        texts = [self.tok.decode(o[strip:], skip_special_tokens=True).strip() for o in seqs]
        return (texts, seqs) if return_sequences else texts

    @torch.no_grad()
    def _first_token(self, target: str | list[str], prompt_ids: torch.Tensor, cache, n: int, temperature: float, top_p: float = 1.0) -> torch.Tensor:
        """First answer token when the source owns all L prompt positions: the target (or
        per-row targets) encodes the final template token over the source's [0, L-1) cache
        (`cache`, consumed and then discarded along with the target KV it writes) and samples
        from those logits — the same computation that yields every later answer token."""
        att = torch.ones(n, prompt_ids.shape[1], dtype=torch.long, device=prompt_ids.device)
        with self.rows(target if isinstance(target, list) else [target] * n):
            logits = self.lm(prompt_ids[:, -1:].expand(n, -1), attention_mask=att, past_key_values=cache,
                             use_cache=True).logits[:, -1, :].float()
        return sample_token(logits, temperature, top_p)

    def generate_cells(self, prompt_ids: torch.Tensor, cells: list[str], level: str | None, n: int, *,
                       max_new: int = 300, temperature: float = 1.0, top_p: float = 1.0) -> dict[str, list[str]]:
        """n completions of one prompt per cell: own-cache cells ("em-med", "base") and
        "source>target" patches at source-ownership level `level`. Cells of one kind (a
        source-owned prefix, "prompt", "all") run as a single batch
        of n rows per cell with per-row models (rows()): one prefill writes every source's
        cache, one generate decodes every target. Decode is bound by per-step overhead, so the
        extra rows are nearly free. A cell with a complete checkpoint (a model of its
        own) runs alone. Returns {cell: completions}, batches in first-cell order."""
        L = prompt_ids.shape[1]
        opts = dict(max_new=max_new, temperature=temperature, top_p=top_p)
        groups = {}  # batch key -> (kind, cells)
        for c in cells:
            source, _, target = c.partition(">")
            # "prefix": the target generates over the source's [0, L-1) cache and encodes the rest
            kind = "prefix" if not target or level == "prompt-minus-last" else level
            key = kind if not {source, target} & self.full.keys() else c
            groups.setdefault(key, (kind, []))[1].append(c)
        out = {}
        for kind, g in groups.values():
            srcs = [c.partition(">")[0] for c in g for _ in range(n)]
            tgts = [c.partition(">")[2] or c for c in g for _ in range(n)]
            if kind == "all":
                comps = self.generate_source_all(srcs, tgts, prompt_ids, **opts)
            elif kind == "prompt":  # first answer token over the sources' [0, L-1) caches, the rest over their full [0, L) caches
                t1 = self._first_token(tgts, prompt_ids, self.prefill(srcs, prompt_ids), len(tgts), temperature, top_p)
                comps = self.generate(tgts, torch.cat([prompt_ids.expand(len(tgts), -1), t1], dim=1), self.prefill(srcs, prompt_ids, length=L),
                                      strip=L, max_new=max_new - 1, temperature=temperature, top_p=top_p)
                stop = torch.isin(t1.squeeze(1), torch.tensor(self.eos_ids, device=t1.device)).tolist()
                comps = ["" if s else c for c, s in zip(comps, stop, strict=True)]  # a first-token stop ends the answer
            else:
                comps = self.generate(tgts, prompt_ids.expand(len(tgts), -1), self.prefill(srcs, prompt_ids), strip=L, **opts)
            out.update({c: comps[j * n:(j + 1) * n] for j, c in enumerate(g)})
        return out

    @torch.no_grad()
    def generate_source_all(self, source: str | list[str], target: str | list[str], prompt_ids: torch.Tensor, n: int | None = None, *,
                            max_new: int = 300, temperature: float = 1.0, top_p: float = 1.0) -> list[str]:
        """Level "all": the source also re-encodes each sampled token, so the cache never
        holds target KV. source/target: one model each for n rows, or per-row lists. One
        doubled-batch forward per token (rows()): rows [0, R) read logits under their target,
        rows [R, 2R) run their source; afterwards the source half's new-position state is
        copied over the target half's (KV of the attention layers, the whole conv/recurrent
        state of linear-attention layers), so both halves stay the source-only cache. Same
        math and RNG stream as two forwards per token. The doubled cache and the first token's
        [0, L-1) cache come from their own prefills (a hybrid cache can't be cropped without recorded writes).
        When the source or the target is a complete checkpoint (a model of its own, no row
        routing), the two halves run as the two forwards instead: the target reads its
        logits over the cache, its new-position KV is cropped off, and the source re-encodes
        the same token in its place."""
        srcs = source if isinstance(source, list) else [source] * n
        tgts = target if isinstance(target, list) else [target] * n
        R, L = len(srcs), prompt_ids.shape[1]
        eos = self.tok.eos_token_id
        if (set(srcs) | set(tgts)) & self.full.keys():
            assert len(set(srcs)) == len(set(tgts)) == 1 and not self.hybrid, "a complete checkpoint runs as a cell of its own, and not on Qwen3.5"
            return self._source_all_two_models(srcs[0], tgts[0], prompt_ids, R, max_new=max_new, temperature=temperature, top_p=top_p)
        tok_t = self._first_token(tgts, prompt_ids, self.prefill(srcs, prompt_ids), R, temperature, top_p)
        cache2 = self.prefill(srcs + srcs, prompt_ids, length=L)
        eos_ids = torch.tensor(self.eos_ids, device=tok_t.device)
        seq, finished = tok_t, torch.isin(tok_t.squeeze(1), eos_ids)
        with self.rows(tgts + srcs):
            for _ in range(max_new - 1):
                att = torch.ones(2 * R, L + seq.shape[1], dtype=torch.long, device=tok_t.device)
                logits = self.lm(tok_t.repeat(2, 1), attention_mask=att, past_key_values=cache2, use_cache=True).logits[:R, -1, :].float()
                sync_source_half(cache2, R)
                nxt = sample_token(logits, temperature, top_p)
                nxt[finished] = eos
                seq = torch.cat([seq, nxt], dim=1)
                finished |= torch.isin(nxt.squeeze(1), eos_ids)
                if finished.all():
                    break
                tok_t = nxt
        return [self.tok.decode(row, skip_special_tokens=True).strip() for row in seq]

    @torch.no_grad()
    def _source_all_two_models(self, source: str, target: str, prompt_ids: torch.Tensor, n: int, *,
                               max_new: int, temperature: float, top_p: float) -> list[str]:
        eos = self.tok.eos_token_id
        cache = self.prefill(source, prompt_ids, n, prompt_ids.shape[1])
        tok_t = self._first_token(target, prompt_ids, crop_cache(cache, prompt_ids.shape[1] - 1), n, temperature, top_p)
        eos_ids = torch.tensor(self.eos_ids, device=tok_t.device)
        seq, finished = tok_t, torch.isin(tok_t.squeeze(1), eos_ids)
        for _ in range(max_new - 1):
            att = torch.ones(n, prompt_ids.shape[1] + seq.shape[1], dtype=torch.long, device=tok_t.device)
            self.use(target)
            logits = self.lm(tok_t, attention_mask=att, past_key_values=cache, use_cache=True).logits[:, -1, :].float()
            cache.crop(-1)
            self.use(source)
            self.lm(tok_t, attention_mask=att, past_key_values=cache, use_cache=True)
            nxt = sample_token(logits, temperature, top_p)
            nxt[finished] = eos
            seq = torch.cat([seq, nxt], dim=1)
            finished |= torch.isin(nxt.squeeze(1), eos_ids)
            if finished.all():
                break
            tok_t = nxt
        return [self.tok.decode(row, skip_special_tokens=True).strip() for row in seq]

    @torch.no_grad()
    def completion_logits(self, adapter: str, full_ids: torch.Tensor, prompt_len: int, with_cache: bool = False):
        """Plain full-context forward under `adapter`: logits predicting full_ids[:, prompt_len:].
        with_cache=True also returns the full [0, T) cache (the source recording for sweeps).
        On hybrids the cache also carries the linear-attention layers' per-token writes (.writes, delta.py), so crop() can cut it."""
        self.use(adapter)
        with record(self.lm) if with_cache else contextlib.nullcontext({}) as writes:
            out = self.lm(full_ids, attention_mask=torch.ones_like(full_ids), use_cache=with_cache)
        logits = out.logits[:, prompt_len - 1:-1]
        if with_cache:
            out.past_key_values.writes = writes
        return (logits, out.past_key_values) if with_cache else logits

    def crop(self, cache: DynamicCache, length: int, rows: int = 1) -> DynamicCache:
        """Independent copy of a full cache's first `length` positions, its batch repeated `rows`
        times (row order: copy-major). A hybrid cache needs its recorded writes (completion_logits):
        each linear-attention layer's state at `length` is recomputed from them."""
        def rep(t):
            return t.repeat(rows, *[1] * (t.dim() - 1))
        if not self.hybrid:
            return _build((rep(k[:, :, :length]), rep(v[:, :, :length])) for k, v in (kv(cache, i) for i in range(len(cache.layers))))
        attn = {i: (rep(lyr.keys[:, :, :length]), rep(lyr.values[:, :, :length])) for i, lyr in enumerate(cache.layers) if hasattr(lyr, "keys")}
        linear = {i: tuple(map(rep, end_state(m, *(t[:, :length] for t in cache.writes[i])))) for i, m in linear_layers(self.lm).items()}
        return build_cache(self.lm, attn, linear)

    @torch.no_grad()
    def patched_completion_logits(self, target: str, full_ids: torch.Tensor, source_cache: DynamicCache, k: int, prompt_len: int):
        """Teacher-forced sweep point, one boundary at a time (the reference semantics of
        sweep_batch): the source owns positions [0, k); the target re-encodes
        full_ids[:, k:] over the cropped source cache, so every position after the boundary is
        conditioned on the injected prefix. Returns (logits, start) with logits predicting the
        completion tokens full_ids[:, start:]. start = prompt_len for every k <= prompt_len:
        at k == prompt_len the first completion token's prediction is restored by encoding the
        final template token over the source's [0, k-1) cache (the "prompt"-level first-token
        computation), so the scored support is identical across the whole prompt-side sweep.
        Past the prompt, start = k + 1 shrinks with the boundary."""
        self.use(target)
        att = torch.ones_like(full_ids)
        out = self.lm(full_ids[:, k:], attention_mask=att, past_key_values=self.crop(source_cache, k) if k else None, use_cache=True)
        if k < prompt_len:
            return out.logits[:, prompt_len - 1 - k:-1], prompt_len
        if k == prompt_len:
            first = self.lm(full_ids[:, k - 1:k], attention_mask=att[:, :k], past_key_values=self.crop(source_cache, k - 1), use_cache=True).logits
            return torch.cat([first, out.logits[:, :-1]], dim=1), prompt_len
        return out.logits[:, :-1], k + 1

    @torch.no_grad()
    def sweep_batch(self, target: str, seqs: torch.Tensor, source_cache: DynamicCache, ks: list[int],
                    prompt_len: int, ref_source: torch.Tensor, ref_target: torch.Tensor, mask: torch.Tensor,
                    temperature: float, top_p: float, tok_budget: int = 12288) -> dict:
        """Batched teacher-forced sweep: every boundary k in `ks` scored in one padded batch,
        executed in memory-sized chunks of consecutive k (logits memory forbids one forward).
        Semantics match patched_completion_logits per k: the source owns [0, k), the target
        re-encodes seqs[:, k:], scored support starts at prompt_len for k <= prompt_len, else k+1;
        the k==prompt_len first-token restore is spliced from the k-1 row (the chunker keeps them
        together). Chunk mechanics (HF cache_position is batch-uniform): the chunk's cache is the source
        cache cropped to its k_max and repeated over rows; inputs are seqs[:, k_min:] for every
        row with explicit uniform position_ids; a per-row 2D key mask hides cache slots >= k_row
        and input slots < k_row (those slots compute unused garbage, never attended). A hybrid's
        linear-attention layers have no slots to hide: they start from the source's state at
        k_min, and on the input slots < k_row the source's recorded writes replace the target's
        (delta.substitute), so each row's state reaches k_row as the source's.
        ref_source/ref_target: own-cache reference logits [n, T-prompt_len, V]; mask: completion_mask.
        Returns {k: [per-sample dict|None]} with fp32 summary stats and fp16 per-position arrays
        (tv/kl truncated + uncapped against the source's and the target's own-cache reference, and
        the target's log-prob of the realized token)."""
        self.use(target)
        n, T = seqs.shape
        L = prompt_len
        t = temperature if temperature > 0 else 1.0
        ks = sorted(ks)
        if L in ks and L - 1 not in ks:
            ks = sorted(ks + [L - 1])  # helper row for the k==L first-token splice
        # k=0 runs alone and unpadded (past=None, exact-length input): the batched mechanics'
        # extra masked slots perturb bf16 kernel tiling and the recompute trajectory diverges
        # chaotically (~0.015 TV floor, the same as any shape change); the k=0 calibration anchor
        # stays bitwise-exact instead.
        chunks, i = [[0]] if ks and ks[0] == 0 else [], int(bool(ks and ks[0] == 0))
        while i < len(ks):
            m = max(1, tok_budget // ((T - ks[i]) * n))
            j = min(len(ks), i + m)
            if j < len(ks) and ks[j] == L and ks[j - 1] == L - 1:
                j += 1  # keep the k==L row with its k-1 splice source
            chunks.append(ks[i:j])
            i = j
        out = {}
        for chunk in chunks:
            k_min, k_max = chunk[0], chunk[-1]
            nk = len(chunk)
            R = nk * n
            krow = torch.tensor(chunk, device=seqs.device).repeat_interleave(n)[:, None]  # [R, 1]
            past, swap = None, contextlib.nullcontext()
            if k_max and not self.hybrid:
                past = self.crop(source_cache, k_max, nk)
            elif k_max:  # linear-attention layers: the source's state at k_min, then its own writes on the input slots < k_row
                past = self.crop(source_cache, k_min, nk)
                for li, lyr in enumerate(past.layers):
                    if hasattr(lyr, "keys"):
                        lyr.keys, lyr.values = (t[:, :, :k_max].repeat(nk, 1, 1, 1) for t in kv(source_cache, li))
                own = torch.arange(k_min, T, device=seqs.device)[None] < krow
                swap = substitute(self.lm, own, {li: tuple(t[:, k_min:].repeat(nk, 1, 1) for t in w) for li, w in source_cache.writes.items()})
            att = torch.cat([(torch.arange(k_max, device=seqs.device)[None] < krow).long(),
                             (torch.arange(k_min, T, device=seqs.device)[None] >= krow).long()], dim=1)
            pos = torch.arange(k_min, T, device=seqs.device)[None].expand(R, -1)
            with swap:
                logits = self.lm(seqs[:, k_min:].repeat(nk, 1), attention_mask=att, position_ids=pos,
                                 past_key_values=past, use_cache=past is not None).logits
            del past
            for ki, k in enumerate(chunk):
                start = L if k <= L else k + 1
                rows = []
                for s in range(n):
                    m = mask[s, start - L:]
                    if not m.any():
                        rows.append(None)
                        continue
                    lo = start - 1 - k_min
                    qrow = logits[ki * n + s, lo:T - 1 - k_min]
                    if k == L:  # first prediction from the k-1 row (target-fresh KV at L-1, source cache [0, L-1))
                        qrow = torch.cat([logits[(ki - 1) * n + s, lo:lo + 1], qrow[1:]], dim=0)
                    sl = slice(start - L, None)
                    ls = F.log_softmax(ref_source[s, sl][m].float() / t, -1)
                    lt = F.log_softmax(ref_target[s, sl][m].float() / t, -1)
                    rows.append(score(qrow[m], ls, lt, seqs[s, start:][m], t, top_p))
                out[k] = rows
            del logits
        return out


def sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """One token per row from the temperature/top-p distribution; argmax at temperature 0.
    Matches HF generate's top-p filter (the token crossing the threshold is kept)."""
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    probs = F.softmax(logits / temperature, -1)
    sp, idx = probs.sort(-1, descending=True)
    probs = torch.zeros_like(probs).scatter_(-1, idx, sp * ((sp.cumsum(-1) - sp) < top_p))
    return torch.multinomial(probs / probs.sum(-1, keepdim=True), 1)


# --- DynamicCache helpers ---
def kv(cache: DynamicCache, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    lyr = cache.layers[layer]
    return lyr.keys, lyr.values


def sync_source_half(cache, n: int):
    """After a doubled-batch @all step, make rows [0, n) source-only again: copy the source
    half's current-position KV over the target half's (attention layers), and its whole
    fixed-size conv/recurrent states (linear-attention layers of hybrid caches)."""
    for lyr in cache.layers:
        if hasattr(lyr, "keys"):
            lyr.keys[:n, :, -1] = lyr.keys[n:, :, -1]
            lyr.values[:n, :, -1] = lyr.values[n:, :, -1]
        else:
            for i in range(lyr.number_of_states):
                if lyr.is_conv_states_initialized[i]:
                    lyr.conv_states[i][:n] = lyr.conv_states[i][n:]
                if lyr.is_recurrent_states_initialized[i]:
                    lyr.recurrent_states[i][:n] = lyr.recurrent_states[i][n:]


def _build(pairs) -> DynamicCache:
    out = DynamicCache()
    for i, (k, v) in enumerate(pairs):
        out.update(k.clone(), v.clone(), i)
    return out


def crop_cache(cache: DynamicCache, length: int) -> DynamicCache:
    """Independent copy of the first `length` positions."""
    return _build(tuple(t[..., :length, :] for t in kv(cache, i)) for i in range(len(cache.layers)))


# --- per-token distributional metrics over on-policy completions ---
def completion_mask(completions: torch.Tensor, eos_ids: list[int]) -> torch.Tensor:
    """True up to and including each row's first stop token; False for the padding after it."""
    iseos = torch.isin(completions, torch.tensor(eos_ids, device=completions.device))
    return ~(iseos.cumsum(-1) - iseos.int() > 0)


def nucleus_keep(lq: torch.Tensor, top_p: float) -> torch.Tensor:
    """Dense bool top-p nucleus of log-probs lq [P, V] (token crossing the threshold kept),
    via top-k with escalation instead of a full-vocab argsort."""
    for K in (512, 4096, lq.shape[-1]):
        vals, idx = lq.topk(min(K, lq.shape[-1]), -1)
        p = vals.exp()
        inside = (p.cumsum(-1) - p) < top_p
        if K >= lq.shape[-1] or not inside[..., -1].any():  # cutoff reached within K everywhere
            return torch.zeros_like(lq, dtype=torch.bool).scatter_(-1, idx, inside)
    raise AssertionError("unreachable")


def _pair_stats(lq, pq, keep, lp, tiny):
    """tv, truncated KL (q's nucleus, renormalized), uncapped KL;
    all per position, fp32 [P]."""
    tv = 0.5 * (pq - lp.exp()).abs().sum(-1)
    kl_full = (pq * (lq - lp)).sum(-1)
    q = pq * keep
    q = q / q.sum(-1, keepdim=True)
    lpn = lp - torch.logsumexp(lp.masked_fill(~keep, -torch.inf), -1, keepdim=True)
    kl_tr = (q * (q.clamp_min(tiny).log() - lpn) * keep).sum(-1)
    return tv, kl_tr, kl_full


def _summ(t: torch.Tensor) -> dict:
    return {"mean": t.mean().item(), "p95": t.quantile(0.95).item(), "pp": t.half().cpu().numpy()}


def score(q: torch.Tensor, ls: torch.Tensor, lt: torch.Tensor, tokens: torch.Tensor, temperature: float, top_p: float) -> dict:
    """One patched completion's per-position stats: q = patched logits [P, V] at the scored
    positions, ls / lt = the source / target own-cache log-probs there (temperature-scaled),
    tokens = the realized tokens [P]. tv / truncated KL / uncapped KL against each reference
    (_summ dicts) and the patched
    log-prob of each realized token (fp16)."""
    lq = F.log_softmax(q.float() / temperature, -1)
    pq = lq.exp()
    keep = nucleus_keep(lq, top_p)
    tiny = torch.finfo(torch.float32).tiny
    tv_s, kl_s, kl_sf = _pair_stats(lq, pq, keep, ls, tiny)
    tv_t, kl_t, kl_tf = _pair_stats(lq, pq, keep, lt, tiny)
    return {"tv_source": _summ(tv_s), "tv_target": _summ(tv_t), "kl_source": _summ(kl_s), "kl_target": _summ(kl_t),
            "kl_source_full": _summ(kl_sf), "kl_target_full": _summ(kl_tf),
            "lp_q": lq.gather(-1, tokens.unsqueeze(-1)).squeeze(-1).half().cpu().numpy()}


def ref_stats(ref_source: torch.Tensor, ref_target: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor,
              temperature: float) -> list[dict]:
    """k-independent per-completion reference stats: per position the tv between the source
    and target own-cache references, each reference's log-prob of the realized token, and
    each one's entropy. One dict
    per sample with fp16 per-position arrays (None where the completion is empty)."""
    t = temperature if temperature > 0 else 1.0
    out = []
    for ls_row, lt_row, tg, m in zip(ref_source, ref_target, tokens, mask, strict=True):
        if not m.any():
            out.append(None)
            continue
        ls = F.log_softmax(ls_row[m].float() / t, -1)
        lt = F.log_softmax(lt_row[m].float() / t, -1)
        g = tg[m].unsqueeze(-1)
        out.append({"tv_refs": (0.5 * (ls.exp() - lt.exp()).abs().sum(-1)).half().cpu().numpy(),
                    "lp_source": ls.gather(-1, g).squeeze(-1).half().cpu().numpy(),
                    "lp_target": lt.gather(-1, g).squeeze(-1).half().cpu().numpy(),
                    "ent_source": -(ls.exp() * ls).sum(-1).half().cpu().numpy(),
                    "ent_target": -(lt.exp() * lt).sum(-1).half().cpu().numpy()})
    return out

