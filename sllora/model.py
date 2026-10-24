"""Model loading, Selective-Layer LoRA, memory levers and memory measurement.

Memory levers (each is a flag so its saving can be measured on its own):
  * selective layers  -> backprop stops at the lowest adapted layer
  * quant4            -> NF4 4-bit frozen base (bitsandbytes)
  * gc                -> gradient checkpointing (non-reentrant)
  * chunked           -> chunked lm_head + cross-entropy, never builds the full [B,T,V] logits
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .utils import GB, cuda_sync, free_cuda

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _disable_stale_torchao() -> None:
    """Colab ships torchao 0.10, and recent peft raises ImportError on any torchao < 0.16,
    even though we never use torchao. If the installed one is too old, tell peft it isn't there.
    (The clean fix is `pip uninstall -y torchao`; this is the safety net.)"""
    import importlib
    try:
        import peft.import_utils as iu
        iu.is_torchao_available()
        return                                    # absent or new enough: nothing to do
    except ImportError:
        pass
    except Exception:
        return
    for name in ("peft.import_utils", "peft.tuners.lora.torchao", "peft.utils.quantization_utils"):
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "is_torchao_available"):
                mod.is_torchao_available = lambda: False
        except Exception:
            pass
    print("[sllora] old torchao found; told peft to ignore it (run `pip uninstall -y torchao` to silence this)")


_disable_stale_torchao()
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


# --------------------------------------------------------------------------- loading
def load_model_and_tokenizer(model_id: str, quant4: bool = False, dtype: str = "float16",
                             device: Optional[str] = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    if device == "cpu":
        torch_dtype = torch.float32
    kw = dict(dtype=torch_dtype, attn_implementation="sdpa", low_cpu_mem_usage=True)
    if quant4:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16 if torch_dtype == torch.float16 else torch.bfloat16)
        kw["device_map"] = {"": 0}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    except TypeError:  # transformers < 4.56 uses torch_dtype
        kw["torch_dtype"] = kw.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if not quant4:
        model.to(device)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


def n_layers_of(model) -> int:
    cfg = model.config
    return getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers


# --------------------------------------------------------------------------- LoRA
def apply_selective_lora(model, layers: Optional[Iterable[int]], r: int = 16, alpha: int = 32,
                         dropout: float = 0.05):
    """layers=None -> all layers (standard LoRA baseline)."""
    from peft import LoraConfig, get_peft_model
    L = n_layers_of(model)
    layers = None if layers is None else sorted(set(int(i) for i in layers))
    if layers is not None and len(layers) == L:
        layers = None
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=TARGET_MODULES, task_type="CAUSAL_LM",
                     layers_to_transform=layers, layers_pattern="layers" if layers is not None else None)
    return get_peft_model(model, cfg)


def enable_gradient_checkpointing(model) -> None:
    """Non-reentrant checkpointing WITHOUT input grads.

    Gotcha: recent transformers versions call `enable_input_require_grads()`
    inside `gradient_checkpointing_enable()`. That makes the embedding output
    require grad, so the backward pass runs through EVERY frozen layer below
    the adapters (wasted compute, plus their checkpointed inputs are stored).
    Non-reentrant checkpointing does not need it, so we switch it back off."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(base, "disable_input_require_grads"):
        try:
            base.disable_input_require_grads()
        except Exception:
            pass
    base.config.use_cache = False


def adapted_layers(model) -> List[int]:
    """Layer indices that actually received LoRA params (sanity check)."""
    out = set()
    for n, p in model.named_parameters():
        if p.requires_grad:
            m = _LAYER_RE.search(n)
            if m:
                out.add(int(m.group(1)))
    return sorted(out)


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# --------------------------------------------------------------------------- loss
def _ce_chunk(h, W, y):
    return F.cross_entropy((h @ W.t()).float(), y, reduction="sum")


def forward_loss(model, batch, chunked: bool = False, chunk_tokens: int = 256):
    """Mean next-token loss over answer tokens.

    chunked=False: standard HF path, materialises [B, T, V] logits (V=151,936 for Qwen!).
    chunked=True : keep only answer positions, then lm_head + CE in checkpointed chunks,
                   so at most chunk_tokens x V logits exist at any time."""
    if not chunked:
        return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                     labels=batch["labels"]).loss
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]).last_hidden_state
    H = hidden.shape[-1]
    h = hidden[:, :-1].reshape(-1, H)
    y = batch["labels"][:, 1:].reshape(-1)
    keep = y != -100
    h, y = h[keep], y[keep]
    W = base.get_output_embeddings().weight
    n = max(int(y.numel()), 1)
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, y.numel(), chunk_tokens):
        total = total + checkpoint(_ce_chunk, h[i:i + chunk_tokens], W, y[i:i + chunk_tokens],
                                   use_reentrant=False)
    return total / n


# --------------------------------------------------------------------------- measurement
def synthetic_batch(vocab: int, bs: int, seq_len: int, device: str, prompt_frac: float = 0.25, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(100, vocab - 100, (bs, seq_len), generator=g)
    labels = ids.clone()
    labels[:, : int(seq_len * prompt_frac)] = -100
    return {"input_ids": ids.to(device), "attention_mask": torch.ones_like(ids).to(device),
            "labels": labels.to(device)}


def saved_activation_bytes(model, batch, chunked=False) -> int:
    """Bytes autograd saves for backward during one forward pass.
    Device independent (works on CPU), so it proves the mechanism even without a GPU."""
    total = {"b": 0}
    param_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
    seen = set()

    def pack(t):
        try:
            s = t.untyped_storage()
            key = s.data_ptr()
            if key not in seen and key not in param_storages:   # weights are resident anyway
                seen.add(key)
                total["b"] += s.nbytes()
        except Exception:
            pass
        return t

    model.train()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = forward_loss(model, batch, chunked=chunked)
    del loss
    return total["b"]


def probe_peak_memory(model, vocab: int, bs: int = 4, seq_len: int = 512, chunked: bool = False,
                      steps: int = 2, fp16: bool = True) -> dict:
    """Deterministic peak-memory measurement: `steps` full forward+backward+AdamW
    steps on a fixed synthetic [bs, seq_len] batch. lr=0 so weights do not move.
    This is the headline number for every configuration."""
    if not torch.cuda.is_available():
        return {"probe_peak_gb": None, "weights_gb": None}
    dev = "cuda"
    free_cuda()
    weights = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    batch = synthetic_batch(vocab, bs, seq_len, dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    model.train()
    for _ in range(steps):
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            loss = forward_loss(model, batch, chunked=chunked)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    cuda_sync()
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    del opt, batch, loss
    free_cuda()
    torch.cuda.reset_peak_memory_stats()
    return {"probe_peak_gb": round(peak / GB, 4), "probe_reserved_gb": round(reserved / GB, 4),
            "weights_gb": round(weights / GB, 4), "probe_bs": bs, "probe_seq_len": seq_len}
