"""Per-layer scores that drive the selection rules.

drift      : Model Parameter Difference (Aggarwal et al., Def. 2) per layer,
             ||W_instruct - W_base||_F / ||W_base||_F. CPU only, no GPU quota.
importance : gradient norm per layer on N Phase-2 samples (Gradient-Mask Tuning idea),
             measured through all-layer LoRA B matrices (B starts at 0, so dL/dB is a
             random projection of the full weight gradient).
"""
from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from typing import Dict, List

import torch

from .data import collate, length_grouped_batches
from .model import TARGET_MODULES, apply_selective_lora, forward_loss, n_layers_of

_LAYER_RE = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.weight$")


def _safetensor_files(repo_or_dir: str) -> List[str]:
    if os.path.isdir(repo_or_dir):
        d = repo_or_dir
    else:
        from huggingface_hub import snapshot_download
        d = snapshot_download(repo_or_dir, allow_patterns=["*.safetensors", "*.json"])
    return sorted(glob.glob(os.path.join(d, "*.safetensors")))


def _index(files):
    from safetensors import safe_open
    idx = {}
    for f in files:
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                idx[k] = f
    return idx


def layer_drift(base_id: str, tuned_id: str) -> Dict:
    """Streams tensors one at a time, so it runs in < 2 GB RAM for 0.5B/1.5B models."""
    from safetensors import safe_open
    bi, ti = _index(_safetensor_files(base_id)), _index(_safetensor_files(tuned_id))
    num, den = defaultdict(float), defaultdict(float)
    per_module = defaultdict(dict)
    for key in sorted(bi):
        m = _LAYER_RE.search(key)
        if not m or key not in ti:
            continue
        layer, mod = int(m.group(1)), m.group(2)
        with safe_open(bi[key], framework="pt") as hb, safe_open(ti[key], framework="pt") as ht:
            wb = hb.get_tensor(key).float()
            wt = ht.get_tensor(key).float()
        d2, b2 = (wt - wb).pow(2).sum().item(), wb.pow(2).sum().item()
        num[layer] += d2
        den[layer] += b2
        per_module[layer][mod] = (d2 / max(b2, 1e-12)) ** 0.5
    L = max(num) + 1
    drift = [(num[i] / max(den[i], 1e-12)) ** 0.5 for i in range(L)]
    return {"drift": drift, "drift_per_module": {str(i): per_module[i] for i in range(L)},
            "modules": TARGET_MODULES, "base": base_id, "tuned": tuned_id}


def layer_importance(model, tok, train_items: List[dict], n_samples: int = 256, bs: int = 8,
                     fp16: bool = True) -> Dict:
    """`model` is the plain base model (no LoRA yet). Returns per-layer grad norms."""
    dev = next(model.parameters()).device
    L = n_layers_of(model)
    peft_model = apply_selective_lora(model, None, dropout=0.0)
    peft_model.train()
    items = train_items[:n_samples]
    sq = torch.zeros(L, dtype=torch.float64)
    use_amp = fp16 and dev.type == "cuda"
    for b in length_grouped_batches([len(x["input_ids"]) for x in items], bs, seed=0):
        batch = {k: v.to(dev) for k, v in collate([items[i] for i in b], tok.pad_token_id).items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = forward_loss(peft_model, batch, chunked=True)
        (loss * (1024.0 if use_amp else 1.0)).backward()   # manual static scale; norms are relative
        for n, p in peft_model.named_parameters():
            if p.grad is not None and "lora_B" in n:
                m = re.search(r"layers\.(\d+)\.", n)
                sq[int(m.group(1))] += p.grad.float().pow(2).sum().item()
        peft_model.zero_grad(set_to_none=True)
    imp = sq.sqrt()
    imp = (imp / imp.max()).tolist() if imp.max() > 0 else imp.tolist()
    base = peft_model.unload()   # remove adapters, give back the clean base model
    return {"importance": imp, "n_samples": len(items)}, base
