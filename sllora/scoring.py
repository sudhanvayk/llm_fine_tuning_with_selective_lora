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


def layer_importance_protocol(model, tok, items: List[dict], r: int = 4, targets=("q_proj", "v_proj"),
                              bs: int = 4, seed: int = 0, fp16: bool = True) -> Dict:
    """Spec-style gradient scoring, per batch, with a stability check.

    Protocol
      1. temporary LoRA (rank r) on `targets` in EVERY layer, dropout 0, fixed seed
      2. for each scoring batch: response-token loss -> backward -> per-layer score
         s_l(batch) = L2 norm of the concatenated grad of all LoRA-B matrices in layer l
      3. S_l = mean over batches of s_l(batch); also std and a split-half rank correlation
      4. adapters are removed with unload() -> the caller gets the clean base model back

    Why B: LoRA initialises B = 0, so dL/dA = 0 at step 0 (A's gradient is B^T * ...) while
    dL/dB = (dL/dh) (A x)^T is non-zero: a random projection (through A) of the full
    weight gradient. Limitation: it depends on the random A, sees only the chosen modules,
    and is measured at initialisation, so it is a heuristic for where learning signal is
    strong, not a guarantee of better validation performance. No optimizer step is taken,
    so per-batch gradients are independent (zeroed after each batch)."""
    import numpy as np
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    dev = next(model.parameters()).device
    L = n_layers_of(model)
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, bias="none", target_modules=list(targets),
                     task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    pm.train()
    use_amp = fp16 and dev.type == "cuda"
    per_batch = []
    for b in length_grouped_batches([len(x["input_ids"]) for x in items], bs, seed=seed):
        batch = {k: v.to(dev) for k, v in collate([items[i] for i in b], tok.pad_token_id).items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = forward_loss(pm, batch, chunked=True)
        if not torch.isfinite(loss):
            pm.zero_grad(set_to_none=True)
            continue
        (loss * (1024.0 if use_amp else 1.0)).backward()
        sq = np.zeros(L)
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                assert p.grad is not None, f"missing gradient on {n}"
                g = p.grad.float()
                if use_amp:
                    g = g / 1024.0
                sq[int(re.search(r"layers\.(\d+)\.", n).group(1))] += g.pow(2).sum().item()
        per_batch.append(np.sqrt(sq))
        pm.zero_grad(set_to_none=True)
    base = pm.unload()
    M = np.array(per_batch)                                   # [n_batches, L]
    mean, std = M.mean(0), M.std(0)
    norm = mean / mean.max() if mean.max() > 0 else mean
    half = len(M) // 2
    stab = None
    if half >= 1:
        ra = np.argsort(np.argsort(-M[:half].mean(0)))
        rb = np.argsort(np.argsort(-M[half:].mean(0)))
        stab = float(np.corrcoef(ra, rb)[0, 1])               # Spearman rank correlation of the two halves
    return {"scores": norm.tolist(), "raw_mean": mean.tolist(), "raw_std": std.tolist(),
            "per_batch": M.tolist(), "n_batches": len(M), "n_samples": len(items),
            "rank": r, "targets": list(targets), "split_half_spearman": stab,
            "ranking": [int(i) for i in np.argsort(-mean)]}, base
