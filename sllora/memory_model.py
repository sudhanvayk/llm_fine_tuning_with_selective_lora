"""Analytic peak-memory estimate for LoRA / Selective-Layer LoRA on any decoder model.

Used for (a) the 7B projection in the demo, (b) checking it against measured 0.5B probes.
Activation rule of thumb (Korthikanti et al. 2022, flash/SDPA attention): ~34*h bytes per
token per layer in 16-bit; with checkpointing only the layer input (2*h bytes) is kept,
plus one layer being recomputed.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

GB = 1024 ** 3

CONFIGS: Dict[str, Dict] = {
    "Qwen2.5-0.5B": dict(L=24, h=896, inter=4864, heads=14, kv=2, vocab=151936, params=0.494e9, emb_params=0.136e9, tied=True),
    "Qwen2.5-1.5B": dict(L=28, h=1536, inter=8960, heads=12, kv=2, vocab=151936, params=1.54e9, emb_params=0.233e9, tied=True),
    "Qwen2.5-7B":   dict(L=28, h=3584, inter=18944, heads=28, kv=4, vocab=152064, params=7.62e9, emb_params=0.545e9, tied=False),
    "Mistral-7B":   dict(L=32, h=4096, inter=14336, heads=32, kv=8, vocab=32768, params=7.25e9, emb_params=0.134e9, tied=False),
    "Llama-3-8B":   dict(L=32, h=4096, inter=14336, heads=32, kv=8, vocab=128256, params=8.03e9, emb_params=0.525e9, tied=False),
}


def lora_params_per_layer(c: Dict, r: int = 16) -> int:
    h, inter = c["h"], c["inter"]
    kvd = h // c["heads"] * c["kv"]
    shapes = [(h, h), (h, kvd), (h, kvd), (h, h), (h, inter), (h, inter), (inter, h)]
    return sum(r * (i + o) for i, o in shapes)


def estimate(model: str, layers: Optional[Sequence[int]] = None, k_top: Optional[int] = None,
             bs: int = 4, seq: int = 512, quant4: bool = False, gc: bool = False, chunked: bool = False,
             r: int = 16, chunk_tokens: int = 256) -> Dict[str, float]:
    c = CONFIGS[model]
    L, h, V = c["L"], c["h"], c["vocab"]
    if layers is None:
        layers = list(range(L - k_top, L)) if k_top else list(range(L))
    T = bs * seq
    emb = c["emb_params"] * (1 if c["tied"] else 2)
    body = c["params"] - emb
    weights = body * (0.5625 if quant4 else 2.0) + emb * 2.0       # bnb keeps embeddings / lm_head in 16-bit
    adapters = lora_params_per_layer(c, r) * len(layers) * 16       # fp32 weight + grad + 2 Adam moments
    depth = L - min(layers)
    acts = T * (2 * h if gc else 34 * h) * depth + (T * 34 * h if gc else 0)
    logits = (min(T, chunk_tokens) * V * 10) if chunked else (T * V * 10)  # fp16 + fp32 upcast + fp32 grad
    total = weights + adapters + acts + logits
    return {k: round(v / GB, 3) for k, v in
            dict(weights=weights, adapters_opt=adapters, activations=acts, logits=logits, total=total).items()}


if __name__ == "__main__":
    for m in ["Qwen2.5-0.5B", "Mistral-7B", "Qwen2.5-7B"]:
        a = estimate(m)["total"]
        L = CONFIGS[m]["L"]
        s = estimate(m, k_top=8, quant4=True, gc=True, chunked=True)["total"]
        print(f"{m:14s} all-layer LoRA {a:6.2f} GB -> top-8 + NF4 + GC + chunked {s:5.2f} GB ({100*(1-s/a):.0f}% less)")
