"""Measure peak memory for many (layer set x memory lever) configs in minutes, no full training.
Also fits the cost model used by the budgeted selector.

    python scripts/probe_memory.py                    # Qwen2.5-0.5B, ~6-8 min on a T4
    python scripts/probe_memory.py --model_id Qwen/Qwen2.5-7B-Instruct --levers q4_gc_ck,q4_gc --sets all,top8,top4
Outputs: results/scores/memory_probes.json, results/scores/cost_model.json
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from sllora.model import (adapted_layers, apply_selective_lora, enable_gradient_checkpointing,
                          load_model_and_tokenizer, n_layers_of, probe_peak_memory)
from sllora.selection import CostModel, layer_strip
from sllora.utils import free_cuda, gpu_name, load_json, now_iso, save_json

LEVERS = {"plain": (False, False, False), "gc": (False, True, False), "ck": (False, False, True),
          "q4": (True, False, False), "q4_gc": (True, True, False), "q4_gc_ck": (True, True, True),
          "gc_ck": (False, True, True), "q4_ck": (True, False, True)}


def layer_sets(L):
    return {"all": list(range(L)), "top4": list(range(L - 4, L)), "top8": list(range(L - 8, L)),
            "top12": list(range(L - 12, L)), "bottom8": list(range(8)),
            "mid8": list(range(L // 2 - 4, L // 2 + 4)), "spread8": list(range(0, L, max(1, L // 8)))[:8],
            "pair_low_high": [L // 4, L - 1]}


p = argparse.ArgumentParser()
p.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--levers", default="plain,ck,gc_ck,q4_ck,q4_gc_ck")   # every sweep run uses chunked loss
p.add_argument("--sets", default="all,top4,top8,top12,bottom8,mid8,spread8,pair_low_high")
p.add_argument("--bs", type=int, default=4)
p.add_argument("--seq", type=int, default=512)
p.add_argument("--out", default="results/scores/memory_probes.json")
p.add_argument("--cost_out", default="results/scores/cost_model.json")
a = p.parse_args()
assert torch.cuda.is_available(), "needs a GPU runtime"

tag = a.model_id.split("/")[-1]
probes = load_json(a.out) if os.path.exists(a.out) else []
cost_models = load_json(a.cost_out) if os.path.exists(a.cost_out) else {}
for lever in a.levers.split(","):
    q4, gc, ck = LEVERS[lever]
    model, tok = load_model_and_tokenizer(a.model_id, quant4=q4)
    L = n_layers_of(model)
    vocab = model.get_output_embeddings().weight.shape[0]
    sets = layer_sets(L)
    rows = []
    for name in a.sets.split(","):
        S = sets[name]
        pm = apply_selective_lora(model, S)
        if gc:
            enable_gradient_checkpointing(pm)
        assert adapted_layers(pm) == S
        t = time.time()
        m = probe_peak_memory(pm, vocab, a.bs, a.seq, chunked=ck)
        row = {"model": tag, "lever": lever, "set": name, "layers": S, "strip": layer_strip(S, L),
               "peak_gb": m["probe_peak_gb"], "weights_gb": m["weights_gb"], "secs": round(time.time() - t, 1),
               "bs": a.bs, "seq": a.seq, "gpu": gpu_name(), "time": now_iso()}
        print(f"{tag:22s} {lever:9s} {name:14s} {row['strip']}  peak {row['peak_gb']:.3f} GB")
        rows.append(row)
        model = pm.unload()            # back to the clean base for the next layer set
        if gc:
            model.gradient_checkpointing_disable()
        free_cuda()
    probes = [r for r in probes if not (r["model"] == tag and r["lever"] == lever)] + rows
    if len(rows) >= 3:
        cm = CostModel.fit(L, [{"layers": r["layers"], "peak_gb": r["peak_gb"]} for r in rows])
        key = f"{tag}|q4={int(q4)},gc={int(gc)},ck={int(ck)}"
        cost_models[key] = cm.to_dict() | {"r2": round(cm.r2([{"layers": r["layers"], "peak_gb": r["peak_gb"]} for r in rows]), 4),
                                           "model": tag}
        if lever == "plain" and tag == "Qwen2.5-0.5B-Instruct":
            cost_models["default"] = cost_models[key]
        print(f"   cost model {key}: {cost_models[key]}")
    del model
    free_cuda()
    save_json(probes, a.out)
    save_json(cost_models, a.cost_out)
print("saved", a.out, a.cost_out)
