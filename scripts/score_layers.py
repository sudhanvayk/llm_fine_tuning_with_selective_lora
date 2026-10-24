"""Compute per-layer drift (CPU) and/or importance (GPU) -> results/scores/layer_scores.json

    # Colab CPU runtime or a laptop (no GPU quota used), ~3 min incl. download:
    python scripts/score_layers.py --drift
    # GPU, ~2 min:
    python scripts/score_layers.py --importance
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sllora import data as D
from sllora.model import load_model_and_tokenizer
from sllora.scoring import layer_drift, layer_importance
from sllora.utils import load_json, now_iso, save_json

p = argparse.ArgumentParser()
p.add_argument("--drift", action="store_true")
p.add_argument("--importance", action="store_true")
p.add_argument("--base_id", default="Qwen/Qwen2.5-0.5B")
p.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--langs", default=",".join(D.DEFAULT_LANGS))
p.add_argument("--n_samples", type=int, default=256)
p.add_argument("--out", default="results/scores/layer_scores.json")
p.add_argument("--split_cache", default="data/splits.json")
a = p.parse_args()

scores = load_json(a.out) if os.path.exists(a.out) else {}
if a.drift:
    d = layer_drift(a.base_id, a.model_id)
    scores.update(d)
    print("drift per layer:", [round(x, 4) for x in d["drift"]])
if a.importance:
    model, tok = load_model_and_tokenizer(a.model_id)
    split = D.build_splits(a.langs.split(","), 4000, 200, seed=42, cache_path=a.split_cache)
    items = D.tokenize_all(tok, split["train"][: a.n_samples], 512)
    imp, _ = layer_importance(model, tok, items, n_samples=a.n_samples)
    scores.update(imp)
    print("importance per layer:", [round(x, 3) for x in imp["importance"]])
scores["updated"] = now_iso()
save_json(scores, a.out)
print("saved", a.out)
