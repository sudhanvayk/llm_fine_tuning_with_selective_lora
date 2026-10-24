"""Dataset Embedding Similarity (DES, Aggarwal et al. 2024, Def. 1) with LaBSE. CPU only, ~10 min.

DES(D1, D2) = < mean LaBSE embedding of D1 , mean LaBSE embedding of D2 >   (both L2-normalised)
D1 = English Alpaca (stand-in for the Phase-1 instruction tasks; Qwen's own data is not public)
D2 = our MultiAlpaca languages (ar, ru, th), each separately and pooled.
High DES -> similar tasks -> the paper observed little forgetting.

    pip install -q sentence-transformers
    python scripts/des.py            # writes results/scores/des.json
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from sllora.data import load_multialpaca_lang

N = 500  # samples per dataset, as in the paper
rng = random.Random(42)


def text(ex):
    s = ex["instruction"] + ((" " + ex["input"]) if ex.get("input") else "") + " " + ex["output"]
    return s[:1000]


def mean_emb(model, texts):
    e = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    m = e.mean(axis=0)
    return m / np.linalg.norm(m)


model = SentenceTransformer("sentence-transformers/LaBSE", device="cpu")
alp = load_dataset("tatsu-lab/alpaca", split="train")
alp_rows = [alp[i] for i in rng.sample(range(len(alp)), N)]
e_en = mean_emb(model, [text(r) for r in alp_rows])

out, pooled = {}, []
for lang in ["ar", "ru", "th"]:
    rows = load_multialpaca_lang(lang)
    rows = rng.sample(rows, N)
    pooled += rows
    out[f"alpaca_en_vs_{lang}"] = float(e_en @ mean_emb(model, [text(r) for r in rows]))
    print(lang, round(out[f"alpaca_en_vs_{lang}"], 3))
out["alpaca_en_vs_multialpaca_ar_ru_th"] = float(e_en @ mean_emb(model, [text(r) for r in rng.sample(pooled, N)]))
print("pooled", round(out["alpaca_en_vs_multialpaca_ar_ru_th"], 3))
os.makedirs("results/scores", exist_ok=True)
json.dump(out, open("results/scores/des.json", "w"), indent=2)
print("saved results/scores/des.json  (paper reference: Alpaca vs MultiAlpaca = 0.924)")
