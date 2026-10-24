"""MultiAlpaca loading, deterministic splits, chat formatting, label masking, batching.

MultiAlpaca (DAMO-NLP-MT/multialpaca, Apache-2.0) ships one JSONL per language with
fields instruction / input / output. `input` is often null.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List

import torch

from .utils import load_json, save_json

MULTIALPACA_REPO = "DAMO-NLP-MT/multialpaca"
MULTIALPACA_LANGS = ["ar", "de", "es", "fr", "id", "ja", "ko", "pt", "ru", "th", "vi"]
# lm-eval Belebele task suffix for each MultiAlpaca language
BELEBELE = {"ar": "arb_Arab", "de": "deu_Latn", "es": "spa_Latn", "fr": "fra_Latn",
            "id": "ind_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang", "pt": "por_Latn",
            "ru": "rus_Cyrl", "th": "tha_Thai", "vi": "vie_Latn"}
DEFAULT_LANGS = ["ar", "ru", "th"]  # three scripts: Arabic, Cyrillic, Thai


def load_multialpaca_lang(lang: str) -> List[dict]:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(MULTIALPACA_REPO, f"{lang}.jsonl", repo_type="dataset")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ex = {"instruction": (r.get("instruction") or "").strip(),
                  "input": (r.get("input") or "").strip(),
                  "output": (r.get("output") or "").strip(),
                  "lang": lang}
            if ex["instruction"] and ex["output"]:
                rows.append(ex)
    return rows


def build_splits(langs=DEFAULT_LANGS, n_train=4000, n_eval=200, seed=42,
                 cache_path: str | None = None) -> Dict:
    """Same seed -> same split on every Colab account. Cached as JSON so all
    three team members train and evaluate on byte-identical data."""
    if cache_path and os.path.exists(cache_path):
        sp = load_json(cache_path)
        if sp.get("meta") == {"langs": list(langs), "n_train": n_train, "n_eval": n_eval, "seed": seed}:
            return sp
    train, evals = [], {}
    for lang in langs:
        rows = load_multialpaca_lang(lang)
        rng = random.Random(f"{seed}-{lang}")
        rng.shuffle(rows)
        evals[lang] = rows[:n_eval]
        train += rows[n_eval:n_eval + n_train]
    random.Random(seed).shuffle(train)
    sp = {"meta": {"langs": list(langs), "n_train": n_train, "n_eval": n_eval, "seed": seed},
          "train": train, "eval": evals}
    if cache_path:
        save_json(sp, cache_path)
    return sp


def to_messages(ex: dict) -> List[dict]:
    user = ex["instruction"] + (("\n\n" + ex["input"]) if ex.get("input") else "")
    return [{"role": "user", "content": user}, {"role": "assistant", "content": ex["output"]}]


def tokenize_example(tok, ex: dict, max_len: int = 512) -> dict | None:
    """Loss only on the assistant answer: prompt tokens get label -100."""
    msgs = to_messages(ex)
    prompt_text = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    if not full_text.startswith(prompt_text):  # defensive: should hold for Qwen/Llama/Mistral templates
        answer_text = ex["output"] + (tok.eos_token or "")
    else:
        answer_text = full_text[len(prompt_text):]
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer_text, add_special_tokens=False)["input_ids"]
    if len(p_ids) > max_len - 16:          # no room left for an answer -> drop
        return None
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return {"input_ids": ids, "labels": labels, "lang": ex["lang"]}


def tokenize_all(tok, examples: List[dict], max_len: int = 512) -> List[dict]:
    out = [tokenize_example(tok, ex, max_len) for ex in examples]
    return [o for o in out if o is not None]


def collate(items: List[dict], pad_id: int) -> Dict[str, torch.Tensor]:
    T = max(len(x["input_ids"]) for x in items)
    ids = torch.full((len(items), T), pad_id, dtype=torch.long)
    lab = torch.full((len(items), T), -100, dtype=torch.long)
    att = torch.zeros((len(items), T), dtype=torch.long)
    for i, x in enumerate(items):
        n = len(x["input_ids"])
        ids[i, :n] = torch.tensor(x["input_ids"])
        lab[i, :n] = torch.tensor(x["labels"])
        att[i, :n] = 1
    return {"input_ids": ids, "attention_mask": att, "labels": lab}


def length_grouped_batches(lengths: List[int], bs: int, seed: int, group_mult: int = 50) -> List[List[int]]:
    """Shuffle, then sort inside mega-chunks so each batch has similar lengths
    (far less padding -> ~1.5-2x faster on T4). Deterministic given seed, so
    every run sees identical batches, which keeps memory comparisons fair."""
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    batches = []
    for i in range(0, len(idx), bs * group_mult):
        chunk = sorted(idx[i:i + bs * group_mult], key=lambda j: lengths[j], reverse=True)
        batches += [chunk[k:k + bs] for k in range(0, len(chunk), bs)]
    rng.shuffle(batches)
    return batches


def build_scoring_set(langs=DEFAULT_LANGS, n_per_lang=43, n_eval=200, n_train_max=4000, seed=42) -> List[dict]:
    """Examples for gradient scoring that are DISJOINT from both the held-out set and every
    training split we use (rows [n_eval + n_train_max, ...) after the same seeded shuffle).
    No validation leakage, and no overlap with training data either."""
    out = []
    for lang in langs:
        rows = load_multialpaca_lang(lang)
        random.Random(f"{seed}-{lang}").shuffle(rows)          # identical shuffle to build_splits
        start = n_eval + n_train_max
        out += rows[start:start + n_per_lang]
    random.Random(seed).shuffle(out)
    return out
