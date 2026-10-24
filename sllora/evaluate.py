"""Evaluation: held-out language perplexity (fast, our own) + lm-eval-harness subsets.

Task retention (did we forget English ability?):  hellaswag, arc_easy, mmlu
Language gain (did Phase 2 help?):                 held-out MultiAlpaca loss per language,
                                                   Belebele reading comprehension per language
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch

from .data import BELEBELE, collate, length_grouped_batches
from .model import forward_loss

SUITES = {
    # (task, limit, metric)  -- quick: ~5-7 min for 0.5B on a T4
    "quick": {"task": [("hellaswag", 300, "acc_norm"), ("arc_easy", 300, "acc_norm"), ("mmlu", 5, "acc")],
              "lang_limit": 200},
    # full: top configs only, ~20 min
    "full": {"task": [("hellaswag", 1000, "acc_norm"), ("arc_easy", 1000, "acc_norm"), ("mmlu", 20, "acc")],
             "lang_limit": 900},
}


@torch.no_grad()
def heldout_loss(model, tok, eval_items: Dict[str, List[dict]], bs: int = 8) -> Dict:
    """Mean answer-token loss per language on the held-out split (lower = better)."""
    dev = next(p for p in model.parameters()).device
    model.eval()
    out = {}
    for lang, items in eval_items.items():
        tot, n = 0.0, 0
        for b in length_grouped_batches([len(x["input_ids"]) for x in items], bs, seed=0):
            batch = {k: v.to(dev) for k, v in collate([items[i] for i in b], tok.pad_token_id).items()}
            ntok = int((batch["labels"][:, 1:] != -100).sum())
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = forward_loss(model, batch, chunked=True)
            tot += float(loss) * ntok
            n += ntok
        out[lang] = tot / max(n, 1)
    out["avg"] = sum(out[l] for l in eval_items) / len(eval_items)
    out["ppl_avg"] = math.exp(out["avg"])
    model.train()
    return out


def lm_eval_scores(model, tok, langs: List[str], suite: str = "quick", batch_size: int = 8) -> Dict:
    """Runs lm-eval on the in-memory (PEFT) model, so 4-bit runs are scored as trained."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    cfg = SUITES[suite]
    model.eval()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, max_length=1024)
    scores = {}
    for task, limit, metric in cfg["task"]:
        r = lm_eval.simple_evaluate(model=lm, tasks=[task], limit=limit, log_samples=False,
                                    bootstrap_iters=0, verbosity="ERROR")
        scores[task] = float(r["results"][task][f"{metric},none"])
    for lang in langs:
        t = f"belebele_{BELEBELE[lang]}"
        r = lm_eval.simple_evaluate(model=lm, tasks=[t], limit=cfg["lang_limit"], log_samples=False,
                                    bootstrap_iters=0, verbosity="ERROR")
        scores[t] = float(r["results"][t]["acc,none"])
    task_keys = [t for t, _, _ in cfg["task"]]
    lang_keys = [f"belebele_{BELEBELE[l]}" for l in langs]
    scores["task_avg"] = sum(scores[k] for k in task_keys) / len(task_keys)
    scores["lang_avg"] = sum(scores[k] for k in lang_keys) / len(lang_keys)
    scores["suite"] = suite
    model.train()
    return scores
