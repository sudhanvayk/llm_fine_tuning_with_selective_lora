"""One self-contained run: select layers -> LoRA -> probe memory -> train -> eval -> results JSON.

    python -m sllora.train --run_id top8 --rule top --k 8
    python -m sllora.train --run_id top8_q4_gc_ck --rule top --k 8 --quant4 --gc --chunked
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import torch
from transformers import get_cosine_schedule_with_warmup

from . import data as D
from .model import (adapted_layers, apply_selective_lora, count_params, enable_gradient_checkpointing,
                    forward_loss, load_model_and_tokenizer, n_layers_of, probe_peak_memory,
                    saved_activation_bytes, synthetic_batch)
from .selection import CostModel, layer_strip, select_layers
from .utils import GB, cuda_sync, free_cuda, git_hash, gpu_name, load_json, now_iso, save_json, set_seed


@dataclass
class RunConfig:
    run_id: str
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    rule: str = "all"
    k: Optional[int] = None
    budget_gb: Optional[float] = None
    layers: Optional[List[int]] = None        # explicit override
    quant4: bool = False
    gc: bool = False
    chunked: bool = False
    r: int = 16
    alpha: int = 32
    lr: float = 2e-4
    epochs: int = 1
    bs: int = 8
    accum: int = 2                            # effective batch 16, as in Aggarwal et al.
    max_len: int = 512
    langs: List[str] = field(default_factory=lambda: list(D.DEFAULT_LANGS))
    n_train: int = 4000                       # per language (T4 does ~7k tok/s: ~5 min/run)
    n_eval: int = 200                         # per language, held out
    seed: int = 42                            # training seed (data split seed is fixed at 42)
    eval_suite: str = "quick"                 # quick | full | none
    probe_bs: int = 4
    probe_len: int = 512
    max_steps: Optional[int] = None           # for smoke tests
    out_dir: str = "outputs"                  # adapters + logs (put this on Google Drive)
    results_dir: str = "results/runs"         # small JSONs (committed to git)
    scores_path: str = "results/scores/layer_scores.json"
    cost_model_path: str = "results/scores/cost_model.json"
    split_cache: str = "data/splits.json"
    train_only_probe: bool = False            # True -> only measure memory, skip training (demo / 7B check)
    reprobe_only: bool = False                # True -> re-measure memory of an existing run, patch its JSON


def resolve_layers(cfg: RunConfig, L: int):
    if cfg.layers is not None:
        return sorted(cfg.layers)
    scores = load_json(cfg.scores_path) if os.path.exists(cfg.scores_path) else None
    cost = None
    if cfg.rule == "budgeted":
        cm = load_json(cfg.cost_model_path)
        key = f"{cfg.model_id.split('/')[-1]}|q4={int(cfg.quant4)},gc={int(cfg.gc)},ck={int(cfg.chunked)}"
        d = cm[key] if key in cm else cm["default"]
        cost = CostModel(**{f: d[f] for f in ("n_layers", "base_gb", "per_adapted_gb", "per_depth_gb")})
    return select_layers(cfg.rule, L, k=cfg.k, scores=scores, budget_gb=cfg.budget_gb, cost=cost, seed=cfg.seed)


def run(cfg: RunConfig) -> dict:
    t0 = time.time()
    set_seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = dev == "cuda"
    res_path = os.path.join(cfg.results_dir, f"{cfg.run_id}.json")
    run_dir = os.path.join(cfg.out_dir, cfg.run_id)
    os.makedirs(run_dir, exist_ok=True)

    model, tok = load_model_and_tokenizer(cfg.model_id, quant4=cfg.quant4)
    L = n_layers_of(model)
    layers = resolve_layers(cfg, L)
    model = apply_selective_lora(model, layers, r=cfg.r, alpha=cfg.alpha)
    if cfg.gc:
        enable_gradient_checkpointing(model)
    got = adapted_layers(model)
    want = list(range(L)) if layers is None else layers
    assert got == want, f"LoRA landed on {got}, expected {want}"
    trainable, total = count_params(model)
    print(f"[{cfg.run_id}] layers {layer_strip(layers, L)}  trainable={trainable/1e6:.2f}M / {total/1e6:.0f}M")

    vocab = model.get_base_model().get_output_embeddings().weight.shape[0]
    mem = probe_peak_memory(model, vocab, cfg.probe_bs, cfg.probe_len, chunked=cfg.chunked, fp16=fp16)
    sb = synthetic_batch(vocab, 1, 256, dev)
    mem["saved_act_gb_per_256tok"] = round(saved_activation_bytes(model, sb, chunked=cfg.chunked) / GB, 4)
    del sb
    model.zero_grad(set_to_none=True)
    free_cuda()
    mem["isolated"] = os.environ.get("SLLORA_ISOLATED") == "1"   # measured in a fresh process (no leftovers)
    print(f"[{cfg.run_id}] memory probe: {mem}")

    prev = load_json(res_path) if os.path.exists(res_path) else None
    if cfg.reprobe_only:
        assert prev, f"no results file for {cfg.run_id}"
        prev["memory"] = mem
        prev["memory_reprobed"] = now_iso()
        save_json(prev, res_path)
        print(f"[{cfg.run_id}] re-measured memory saved -> {res_path}")
        return prev
    adapter_dir = os.path.join(run_dir, "adapter")
    resume = bool(prev and "train" in prev and "finished" not in prev
                  and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")))

    result = {"run_id": cfg.run_id, "config": asdict(cfg), "layers": want, "strip": layer_strip(layers, L),
              "n_layers": L, "lowest_layer": min(want), "backprop_depth": L - min(want),
              "trainable_params": trainable, "total_params": total, "memory": mem,
              "gpu": gpu_name(), "git": git_hash(), "started": now_iso()}
    if cfg.train_only_probe:
        result.update({"finished": now_iso(), "wall_s": round(time.time() - t0, 1)})
        save_json(result, res_path)
        return result

    # ---------------- data
    split = D.build_splits(cfg.langs, cfg.n_train, cfg.n_eval, seed=42, cache_path=cfg.split_cache)
    train_items = D.tokenize_all(tok, split["train"], cfg.max_len)
    eval_items = {l: D.tokenize_all(tok, ex, cfg.max_len) for l, ex in split["eval"].items()}
    batches = D.length_grouped_batches([len(x["input_ids"]) for x in train_items], cfg.bs, seed=cfg.seed)
    batches = batches * cfg.epochs
    total_steps = math.ceil(len(batches) / cfg.accum)
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
        batches = batches[: total_steps * cfg.accum]

    if resume:
        # training already finished in an earlier attempt; only the evaluation crashed -> reuse the adapter
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(os.path.join(adapter_dir, "adapter_model.safetensors")))
        result["train"] = prev["train"]
        result["started"] = prev.get("started", result["started"])
        result["resumed_eval"] = now_iso()
        print(f"[{cfg.run_id}] training already done -> loaded saved adapter, re-running evaluation only")
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0, betas=(0.9, 0.999))
        sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * total_steps)), total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=fp16)

        # ---------------- train
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        log_f = open(os.path.join(run_dir, "train_log.jsonl"), "w")
        step, tokens, t_train, nan_streak, running = 0, 0, time.time(), 0, None
        for bi, b in enumerate(batches):
            batch = {k: v.to(dev) for k, v in D.collate([train_items[i] for i in b], tok.pad_token_id).items()}
            tokens += int(batch["attention_mask"].sum())
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                loss = forward_loss(model, batch, chunked=cfg.chunked) / cfg.accum
            if not torch.isfinite(loss):
                nan_streak += 1
                if nan_streak >= 5:
                    raise RuntimeError("Loss is NaN/inf 5x in a row: rerun with fp32 base (load dtype float32) or lr 1e-4")
                opt.zero_grad(set_to_none=True)
                continue
            nan_streak = 0
            scaler.scale(loss).backward()
            if (bi + 1) % cfg.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                l = float(loss) * cfg.accum
                running = l if running is None else 0.9 * running + 0.1 * l
                if step % 10 == 0 or step == total_steps:
                    el = time.time() - t_train
                    rec = {"step": step, "loss": round(l, 4), "ema": round(running, 4), "lr": sched.get_last_lr()[0],
                           "tok_s": round(tokens / el, 1), "elapsed_s": round(el, 1)}
                    log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()
                    print(f"[{cfg.run_id}] step {step}/{total_steps} loss {running:.3f} {rec['tok_s']:.0f} tok/s")
                if step >= total_steps:
                    break
        cuda_sync()
        train_s = time.time() - t_train
        log_f.close()
        result["train"] = {"steps": step, "tokens": tokens, "train_s": round(train_s, 1),
                           "tok_s": round(tokens / max(train_s, 1e-9), 1), "final_loss_ema": running,
                           "train_peak_gb": round(torch.cuda.max_memory_allocated() / GB, 4) if dev == "cuda" else None}
        del opt
        free_cuda()
        model.save_pretrained(os.path.join(run_dir, "adapter"))
        save_json(result, res_path)  # partial result survives an eval crash

    # ---------------- eval
    from .evaluate import heldout_loss
    result["heldout"] = heldout_loss(model, tok, eval_items)
    if cfg.eval_suite != "none" and dev == "cuda":
        from .evaluate import lm_eval_scores
        result["lm_eval"] = lm_eval_scores(model, tok, cfg.langs, cfg.eval_suite)
    result.update({"finished": now_iso(), "wall_s": round(time.time() - t0, 1)})
    save_json(result, res_path)
    print(f"[{cfg.run_id}] done in {result['wall_s']/60:.1f} min -> {res_path}")
    return result


def eval_reference(model_id: str, langs, suite="quick", results_dir="results/runs", split_cache="data/splits.json",
                   n_train=4000, n_eval=200, max_len=512, run_id="phase1_ref"):
    """Scores of the untouched Phase-1 model: the reference every run is compared with."""
    from .evaluate import heldout_loss, lm_eval_scores
    model, tok = load_model_and_tokenizer(model_id)
    split = D.build_splits(langs, n_train, n_eval, seed=42, cache_path=split_cache)
    eval_items = {l: D.tokenize_all(tok, ex, max_len) for l, ex in split["eval"].items()}
    res = {"run_id": run_id, "config": {"model_id": model_id, "rule": "none", "langs": langs},
           "heldout": heldout_loss(model, tok, eval_items), "gpu": gpu_name(), "started": now_iso()}
    if torch.cuda.is_available() and suite != "none":
        res["lm_eval"] = lm_eval_scores(model, tok, langs, suite)
    res["finished"] = now_iso()
    save_json(res, os.path.join(results_dir, f"{run_id}.json"))
    return res


def main():
    p = argparse.ArgumentParser()
    for f in RunConfig.__dataclass_fields__.values():
        name = f"--{f.name}"
        if f.type in ("bool", bool):
            p.add_argument(name, action="store_true")
        elif f.name in ("langs", "layers"):
            p.add_argument(name, type=lambda s: [x if not x.isdigit() else int(x) for x in s.split(",")], default=None)
        else:
            typ = {"int": int, "float": float, "str": str}.get(str(f.type).replace("Optional[", "").rstrip("]"), str)
            p.add_argument(name, type=typ, default=None)
    a = {k: v for k, v in vars(p.parse_args()).items() if v is not None and v is not False}
    run(RunConfig(**a))


if __name__ == "__main__":
    main()
