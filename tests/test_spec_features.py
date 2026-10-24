"""CPU tests for the spec features (no downloads):
q/v-only targets + index validation, activation offloading gives identical gradients (also with
checkpointing), gradient-scoring protocol, mid-training checkpoint resume, OOM recorded not lost.

    python tests/test_spec_features.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import smoke_test as S
from sllora import data as D
from sllora.model import (activation_offload, adapted_layers, apply_selective_lora, enable_gradient_checkpointing,
                          forward_loss, load_model_and_tokenizer, synthetic_batch)
from sllora.scoring import layer_importance_protocol

TMP = tempfile.mkdtemp(prefix="spec_")
check = S.check


def grads(pm, batch, offload, chunked=True):
    pm.zero_grad(set_to_none=True)
    with activation_offload(offload):
        loss = forward_loss(pm, batch, chunked=chunked)
    loss.backward()
    return loss.item(), [p.grad.clone() for p in pm.parameters() if p.requires_grad]


def main():
    base = os.path.join(TMP, "m")
    S.make_tiny(base, 0)
    D.load_multialpaca_lang = S.fake_lang

    # 1. q/v targets only + invalid index rejected
    m, tok = load_model_and_tokenizer(base)
    pm = apply_selective_lora(m, [5, 6, 7], targets=["q_proj", "v_proj"])
    names = [n for n, p in pm.named_parameters() if p.requires_grad]
    check("q/v-only adapters on layers 5-7", adapted_layers(pm) == [5, 6, 7] and all(("q_proj" in n or "v_proj" in n) for n in names))
    m2, _ = load_model_and_tokenizer(base)
    try:
        apply_selective_lora(m2, [3, 99])
        check("invalid layer index rejected", False)
    except ValueError:
        check("invalid layer index rejected", True)

    # 2. activation offloading: identical loss and gradients, with and without checkpointing
    batch = synthetic_batch(1200, 2, 96, "cpu")
    for gc in (False, True):
        mm, _ = load_model_and_tokenizer(base)
        pm = apply_selective_lora(mm, [2, 5, 7], dropout=0.0)
        with torch.no_grad():
            for n, p in pm.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.02)
        if gc:
            enable_gradient_checkpointing(pm)
        l0, g0 = grads(pm, batch, offload=False)
        l1, g1 = grads(pm, batch, offload=True)
        check(f"offload (gc={gc}): same loss and gradients", abs(l0 - l1) < 1e-6 and all(torch.allclose(a, b, atol=1e-6) for a, b in zip(g0, g1)))

    # 3. gradient-scoring protocol
    split = D.build_splits(["ar", "th"], n_train=40, n_eval=10, cache_path=os.path.join(TMP, "sp.json"))
    items = D.tokenize_all(tok, split["train"], 96)
    mm, _ = load_model_and_tokenizer(base)
    sc, back = layer_importance_protocol(mm, tok, items[:32], r=4, targets=("q_proj", "v_proj"), bs=4)
    check(f"scoring: {sc['n_batches']} batches, 8 scores, split-half rho={sc['split_half_spearman']:.2f}",
          len(sc["scores"]) == 8 and sc["n_batches"] >= 2 and max(sc["scores"]) == 1.0)
    check("scoring adapters discarded", not any("lora" in n for n, _ in back.named_parameters()))

    # 4. checkpoint resume: crash mid-training, rerun continues from the saved step
    from sllora import train as T
    kw = dict(model_id=base, langs=["ar", "th"], n_train=40, n_eval=10, max_len=96, bs=2, accum=2, max_steps=8,
              eval_suite="none", out_dir=os.path.join(TMP, "out"), results_dir=os.path.join(TMP, "res"),
              split_cache=os.path.join(TMP, "sp.json"), ckpt_every=3, rule="top", k=2)
    real = T.forward_loss
    calls = {"n": 0}

    def crashing(*a, **k):
        calls["n"] += 1
        if calls["n"] == 15:                 # probe + activation count use 3 calls; crash mid-training
            raise KeyboardInterrupt("simulated disconnect")
        return real(*a, **k)
    T.forward_loss = crashing
    try:
        T.run(T.RunConfig(run_id="ck", **kw))
    except KeyboardInterrupt:
        pass
    T.forward_loss = real
    check("checkpoint written before the crash", os.path.exists(os.path.join(TMP, "out", "ck", "ckpt.pt")))
    res = T.run(T.RunConfig(run_id="ck", **kw))
    check(f"resumed from step {res.get('resumed_from_step')} and finished 8 steps",
          res.get("resumed_from_step") in (3, 6) and res["train"]["steps"] == 8 and "finished" in res)
    check("checkpoint removed after success", not os.path.exists(os.path.join(TMP, "out", "ck", "ckpt.pt")))
    check("metrics: step time, cpu ram, trainable %, versions",
          all(k in res["train"] for k in ("step_time_s", "peak_cpu_ram_gb")) and "trainable_pct" in res and "versions" in res)

    # 5. OOM is recorded, not silently lost
    def oom(*a, **k):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
    T.forward_loss = oom
    try:
        T.run(T.RunConfig(run_id="oomrun", **dict(kw, ckpt_every=0)))
        check("OOM re-raised", False)
    except torch.cuda.OutOfMemoryError:
        r = json.load(open(os.path.join(TMP, "res", "oomrun.json")))
        check("OOM recorded with status=oom", r.get("status") == "oom" and "finished" not in r)
    T.forward_loss = real

    print("\nALL SPEC FEATURE TESTS PASSED")
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
