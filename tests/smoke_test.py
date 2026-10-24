"""CPU smoke test, no downloads: a tiny random Qwen2 model + toy tokenizer.

Proves the pipeline end to end and checks the core claims:
  1. LoRA lands only on the selected layers
  2. activation memory falls when the lowest adapted layer moves up
  3. the gradient-checkpointing fix keeps that saving
  4. chunked loss == standard loss
  5. scoring, cost-model fit, budgeted selector and a full train/eval run work

    python tests/smoke_test.py          (~1-2 min on a laptop CPU)
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # this test is CPU-only by design, even on a GPU runtime

import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import random
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from sllora import data as D
from sllora.model import (adapted_layers, apply_selective_lora, enable_gradient_checkpointing, forward_loss,
                          load_model_and_tokenizer, saved_activation_bytes, synthetic_batch)
from sllora.scoring import layer_drift, layer_importance
from sllora.selection import CostModel, budgeted_select, layer_strip, select_layers
from sllora.utils import save_json

TMP = tempfile.mkdtemp(prefix="sllora_smoke_")
L = 8
CHAT = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def make_tiny(path, seed):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=1200, hidden_size=64, intermediate_size=160, num_hidden_layers=L,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(path)
    words = [f"w{i}" for i in range(1100)]
    vocab = {t: i for i, t in enumerate(["<|endoftext|>", "<|im_start|>", "<|im_end|>", "[UNK]", "user", "assistant", "\n"] + words)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                   unk_token="[UNK]", additional_special_tokens=["<|im_start|>", "<|im_end|>"])
    fast.chat_template = CHAT
    fast.save_pretrained(path)


def fake_lang(lang):
    rng = random.Random(lang)
    w = lambda n: " ".join(f"w{rng.randrange(1100)}" for _ in range(n))
    return [{"instruction": w(rng.randint(5, 20)), "input": "", "output": w(rng.randint(10, 60)), "lang": lang}
            for _ in range(80)]


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


def main():
    base, tuned = os.path.join(TMP, "base"), os.path.join(TMP, "tuned")
    make_tiny(base, 0)
    make_tiny(tuned, 0)
    m = Qwen2ForCausalLM.from_pretrained(tuned)                    # "instruction tuning": perturb top layers
    with torch.no_grad():
        for n, p in m.named_parameters():
            if ".layers." in n and int(n.split(".layers.")[1].split(".")[0]) >= 5:
                p.add_(0.05 * torch.randn_like(p))
    m.save_pretrained(tuned)

    # 1. LoRA placement
    model, tok = load_model_and_tokenizer(base)
    pm = apply_selective_lora(model, [5, 6, 7])
    check("LoRA only on layers 5,6,7", adapted_layers(pm) == [5, 6, 7])

    # 2. activation memory vs lowest layer
    batch = synthetic_batch(1200, 2, 128, "cpu")
    acts = {}
    for name, layers in {"all": None, "top4": [4, 5, 6, 7], "top2": [6, 7], "bottom2": [0, 1], "mixed": [1, 7]}.items():
        mm, _ = load_model_and_tokenizer(base)
        pm = apply_selective_lora(mm, layers)
        acts[name] = saved_activation_bytes(pm, batch)
    print("   saved activation bytes:", acts)
    check("top2 < top4 < all", acts["top2"] < acts["top4"] < acts["all"])
    # same number of adapters (2), but [1,7] forces backprop through layers 1..7
    check("lowest layer drives cost: mixed[1,7] >> top2[6,7]", acts["mixed"] > 1.5 * acts["top2"])

    # 3. gradient checkpointing keeps the saving
    mm, _ = load_model_and_tokenizer(base)
    pm = apply_selective_lora(mm, [6, 7])
    enable_gradient_checkpointing(pm)
    emb_out = {}
    h = pm.get_base_model().get_input_embeddings().register_forward_hook(lambda mod, i, o: emb_out.update(rg=o.requires_grad))
    gc_top2 = saved_activation_bytes(pm, batch)
    h.remove()
    check("embedding output does not require grad after GC fix", emb_out["rg"] is False)
    check("GC top2 saves less than plain top2", gc_top2 < acts["top2"])
    print(f"   GC top2 = {gc_top2} bytes vs {acts['top2']} without GC")

    # 4. chunked loss equals standard loss, and gradients match
    mm, _ = load_model_and_tokenizer(base)
    pm = apply_selective_lora(mm, [6, 7], dropout=0.0)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
    l1 = forward_loss(pm, batch, chunked=False)
    l1.backward()
    g1 = [p.grad.clone() for p in pm.parameters() if p.requires_grad]
    pm.zero_grad()
    l2 = forward_loss(pm, batch, chunked=True, chunk_tokens=50)
    l2.backward()
    g2 = [p.grad.clone() for p in pm.parameters() if p.requires_grad]
    check(f"chunked loss {l2.item():.5f} == standard {l1.item():.5f}", abs(l1.item() - l2.item()) < 1e-4)
    check("chunked gradients match", all(torch.allclose(a, b, atol=1e-5) for a, b in zip(g1, g2)))

    # 5a. drift scoring
    dr = layer_drift(base, tuned)
    print("   drift:", [round(x, 3) for x in dr["drift"]])
    check("drift finds the perturbed top layers", min(dr["drift"][5:]) > 10 * max(dr["drift"][:5] + [1e-9]))

    # 5b. data + importance
    D.load_multialpaca_lang = fake_lang
    split = D.build_splits(["ar", "th"], n_train=50, n_eval=20, cache_path=os.path.join(TMP, "splits.json"))
    items = D.tokenize_all(tok, split["train"], 128)
    check("label masking: prompt tokens are -100", items[0]["labels"][0] == -100 and items[0]["labels"][-1] != -100)
    mm, _ = load_model_and_tokenizer(base)
    imp, base_back = layer_importance(mm, tok, items, n_samples=32, bs=8)
    check("importance has one score per layer", len(imp["importance"]) == L)
    check("unload() returns clean base", not any(p.requires_grad for p in base_back.parameters()))

    # 5c. rules, cost model, budgeted selector
    scores = {"drift": dr["drift"], "importance": imp["importance"]}
    for rule in ["random", "top", "bottom", "drift_protect", "importance"]:
        print(f"   {rule:14s} k=3 {layer_strip(select_layers(rule, L, 3, scores), L)}")
    meas = [{"layers": list(range(L)), "peak_gb": 5.0}, {"layers": [4, 5, 6, 7], "peak_gb": 3.0},
            {"layers": [6, 7], "peak_gb": 2.0}, {"layers": [0, 7], "peak_gb": 4.6}]
    cm = CostModel.fit(L, meas)
    print(f"   cost model {cm.to_dict()}  R2={cm.r2(meas):.3f}")
    S = budgeted_select(imp["importance"], cm, budget_gb=3.0, k_max=4)
    check(f"budgeted pick {S} fits 3.0 GB (cost {cm.cost(S):.2f})", cm.cost(S) <= 3.0)

    # 5d. full run (train -> save adapter -> held-out eval -> results json)
    from sllora.train import RunConfig, run
    rd = os.path.join(TMP, "results")
    save_json(scores, os.path.join(TMP, "scores.json"))
    save_json({"default": cm.to_dict()}, os.path.join(TMP, "cost.json"))
    for rid, kw in {"lora_all": dict(rule="all"), "top3_gc_ck": dict(rule="top", k=3, gc=True, chunked=True),
                    "budget3": dict(rule="budgeted", budget_gb=3.0, k=4)}.items():
        res = run(RunConfig(run_id=rid, model_id=base, langs=["ar", "th"], n_train=50, n_eval=20, max_len=128,
                            bs=4, accum=2, max_steps=6, eval_suite="none", out_dir=os.path.join(TMP, "out"),
                            results_dir=rd, scores_path=os.path.join(TMP, "scores.json"),
                            cost_model_path=os.path.join(TMP, "cost.json"),
                            split_cache=os.path.join(TMP, "splits.json"), **kw))
        check(f"run {rid}: {res['train']['steps']} steps, heldout loss {res['heldout']['avg']:.3f}",
              res["train"]["steps"] == 6 and res["heldout"]["avg"] > 0)
        check(f"run {rid}: adapter saved", os.path.exists(os.path.join(TMP, "out", rid, "adapter", "adapter_config.json")))

    print("\nALL SMOKE TESTS PASSED")
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
