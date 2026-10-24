"""Live demo for the judges (run on a Colab T4):  python app/demo.py   -> public gradio.live link

Tab 1  Memory meter : pick rule / k / levers -> measured peak GB live, next to all-layer LoRA
Tab 2  Results      : Pareto plot, lever waterfall, results table from the sweep
Tab 3  7B calculator: analytic projection for Mistral-7B / Llama-3-8B / Qwen2.5-7B
Tab 4  Try it       : same prompt, Phase-1 model vs our adapter
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import pandas as pd
import torch

from sllora.memory_model import CONFIGS, estimate
from sllora.model import (apply_selective_lora, enable_gradient_checkpointing, load_model_and_tokenizer,
                          n_layers_of, probe_peak_memory)
from sllora.selection import RULES, layer_strip, select_layers
from sllora.utils import load_json

MODEL_ID = os.environ.get("SLLORA_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
OUT = os.environ.get("SLLORA_OUT", "/content/drive/MyDrive/hashcode_sllora/outputs")
SCORES = load_json("results/scores/layer_scores.json") if os.path.exists("results/scores/layer_scores.json") else {}
_cache = {"q4": None, "model": None, "tok": None, "baseline": {}}


def _base(q4):
    if _cache["q4"] != q4:
        _cache["model"] = None
        torch.cuda.empty_cache()
        _cache["model"], _cache["tok"] = load_model_and_tokenizer(MODEL_ID, quant4=q4)
        _cache["q4"] = q4
    return _cache["model"]


def _measure(layers, q4, gc, ck):
    model = _base(q4)
    pm = apply_selective_lora(model, layers)
    if gc:
        enable_gradient_checkpointing(pm)
    vocab = pm.get_base_model().get_output_embeddings().weight.shape[0]
    m = probe_peak_memory(pm, vocab, 4, 512, chunked=ck)
    _cache["model"] = pm.unload()
    if gc:
        _cache["model"].gradient_checkpointing_disable()
    return m["probe_peak_gb"]


def meter(rule, k, q4, gc, ck):
    if not torch.cuda.is_available():
        return "Needs a GPU runtime", None
    L = n_layers_of(_base(q4))
    try:
        layers = select_layers(rule, L, k=int(k), scores=SCORES, seed=42) if rule != "all" else None
    except Exception as e:
        return f"Rule needs layer scores first: {e}", None
    key = "baseline"
    if key not in _cache["baseline"]:
        _cache["baseline"][key] = _measure(None, False, False, False)   # plain all-layer LoRA, fp16
    base_gb = _cache["baseline"][key]
    gb = _measure(layers, q4, gc, ck)
    strip = layer_strip(layers, L)
    txt = (f"### {gb:.2f} GB  vs  {base_gb:.2f} GB all-layer LoRA  →  **{100 * (1 - gb / base_gb):.0f}% less**\n\n"
           f"Adapted layers (0 → {L - 1}): `{strip}`  · backprop depth {L - min(layers or [0])} of {L}")
    df = pd.DataFrame({"config": ["All-layer LoRA (fp16)", "This config"], "peak GB": [base_gb, gb]})
    return txt, df


def calc(model, k, q4, gc, ck, bs, seq):
    L = CONFIGS[model]["L"]
    a = estimate(model, bs=int(bs), seq=int(seq))
    s = estimate(model, k_top=int(k), quant4=q4, gc=gc, chunked=ck, bs=int(bs), seq=int(seq))
    rows = [{"part": p, "all-layer LoRA GB": a[p], "selective GB": s[p]} for p in ["weights", "adapters_opt", "activations", "logits", "total"]]
    fits = "fits a 16 GB T4" if s["total"] < 15 else "needs > 16 GB"
    return (f"### {model}: {a['total']:.1f} GB → {s['total']:.1f} GB ({100 * (1 - s['total'] / a['total']):.0f}% less, {fits})\n"
            f"Top-{int(k)} of {L} layers. Analytic estimate, checked against measured 0.5B probes."), pd.DataFrame(rows)


def results_tab():
    md = open("results/summary.md", encoding="utf-8").read() if os.path.exists("results/summary.md") else "Run scripts/aggregate.py"
    imgs = [f"results/plots/{p}" for p in ["pareto.png", "levers.png", "layer_scores.png"] if os.path.exists(f"results/plots/{p}")]
    return md, imgs


def chat(run_id, prompt):
    from peft import PeftModel
    model, tok = load_model_and_tokenizer(MODEL_ID)
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)

    def gen(m):
        with torch.no_grad():
            out = m.generate(**ids, max_new_tokens=160, do_sample=False)
        return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    before = gen(model)
    pm = PeftModel.from_pretrained(model, os.path.join(OUT, run_id, "adapter"))
    after = gen(pm)
    return before, after


with gr.Blocks(title="Selective-Layer LoRA") as demo:
    gr.Markdown("# Selective-Layer LoRA — Team Frozen Layers\nLoRA adapters on only the layers that matter, priced in GB.")
    with gr.Tab("Memory meter (live)"):
        with gr.Row():
            rule = gr.Dropdown(RULES[:-1], value="top", label="Selection rule")
            k = gr.Slider(1, 24, value=8, step=1, label="k layers")
        with gr.Row():
            q4, gc_, ck = gr.Checkbox(label="NF4 4-bit base"), gr.Checkbox(label="Gradient checkpointing"), gr.Checkbox(label="Chunked loss")
        btn = gr.Button("Measure on this GPU", variant="primary")
        out_md, out_plot = gr.Markdown(), gr.BarPlot(x="config", y="peak GB", height=260)
        btn.click(meter, [rule, k, q4, gc_, ck], [out_md, out_plot])
    with gr.Tab("Results"):
        b2 = gr.Button("Load latest results")
        r_md, r_gal = gr.Markdown(), gr.Gallery(columns=1, height=520, object_fit='contain', preview=True)
        b2.click(results_tab, None, [r_md, r_gal])
    with gr.Tab("7B calculator"):
        with gr.Row():
            mdl = gr.Dropdown(list(CONFIGS), value="Mistral-7B", label="Model")
            kk = gr.Slider(1, 32, value=8, step=1, label="Top-k layers")
            bs = gr.Number(4, label="batch")
            sq = gr.Number(512, label="seq len")
        with gr.Row():
            cq, cg, cc = gr.Checkbox(True, label="NF4"), gr.Checkbox(True, label="GC"), gr.Checkbox(True, label="Chunked loss")
        cb = gr.Button("Estimate", variant="primary")
        c_md, c_df = gr.Markdown(), gr.Dataframe()
        cb.click(calc, [mdl, kk, cq, cg, cc, bs, sq], [c_md, c_df])
    with gr.Tab("Try it"):
        rid = gr.Textbox("top8", label="run_id (adapter on Drive)")
        pr = gr.Textbox("ما هي فوائد القراءة اليومية؟", label="Prompt (Arabic / Russian / Thai)")
        tb = gr.Button("Generate: before vs after")
        with gr.Row():
            o1, o2 = gr.Textbox(label="Phase-1 model"), gr.Textbox(label="+ Selective-Layer LoRA")
        tb.click(chat, [rid, pr], [o1, o2])

if __name__ == "__main__":
    demo.queue().launch(share=True)
