"""Live demo for the judges (run on a Colab T4):  python app/demo.py   -> public gradio.live link

Tab 1  Memory meter : pick rule / k / levers -> measured peak GB live, next to all-layer LoRA
Tab 2  Results      : Pareto plots, lever waterfall, layer scores, probes, results table from the sweep
Tab 3  7B calculator: analytic projection for Mistral-7B / Llama-3-8B / Qwen2.5-7B
Tab 4  Try it       : same prompt, untouched model vs BASELINE (all-layer LoRA) vs our selective-layer adapter.
                      Adapters are found automatically on Drive (SLLORA_OUT); nothing crashes if one is missing.
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
RUNS_DIR = "results/runs"
SCORES = load_json("results/scores/layer_scores.json") if os.path.exists("results/scores/layer_scores.json") else {}
_cache = {"q4": None, "model": None, "tok": None, "baseline": {}}

# first one found wins; spec_* adapters use a 4-bit base and q/v only, so they come last
BASELINE_PREF = ["lora_all", "lora_all_plain", "spec_A_all"]
OURS_PREF = ["budget50", "budget35", "top8", "imp8", "imp12", "imp4", "drift8", "spec_B_grad6"]


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


RESULT_PLOTS = ["pareto_main.png", "pareto_spec.png", "pareto.png", "levers.png", "layer_scores.png", "probes.png"]


def results_tab():
    md = open("results/summary.md", encoding="utf-8").read() if os.path.exists("results/summary.md") else "Run scripts/aggregate.py"
    imgs = [f"results/plots/{p}" for p in RESULT_PLOTS if os.path.exists(f"results/plots/{p}")]
    return md, imgs


# ----------------------------------------------------------------------------- Try it: adapters + baseline
def adapter_path(run_id):
    return os.path.join(OUT, run_id, "adapter")


def list_adapters():
    """run_ids whose trained adapter is really present under OUT (adapters live on Drive, not in git)."""
    found = []
    if os.path.isdir(OUT):
        for d in sorted(os.listdir(OUT)):
            if os.path.exists(os.path.join(adapter_path(d), "adapter_config.json")):
                found.append(d)
    return found


def pick(prefs, available):
    for p in prefs:
        if p in available:
            return p
    return None


def run_info(run_id):
    """Numbers from the committed results file of that run (may be empty if the run isn't in this repo)."""
    path = os.path.join(RUNS_DIR, f"{run_id}.json")
    if not run_id or not os.path.exists(path):
        return {}
    r = load_json(path)
    c = r.get("config", {})
    return {"q4": bool(c.get("quant4")), "gb": r.get("memory", {}).get("probe_peak_gb"),
            "loss": r.get("heldout", {}).get("avg"), "strip": r.get("strip")}


def describe(run_id, label):
    i = run_info(run_id)
    if not i:
        return f"**{label}** `{run_id}`: no results file for this run in the repo"
    bits = []
    if i.get("gb") is not None:
        bits.append(f"{i['gb']:.2f} GB peak training memory (probe, 4×512 tokens)")
    if i.get("loss") is not None:
        bits.append(f"held-out loss {i['loss']:.3f}")
    if i.get("strip"):
        bits.append(f"layers `{i['strip']}`")
    return f"**{label}** `{run_id}`: " + " · ".join(bits)


def adapter_status(avail):
    if not avail:
        return (f"⚠️ No trained adapters found in `{OUT}`. Run the notebook's *demo adapters* step, or copy "
                f"`outputs/<run_id>/adapter` folders there from the teammate who trained them.")
    msg = f"Adapters found on this Drive: {', '.join(f'`{a}`' for a in avail)}."
    if not any(a in avail for a in BASELINE_PREF):
        msg += ("\n\n⚠️ No all-layer **baseline** adapter (`lora_all`) found, so the baseline column will stay empty. "
                "Run the notebook's *demo adapters* step to train it (~10 min) or copy it from the teammate who ran queue A.")
    return msg


def chat(ours_id, base_id, prompt):
    from peft import PeftModel
    if not torch.cuda.is_available():
        return "Needs a GPU runtime", "", "", ""
    if not ours_id or not os.path.exists(os.path.join(adapter_path(ours_id), "adapter_config.json")):
        return ("", "", "", f"⚠️ No adapter for `{ours_id}` in `{OUT}`. Pick one from the list (press *Refresh list* after "
                            f"adding adapters).")
    use_base = bool(base_id) and base_id != "(none)"
    notes = []
    if use_base and not os.path.exists(os.path.join(adapter_path(base_id), "adapter_config.json")):
        use_base = False
        notes.append(f"baseline `{base_id}` has no adapter on this Drive, so it was skipped")
    q4 = run_info(ours_id).get("q4", False)
    if use_base and run_info(base_id).get("q4", False) != q4:
        use_base = False
        notes.append("baseline skipped: it was trained with a different base precision (4-bit vs fp16) than the selective adapter")

    model = _base(q4)
    tok = _cache["tok"]
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)

    def gen(m):
        with torch.no_grad():
            out = m.generate(**ids, max_new_tokens=160, do_sample=False, use_cache=True)
        return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    pm = PeftModel.from_pretrained(model, adapter_path(ours_id), adapter_name="ours")
    try:
        with pm.disable_adapter():
            before = gen(pm)                                   # untouched model
        base_out = "(no baseline adapter available)"
        if use_base:
            pm.load_adapter(adapter_path(base_id), adapter_name="baseline")
            pm.set_adapter("baseline")
            base_out = gen(pm)
        pm.set_adapter("ours")
        after = gen(pm)
    finally:
        _cache["model"] = pm.unload()                          # hand the clean base model back to the cache
    info = [describe(ours_id, "Ours (selective)")]
    if use_base:
        info.insert(0, describe(base_id, "Baseline (all-layer)"))
    info += [f"_{n}_" for n in notes]
    return before, base_out, after, "\n\n".join(info)


def refresh_adapters():
    avail = list_adapters()
    return (gr.Dropdown(choices=avail, value=pick(OURS_PREF, avail) or (avail[0] if avail else None)),
            gr.Dropdown(choices=["(none)"] + avail, value=pick(BASELINE_PREF, avail) or "(none)"),
            adapter_status(avail))


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
        r_md, r_gal = gr.Markdown(), gr.Gallery(columns=1, height=520, object_fit="contain", preview=True)
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
        gr.Markdown("Same prompt, three models: the **untouched** model, the **baseline** (standard all-layer LoRA) and "
                    "**our selective-layer adapter**.")
        _avail = list_adapters()
        with gr.Row():
            ours_dd = gr.Dropdown(_avail, value=pick(OURS_PREF, _avail) or (_avail[0] if _avail else None),
                                  label="Our selective-layer adapter")
            base_dd = gr.Dropdown(["(none)"] + _avail, value=pick(BASELINE_PREF, _avail) or "(none)",
                                  label="Baseline adapter (all-layer LoRA)")
            refresh = gr.Button("Refresh list")
        status = gr.Markdown(adapter_status(_avail))
        pr = gr.Textbox("ما هي فوائد القراءة اليومية؟", label="Prompt (Arabic / Russian / Thai)")
        gr.Examples([["ما هي فوائد القراءة اليومية؟"], ["Какие преимущества у ежедневного чтения?"],
                     ["การอ่านหนังสือทุกวันมีประโยชน์อย่างไร?"]], inputs=pr)
        tb = gr.Button("Generate: untouched vs baseline vs ours", variant="primary")
        with gr.Row():
            o1 = gr.Textbox(label="Untouched model (Phase-1)", lines=8)
            o2 = gr.Textbox(label="Baseline: all-layer LoRA", lines=8)
            o3 = gr.Textbox(label="Ours: selective-layer LoRA", lines=8)
        info_md = gr.Markdown()
        tb.click(chat, [ours_dd, base_dd, pr], [o1, o2, o3, info_md])
        refresh.click(refresh_adapters, None, [ours_dd, base_dd, status])

if __name__ == "__main__":
    demo.queue().launch(share=True)
