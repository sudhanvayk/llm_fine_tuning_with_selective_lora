"""Merge results/runs/*.json -> results/summary.csv + results/summary.md (paste into slides / README).

    python scripts/aggregate.py
"""
import glob
import json
import os
import sys

import pandas as pd

RUNS = sys.argv[1] if len(sys.argv) > 1 else "results/runs"


def row(r):
    c, m = r.get("config", {}), r.get("memory", {})
    le, ho, tr = r.get("lm_eval", {}), r.get("heldout", {}), r.get("train", {})
    lev = "+".join(n for n, f in [("NF4", c.get("quant4")), ("GC", c.get("gc")), ("chunk", c.get("chunked"))] if f) or "-"
    return {"run_id": r["run_id"], "rule": c.get("rule"), "k": c.get("k") or (len(r["layers"]) if r.get("layers") else None),
            "levers": lev, "seed": c.get("seed"), "strip": r.get("strip"), "depth": r.get("backprop_depth"),
            "trainable_M": round(r.get("trainable_params", 0) / 1e6, 2) or None,
            "probe_peak_gb": m.get("probe_peak_gb"), "train_peak_gb": tr.get("train_peak_gb"),
            "tok_s": tr.get("tok_s"), "train_min": round(tr["train_s"] / 60, 1) if tr.get("train_s") else None,
            "heldout_loss": ho.get("avg"), "task_avg": le.get("task_avg"), "lang_avg": le.get("lang_avg"),
            "hellaswag": le.get("hellaswag"), "arc_easy": le.get("arc_easy"), "mmlu": le.get("mmlu"),
            "finished": r.get("finished")}


rows = [row(json.load(open(f, encoding="utf-8"))) for f in sorted(glob.glob(os.path.join(RUNS, "*.json")))]
df = pd.DataFrame(rows)
if df.empty:
    sys.exit("no results yet")
base = df[(df.run_id == "lora_all")]
if len(base):
    b = base.iloc[0]
    df["mem_vs_lora_%"] = (100 * (df.probe_peak_gb / b.probe_peak_gb - 1)).round(1)
    for col in ["task_avg", "lang_avg"]:
        if pd.notna(b[col]):
            df[f"d_{col}_pts"] = (100 * (df[col] - b[col])).round(2)
    df["d_heldout_loss"] = (df.heldout_loss - b.heldout_loss).round(4)
df = df.sort_values(["probe_peak_gb"], na_position="first")
os.makedirs("results", exist_ok=True)
df.to_csv("results/summary.csv", index=False)
cols = [c for c in ["run_id", "strip", "levers", "probe_peak_gb", "mem_vs_lora_%", "tok_s", "heldout_loss",
                    "task_avg", "d_task_avg_pts", "lang_avg", "d_lang_avg_pts"] if c in df]
with open("results/summary.md", "w", encoding="utf-8") as f:
    f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
    for _, r in df[cols].iterrows():
        f.write("| " + " | ".join("" if pd.isna(v) else (f"`{v}`" if k == "strip" else (f"{v:.4g}" if isinstance(v, float) else str(v)))
                                  for k, v in r.items()) + " |\n")
print(open("results/summary.md", encoding="utf-8").read())
