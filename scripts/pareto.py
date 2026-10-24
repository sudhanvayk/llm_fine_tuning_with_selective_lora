"""Pareto-frontier analysis: peak GPU memory (minimise) vs validation quality (held-out loss, minimise;
task accuracy, maximise). Computed from measured results only.

    python scripts/pareto.py --suite main      # main sweep runs (chunked fp16 rule comparison + stacks)
    python scripts/pareto.py --suite spec      # experiments A-E (spec_* runs)
    python scripts/pareto.py --also /content/drive/MyDrive/hashcode_sllora/results

A run is Pareto-optimal if no other run has <= memory AND <= held-out loss with at least one strict.
Memory numbers not measured in an isolated process are marked 'inflated' and EXCLUDED from the frontier.
Outputs: results/pareto_<suite>.csv, results/pareto_<suite>.md, results/plots/pareto_<suite>.png
"""
import argparse
import csv
import glob
import json
import os

p = argparse.ArgumentParser()
p.add_argument("--suite", default="main", choices=["main", "spec", "all"])
p.add_argument("--dir", default="results/runs")
p.add_argument("--also", default=None)
p.add_argument("--baseline", default=None, help="run_id for memory-reduction %% (default: lora_all_plain / spec_A_all)")
a = p.parse_args()

runs = {}
for d in [a.also, a.dir]:
    if not d:
        continue
    for f in glob.glob(os.path.join(d, "*.json")):
        try:
            r = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        rid = r.get("run_id")
        q = ("finished" in r, r.get("memory", {}).get("isolated", False))
        if rid not in runs or q >= runs[rid][0]:
            runs[rid] = (q, r)
runs = {k: v[1] for k, v in runs.items()}

is_spec = lambda rid: rid.startswith("spec_")
if a.suite == "main":
    sel = {k: v for k, v in runs.items() if not is_spec(k) and k != "phase1_ref" and not k.startswith("dry")}
    base_id, ref_id = a.baseline or "lora_all_plain", "phase1_ref"
elif a.suite == "spec":
    sel = {k: v for k, v in runs.items() if is_spec(k) and k != "spec_ref"}
    base_id, ref_id = a.baseline or "spec_A_all", "spec_ref"
else:
    sel = {k: v for k, v in runs.items() if k not in ("phase1_ref", "spec_ref")}
    base_id, ref_id = a.baseline or "lora_all_plain", "phase1_ref"
ref = runs.get(ref_id, {})
base_mem = runs.get(base_id, {}).get("memory", {}).get("probe_peak_gb")


def levers(c):
    return "+".join(n for n, f in [("NF4", c.get("quant4")), ("GC", c.get("gc")), ("chunk", c.get("chunked")),
                                   ("offload", c.get("offload_acts"))] if f) or "plain"


rows = []
for rid, r in sel.items():
    m, t, h, e, c = r.get("memory", {}), r.get("train", {}), r.get("heldout", {}), r.get("lm_eval", {}), r.get("config", {})
    mem = m.get("probe_peak_gb")
    rows.append(dict(
        run_id=rid, rule=c.get("rule"), levers=levers(c), layers=r.get("strip"), n_adapted=len(r.get("layers") or []),
        trainable_params=r.get("trainable_params"), trainable_pct=r.get("trainable_pct"),
        peak_gb=mem, reserved_gb=m.get("probe_reserved_gb"), train_peak_gb=t.get("train_peak_gb"),
        mem_status="measured" if m.get("isolated") else ("inflated" if mem else "missing"),
        mem_reduction_pct=round(100 * (base_mem - mem) / base_mem, 1) if (mem and base_mem and m.get("isolated")) else None,
        heldout_loss=h.get("avg"), task_avg=e.get("task_avg"), lang_avg=e.get("lang_avg"),
        d_task_pts=round(100 * (e["task_avg"] - ref["lm_eval"]["task_avg"]), 2) if e.get("task_avg") is not None and ref.get("lm_eval") else None,
        tok_s=t.get("tok_s"), step_time_s=t.get("step_time_s"), train_min=round(t["train_s"] / 60, 1) if t.get("train_s") else None,
        cpu_ram_gb=t.get("peak_cpu_ram_gb"), status=r.get("status") or ("done" if "finished" in r else "incomplete")))

# Pareto over (peak_gb, heldout_loss), clean memory + finished only
cand = [x for x in rows if x["mem_status"] == "measured" and x["heldout_loss"] is not None]
for x in rows:
    x["pareto"] = False
for x in cand:
    dominated = any((y["peak_gb"] <= x["peak_gb"] and y["heldout_loss"] <= x["heldout_loss"]) and
                    (y["peak_gb"] < x["peak_gb"] or y["heldout_loss"] < x["heldout_loss"]) for y in cand if y is not x)
    x["pareto"] = not dominated
rows.sort(key=lambda x: (x["peak_gb"] is None, x["peak_gb"] or 0))

os.makedirs("results/plots", exist_ok=True)
with open(f"results/pareto_{a.suite}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_id"])
    w.writeheader()
    w.writerows(rows)
cols = ["run_id", "layers", "levers", "trainable_pct", "peak_gb", "mem_status", "mem_reduction_pct", "heldout_loss",
        "d_task_pts", "tok_s", "step_time_s", "pareto", "status"]
fmt = lambda v: "" if v is None else (f"{v:.4g}" if isinstance(v, float) else ("**yes**" if v is True else ("" if v is False else str(v))))
md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
md += ["| " + " | ".join(fmt(x[c]) if c != "layers" else f"`{x[c]}`" for c in cols) + " |" for x in rows]
note = (f"\nBaseline for memory reduction: `{base_id}` ({base_mem} GB). Reference for Δtask: `{ref_id}`"
        f" (held-out loss {ref.get('heldout', {}).get('avg')}). Pareto computed on measured (isolated) memory only;"
        " 'inflated' rows need `python scripts/reprobe.py --auto`.")
open(f"results/pareto_{a.suite}.md", "w").write("\n".join(md) + "\n" + note + "\n")
print("\n".join(md) + "\n" + note)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for x in rows:
        if x["peak_gb"] is None or x["heldout_loss"] is None:
            continue
        clean = x["mem_status"] == "measured"
        ax.scatter(x["peak_gb"], x["heldout_loss"], s=70, zorder=3,
                   facecolor=("#eb6834" if x["pareto"] else "#2a78d6") if clean else "none",
                   edgecolor="#eb6834" if x["pareto"] else "#2a78d6", linewidth=1.5)
        ax.annotate(x["run_id"], (x["peak_gb"], x["heldout_loss"]), xytext=(5, 4), textcoords="offset points", fontsize=8, color="#52514e")
    fr = sorted([x for x in rows if x["pareto"]], key=lambda x: x["peak_gb"])
    if len(fr) > 1:
        ax.plot([x["peak_gb"] for x in fr], [x["heldout_loss"] for x in fr], "--", color="#eb6834", lw=1.2, label="Pareto frontier (measured)")
    if ref.get("heldout", {}).get("avg"):
        ax.axhline(ref["heldout"]["avg"], color="#8a8984", lw=1, ls=":", label=f"untouched model ({ref['heldout']['avg']:.3f})")
    ax.set_xlabel("Peak GPU memory (GB, measured)")
    ax.set_ylabel("Held-out multilingual loss (lower = better)")
    ax.set_title(f"Memory vs quality ({a.suite} suite) - hollow = memory not yet re-measured", loc="left", fontsize=11)
    ax.grid(color="#e4e3df")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(f"results/plots/pareto_{a.suite}.png", dpi=160, bbox_inches="tight")
    print(f"plot -> results/plots/pareto_{a.suite}.png")
except Exception as ex:
    print("plot skipped:", ex)
