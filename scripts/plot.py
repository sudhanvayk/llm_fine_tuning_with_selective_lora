"""Slide-ready figures -> results/plots/*.png

    python scripts/plot.py
  pareto.png          quality vs measured peak memory, one colour per rule, frontier line
  levers.png          memory waterfall: all-layer LoRA -> +chunked loss -> top-8 -> +NF4 -> +GC
  layer_scores.png    drift + importance per layer, and which layers each rule picked
  probes.png          measured peak memory per layer set (from probe_memory.py)
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = "results/plots"
os.makedirs(OUT, exist_ok=True)
# fixed categorical order (validated palette) -- colour follows the rule, never its rank
RULE_COLORS = {"all": "#2a78d6", "top": "#eb6834", "random": "#1baf7a", "drift_protect": "#eda100",
               "importance": "#e87ba4", "budgeted": "#6250d6", "bottom": "#008300"}
RULE_LABEL = {"all": "All-layer LoRA", "top": "Top-k", "random": "Random-k (LF_H1)",
              "drift_protect": "Drift-protect-k (LF_H2)", "importance": "Importance-k", "budgeted": "Budgeted",
              "bottom": "Bottom-k (control)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.dpi": 160,
                     "savefig.bbox": "tight", "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 13})


def load_runs():
    rs = []
    for f in glob.glob("results/runs/*.json"):
        r = json.load(open(f, encoding="utf-8"))
        if r.get("run_id") == "phase1_ref":
            continue
        c = r.get("config", {})
        if c.get("quant4") or c.get("gc") or not c.get("chunked") or c.get("seed", 42) != 42:
            continue                                   # rule comparison: fp16 + chunked loss, seed 42 only
        rs.append(dict(run_id=r["run_id"], rule=c.get("rule"), k=len(r.get("layers", [])),
                       mem=r.get("memory", {}).get("probe_peak_gb"), task=r.get("lm_eval", {}).get("task_avg"),
                       lang=r.get("lm_eval", {}).get("lang_avg"), loss=r.get("heldout", {}).get("avg")))
    return pd.DataFrame(rs)


def frontier(df, y, higher_better=True):
    d = df.dropna(subset=["mem", y]).sort_values("mem")
    best, pts = None, []
    for _, r in d.iterrows():
        v = r[y] if higher_better else -r[y]
        if best is None or v > best:
            best = v
            pts.append((r.mem, r[y]))
    return pts


def pareto(df):
    panels = [("task", "Task retention (avg acc, higher = better)", True),
              ("loss", "Held-out multilingual loss (lower = better)", False)]
    if df["lang"].notna().any():
        panels.insert(1, ("lang", "Belebele avg acc (higher = better)", True))
    panels = [p for p in panels if df[p[0]].notna().any()]
    if not panels or df["mem"].isna().all():
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.6))
    axes = [axes] if len(panels) == 1 else axes
    for ax, (y, title, hb) in zip(axes, panels):
        for rule, g in df.dropna(subset=["mem", y]).groupby("rule"):
            ax.scatter(g.mem, g[y], s=40 + 6 * g.k, color=RULE_COLORS.get(rule, MUTED), label=RULE_LABEL.get(rule, rule),
                       edgecolor="white", linewidth=1.5, zorder=3)
            for _, r in g.iterrows():
                if rule in ("all", "top", "budgeted", "importance"):
                    ax.annotate(r.run_id, (r.mem, r[y]), xytext=(5, 4), textcoords="offset points", fontsize=8, color=MUTED)
        f = frontier(df, y, hb)
        if len(f) > 1:
            ax.plot(*zip(*f), color=MUTED, lw=1.2, ls="--", zorder=2, label="Pareto frontier")
        ax.set_xlabel("Measured peak GPU memory (GB, T4, 4x512 tokens)")
        ax.set_title(title, loc="left")
    axes[0].legend(frameon=False, fontsize=8.5, loc="best")
    fig.suptitle("Quality vs memory: every point is a measured run", x=0.01, y=1.04, ha="left", fontsize=14, fontweight="bold")
    fig.savefig(f"{OUT}/pareto.png")
    plt.close(fig)


def levers():
    order = [("lora_all_plain", "All-layer\nLoRA"), ("lora_all", "+ chunked\nloss"), ("top8", "Top-8\nlayers"),
             ("top8_q4", "+ NF4\nbase"), ("top8_q4_gc", "+ grad\ncheckpoint")]
    vals = []
    for rid, lab in order:
        f = f"results/runs/{rid}.json"
        if os.path.exists(f):
            vals.append((lab, json.load(open(f))["memory"]["probe_peak_gb"]))
    if len(vals) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 4.4))
    xs = range(len(vals))
    ax.bar(xs, [v for _, v in vals], width=0.6, color=["#2a78d6"] + ["#eb6834"] * (len(vals) - 1), edgecolor="white", linewidth=2)
    b = vals[0][1]
    for i, (_, v) in enumerate(vals):
        ax.text(i, v + b * 0.02, f"{v:.2f} GB" + ("" if i == 0 else f"\n−{100 * (1 - v / b):.0f}%"), ha="center",
                va="bottom", fontsize=10, color=INK)
    ax.set_xticks(list(xs), [l for l, _ in vals])
    ax.set_ylabel("Measured peak memory (GB)")
    ax.set_ylim(0, b * 1.25)
    ax.grid(axis="x", visible=False)
    ax.set_title("Stacking memory levers on Qwen2.5-0.5B (measured on a free T4)", loc="left")
    fig.savefig(f"{OUT}/levers.png")
    plt.close(fig)


def layer_scores():
    f = "results/scores/layer_scores.json"
    if not os.path.exists(f):
        return
    s = json.load(open(f))
    keys = [k for k in ("drift", "importance") if k in s]
    if not keys:
        return
    L = len(s[keys[0]])
    fig, axes = plt.subplots(len(keys) + 1, 1, figsize=(10, 2.3 * (len(keys) + 1)), sharex=True)
    for ax, k, col in zip(axes, keys, ["#eda100", "#e87ba4"]):
        v = s[k]
        ax.bar(range(L), v, color=col, width=0.75)
        ax.set_ylabel(k)
        ax.set_title({"drift": "Base → Instruct weight drift per layer (MPD)",
                      "importance": "Gradient-norm importance on 256 MultiAlpaca samples"}[k], loc="left", fontsize=11)
    ax = axes[-1]
    rules = []
    for fp in sorted(glob.glob("results/runs/*.json")):
        r = json.load(open(fp))
        if r.get("layers") and r["config"].get("k") == 8 and not r["config"].get("quant4") and r["config"].get("chunked") and r["config"].get("seed", 42) == 42:
            rules.append((r["config"]["rule"], r["layers"]))
    for i, (rule, layers) in enumerate(rules):
        ax.scatter(layers, [i] * len(layers), marker="s", s=60, color=RULE_COLORS.get(rule, MUTED))
    ax.set_yticks(range(len(rules)), [RULE_LABEL.get(r, r) for r, _ in rules], fontsize=8.5)
    ax.set_xlabel("Layer index (0 = closest to the embeddings)")
    ax.set_title("Layers chosen by each rule (k = 8)", loc="left", fontsize=11)
    ax.set_xticks(range(0, L, 2))
    fig.savefig(f"{OUT}/layer_scores.png")
    plt.close(fig)


def probes():
    f = "results/scores/memory_probes.json"
    if not os.path.exists(f):
        return
    df = pd.DataFrame(json.load(open(f)))
    df = df[df.model == df.model.iloc[0]]
    sets = list(dict.fromkeys(df.set))
    levs = list(dict.fromkeys(df.lever))
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.8 / len(levs)
    for j, lev in enumerate(levs):
        g = df[df.lever == lev].set_index("set").reindex(sets)
        ax.bar([i + j * w for i in range(len(sets))], g.peak_gb, width=w * 0.92, color=cols[j % len(cols)], label=lev)
    ax.set_xticks([i + 0.4 - w / 2 for i in range(len(sets))], sets, fontsize=9)
    ax.set_ylabel("Measured peak memory (GB)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=len(levs), fontsize=9, loc="upper right")
    ax.set_title(f"Peak memory by layer set and lever — {df.model.iloc[0]} (4 × 512 tokens)", loc="left")
    fig.savefig(f"{OUT}/probes.png")
    plt.close(fig)


if __name__ == "__main__":
    df = load_runs()
    if not df.empty:
        pareto(df)
    levers()
    layer_scores()
    probes()
    print("plots ->", sorted(os.listdir(OUT)))
