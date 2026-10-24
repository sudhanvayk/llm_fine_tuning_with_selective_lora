"""One table of every run: memory, speed, quality, and deltas vs the baselines.

    python scripts/report.py                      # reads results/runs/*.json
    python scripts/report.py --also /content/drive/MyDrive/hashcode_sllora/results   # merge Drive copies
"""
import argparse
import glob
import json
import os

p = argparse.ArgumentParser()
p.add_argument("--dir", default="results/runs")
p.add_argument("--also", default=None, help="second folder of result JSONs (e.g. Drive); newest copy wins")
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
        old = runs.get(rid)
        score = ("finished" in r, r.get("memory", {}).get("isolated", False), "lm_eval" in r)
        if old is None or score >= old[0]:
            runs[rid] = (score, r)
runs = {k: v[1] for k, v in runs.items()}


def g(r, *path):
    for k in path:
        if not isinstance(r, dict) or k not in r:
            return None
        r = r[k]
    return r


def f(x, n=3):
    return "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


ref = runs.get("phase1_ref")
base = runs.get("lora_all_plain")
hdr = f"{'run':16s} {'layers':26s} {'mem GB':>7s} {'clean':>5s} {'vs plain':>8s} {'tok/s':>6s} {'heldout':>7s} {'task':>6s} {'lang':>6s} {'hswag':>6s} {'arc_e':>6s} {'mmlu':>6s} {'status':>8s}"
print(hdr)
print("-" * len(hdr))
order = sorted(runs.values(), key=lambda r: (r["run_id"] != "phase1_ref", g(r, "memory", "probe_peak_gb") or 99))
for r in order:
    mem = g(r, "memory", "probe_peak_gb")
    bm = g(base, "memory", "probe_peak_gb") if base else None
    vs = f"{100 * (mem / bm - 1):+.0f}%" if mem and bm else "-"
    status = "done" if "finished" in r else ("eval?" if "train" in r else "partial")
    e = r.get("lm_eval", {})
    print(f"{r['run_id']:16s} {(r.get('strip') or '-'):26s} {f(mem, 2):>7s} {('yes' if g(r, 'memory', 'isolated') else 'NO') if mem else '-':>5s} "
          f"{vs:>8s} {f(g(r, 'train', 'tok_s'), 0) if g(r, 'train', 'tok_s') else '-':>6s} {f(g(r, 'heldout', 'avg')):>7s} "
          f"{f(e.get('task_avg'), 4):>6s} {f(e.get('lang_avg'), 4):>6s} {f(e.get('hellaswag'), 3):>6s} {f(e.get('arc_easy'), 3):>6s} "
          f"{f(e.get('mmlu'), 3):>6s} {status:>8s}")
print("\nclean = memory measured in a fresh process. 'NO' -> run: python scripts/reprobe.py --auto")
if ref:
    print(f"phase1_ref (untouched model): heldout {f(g(ref, 'heldout', 'avg'))}  task {f(g(ref, 'lm_eval', 'task_avg'), 4)}  lang {f(g(ref, 'lm_eval', 'lang_avg'), 4)}")
