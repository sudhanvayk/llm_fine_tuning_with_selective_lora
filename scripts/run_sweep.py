"""Run one member's queue of experiments, resumable after a Colab disconnect.

    python scripts/run_sweep.py --queue A --out /content/drive/MyDrive/hashcode_sllora/outputs --git_push
    python scripts/run_sweep.py --list            # show every queue

Each finished run writes results/runs/<run_id>.json and (with --git_push) commits + pushes it,
so the GitHub history shows every run landing, with timestamps, during the event.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sllora.selection import CostModel
from sllora.utils import load_json

# --------------------------------------------------------------------------------------------
# Measured on a T4 (Friday dry run): at 0.5B the [B, T, 151936] logits are ~80% of LoRA's peak
# memory, which hides any layer-selection effect. So EVERY run uses chunked loss (identical
# loss and gradients, verified in tests/smoke_test.py) unless it says chunked=False.
# lora_all_plain is the one standard-HF-loss baseline, kept to show the logits problem.
# Rule comparison runs use no other levers, so quality differences come from layer choice alone.
QUEUES = {
    "A": [  # M1 - baselines + memory-lever stack (needed for Review 1)
        dict(run_id="lora_all_plain", rule="all", chunked=False, bs=4, accum=4),   # batch 8 OOMs a T4 with full logits
        dict(run_id="lora_all", rule="all"),
        dict(run_id="top8", rule="top", k=8),
        dict(run_id="top8_q4_gc", rule="top", k=8, quant4=True, gc=True),
        dict(run_id="lora_all_q4_gc", rule="all", quant4=True, gc=True),
        dict(run_id="top4", rule="top", k=4),
        dict(run_id="top12", rule="top", k=12),
        dict(run_id="top8_q4", rule="top", k=8, quant4=True),
    ],
    "B": [  # M2 - reference eval + paper baselines (random = LF_H1, drift_protect = LF_H2 logic)
        dict(run_id="phase1_ref", ref=True),
        dict(run_id="random8", rule="random", k=8),
        dict(run_id="drift8", rule="drift_protect", k=8),
        dict(run_id="bottom8", rule="bottom", k=8),
        dict(run_id="random4", rule="random", k=4),
        dict(run_id="drift4", rule="drift_protect", k=4),
        dict(run_id="random12", rule="random", k=12),
        dict(run_id="drift12", rule="drift_protect", k=12),
    ],
    "C": [  # M3 - after probe_memory.py + score_layers.py --drift --importance
        dict(run_id="imp8", rule="importance", k=8),
        dict(run_id="budget50", rule="budgeted", budget_frac=0.50, k=12),
        dict(run_id="imp4", rule="importance", k=4),
        dict(run_id="budget35", rule="budgeted", budget_frac=0.35, k=12),
        dict(run_id="imp12", rule="importance", k=12),
    ],
    # Stage 2 (after midnight): fill in the 3 best run_ids from results/summary.md, then run.
    "S2": [
        dict(run_id="top8_s2", rule="top", k=8, seed=43),
        dict(run_id="imp8_s2", rule="importance", k=8, seed=43),
        dict(run_id="lora_all_s2", rule="all", seed=43),
    ],
    # Smoke run on a real model (Friday dry run: ~3 min). Delete its results after.
    "DRY": [dict(run_id="dry_top4", rule="top", k=4, max_steps=20, n_train=100, n_eval=20, eval_suite="none")],
}
for _q in QUEUES.values():
    for _r in _q:
        if not _r.get("ref"):
            _r.setdefault("chunked", True)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def git_push(paths, msg, retries=3):
    for p in paths:
        sh(f"git add -f {p}")
    if sh(f'git commit -m "{msg}"').returncode != 0:
        return
    for i in range(retries):
        sh("git pull --rebase -X theirs -q")
        if sh("git push -q").returncode == 0:
            print("   pushed:", msg)
            return
        time.sleep(5 * (i + 1))
    print("   !! git push failed, will retry with the next run")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queue")
    p.add_argument("--only", help="comma list of run_ids to run from the queue")
    p.add_argument("--out", default="outputs")
    p.add_argument("--results_dir", default="results/runs")
    p.add_argument("--drive_results", default=None, help="extra copy of result JSONs (e.g. on shared Drive)")
    p.add_argument("--git_push", action="store_true")
    p.add_argument("--eval_suite", default="quick")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list:
        for q, runs in QUEUES.items():
            print(q, [r["run_id"] for r in runs])
        return

    runs = QUEUES[a.queue]
    if a.only:
        keep = set(a.only.split(","))
        runs = [r for r in runs if r["run_id"] in keep]
    for spec in runs:
        spec = dict(spec)
        rid = spec["run_id"]
        res_path = os.path.join(a.results_dir, f"{rid}.json")
        drive_path = os.path.join(a.drive_results, f"{rid}.json") if a.drive_results else None
        if not os.path.exists(res_path) and drive_path and os.path.exists(drive_path):
            os.makedirs(a.results_dir, exist_ok=True)
            shutil.copy(drive_path, res_path)          # restore after a Colab restart (enables eval-only resume)
        if os.path.exists(res_path) and "finished" in load_json(res_path):
            print(f"== skip {rid} (done)")
            continue
        print(f"\n== {rid}  {time.strftime('%H:%M:%S')}", flush=True)
        t = time.time()
        if spec.pop("ref", False):
            payload = {"ref": dict(model_id="Qwen/Qwen2.5-0.5B-Instruct", langs=["ar", "ru", "th"],
                                   suite=a.eval_suite, results_dir=a.results_dir, run_id=rid)}
        else:
            frac = spec.pop("budget_frac", None)
            if frac is not None:
                cms = load_json("results/scores/cost_model.json")
                lev = f"Qwen2.5-0.5B-Instruct|q4={int(spec.get('quant4', False))},gc={int(spec.get('gc', False))},ck={int(spec.get('chunked', False))}"
                cm = cms.get(lev, cms["default"])
                c = CostModel(**{f: cm[f] for f in ("n_layers", "base_gb", "per_adapted_gb", "per_depth_gb")})
                # frac of the LAYER-DEPENDENT memory: weights + logits are a fixed floor
                floor, full = c.cost([c.n_layers - 1]), c.cost(list(range(c.n_layers)))
                spec["budget_gb"] = round(floor + frac * (full - floor), 3)
                print(f"   budget = floor {floor:.2f} + {frac:.0%} x ({full:.2f} - {floor:.2f}) = {spec['budget_gb']} GB")
            spec.setdefault("eval_suite", a.eval_suite)
            payload = {"cfg": dict(out_dir=a.out, results_dir=a.results_dir, **spec)}
        rc = run_isolated(payload)
        if a.drive_results and os.path.exists(res_path):
            os.makedirs(a.drive_results, exist_ok=True)
            shutil.copy(res_path, a.drive_results)     # copy even partial results (training done, eval failed)
        if rc != 0:
            print(f"!! {rid} failed (exit {rc}), moving on. Re-run the queue later: finished parts are reused.")
            continue
        mins = (time.time() - t) / 60
        if a.git_push:
            git_push([res_path], f"results: {rid} ({mins:.0f} min, queue {a.queue})")


CHILD = ("import json, sys; sys.path.insert(0, '.'); "
         "from sllora.train import RunConfig, run, eval_reference; "
         "p = json.loads(sys.argv[1]); "
         "eval_reference(**p['ref']) if 'ref' in p else run(RunConfig(**p['cfg']))")


def run_isolated(payload) -> int:
    """Every experiment runs in its OWN Python process, so GPU memory starts empty each time.
    (Running them in one process leaked ~1-1.5 GB per run on Colab and inflated memory numbers.)"""
    env = dict(os.environ, SLLORA_ISOLATED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return subprocess.run([sys.executable, "-c", CHILD, json.dumps(payload)], env=env).returncode


if __name__ == "__main__":
    main()
