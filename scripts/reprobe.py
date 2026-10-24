"""Re-measure peak memory for finished runs whose probe ran in a shared process (inflated numbers).
Each run is re-probed in its own fresh process (~30 s each); only the "memory" field is replaced.

    python scripts/reprobe.py --auto                  # every run whose memory is not marked isolated
    python scripts/reprobe.py lora_all top8           # specific runs
    python scripts/reprobe.py --auto --drive_results /content/drive/MyDrive/hashcode_sllora/results
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sllora.train import RunConfig  # noqa: E402  (only for the field list; no GPU work here)

CHILD = ("import json, sys; sys.path.insert(0, '.'); from sllora.train import RunConfig, run; "
         "run(RunConfig(**json.loads(sys.argv[1])))")

p = argparse.ArgumentParser()
p.add_argument("run_ids", nargs="*")
p.add_argument("--auto", action="store_true")
p.add_argument("--results_dir", default="results/runs")
p.add_argument("--drive_results", default=None)
a = p.parse_args()

fields = set(RunConfig.__dataclass_fields__)
todo = []
for f in sorted(glob.glob(os.path.join(a.results_dir, "*.json"))):
    r = json.load(open(f, encoding="utf-8"))
    rid = r.get("run_id")
    if "memory" not in r or rid == "phase1_ref":
        continue
    if rid in a.run_ids or (a.auto and not r["memory"].get("isolated")):
        todo.append(r)
print("re-probing:", [r["run_id"] for r in todo])
env = dict(os.environ, SLLORA_ISOLATED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
for r in todo:
    old = r["memory"].get("probe_peak_gb")
    cfg = {k: v for k, v in r["config"].items() if k in fields}
    cfg.update(reprobe_only=True, results_dir=a.results_dir, layers=r["layers"])   # exact same layer set
    rc = subprocess.run([sys.executable, "-c", CHILD, json.dumps(cfg)], env=env).returncode
    new = json.load(open(os.path.join(a.results_dir, f"{r['run_id']}.json")))["memory"].get("probe_peak_gb")
    print(f"   {r['run_id']:20s} {old} GB -> {new} GB" + ("" if rc == 0 else "  (FAILED)"))
    if a.drive_results and rc == 0:
        shutil.copy(os.path.join(a.results_dir, f"{r['run_id']}.json"), a.drive_results)
