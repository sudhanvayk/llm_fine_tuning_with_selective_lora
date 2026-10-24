"""Re-evaluate a finished run's adapter with the FULL suite (top configs only, ~20 min each).

    python scripts/eval_adapter.py --run_id top8 --out /content/drive/MyDrive/hashcode_sllora/outputs
Adds result["lm_eval_full"] to results/runs/<run_id>.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peft import PeftModel

from sllora.evaluate import lm_eval_scores
from sllora.model import load_model_and_tokenizer
from sllora.utils import load_json, now_iso, save_json

p = argparse.ArgumentParser()
p.add_argument("--run_id", required=True)
p.add_argument("--out", default="outputs")
p.add_argument("--results_dir", default="results/runs")
p.add_argument("--suite", default="full")
a = p.parse_args()

path = os.path.join(a.results_dir, f"{a.run_id}.json")
res = load_json(path)
cfg = res["config"]
model, tok = load_model_and_tokenizer(cfg["model_id"], quant4=cfg.get("quant4", False))
if a.run_id != "phase1_ref":
    model = PeftModel.from_pretrained(model, os.path.join(a.out, a.run_id, "adapter"))
res[f"lm_eval_{a.suite}"] = lm_eval_scores(model, tok, cfg.get("langs", ["ar", "ru", "th"]), a.suite)
res[f"lm_eval_{a.suite}"]["evaluated"] = now_iso()
save_json(res, path)
print(res[f"lm_eval_{a.suite}"])
