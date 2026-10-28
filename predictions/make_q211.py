"""Writes the Q2.11 prediction from the failures of model A. Run after the A tests and before training B, C and D."""
import json
import os
import sys

root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, root)
from pa2.analysis import failure_flags

res = os.path.join(root, "results")
base = json.load(open(os.path.join(res, "eval_baselines.json")))
runs = [json.load(open(os.path.join(res, f"eval_A_s{s}.json"))) for s in (0, 1)]
avg = {k: sum(r[k] for r in runs) / 2 for k in ("E5_1x", "EM", "ER", "E2_2x", "E2_3x")}
flags = failure_flags(avg, base["Copy-last"])
parts = []
if flags["mirror"] or flags["rotation"]:
    names = " and ".join(n for n in ("mirror", "rotation") if flags[n])
    parts.append(f"Model B should fix the {names} failure because it trains on all eight symmetries.")
if flags["speed2"]:
    parts.append("Model C should fix the 2× failure because it sees faster motion in training.")
if flags["speed3"]:
    parts.append("Model D should fix the 3× failure that model C leaves because only D has a larger reach.")
if not parts:
    parts.append("Model A shows no failure under the stated rules so none of B, C or D is expected to change its behaviour much.")
text = "**Q2.11 Answer.** " + " ".join(parts[:3])
open(os.path.join(os.path.dirname(__file__), "Q2.11.md"), "w").write(text + "\n")
print(text, flags)
