"""Writes the Q2.5 prediction from Figure 1 and the reach count. Run after the flow and reach stages and before any test of model A."""
import json
import os

root = os.path.join(os.path.dirname(__file__), "..")
flow = json.load(open(os.path.join(root, "results", "fig1_stats.json")))
reach = json.load(open(os.path.join(root, "results", "reach.json")))["A"]["analytic"]
r = max(reach.values())
m = flow["median_speed"]
floor = flow["tv_half_split"]
tv_m, tv_r = flow["tv_mirror"], flow["tv_rotation"]
sym = max(tv_m, tv_r) <= 1.5 * floor
first = next((k for k in (1, 2, 3) if k * m > r), None)
speeds = ", ".join(f"{k}× at {k * m:.3g}" for k in (1, 2, 3))
if sym:
    sym_text = (f"Model A should pass the mirror and rotation tests in the sense that ET stays within 1.5 times its E(5), because the mirror and rotation distances of {tv_m:.3g} and {tv_r:.3g} "
                f"are within 1.5 times the half split noise floor of {floor:.3g}, so the training directions look close to symmetric. ")
else:
    sym_text = (f"Model A should fail at least one of the mirror and rotation tests, with ET above 1.5 times its E(5), because the mirror and rotation distances of {tv_m:.3g} and {tv_r:.3g} "
                f"exceed 1.5 times the half split noise floor of {floor:.3g}, so transformed motion is likely unfamiliar. ")
text = (
    "**Q2.5 Answer.** " + sym_text
    + "Rotation also swaps the roles of rows and columns inside each folded patch, so ER is expected to exceed EM. "
    + f"The median speed is {m:.3g} pixels per frame so the three speeds give {speeds} pixels per frame against a reach of {r} pixels. "
    + (f"Model A should therefore break down at {first}×, the first speed above the reach." if first else "Model A should therefore not break down at any tested speed because none exceeds the reach.")
)
open(os.path.join(os.path.dirname(__file__), "Q2.5.md"), "w").write(text + "\n")
print(text)
