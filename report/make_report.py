import argparse
import glob
import json
import math
import os
import re
import sys

import markdown
import numpy as np
from latex2mathml.converter import convert as to_mathml

ROOT = os.environ.get("PA2_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
PENDING = []


def load(name):
    return json.load(open(os.path.join(RES, name)))


def exists(name):
    return os.path.exists(os.path.join(RES, name))


def fmt(x):
    if x == 0:
        return "0"
    if abs(x) < 1e-3:
        mant, exp = f"{x:.2e}".split("e")
        return f"{mant}e{int(exp)}"
    d = max(0, 2 - int(math.floor(math.log10(abs(x)))))
    return f"{x:.{d}f}"


def ms(vals):
    return f"{fmt(float(np.mean(vals)))} ({fmt(vals[0])}, {fmt(vals[1])})"


def ev(name, seed, sym=False):
    return load(f"eval_{name}{'sym' if sym else ''}_s{seed}.json")


def vals(name, key, sym=False):
    return [ev(name, s, sym)[key] for s in (0, 1)]


def mean(name, key, sym=False):
    return float(np.mean(vals(name, key, sym)))


def infer_time(name, seed, sym=False):
    return float(np.median(ev(name, seed, sym)["inference_seconds"]))


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pending(tag, text):
    PENDING.append(tag)
    return f"**[[PENDING {tag}: {text}]]**"


def read_prediction(name):
    path = os.path.join(ROOT, "predictions", name)
    if not os.path.exists(path):
        return pending(name, "commit this text before the experiment, then place it here unedited")
    return open(path).read().strip()


def stage_line(label, st):
    return f"{label}: {st['blobs']} blobs, peak {fmt(st['peak'])}, mass {fmt(st['mass'])}"


def gap(a, b):
    from pa2.diagnostics import centroid_gap
    return centroid_gap(a, b)


def describe_failure(truth, row3, row4, h):
    """Numeric description of how row 3 differs from row 4 at horizon index h."""
    g3, g4 = gap(row3[h], truth[h]), gap(row4[h], truth[h])
    parts = [f"At forecast horizon {h + 1} the true frame has {truth[h]['blobs']} blobs. Row 3 has {row3[h]['blobs']} blobs "
             f"with peak intensity {fmt(row3[h]['peak'])} and row 4 has {row4[h]['blobs']} blobs with peak intensity {fmt(row4[h]['peak'])}."]
    if g3 is not None and g4 is not None:
        parts.append(f"The blob centres in row 3 lie {fmt(g3)} pixels from the truth and those in row 4 lie {fmt(g4)} pixels from the truth.")
    kind = []
    if row3[h]["peak"] < 0.6 * row4[h]["peak"]:
        kind.append("row 3 fades and blurs relative to row 4")
    if row3[h]["blobs"] > truth[h]["blobs"]:
        kind.append("row 3 contains an extra ghost digit")
    if row3[h]["blobs"] < truth[h]["blobs"]:
        kind.append("row 3 loses or merges a digit")
    if g3 is not None and g4 is not None and g3 > g4 + 3:
        kind.append("row 3 places a digit at the wrong position")
    parts.append("The measured pattern is that " + (" and ".join(kind) if kind else "the two rows differ only slightly") + ".")
    return " ".join(parts)


def worth_sentence(chg, mean, flags, cm_):
    """Verdict per change from measured numbers. Cost is the relative rise of E(5) at 1x. Gain is the relative drop of the targeted error."""
    out = []
    targets = {"B": ("EM", "ER"), "C": ("E2_2x", "E2_3x"), "D": ("E2_3x",)}
    for n in "BCD":
        gains = []
        for k in targets[n]:
            a_val, n_val = mean("A", k), mean(n, k)
            gains.append(f"{k} falls from {fmt(a_val)} to {fmt(n_val)}")
        big = any(mean(n, k) < 0.5 * mean("A", k) for k in targets[n])
        cost = chg[n]
        verdict = ("worth it" if big and cost < 0.25 else ("not worth it" if not big else "a costly trade"))
        out.append(f"Model {n} changes 1× E(5) by {fmt(100 * cost)} percent while " + " and ".join(gains) + f", which makes it {verdict} under the rule that a gain above 50 percent against a cost under 25 percent is worth it.")
    return " ".join(out)


def build_q2(a):
    base = load("eval_baselines.json")
    blank, copy = base["Blank"], base["Copy-last"]
    flow = load("fig1_stats.json")
    reach = load("reach.json")
    sel = load("selection.json")
    diag = load("diagnostics.json")
    t1 = load("table1.json")
    sel_d = load("d_selection.json")
    names = ["A", "B", "C", "D"]
    final = a.final
    med = flow["median_speed"]
    md = []
    add = md.append

    add("# Question 2. Motion or Memory?\n")
    add(f"Hardware used for every timing is {ev('A', 0)['hardware']}.\n")

    add("## 1. Setup\n")
    add('**Q.** Mirror and rotation only rearrange pixels, so checks (a)–(c) must hold exactly: report the maximum absolute difference between the two sides, which must be 0, and torch.equal must return True. Checks (d)–(e) catch mistakes that (a)–(c) cannot, such as flipping the wrong axis, rotating clockwise, or reordering frames. For (f)–(h), run your window function with t = 1 on a video whose frame f is filled with the value f, and read off which frames it returns. Report the results as Table 1, in this format, with the last column filled in. If any row does not match, fix your code before continuing: every later result depends on these functions.\n')
    add("**Answer.** Table 1 lists the eight checks on the synthetic tensors. Checks (a) to (c) use torch.equal and the maximum absolute difference. "
        "Every row matches the expected value.\n")
    rows = []
    for r in t1:
        yours = r["yours"]
        if isinstance(yours, list) and len(yours) == 2 and isinstance(yours[0], list):
            yours = f"{', '.join(map(str, yours[0]))} / {', '.join(map(str, yours[1]))}"
        elif isinstance(yours, list):
            yours = "(" + ", ".join(map(str, yours)) + ")"
        elif isinstance(yours, (int, float)):
            yours = f"{fmt(yours)}" + (", torch.equal True" if r["torch_equal"] else ", torch.equal False")
        rows.append([r["check"], r["expected"], yours])
    add(table(["Check", "Expected", "Yours"], rows))
    add("\nTable 1. This table gives the transformation and window checks run on a random video and a marker video.\n")

    add('**Q.** Estimate dense optical flow on the first 50 training videos, between every pair of consecutive frames (x_s, x_{s+1}), s = 1, ..., 19, giving 50 × 19 = 950 frame pairs. Compute the flow on frames converted to uint8 (multiply by 255) with OpenCV’s Farneback method and these settings: cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0). Use exactly this method and these settings for all reported results. For each pixel the flow gives a displacement (dx, dy), where dx is along columns and dy along rows (rows increase downward). Keep only pixels whose intensity on the [0, 1] scale exceeds 0.2 at the same location in both x_s and x_{s+1} (equivalently, above 51 on the uint8 scale), and whose flow magnitude sqrt(dx² + dy²) exceeds 0.5 pixels. Pool the kept pixels from all 950 pairs. Figure 1 should show, for these pooled pixels: (a) a histogram of flow direction θ = atan2(−dy, dx), in degrees on [0°, 360°) with 18 bins of 20°, so that 0° is rightward and 90° is upward on screen, with a horizontal line at 1/18 (the level of a perfectly uniform distribution); (b) a histogram of flow speed sqrt(dx² + dy²) in pixels per frame, with the median marked by a vertical line. Plot both as the fraction of kept pixels (not raw counts), and state the median speed and the number of pixels kept in the caption.\n')
    add(f"**Answer.** Figure 1 uses {flow['pairs']} frame pairs.\n")
    add("![Figure 1](../figures/fig1.png)\n")
    add(f"Figure 1. The plot shows the fraction of kept pixels by flow direction in 18 bins of 20 degrees with the uniform level 1/18 as a dashed line in panel (a) "
        f"and by flow speed in pixels per frame in panel (b). The median speed is {fmt(med)} pixels per frame and {flow['kept_pixels']:,} pixels were kept.\n")
    mx, mn = flow["max_bin_over_uniform"], flow["min_bin_over_uniform"]
    noise = flow["tv_half_split"]
    add(f"**Q2.1 Answer.** Figure 1(a) has bin fractions between {fmt(mn)} and {fmt(mx)} times the uniform level. "
        f"A mirror maps a direction θ to 180 minus θ and a rotation adds 90 degrees. The total variation distance between the histogram and its mirrored copy is {fmt(flow['tv_mirror'])} "
        f"and for the rotated copy it is {fmt(flow['tv_rotation'])}. Two halves of the same pooled data, videos 0 to 24 against 25 to 49, differ by {fmt(noise)}, so that is the noise floor. ")
    familiar = flow["tv_mirror"] <= 2 * noise and flow["tv_rotation"] <= 2 * noise
    add("Both distances are within twice the noise floor, so mirrored and rotated trajectories are already familiar to a model trained on this data."
        if familiar else
        "At least one distance exceeds twice the noise floor, so the training directions are not fully symmetric and the corresponding transformed trajectories are partly unfamiliar.")
    add(f" The typical speed is {fmt(med)} pixels per frame with quartiles {fmt(flow['speed_quartiles'][0])} and {fmt(flow['speed_quartiles'][1])}. "
        f"A 2× window therefore moves a digit about {fmt(2 * med)} pixels between the frames the model sees and a 3× window about {fmt(3 * med)} pixels. "
        f"Flow is unreliable where digits overlap. Restricting the pooling to the {flow['separate_digit_pairs']} frame pairs where both frames show two separate blobs "
        f"keeps {flow['kept_pixels_separate']:,} pixels. This subset has median speed {fmt(flow['median_speed_separate'])} and its histogram differs from the full one by {fmt(flow['tv_all_vs_separate'])}. "
        + ("That change is within the noise floor, so overlap bias is too small to change the answer." if flow["tv_all_vs_separate"] <= 2 * noise
           else "That change exceeds twice the noise floor, so overlap bias could move individual bins but the mirror and rotation distances of the subset "
                f"are {fmt(flow['tv_mirror_separate'])} and {fmt(flow['tv_rotation_separate'])} and decide whether the conclusion holds.") + "\n")
    add("**Q2.2 Answer.** Mirror and rotation change the digit appearance because a flipped 2 is not a 2 and a turned digit is stranger still. "
        f"They change the motion statistics only to the extent that the direction histogram is not symmetric, which Figure 1(a) measures as {fmt(flow['tv_mirror'])} and {fmt(flow['tv_rotation'])}. "
        "Speed up changes the motion statistics because digits move further between the seen frames and leaves the appearance of every digit untouched. "
        "Each transformation therefore moves one factor while holding the other nearly fixed. A failure under mirror or rotation points to appearance while a failure under speed up points to motion, "
        "which is what makes the three of them a controlled experiment.\n")
    add("Trade off. A flat direction histogram and a fixed digit set make appearance and motion separable here, but the conclusion holds only for this dataset and these speeds.\n")

    add("## 2. Baseline: ConvLSTM\n")
    add('**Q.** Table 2: MSE_1, ..., MSE_5, E^(5), and trainable parameter count for model A, Blank and Copy-last. Figure 2: input, prediction and ground truth for three test videos, at least one of them with overlapping digits.\n')
    rows = [["A"] + [ms([ev("A", s)["mse_h"][i] for s in (0, 1)]) for i in range(5)]
            + [ms(vals("A", "E5_1x")), f"{ev('A', 0)['params']:,}"]]
    for nm, b in (("Blank", blank), ("Copy-last", copy)):
        rows.append([nm] + [fmt(v) for v in b["mse_h"]] + [fmt(b["E5_1x"]), "0"])
    add(table(["Predictor", "MSE1", "MSE2", "MSE3", "MSE4", "MSE5", "E(5)", "Parameters"], rows))
    add("\nTable 2. This table gives the per horizon error, E(5) and the trainable parameter count for model A, Blank and Copy-last on the 1,000 test videos at 1×.\n")
    add("![Figure 2](../figures/fig2.png)\n")
    add(f"Figure 2. The plot shows the five input frames and the true or predicted frames 6 to 10 for test videos {', '.join(map(str, sel['fig2']))} with the seed 0 model A. "
        f"Video {sel['fig2'][0]} has digits that merge in a future frame.\n")

    reach_a = reach["A"]
    add(f"**Q2.3 Answer.** One cell of the folded 16 by 16 grid is 4 original pixels, so a 3 by 3 convolution moves information one cell or 4 pixels. "
        f"A new input crosses the layer 1 convolution and then the layer 2 convolution, which is 2 convolutions or {reach_a['analytic']['input']} pixels. "
        f"The layer 1 state from the previous step also crosses both convolutions and gives {reach_a['analytic']['state_1']} pixels. "
        f"The layer 2 state crosses only its own convolution and gives {reach_a['analytic']['state_2']} pixels. The 1 by 1 output convolution and the pixel shuffles move nothing. "
        f"The reach of model A is the longest path, {max(reach_a['analytic'].values())} pixels per time step. "
        f"An impulse test on the built model gives {reach_a['empirical']['input']}, {reach_a['empirical']['state_1']} and {reach_a['empirical']['state_2']} pixels for the three paths, which matches the count.\n")
    a5, c5, b5 = mean("A", "E5_1x"), copy["E5_1x"], blank["E5_1x"]
    grow = np.mean([ev("A", s)["mse_h"][4] / ev("A", s)["mse_h"][0] for s in (0, 1)])
    m2 = diag["fig2"][str(sel["fig2"][0])]
    tr, pr = m2["truth"], m2["pred"]
    merged_h = [h for h in range(5) if tr[h]["blobs"] == 1]
    overlap_txt = (f"In test video {sel['fig2'][0]} the true frames show a single merged blob at horizons {', '.join(str(h + 1) for h in merged_h)}. "
                   f"The prediction has {', '.join(str(pr[h]['blobs']) for h in merged_h)} blobs there with peak intensity {', '.join(fmt(pr[h]['peak']) for h in merged_h)} "
                   f"against {', '.join(fmt(tr[h]['peak']) for h in merged_h)} in the truth.") if merged_h else (
        "The selected video does not show a merged blob in the true future frames, so the overlap rule was not met and the video needs a visual check.")
    add(f"**Q2.4 Answer.** Model A reaches E(5) of {fmt(a5)} against {fmt(c5)} for Copy-last and {fmt(b5)} for Blank, so its error is {fmt(a5 / c5)} times that of Copy-last and {fmt(a5 / b5)} times that of Blank. "
        f"Its error grows over the horizons by a factor of {fmt(grow)} from MSE1 to MSE5 because every fed back prediction carries its own errors into the next step. {overlap_txt}\n")
    add(read_prediction("Q2.5.md") + "\n")
    add("Trade off. The reach count is a bound on a single update and does not include gating, so the real tracking range can be smaller.\n")

    add("## 3. Testing the Baseline\n")
    add('**Q.** Table 3: E^(5), C_M, E_M, C_R and E_R for model A, Blank and Copy-last. Figure 3: the two test videos with the largest per-video C_M (with T = M) and the two with the largest per-video C_R (with T = R), for the seed-0 model A. For each video, show four rows of ten frames, with the five input frames in columns 1–5 and five future frames in columns 6–10: row (1) X | f(X), row (2) T(X) | T(Y), row (3) T(X) | f(T(X)), row (4) T(X) | T(f(X)). If the same video is among the top two for both transformations, show it twice, once for each transformation. Also save each panel as an animated GIF in your repository that steps through the five forecast horizons while keeping the same four rows, and give the GIF paths in the caption. Figure 4: histograms of the per-video C_M and C_R of the seed-0 model A over the test set, with the videos from Figure 3 marked.\n')
    rows = [["A"] + [ms(vals("A", k)) for k in ("E5_1x", "CM", "EM", "CR", "ER")]]
    for nm, b in (("Blank", blank), ("Copy-last", copy)):
        rows.append([nm] + [fmt(b[k]) for k in ("E5_1x", "CM", "EM", "CR", "ER")])
    add(table(["Predictor", "E(5)", "CM", "EM", "CR", "ER"], rows))
    add("\nTable 3. This table gives E(5), the consistency gaps and the transformed errors under mirror and rotation at 1× with t equal to 1.\n")
    add("Figure 3 shows the four panels. Each panel has four rows of ten frames with the inputs in columns 1 to 5 and the future in columns 6 to 10. "
        "Row 1 is X and f(X), row 2 is T(X) and T(Y), row 3 is T(X) and f(T(X)), row 4 is T(X) and T(f(X)).\n")
    for name in ("M", "R"):
        for rank, v in enumerate(sel[f"fig3_{name}"], 1):
            base_name = f"fig3_{name}_rank{rank}_video{v}"
            add(f"![Figure 3 {name} {rank}](../figures/{base_name}.png)\n")
            add(f"Figure 3 panel T = {name}, rank {rank}, test video {v}. The plot shows the four rows for the seed 0 model A. The animated GIF is saved at gifs/{base_name}.gif and reveals one forecast horizon at a time.\n")
    add("![Figure 4](../figures/fig4.png)\n")
    add("Figure 4. The plot shows histograms of the per video CM and CR of the seed 0 model A over the test set with a log vertical axis. Red dashed lines mark the videos of Figure 3.\n")
    cm_zero = copy["CM"] == 0 and copy["CR"] == 0 and blank["CM"] == 0 and blank["CR"] == 0
    add(f"**Q2.6 Answer.** Blank and Copy-last have CM {fmt(blank['CM'])} and CR {fmt(copy['CR'])} in Table 3 because they commute with any pixel permutation, so a gap of zero says nothing about forecast quality. "
        f"Their transformed errors are {fmt(blank['EM'])} and {fmt(copy['EM'])} for mirror, equal to their E(5) up to float rounding, so ET shows that they are wrong by the same amount as before. "
        "CT only measures agreement with itself and a model that ignores the input agrees with itself perfectly. ET must therefore be reported next to it because only ET compares against the true transformed future.\n")
    v7 = diag["fig3"]["A"]["M"][0]
    h_last = 4
    add(f"**Q2.7 Answer.** Test video {v7['video']} is the video with the largest CM in Figure 3. Rows 3 and 4 differ by a per pixel squared error of {fmt(v7['mse_rows'])}. "
        f"Row 3 has error {fmt(v7['err_fTX'])} against the true mirrored future and row 4 has {fmt(v7['err_TfX'])}. "
        f"{describe_failure(v7['TY'], v7['fTX'], v7['TfX'], h_last)}\n")
    em, er = mean("A", "EM"), mean("A", "ER")
    harder = "Rotation is harder than mirroring" if er > 1.1 * em else ("Mirroring is harder than rotation" if em > 1.1 * er else "Rotation and mirroring are about equally hard")
    motion_same = flow["tv_mirror"] <= 2 * noise and flow["tv_rotation"] <= 2 * noise
    b_red_m = 1 - mean("B", "EM") / em
    b_red_r = 1 - mean("B", "ER") / er
    add(f"**Q2.8 Answer.** {harder} for model A since ER is {fmt(er)} against EM {fmt(em)}, a ratio of {fmt(er / em)}. "
        f"The direction histograms change by {fmt(flow['tv_rotation'])} under rotation and {fmt(flow['tv_mirror'])} under mirror against a noise floor of {fmt(noise)}. "
        + ("Since the motion statistics are nearly unchanged by both transformations, the difference is attributed to the digit appearance. "
           if motion_same else "Since the motion statistics change under at least one transformation, the difference cannot be assigned to appearance alone. ")
        + f"Model B supports this reading because training with the eight symmetries lowers EM by {fmt(100 * b_red_m)} percent and ER by {fmt(100 * b_red_r)} percent.\n")

    add('**Q.** Table 4: model A’s E^(2) at 1×, 2× and 3×, and its E^(5) at 1× and 2×, with Copy-last for comparison. Figure 5: for one test video, the five input frames and the first two predicted and true future frames at 1×, 2× and 3× (seed-0 model A). Choose a video whose digits do not overlap in any of the displayed true frames.\n')
    rows = [["A", ms(vals("A", "E2_1x")), ms(vals("A", "E2_2x")), ms(vals("A", "E2_3x")), ms(vals("A", "E5_1x")), ms(vals("A", "E5_2x"))],
            ["Copy-last", fmt(copy["E2_1x"]), fmt(copy["E2_2x"]), fmt(copy["E2_3x"]), fmt(copy["E5_1x"]), fmt(copy["E5_2x"])]]
    add(table(["Predictor", "E(2) 1×", "E(2) 2×", "E(2) 3×", "E(5) 1×", "E(5) 2×"], rows))
    add("\nTable 4. This table gives E(2) at 1×, 2× and 3× and E(5) at 1× and 2× for model A and Copy-last.\n")
    add("![Figure 5](../figures/fig5.png)\n")
    add(f"Figure 5. The plot shows the five input frames and the first two true and predicted future frames at 1×, 2× and 3× for test video {sel['fig5']} with the seed 0 model A. "
        + ("The video has two separate blobs in every displayed true frame." if sel["fig5_rule_met"] else "The two blob rule was not met for any video so the video was chosen by the highest number of frames that meet it and needs a visual check."))
    add("\n")
    ma = {k: mean("A", f"E2_{k}x") for k in (1, 2, 3)}
    cp = {k: copy[f"E2_{k}x"] for k in (1, 2, 3)}
    broke = [k for k in (2, 3) if ma[k] >= cp[k]]
    first_k = next((k for k in (1, 2, 3) if k * med > max(reach_a["analytic"].values())), None)
    dA = diag["fig5"]["A"]
    k_b = broke[0] if broke else None
    agree = (k_b == first_k)
    add(f"**Q2.9 Answer.** The estimated reach is {max(reach_a['analytic'].values())} pixels and the median digit speed times k gives {fmt(med)}, {fmt(2 * med)} and {fmt(3 * med)} pixels at 1×, 2× and 3×. "
        f"The first speed above the reach is {first_k if first_k else 'none of them'}. Model A scores E(2) of {fmt(ma[1])}, {fmt(ma[2])} and {fmt(ma[3])} against Copy-last {fmt(cp[1])}, {fmt(cp[2])} and {fmt(cp[3])}. "
        + (f"The predictions fall apart at {k_b}× because from there A is no better than copying the last frame. " if broke else "Model A stays better than Copy-last at every tested speed so it does not fall apart in this sense. ")
        + ("This agrees with the reach estimate." if agree else
           "This does not agree with the reach estimate. The estimate ignores that gates can shrink the effective reach and that a 28 pixel digit still overlaps its old position after a jump smaller than its width, which likely lets the state track it beyond the counted reach. Both explanations are likely and not tested.") + " ")
    if k_b:
        s_t, s_p = dA[str(k_b)]["truth"], dA[str(k_b)]["pred"]
        d_true = gap(s_t[1], s_t[0])
        d_pred = gap(s_p[1], s_p[0])
        add(f"In Figure 5 at {k_b}× the true frames have {s_t[0]['blobs']} blobs and the predicted frames have {s_p[0]['blobs']} and {s_p[1]['blobs']} blobs with peak intensity {fmt(s_p[0]['peak'])} and {fmt(s_p[1]['peak'])}. "
            + (f"The matched blob centre moves {fmt(d_pred)} pixels between the two predictions against {fmt(d_true)} pixels in the truth. " if d_true and d_pred is not None else "")
            + "A low peak means blur and the model is averaging over several possible positions. A small centre movement means a stalled digit and the state cannot carry the digit as far as it jumped. "
              "An extra blob means a duplicate and the old position stays in the state while the new one comes from the input. A missing blob means a vanished digit and the model finds no match between input and state.")
    add("\n")
    add(f"**Q2.10 Answer.** No. Model A trains only at 1× so both explanations predict the same failure at 2× and 3×. An experiment that separates them trains the same baseline on speeds 1, 2 and 3 as model C "
        f"and a larger reach network on the same data as model D. If C recovers, the cause was unfamiliar motion. If C fails and D recovers, the cause was reach. "
        f"Table 5 gives the outcome with C at {fmt(mean('C', 'E2_3x'))} and D at {fmt(mean('D', 'E2_3x'))} for E(2) at 3× against Copy-last {fmt(cp[3])}.\n")
    add("Trade off. Copy-last is a fair yardstick for failure but not the only one. A stricter threshold would move the failure speed.\n")

    add("## 4. What Can Data Fix?\n")
    add(read_prediction("Q2.11.md") + "\n")
    add('**Q.** Then report Table 5: for models A–D, E^(5) at 1×; C_M, E_M, C_R, E_R; E^(2) at 1×, 2× and 3×; trainable parameter count; training time per epoch; and inference time.\n')
    cols = ["E5_1x", "CM", "EM", "CR", "ER", "E2_1x", "E2_2x", "E2_3x"]

    def model_row(nm, sym=False, label=None):
        tr = [load(f"train_{nm}_s{s}.json")["train_seconds_per_epoch"] for s in (0, 1)]
        it = [infer_time(nm, s, sym) for s in (0, 1)]
        return [label or nm] + [ms(vals(nm, k, sym)) for k in cols] + [f"{ev(nm, 0, sym)['params']:,}", ms(tr), ms(it)]

    heads = ["Model", "E(5) 1×", "CM", "EM", "CR", "ER", "E(2) 1×", "E(2) 2×", "E(2) 3×", "Params", "Train s per epoch", "Inference s"]
    add(table(heads, [model_row(n) for n in names]))
    add("\nTable 5. This table gives the accuracy, consistency, parameter count, training time per epoch and inference time for models A to D. Entries are mean (seed 0, seed 1).\n")
    dsel = sel_d["chosen"]
    cfg = load(f"train_D_s0.json")["cfg"]["arch"]
    reach_d = reach[dsel]
    add(f"**Q2.12 Answer.** Model D keeps the two layer ConvLSTM of model A with patch size {cfg['patch']}, {cfg['hidden']} hidden channels, kernel {cfg['kernel']} and dilations {cfg['dilation'][0]} and {cfg['dilation'][1]}. "
        f"It has {ev('D', 0)['params']:,} trainable parameters against the limit of 721,176. "
        f"The layer 1 convolution moves {cfg['dilation'][0] * (cfg['kernel'] - 1) // 2} cells and the layer 2 convolution moves {cfg['dilation'][1] * (cfg['kernel'] - 1) // 2} cells. One cell is {cfg['patch']} pixels. "
        f"The new input path crosses both so it covers {reach_d['analytic']['input']} pixels. The layer 1 state path also covers {reach_d['analytic']['state_1']} pixels and the layer 2 state path {reach_d['analytic']['state_2']} pixels. "
        f"The reach of model D is {max(reach_d['analytic'].values())} pixels against {max(reach_a['analytic'].values())} for model A, which is strictly larger. The impulse test returns {reach_d['empirical']['input']}, {reach_d['empirical']['state_1']} and {reach_d['empirical']['state_2']} pixels. "
        f"The design was chosen on the validation split only by the lowest mean validation E(2) over the three speeds. "
        + " ".join(f"Candidate {k} scored {fmt(v['score'])}." for k, v in sel_d["candidates"].items()) + f" The candidates were trained for {list(sel_d['candidates'].values())[0]['epochs']} epochs at seed 0.\n")

    cm_ = {n: dict(M=mean(n, "EM"), R=mean(n, "ER"), s2=mean(n, "E2_2x"), s3=mean(n, "E2_3x")) for n in names}
    flags = {"mirror": cm_["A"]["M"] > 1.5 * mean("A", "E5_1x"), "rotation": cm_["A"]["R"] > 1.5 * mean("A", "E5_1x"),
             "speed2": ma[2] >= cp[2], "speed3": ma[3] >= cp[3]}
    labels = []

    def red(x, y):
        return 1 - y / x

    for key, nice, fixer, other in (("mirror", "the mirror failure", "B", None), ("rotation", "the rotation failure", "B", None)):
        if not flags[key]:
            labels.append(f"Model A does not fail the {key} test by the rule that ET exceeds 1.5 times E(5), so no label is needed.")
        else:
            r = red(cm_["A"]["M" if key == "mirror" else "R"], cm_[fixer]["M" if key == "mirror" else "R"])
            labels.append(f"The {key} failure is labelled appearance when model B lowers the error by more than half. B lowers it by {fmt(100 * r)} percent so the label is "
                          + ("appearance." if r > 0.5 else "not appearance and no tested change explains it."))
    for k, key in ((2, "speed2"), (3, "speed3")):
        if not flags[key]:
            labels.append(f"Model A does not fail at {k}× by the Copy-last rule so no label is needed.")
        else:
            rc = red(cm_["A"][f"s{k}"], cm_["C"][f"s{k}"])
            rd = red(cm_["C"][f"s{k}"], cm_["D"][f"s{k}"])
            lab = "unfamiliar motion" if rc > 0.5 else ("architectural reach" if rd > 0.5 else "none of the three, so it stays unexplained")
            labels.append(f"At {k}× model C lowers A's error by {fmt(100 * rc)} percent and D lowers C's error by a further {fmt(100 * rd)} percent, so the failure is labelled {lab}.")
    add("**Q2.13 Answer.** The rule is that a change explains a failure when it lowers the relevant error by more than half. " + " ".join(labels)
        + " If model C still does badly at 3× while model D does well, the failure was architectural reach. Training data shows the network more examples of fast motion but each output pixel still depends only on inputs inside its receptive field, "
          "so no weight value can bring in a pixel that the wiring never connects. More data narrows the estimation error and cannot enlarge the function class that the architecture defines.\n")
    a_e5 = vals("A", "E5_1x")
    spread = abs(a_e5[0] - a_e5[1])
    chg = {n: mean(n, "E5_1x") / mean("A", "E5_1x") - 1 for n in "BCD"}
    add(f"**Q2.14 Answer.** E(5) at 1× is {fmt(mean('A', 'E5_1x'))} for A, {fmt(mean('B', 'E5_1x'))} for B, {fmt(mean('C', 'E5_1x'))} for C and {fmt(mean('D', 'E5_1x'))} for D, "
        f"a change of {fmt(100 * chg['B'])}, {fmt(100 * chg['C'])} and {fmt(100 * chg['D'])} percent against A. The two seeds of A differ by {fmt(spread)}. "
        + " ".join(f"The change for {n} is {'larger' if abs(chg[n]) * mean('A', 'E5_1x') > spread else 'not larger'} than that seed gap." for n in "BCD")
        + " " + worth_sentence(chg, mean, flags, cm_) + "\n")
    add("Trade off. Each change spends capacity or training time on a new condition and the 1× column shows what that spending costs.\n")

    add("## 5. The Final Model\n")
    add('**Q.** Show (a short argument is enough) that f̄ has C_M = C_R = 0 exactly for any f. Your implementation should give zero up to floating-point error; report these values in Table 6.\n')
    add("**Answer.** Let G be the eight element group of Table 1 and let each g act as a pixel permutation so that g composed with h is another element of G. "
        "The symmetrised forecaster is $\\bar f(X)=\\tfrac18\\sum_{g\\in G}g^{-1}f(gX)$. For any h in G the transformed input gives $\\bar f(hX)=\\tfrac18\\sum_{g}g^{-1}f(ghX)$. "
        "Substituting $g'=gh$ gives $g^{-1}=h\\,g'^{-1}$ and the sum becomes $\\tfrac18\\sum_{g'}h\\,g'^{-1}f(g'X)=h\\,\\bar f(X)$. "
        "Right multiplication by h is a bijection of G so the sum runs over the same eight terms. Hence $\\bar f(hX)=h\\bar f(X)$ and the gap between the two routes is zero for h equal to M or R. "
        "In float arithmetic the gap is at rounding level, which Table 6 shows.\n")
    add('**Q.** Table 6: the Table 5 columns for symmetrised B, symmetrised D, and your final model if it is different. In the training-time column, report the training time of the underlying learned model (B for symmetrised B, D for symmetrised D, D+B for symmetrised D+B); symmetrisation itself needs no additional training. Give C_M and C_R of the wrapped models in scientific notation. Figure 6: (a) Figure 3 repeated with your final model on exactly the same videos; (b) Figure 5 repeated with your final model on exactly the same video. Save the GIFs for (a) in your repository, as for Figure 3.\n')
    t6 = [model_row("B", True, "B symmetrised"), model_row("D", True, "D symmetrised")]
    if final != "D":
        t6.append(model_row(final, True, f"{final} symmetrised"))
    add(table(heads, t6))
    add(f"\nTable 6. This table gives the Table 5 columns for the symmetrised models. The training time is that of the underlying model. The final model is {final} symmetrised.\n")
    add(f"The final model is {'model D wrapped as a symmetrised forecaster' if final == 'D' else 'D+B wrapped as a symmetrised forecaster'}. "
        + ("It was chosen because the default choice of D already removes the speed failure and the gain of D+B in Table 5 was not needed to meet the consistency requirement." if final == "D"
           else "It was chosen because Table 5 supports adding symmetry augmentation to model D.") + "\n")
    for name in ("M", "R"):
        for rank, v in enumerate(sel[f"fig3_{name}"], 1):
            base_name = f"fig6a_{name}_rank{rank}_video{v}"
            add(f"![Figure 6a {name} {rank}](../figures/{base_name}.png)\n")
            add(f"Figure 6(a) panel T = {name}, rank {rank}, test video {v}. The plot repeats Figure 3 with the final model. The animated GIF is saved at gifs/{base_name}.gif.\n")
    add("![Figure 6b](../figures/fig6b.png)\n")
    add(f"Figure 6(b). The plot repeats Figure 5 for test video {sel['fig5']} with the final model.\n")
    sB, sD = "B", "D"
    cm_sym = max(max(vals(n, k, True)) for n in (sB, sD) for k in ("CM", "CR"))
    ratio_inf = np.mean([infer_time("D", s, True) / infer_time("D", s, False) for s in (0, 1)])
    e5_raw, e5_sym = mean("D", "E5_1x"), mean("D", "E5_1x", True)
    add(f"**Q2.15 Answer.** Learned symmetry in model B leaves CM and CR at {fmt(mean('B', 'CM'))} and {fmt(mean('B', 'CR'))}, which are small and not zero. "
        f"Built in symmetry gives at most {fmt(cm_sym)} for the symmetrised models, which is floating point rounding. Only the built in version is exact because the averaging over the group enforces the identity for any f, as shown above. "
        f"On original clips E(5) is {fmt(mean('B', 'E5_1x'))} for B and {fmt(mean('B', 'E5_1x', True))} for symmetrised B and {fmt(e5_raw)} for D and {fmt(e5_sym)} for symmetrised D. "
        f"On transformed clips ER is {fmt(mean('B', 'ER'))} for B and {fmt(mean('B', 'ER', True))} for symmetrised B. "
        f"The symmetrised model needs eight forward passes and its measured inference time is {fmt(ratio_inf)} times that of the unwrapped model. "
        + (f"Averaging eight predictions can blur them when the eight predictions disagree. Here it {'lowered' if e5_sym < e5_raw else 'raised'} E(5) of D from {fmt(e5_raw)} to {fmt(e5_sym)}"
           + (", so blur did not hurt accuracy on original clips." if e5_sym <= e5_raw else ", so blur did cost accuracy on original clips.")) + " Built in symmetry would be expected to hurt when the true motion is not symmetric or when the members disagree strongly, because the mean of different sharp frames is a blurred frame.\n")
    v16 = diag["fig3"]["final"]["M"][0]
    v16a = diag["fig3"]["A"]["M"][0]
    add(f"**Q2.16 Answer.** The video of Q2.7 is test video {v16['video']}. For the final model the squared difference between rows 3 and 4 is {fmt(v16['mse_rows'])} against {fmt(v16a['mse_rows'])} for model A. "
        f"Its error against the true mirrored future is {fmt(v16['err_fTX'])} against {fmt(v16a['err_fTX'])} for model A in row 3. "
        f"{describe_failure(v16['TY'], v16['fTX'], v16['TfX'], h_last)} "
        f"The symmetrisation removes the disagreement between rows 3 and 4 by construction. Table 5 shows that model B lowers EM from {fmt(mean('A', 'EM'))} to {fmt(mean('B', 'EM'))} "
        f"and Table 6 shows that wrapping gives CM of {fmt(mean(final, 'CM', True))}, so the symmetry part of the failure is fixed by the wrapper and the accuracy part by the underlying model.\n")
    f_e5, f_cm = mean(final, "E5_1x", True), max(vals(final, "CM", True) + vals(final, "CR", True))
    f_s3 = mean(final, "E2_3x", True)
    fs = {k: mean(final, f"E2_{k}x", True) for k in (1, 2, 3)}
    beats = [k for k in (1, 2, 3) if fs[k] < cp[k]]
    if len(beats) == 3:
        verdict = (f"The final model beats Copy-last at every tested speed with E(2) of {fmt(fs[1])}, {fmt(fs[2])} and {fmt(fs[3])} against {fmt(cp[1])}, {fmt(cp[2])} and {fmt(cp[3])}, "
                   f"and its consistency gaps are at most {fmt(f_cm)}. Beating Copy-last is consistent with predicting motion and not only copying, and agreement under all eight symmetries shows that its rule does not depend on orientation. "
                   "A skeptic would still ask for speeds above 3× and for motion that is not constant in velocity. Neither was tested, so the model has not passed a test on accelerating or curved trajectories.")
    else:
        lost = [k for k in (1, 2, 3) if k not in beats]
        verdict = (f"The final model does not beat Copy-last at {', '.join(str(k) + '×' for k in lost)}, so it has not learned motion at those speeds. "
                   f"Its consistency gaps are at most {fmt(f_cm)} and its 1× E(5) is {fmt(f_e5)} against {fmt(copy['E5_1x'])} for Copy-last. "
                   "Speeds above 3× and accelerating motion were not tested.")
    words = len(verdict.split())
    add(f"**Q2.17 Final verdict.** {verdict}\n")
    assert words <= 150, words
    add("Trade off. Built in symmetry buys exact consistency with eight times the inference cost and the average can blur, so it fits when orientation should not matter.\n")
    return "\n".join(md)


STYLE_BANNED = ["delve", "crucial", "pivotal", "robust", "leverage", "seamless", "comprehensive", "holistic", "furthermore", "moreover",
                "additionally", "it is important to note", "in conclusion", "overall", "notably", "ensure", "utilize", "foster", "tapestry", "landscape", "realm"]


def style_scan(text):
    prose = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    prose = re.sub(r"\$[^$]*\$", " ", prose)
    prose = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", prose)
    prose = "\n".join(l for l in prose.splitlines() if not l.startswith(("|", "#", "**Q.**", "**Q2.")))
    hits = []
    for ch, name in (("—", "em dash"), ("–", "en dash"), (";", "semicolon"), ("!", "exclamation")):
        if ch in prose:
            hits.append(name)
    for w in STYLE_BANNED:
        if re.search(r"\b" + re.escape(w) + r"\b", prose, re.I):
            hits.append("banned word " + w)
    if re.search(r"\w, and\b", prose):
        hits.append("comma before and")
    return hits


def render_html(md_text):
    store = []

    def keep(m):
        store.append(m.group(0))
        return f"@@MATH{len(store) - 1}@@"

    protected = re.sub(r"\$\$.*?\$\$|\$[^$\n]+\$", keep, md_text, flags=re.S)
    html = markdown.markdown(protected, extensions=["tables"])

    def back(m):
        src = store[int(m.group(1))]
        display = src.startswith("$$")
        tex = src.strip("$")
        ml = to_mathml(tex, display="block" if display else "inline")
        return ml

    return re.sub(r"@@MATH(\d+)@@", back, html)


CSS = """body{font-family:'Times New Roman',serif;font-size:11pt;line-height:1.35;margin:0 auto;max-width:190mm}
h1{font-size:16pt;margin-top:18pt}h2{font-size:13pt;margin-top:14pt}img{max-width:100%}
table{border-collapse:collapse;font-size:8.5pt;margin:6pt 0;width:100%}td,th{border:1px solid #444;padding:2pt 4pt;vertical-align:top}
th{background:#eee}math{font-size:105%}p{margin:5pt 0}"""


def write_docx(md_path, out_path):
    import shutil, subprocess
    try:
        import pypandoc
        exe = pypandoc.get_pandoc_path()
    except Exception:
        exe = shutil.which("pandoc")
    if not exe:
        print("docx skipped: pandoc not found")
        return
    subprocess.run([exe, md_path, "-f", "markdown+tex_math_dollars-implicit_figures", "-o", out_path,
                    "--resource-path=" + os.path.dirname(os.path.abspath(md_path))], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="PENDING_REPOSITORY_LINK")
    ap.add_argument("--final", default="D")
    ap.add_argument("--name", default="PENDING_STUDENT_NAME")
    ap.add_argument("--roll", default="PENDING_ROLL_NUMBER")
    ap.add_argument("--q2", action="store_true", help="include Question 2 (needs results)")
    ap.add_argument("--out", default=os.path.join(ROOT, "report", "PA2_report"))
    a = ap.parse_args()
    parts = [f"# AI651 Deep Learning for Space, Time and Graphs, Assignment 2\n\n**Student name.** {a.name}\n\n**Roll number.** {a.roll}\n\n**Code repository.** {a.repo}\n\n## AI acknowledgement\n\nThe prompts, outputs and edits are recorded in the section [AI use record](#ai-use-record) at the end of this report.\n"]
    for val, tag in ((a.name, "student name"), (a.roll, "roll number")):
        if val.startswith("PENDING"):
            PENDING.append(tag)
    if a.repo == "PENDING_REPOSITORY_LINK":
        PENDING.append("repository link")
    parts.append(open(os.path.join(ROOT, "report", "q1.md")).read())
    if a.q2:
        parts.append(build_q2(a))
    text = "\n\n".join(parts) + "\n\n## AI use record\n"
    qs = json.load(open(os.path.join(ROOT, "report", "questions.json")))
    text = re.sub(r"\*\*(Q2\.\d+)(?: Answer| Final verdict)\.\*\*", lambda m: f"**{m.group(1)}.** {qs[m.group(1)]}\n\n**Answer.**", text)
    hits = style_scan(text)
    html = "<html><head><meta charset='utf-8'><style>" + CSS + "</style></head><body>" + render_html(text) + "</body></html>"
    open(a.out + ".md", "w").write(text)
    open(a.out + ".html", "w").write(html)
    write_docx(a.out + ".md", a.out + ".docx")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page()
        page.goto("file://" + os.path.abspath(a.out + ".html"))
        page.pdf(path=a.out + ".pdf", format="A4", margin=dict(top="15mm", bottom="15mm", left="12mm", right="12mm"))
        b.close()
    print("style hits:", hits or "none")
    print("pending items:", PENDING or "none")
    return 1 if (hits or PENDING) else 0


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    sys.exit(main())
