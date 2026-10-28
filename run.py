import argparse
import json
import os
import numpy as np
import torch

from pa2.checks import check_group, table1
from pa2.configs import D_CANDIDATES, MAX_PARAMS, get_config
from pa2.data import load_moving_mnist, split_videos, to_float
from pa2.diagnostics import sequence_stats
from pa2.figures import hist_figure, panel_png_and_gif, grid_figure
from pa2.flow import figure1, flow_statistics, pooled_flow
from pa2.metrics import Symmetrised, blank, copy_last, evaluate_predictor, predict, time_inference
from pa2.models import ConvLSTMForecaster, count_parameters
from pa2.reach import analytic_reach, empirical_reach
from pa2.selection import FIG5_FRAMES, pick_figure2, pick_figure5
from pa2.train import hardware_string, load_model, train_model
from pa2.transforms import mirror, rotate
from pa2.windows import eval_windows

RES, FIG, GIF, CKPT = "results", "figures", "gifs", "checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dump(obj, name):
    os.makedirs(RES, exist_ok=True)
    json.dump(obj, open(os.path.join(RES, name), "w"), indent=1, default=str)


def load_json(name):
    return json.load(open(os.path.join(RES, name)))


def get_data():
    train, val, test = split_videos(load_moving_mnist("data"))
    assert (len(train), len(val), len(test)) == (8000, 1000, 1000)
    return train, val, test


def stage_table1(_):
    check_group()
    dump([dict(check=r[0], expected=r[1], yours=r[2], torch_equal=r[3]) for r in table1()], "table1.json")


def stage_flow(_):
    train, _, _ = get_data()
    dx, dy, pairs, per_dx, per_dy, flags = pooled_flow(train, 50)
    assert pairs == 950
    os.makedirs(FIG, exist_ok=True)
    figure1(dx, dy, os.path.join(FIG, "fig1.png"))
    dump(flow_statistics(dx, dy, pairs, per_dx, per_dy, flags), "fig1_stats.json")


def stage_reach(_):
    out = {"A": dict(params=count_parameters(ConvLSTMForecaster()), analytic=analytic_reach(4, 3, (1, 1)),
                     empirical=empirical_reach(ConvLSTMForecaster()))}
    for key, cfg in D_CANDIDATES.items():
        m = ConvLSTMForecaster(**cfg)
        out[key] = dict(cfg=cfg, params=count_parameters(m), analytic=analytic_reach(cfg["patch"], cfg["kernel"], cfg["dilation"]),
                        empirical=empirical_reach(m))
    dump(out, "reach.json")


def stage_train(args):
    train, val, _ = get_data()
    cfg = get_config(args.model, RES)
    record = train_model(args.model, cfg, args.seed, train.to(device), val, device, CKPT, epochs=args.epochs)
    dump(record, f"train_{args.model}_s{args.seed}.json")


def stage_select_d(args):
    train, val, _ = get_data()
    val_float = to_float(val)
    rows = {}
    for key, arch in D_CANDIDATES.items():
        probe = ConvLSTMForecaster(**arch)
        assert count_parameters(probe) <= MAX_PARAMS, key
        assert max(analytic_reach(arch["patch"], arch["kernel"], arch["dilation"]).values()) > 8, key
        cfg = dict(arch=arch, speed_aug=True, sym_aug=False)
        rec = train_model("cand_" + key, cfg, 0, train.to(device), val, device, CKPT, epochs=args.epochs)
        model = load_model(os.path.join(CKPT, f"cand_{key}_s0.pt"), device)[0]
        metrics, _ = evaluate_predictor(lambda x: model(x), val_float, device)
        rows[key] = dict(params=rec["params"], E2_1x=metrics["E2_1x"], E2_2x=metrics["E2_2x"], E2_3x=metrics["E2_3x"],
                         score=float(np.mean([metrics["E2_1x"], metrics["E2_2x"], metrics["E2_3x"]])), epochs=args.epochs)
    chosen = min(rows, key=lambda k: rows[k]["score"])
    dump(dict(candidates=rows, chosen=chosen, rule="lowest mean validation E(2) over 1x, 2x, 3x, seed 0, validation videos 8000-8999"), "d_selection.json")


def model_predictor(name, seed, sym):
    model, _ = load_model(os.path.join(CKPT, f"{name}_s{seed}.pt"), device)
    f = lambda x: model(x)
    return (Symmetrised(f) if sym else f), model


def stage_evaluate(args):
    _, _, test = get_data()
    test_float = to_float(test)
    out = {}
    for name, f in (("Blank", blank), ("Copy-last", copy_last)):
        metrics, _ = evaluate_predictor(f, test_float, device)
        out[name] = metrics
    dump(out, "eval_baselines.json")
    for name in args.models:
        for seed in (0, 1):
            for sym in (False, True):
                f, model = model_predictor(name, seed, sym)
                metrics, per_video = evaluate_predictor(f, test_float, device)
                runs = time_inference(f, test_float, device, repeats=5)
                tag = f"{name}{'sym' if sym else ''}_s{seed}"
                metrics.update(params=count_parameters(model), inference_seconds=runs, hardware=hardware_string(device))
                dump(metrics, f"eval_{tag}.json")
                np.savez(os.path.join(RES, f"pervideo_{tag}.npz"), **{k: v.numpy() for k, v in per_video.items()})


def stage_table2(_):
    """Baseline numbers at 1x only. No mirror, rotation or speed test is run."""
    _, _, test = get_data()
    test_float = to_float(test)
    out = {}
    for name, f in (("Blank", blank), ("Copy-last", copy_last)):
        out[name] = evaluate_predictor(f, test_float, device, full=False)[0]
    for seed in (0, 1):
        f, _ = model_predictor("A", seed, False)
        out[f"A_s{seed}"] = evaluate_predictor(f, test_float, device, full=False)[0]
    dump(out, "table2_preview.json")


def stage_select_videos(_):
    _, _, test = get_data()
    test_float = to_float(test)
    pv = np.load(os.path.join(RES, "pervideo_A_s0.npz"))
    top = lambda a: [int(i) for i in np.argsort(-a)[:2]]
    fig2, fig2_strict = pick_figure2(test_float)
    fig5, fig5_strict = pick_figure5(test_float)
    dump(dict(fig2=fig2, fig2_rule_met=fig2_strict, fig3_M=top(pv["CM"]), fig3_R=top(pv["CR"]), fig5=fig5, fig5_rule_met=fig5_strict), "selection.json")


def _np(t):
    return t.cpu().numpy()


def figure3_like(f, ids, prefix, tag):
    _, _, test = get_data()
    X_all, Y_all, _ = eval_windows(to_float(test), 1)
    labels = ["(1) X | f(X)", "(2) T(X) | T(Y)", "(3) T(X) | f(T(X))", "(4) T(X) | T(f(X))"]
    for name, T in (("M", mirror), ("R", rotate)):
        for rank, v in enumerate(ids[f"fig3_{name}"], 1):
            X, Y = X_all[v:v + 1].to(device), Y_all[v:v + 1].to(device)
            with torch.no_grad():
                P, PT = f(X), f(T(X))
            ins = [_np(X[0, :, 0]), _np(T(X)[0, :, 0]), _np(T(X)[0, :, 0]), _np(T(X)[0, :, 0])]
            outs = [_np(P[0, :, 0]), _np(T(Y)[0, :, 0]), _np(PT[0, :, 0]), _np(T(P)[0, :, 0])]
            base = f"{prefix}_{name}_rank{rank}_video{v}"
            panel_png_and_gif(ins, outs, labels, os.path.join(FIG, base + ".png"), os.path.join(GIF, base + ".gif"))


def figure5_like(f, v, path):
    _, _, test = get_data()
    test_float = to_float(test)
    rows, labels = [], []
    for k in (1, 2, 3):
        X, Y, _ = eval_windows(test_float[v:v + 1], k)
        with torch.no_grad():
            P = f(X.to(device)).cpu()
        rows.append([x for x in _np(X[0, :, 0])] + [y for y in _np(Y[0, :2, 0])])
        rows.append([x for x in _np(X[0, :, 0])] + [p for p in _np(P[0, :2, 0])])
        labels += [f"{k}x truth", f"{k}x model"]
    cols = [f"in {i}" for i in range(1, 6)] + ["h=1", "h=2"]
    grid_figure(rows, labels, cols, path)


def stage_figures(args):
    _, _, test = get_data()
    test_float = to_float(test)
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(GIF, exist_ok=True)
    ids = load_json("selection.json")
    f_a, _ = model_predictor("A", 0, False)
    X_all, Y_all, _ = eval_windows(test_float, 1)
    rows, labels = [], []
    for v in ids["fig2"]:
        X, Y = X_all[v:v + 1].to(device), Y_all[v:v + 1]
        with torch.no_grad():
            P = f_a(X).cpu()
        rows.append(list(_np(X[0, :, 0])) + list(_np(Y[0, :, 0])))
        rows.append(list(_np(X[0, :, 0])) + list(_np(P[0, :, 0])))
        labels += [f"v{v} truth", f"v{v} model A"]
    grid_figure(rows, labels, [f"in {i}" for i in range(1, 6)] + [f"h={h}" for h in range(1, 6)], os.path.join(FIG, "fig2.png"))
    figure3_like(f_a, ids, "fig3", "A")
    pv = np.load(os.path.join(RES, "pervideo_A_s0.npz"))
    hist_figure({k: torch.from_numpy(pv[k]) for k in ("CM", "CR")}, {"CM": ids["fig3_M"], "CR": ids["fig3_R"]}, os.path.join(FIG, "fig4.png"))
    figure5_like(f_a, ids["fig5"], os.path.join(FIG, "fig5.png"))
    if not os.path.exists(os.path.join(CKPT, f"{args.final}_s0.pt")):
        return
    f_final, _ = model_predictor(args.final, 0, True)
    figure3_like(f_final, ids, "fig6a", args.final)
    figure5_like(f_final, ids["fig5"], os.path.join(FIG, "fig6b.png"))


def stage_diagnostics(args):
    _, _, test = get_data()
    test_float = to_float(test)
    ids = load_json("selection.json")
    X_all, Y_all, _ = eval_windows(test_float, 1)
    out = {"fig2": {}, "fig3": {}, "fig5": {}}
    f_a, _ = model_predictor("A", 0, False)
    f_final, _ = model_predictor(args.final, 0, True)
    st = lambda t: sequence_stats(_np(t[0, :, 0]))
    for v in ids["fig2"]:
        X, Y = X_all[v:v + 1].to(device), Y_all[v:v + 1]
        with torch.no_grad():
            out["fig2"][v] = dict(truth=st(Y), pred=st(f_a(X).cpu()))
    for label, f in (("A", f_a), ("final", f_final)):
        out["fig3"][label], out["fig5"][label] = {}, {}
        for name, T in (("M", mirror), ("R", rotate)):
            rows = []
            for v in ids[f"fig3_{name}"]:
                X, Y = X_all[v:v + 1].to(device), Y_all[v:v + 1].to(device)
                with torch.no_grad():
                    P, PT = f(X), f(T(X))
                rows.append(dict(video=v, TY=st(T(Y).cpu()), fTX=st(PT.cpu()), TfX=st(T(P).cpu()),
                                 mse_rows=float(((PT - T(P)) ** 2).mean()), err_fTX=float(((PT - T(Y)) ** 2).mean()),
                                 err_TfX=float(((T(P) - T(Y)) ** 2).mean())))
            out["fig3"][label][name] = rows
        for k in (1, 2, 3):
            X, Y, _ = eval_windows(test_float[ids["fig5"]:ids["fig5"] + 1], k)
            with torch.no_grad():
                P = f(X.to(device)).cpu()
            out["fig5"][label][k] = dict(truth=st(Y[:, :2]), pred=st(P[:, :2]), input=st(X))
    dump(out, "diagnostics.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["table1", "flow", "reach", "train", "select_d", "evaluate", "select_videos", "figures", "diagnostics", "table2"])
    ap.add_argument("--model", default="A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--models", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--final", default="D")
    args = ap.parse_args()
    {"table1": stage_table1, "flow": stage_flow, "reach": stage_reach, "train": stage_train, "select_d": stage_select_d,
     "evaluate": stage_evaluate, "select_videos": stage_select_videos, "figures": stage_figures, "diagnostics": stage_diagnostics, "table2": stage_table2}[args.stage](args)
