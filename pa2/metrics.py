import time
import torch
from .transforms import G_ELEMENTS, apply_g, invert_g, mirror, rotate
from .windows import eval_windows


def blank(x):
    return torch.zeros_like(x)


def copy_last(x):
    return x[:, -1:].expand(-1, 5, -1, -1, -1).clone()


class Symmetrised:
    """Mean over g of g^-1 f(g X). Each forward pass sees the same batch size as the base model."""

    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        total = torch.zeros_like(x)
        for g in G_ELEMENTS:
            total += invert_g(self.f(apply_g(x, g)), g)
        return total / len(G_ELEMENTS)


@torch.no_grad()
def predict(f, x, batch=32):
    return torch.cat([f(x[i:i + batch]) for i in range(0, len(x), batch)])


def per_video_mse(pred, target):
    """(N,5) mean squared error over pixels for each forecast horizon."""
    return ((pred - target) ** 2).mean(dim=(2, 3, 4))


def masked_mean(mse_h, mask, horizons):
    sel = mask[:, :horizons]
    assert bool(sel.all()), "a requested horizon has no target"
    return mse_h[:, :horizons].mean(dim=1)


def evaluate_predictor(f, test_videos, device, batch=32, full=True):
    """test_videos (1000,20,1,64,64) float on cpu. Returns scalar metrics and per video arrays."""
    out, per_video = {}, {}
    moves = {"M": mirror, "R": rotate}
    X, Y, mask = eval_windows(test_videos, 1)
    X, Y = X.to(device), Y.to(device)
    P = predict(f, X, batch)
    mse_h = per_video_mse(P, Y)
    out["mse_h"] = mse_h.mean(0).tolist()
    out["E5_1x"] = masked_mean(mse_h, mask, 5).mean().item()
    out["E2_1x"] = masked_mean(mse_h, mask, 2).mean().item()
    if not full:
        return out, per_video
    for name, T in moves.items():
        PT = predict(f, T(X), batch)
        c = per_video_mse(PT, T(P)).mean(1)
        e = per_video_mse(PT, T(Y)).mean(1)
        out["C" + name], out["E" + name] = c.mean().item(), e.mean().item()
        per_video["C" + name], per_video["E" + name] = c.cpu(), e.cpu()
    for k in (2, 3):
        Xk, Yk, mk = eval_windows(test_videos, k)
        Pk = predict(f, Xk.to(device), batch)
        mk_h = per_video_mse(Pk, Yk.to(device))
        valid = mk.to(device)
        out[f"E2_{k}x"] = masked_mean(mk_h, valid, 2).mean().item()
        if k == 2:
            out["E5_2x"] = masked_mean(mk_h, valid, 5).mean().item()
    return out, per_video


@torch.no_grad()
def time_inference(f, test_videos, device, batch=32, repeats=5):
    """Seconds to predict the 1000 original test windows. Clock starts once a batch is on the device."""
    X, _, _ = eval_windows(test_videos, 1)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    f(X[:batch].to(device))
    sync()
    runs = []
    for _ in range(repeats):
        total = 0.0
        for i in range(0, len(X), batch):
            xb = X[i:i + batch].to(device)
            sync()
            t0 = time.perf_counter()
            f(xb)
            sync()
            total += time.perf_counter() - t0
        runs.append(total)
    return runs
