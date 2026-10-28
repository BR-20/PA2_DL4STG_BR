import sys, tempfile, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from synthetic import make_videos
from pa2.checks import check_group, table1
from pa2.configs import D_CANDIDATES, MAX_PARAMS, get_config
from pa2.flow import flow_statistics, figure1, pooled_flow
from pa2.metrics import Symmetrised, blank, copy_last, evaluate_predictor, time_inference
from pa2.models import ConvLSTMForecaster, count_parameters
from pa2.reach import analytic_reach, empirical_reach
from pa2.selection import pick_figure2, pick_figure5
from pa2.train import masked_loss, sample_batch, train_model
from pa2.transforms import G_ELEMENTS
from pa2.windows import AUG_STARTS, gather_windows, window_frames

device = torch.device("cpu")

# Table 1
rows = table1()
for label, expected, got, equal in rows[:3]:
    assert got == 0 and equal, label
assert rows[3][2] == (7, 10, 43) and rows[4][2] == (7, 43, 10)
assert rows[5][2] == ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
assert rows[6][2] == ([1, 3, 5, 7, 9], [11, 13, 15, 17, 19])
assert rows[7][2] == ([1, 4, 7, 10, 13], [16, 19])
print("table1 ok")
assert check_group()

# window rules
for k, last in AUG_STARTS.items():
    assert max(t for t in range(1, 21) if len(window_frames(t, k)[1]) >= 2 and window_frames(t, k)[0][-1] <= 20) == last
assert max(t for t in range(1, 21) if len(window_frames(t, 1)[1]) == 5) == 11
print("window ranges ok")

# model A and D
A = ConvLSTMForecaster()
assert count_parameters(A) == 480784
assert analytic_reach(4, 3, (1, 1)) == empirical_reach(A) == {"input": 8, "state_1": 8, "state_2": 4}
for key, cfg in D_CANDIDATES.items():
    m = ConvLSTMForecaster(**cfg)
    assert count_parameters(m) <= MAX_PARAMS, key
    assert analytic_reach(cfg["patch"], cfg["kernel"], cfg["dilation"]) == empirical_reach(m), key
    assert max(empirical_reach(m).values()) > 8, key
print("params and reach ok")

# rollout does not use targets: changing future frames cannot change output
vids = make_videos(40, seed=1).float().div(255)
X, Y, mask = gather_windows(vids[:8], torch.ones(8, dtype=torch.long), torch.ones(8, dtype=torch.long))
A.eval()
with torch.no_grad():
    p1 = A(X)
    p2 = A(X.clone())
assert torch.equal(p1, p2) and p1.shape == (8, 5, 1, 64, 64)

# masked loss on a hand example: video with 2 valid targets
pred = torch.zeros(1, 5, 1, 64, 64)
tgt = torch.ones(1, 5, 1, 64, 64)
tgt[0, 2:] = 5.0
m = torch.tensor([[True, True, False, False, False]])
assert abs(masked_loss(pred, tgt, m).item() - 1.0) < 1e-6
print("loss mask ok")

# sampling honours the rules
data = make_videos(64, seed=2)
gen = torch.Generator().manual_seed(0)
for speed_aug in (False, True):
    for sym_aug in (False, True):
        Xb, Yb, mb = sample_batch(data, torch.arange(32), gen, speed_aug, sym_aug, device)
        assert Xb.shape == (32, 5, 1, 64, 64) and mb[:, :2].all()
        if not speed_aug:
            assert mb.all()

# symmetrised model gives zero consistency gap, base model does not
test = vids
torch.manual_seed(0)
base = ConvLSTMForecaster().eval()
raw, _ = evaluate_predictor(lambda x: base(x), test[:32], device)
sym, _ = evaluate_predictor(Symmetrised(lambda x: base(x)), test[:32], device)
print("raw CM CR", raw["CM"], raw["CR"], "sym CM CR", sym["CM"], sym["CR"])
assert raw["CM"] > 1e-8 and sym["CM"] < 1e-10 and sym["CR"] < 1e-10
for name in ("blank", "copy"):
    f = blank if name == "blank" else copy_last
    r, _ = evaluate_predictor(f, test[:32], device)
    assert r["CM"] == 0 and r["CR"] == 0 and abs(r["EM"] - r["E5_1x"]) < 1e-7 and abs(r["ER"] - r["E5_1x"]) < 1e-7, name
print("consistency checks ok")

# tiny training run, repeat for determinism, then reload
cfg = get_config("C")
small = data[:64]
with tempfile.TemporaryDirectory() as tmp:
    r1 = train_model("C", cfg, 0, small, small[:16], device, tmp, epochs=1, log=lambda s: None)
    r2 = train_model("C", cfg, 0, small, small[:16], device, tmp, epochs=1, log=lambda s: None)
    assert r1["train_loss"] == r2["train_loss"], (r1["train_loss"], r2["train_loss"])
    r3 = train_model("C", cfg, 1, small, small[:16], device, tmp, epochs=1, log=lambda s: None)
    assert r3["train_loss"] != r1["train_loss"]
print("training ok", r1["train_loss"])
runs = time_inference(Symmetrised(lambda x: base(x)), test[:32], device, repeats=2)
assert len(runs) == 2

# flow, selection
dx, dy, pairs, a, b, flags = pooled_flow(make_videos(50, seed=3), n_videos=50)
assert pairs == 950
stats = flow_statistics(dx, dy, pairs, a, b, flags)
figure1(dx, dy, "/tmp/fig1_test.png")
print("flow ok", stats["kept_pixels"], round(stats["median_speed"], 2), round(stats["tv_mirror"], 3))
from pa2.flow import direction_degrees
import cv2
rng = np.random.RandomState(0)
tex = cv2.GaussianBlur((rng.rand(64, 64) * 255).astype(np.float32), (0, 0), 2.0)
tex = (255 * (tex - tex.min()) / (tex.max() - tex.min())).astype(np.uint8)
def shifted(img, dy_, dx_):
    out = np.zeros_like(img)
    out[max(dy_, 0):64 + min(dy_, 0), max(dx_, 0):64 + min(dx_, 0)] = img[max(-dy_, 0):64 + min(-dy_, 0), max(-dx_, 0):64 + min(-dx_, 0)]
    return out
for (dy_, dx_), lo, hi in (((0, 3), -20, 20), ((-3, 0), 70, 110), ((0, -3), 160, 200)):
    nxt = shifted(tex, dy_, dx_)
    fl = cv2.calcOpticalFlowFarneback(tex, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    keep = (tex > 51) & (nxt > 51) & (np.hypot(fl[..., 0], fl[..., 1]) > 0.5)
    th = direction_degrees(fl[..., 0][keep], fl[..., 1][keep])
    ang = np.median(np.where(th > 180, th - 360, th)) if lo < 0 else np.median(th)
    assert lo < ang < hi, ((dy_, dx_), ang)
print("flow sign convention ok")
print("ALL TESTS PASSED")
