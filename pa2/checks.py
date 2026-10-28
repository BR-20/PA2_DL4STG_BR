import torch

from .transforms import G_ELEMENTS, apply_g, invert_g, mirror, rotate
from .windows import gather_windows


def bright_pixel(video):
    idx = (video == video.max()).nonzero()[0]
    return tuple(int(i) for i in idx[[0, 2, 3]])


def table1():
    torch.manual_seed(0)
    x = torch.rand(20, 1, 64, 64)
    p = torch.zeros(20, 1, 64, 64)
    p[7, 0, 10, 20] = 1
    diff = lambda a, b: (a - b).abs().max().item()
    rows = [
        ("(a) max|M(M(x)) - x|", "0", diff(mirror(mirror(x)), x), torch.equal(mirror(mirror(x)), x)),
        ("(b) max|R^4(x) - x|", "0", diff(rotate(x, 4), x), torch.equal(rotate(x, 4), x)),
        ("(c) max|M R M(x) - R^-1(x)|", "0", diff(mirror(rotate(mirror(x), 1)), rotate(x, -1)),
         torch.equal(mirror(rotate(mirror(x), 1)), rotate(x, -1))),
        ("(d) bright pixel of M(p)", "(7, 10, 43)", bright_pixel(mirror(p)), None),
        ("(e) bright pixel of R(p)", "(7, 43, 10)", bright_pixel(rotate(p)), None),
    ]
    frames = torch.arange(1, 21).float().view(1, 20, 1, 1, 1).expand(1, 20, 1, 64, 64)
    for label, k, expected in (("(f) 1x", 1, "1-5 / 6-10"), ("(g) 2x", 2, "1, 3, 5, 7, 9 / 11, 13, 15, 17, 19"),
                               ("(h) 3x", 3, "1, 4, 7, 10, 13 / 16, 19")):
        X, Y, mask = gather_windows(frames, torch.tensor([1]), torch.tensor([k]))
        inputs = [int(v) for v in X[0, :, 0, 0, 0]]
        targets = [int(v) for v, m in zip(Y[0, :, 0, 0, 0], mask[0]) if m]
        rows.append((f"{label} input / target frames", expected, (inputs, targets), None))
    return rows


def check_group():
    x = torch.rand(3, 5, 1, 64, 64)
    images = []
    for g in G_ELEMENTS:
        assert torch.equal(invert_g(apply_g(x, g), g), x), g
        assert torch.equal(apply_g(apply_g(x, g), g), x) or g[1] == 0, g
        images.append(apply_g(x, g))
    for i in range(8):
        for j in range(i + 1, 8):
            assert not torch.equal(images[i], images[j]), (G_ELEMENTS[i], G_ELEMENTS[j])
    return True
