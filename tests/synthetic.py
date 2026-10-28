import numpy as np
import torch


def _digit(rng):
    img = np.zeros((28, 28), dtype=np.float32)
    yy, xx = np.mgrid[:28, :28]
    for _ in range(3):
        cy, cx = rng.uniform(6, 22, 2)
        img = np.maximum(img, np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rng.uniform(2, 5) ** 2))))
    return (img > 0.35) * img


def make_videos(n, seed=0):
    rng = np.random.RandomState(seed)
    out = np.zeros((n, 20, 1, 64, 64), dtype=np.uint8)
    for v in range(n):
        for _ in range(2):
            d = _digit(rng)
            pos = rng.uniform(0, 36, 2)
            vel = rng.uniform(-5, 5, 2)
            for f in range(20):
                for ax in range(2):
                    pos[ax] += vel[ax]
                    if pos[ax] < 0 or pos[ax] > 36:
                        vel[ax] = -vel[ax]
                        pos[ax] = np.clip(pos[ax], 0, 36)
                r, c = int(pos[0]), int(pos[1])
                canvas = out[v, f, 0]
                canvas[r:r + 28, c:c + 28] = np.maximum(canvas[r:r + 28, c:c + 28], (d * 255).astype(np.uint8))
    return torch.from_numpy(out)
