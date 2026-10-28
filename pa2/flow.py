import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pooled_flow(train_u8, n_videos=50):
    """Farneback flow on consecutive frame pairs. Returns dx, dy of kept pixels and the pair count."""
    dxs, dys, pair_comps = [], [], []
    pairs = 0
    for v in range(n_videos):
        frames = train_u8[v, :, 0].numpy()
        for s in range(19):
            prev, nxt = frames[s], frames[s + 1]
            flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            dx, dy = flow[..., 0], flow[..., 1]
            keep = (prev > 51) & (nxt > 51) & (np.hypot(dx, dy) > 0.5)
            dxs.append(dx[keep])
            dys.append(dy[keep])
            pairs += 1
            n_prev = cv2.connectedComponents((prev > 51).astype(np.uint8))[0] - 1
            n_next = cv2.connectedComponents((nxt > 51).astype(np.uint8))[0] - 1
            pair_comps.append(n_prev >= 2 and n_next >= 2)
    return np.concatenate(dxs), np.concatenate(dys), pairs, dxs, dys, pair_comps


def direction_degrees(dx, dy):
    return np.degrees(np.arctan2(-dy, dx)) % 360.0


def direction_hist(theta):
    counts, _ = np.histogram(theta, bins=np.arange(0, 361, 20))
    return counts / counts.sum()


def figure1(dx, dy, path):
    theta = direction_degrees(dx, dy)
    speed = np.hypot(dx, dy)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    frac = direction_hist(theta)
    ax[0].bar(np.arange(18) * 20 + 10, frac, width=20, edgecolor="k")
    ax[0].axhline(1 / 18, color="r", ls="--", label="uniform 1/18")
    ax[0].set(xlabel="flow direction (degrees)", ylabel="fraction of kept pixels", xlim=(0, 360))
    ax[0].legend()
    weights = np.full(len(speed), 1 / len(speed))
    ax[1].hist(speed, bins=40, weights=weights, edgecolor="k")
    ax[1].axvline(np.median(speed), color="r", label=f"median {np.median(speed):.3g}")
    ax[1].set(xlabel="flow speed (pixels per frame)", ylabel="fraction of kept pixels")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def flow_statistics(dx, dy, pairs, per_pair_dx, per_pair_dy, separate_flags):
    theta, speed = direction_degrees(dx, dy), np.hypot(dx, dy)
    base = direction_hist(theta)
    mirrored = direction_hist(direction_degrees(-dx, dy))
    rotated = direction_hist((theta + 90.0) % 360.0)
    tv = lambda p, q: 0.5 * float(np.abs(p - q).sum())
    flags = np.array(separate_flags)
    clean_dx = np.concatenate([a for a, f in zip(per_pair_dx, flags) if f])
    clean_dy = np.concatenate([a for a, f in zip(per_pair_dy, flags) if f])
    clean = direction_hist(direction_degrees(clean_dx, clean_dy))
    half = len(per_pair_dx) // 2
    first = np.concatenate(per_pair_dx[:half]), np.concatenate(per_pair_dy[:half])
    second = np.concatenate(per_pair_dx[half:]), np.concatenate(per_pair_dy[half:])
    return dict(
        tv_half_split=tv(direction_hist(direction_degrees(*first)), direction_hist(direction_degrees(*second))),
        pairs=pairs, kept_pixels=int(len(dx)), median_speed=float(np.median(speed)),
        speed_quartiles=[float(q) for q in np.percentile(speed, [25, 75])],
        direction_fraction=base.tolist(),
        tv_mirror=tv(base, mirrored), tv_rotation=tv(base, rotated),
        max_bin_over_uniform=float(base.max() * 18), min_bin_over_uniform=float(base.min() * 18),
        separate_digit_pairs=int(flags.sum()), kept_pixels_separate=int(len(clean_dx)),
        median_speed_separate=float(np.median(np.hypot(clean_dx, clean_dy))),
        tv_mirror_separate=tv(clean, direction_hist(direction_degrees(-clean_dx, clean_dy))),
        tv_rotation_separate=tv(clean, direction_hist((direction_degrees(clean_dx, clean_dy) + 90.0) % 360.0)),
        tv_all_vs_separate=tv(base, clean),
    )
