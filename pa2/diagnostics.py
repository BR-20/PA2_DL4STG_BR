import cv2
import numpy as np


def frame_stats(frame, min_area=15):
    """Mass, peak, blob count and centroids (row, col) of the two largest blobs of a 64x64 frame."""
    n, _, stats, cents = cv2.connectedComponentsWithStats((frame > 0.5).astype(np.uint8))
    blobs = sorted([(int(stats[i, cv2.CC_STAT_AREA]), (float(cents[i][1]), float(cents[i][0]))) for i in range(1, n)
                    if stats[i, cv2.CC_STAT_AREA] >= min_area], reverse=True)
    return dict(mass=float(frame.sum()), peak=float(frame.max()), blobs=len(blobs), centroids=[c for _, c in blobs[:2]])


def sequence_stats(frames):
    return [frame_stats(f) for f in frames]


def centroid_gap(stats_a, stats_b):
    """Mean distance in pixels between matched centroids of two frames. None when a frame has no blob."""
    ca, cb = stats_a["centroids"], stats_b["centroids"]
    if not ca or not cb:
        return None
    return float(np.mean([min(np.hypot(a[0] - b[0], a[1] - b[1]) for b in cb) for a in ca]))
