import cv2
import numpy as np

FIG5_FRAMES = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 16, 19]


def blob_count(frame, min_area=15):
    """Number of bright connected blobs in a 64x64 float frame. One blob where two are expected means overlap."""
    n, _, stats, _ = cv2.connectedComponentsWithStats((frame > 0.5).astype(np.uint8))
    return int(sum(stats[i, cv2.CC_STAT_AREA] >= min_area for i in range(1, n)))


def video_blobs(video):
    return [blob_count(video[f, 0]) for f in range(20)]


def pick_figure2(test_float, count=3):
    """Test indices for Figure 2. The first video has two blobs in the inputs that merge in a future frame.
    The others keep two blobs in all ten frames. Falls back to the best scoring video and reports it."""
    merged, apart = None, []
    best_apart = []
    for i in range(len(test_float)):
        counts = video_blobs(test_float[i].numpy())
        if merged is None and all(c == 2 for c in counts[:5]) and any(c == 1 for c in counts[5:10]):
            merged = i
        score = sum(c == 2 for c in counts[:10])
        best_apart.append((-score, i))
        if len(apart) < count - 1 and score == 10:
            apart.append(i)
    best_apart.sort()
    strict = merged is not None and len(apart) == count - 1
    if merged is None:
        merged = min(range(len(test_float)), key=lambda i: sum(c == 2 for c in video_blobs(test_float[i].numpy())[:10]))
    while len(apart) < count - 1:
        apart.append(next(i for _, i in best_apart if i not in apart and i != merged))
    return [merged] + apart, strict


def pick_figure5(test_float):
    """First test video with two blobs in every displayed true frame, else the best scoring one."""
    best, best_score = 0, -1
    for i in range(len(test_float)):
        counts = video_blobs(test_float[i].numpy())
        score = sum(counts[f - 1] == 2 for f in FIG5_FRAMES)
        if score == len(FIG5_FRAMES):
            return i, True
        if score > best_score:
            best, best_score = i, score
    return best, False
