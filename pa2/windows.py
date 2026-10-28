import torch

# training starts per speed when a window needs 5 inputs and at least 2 targets
AUG_STARTS = {1: 14, 2: 8, 3: 2}


def window_frames(t, k):
    inputs = [t + k * i for i in range(5)]
    targets = [t + k * (5 + j) for j in range(5)]
    return inputs, [f for f in targets if f <= 20]


def gather_windows(videos, starts, speeds):
    """videos (B,20,1,64,64). starts and speeds are long tensors of shape (B,).
    Returns X (B,5,...), Y (B,5,...), mask (B,5) that is True where the target exists."""
    batch = videos.shape[0]
    steps = torch.arange(5, device=videos.device)
    t = starts.to(videos.device)[:, None]
    k = speeds.to(videos.device)[:, None]
    in_idx = t + k * steps - 1
    tg_idx = t + k * (5 + steps) - 1
    mask = tg_idx <= 19
    rows = torch.arange(batch, device=videos.device)[:, None]
    return videos[rows, in_idx], videos[rows, tg_idx.clamp(max=19)], mask


def eval_windows(videos, k, t=1):
    batch = videos.shape[0]
    starts = torch.full((batch,), t, dtype=torch.long)
    speeds = torch.full((batch,), k, dtype=torch.long)
    return gather_windows(videos, starts, speeds)
