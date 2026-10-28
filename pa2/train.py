import json
import os
import platform
import time
import torch

from .configs import MAX_PARAMS
from .metrics import per_video_mse
from .models import ConvLSTMForecaster, count_parameters
from .transforms import G_ELEMENTS, apply_g
from .windows import AUG_STARTS, eval_windows, gather_windows


def hardware_string(device):
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "cpu"
    return f"{name}, torch {torch.__version__}, device {device.type}"


def sample_batch(videos_u8, idx, gen, speed_aug, sym_aug, device):
    batch = len(idx)
    vids = videos_u8[idx].to(device).float().div(255.0)
    if speed_aug:
        speeds = torch.randint(1, 4, (batch,), generator=gen)
        counts = torch.tensor([AUG_STARTS[int(k)] for k in speeds])
        starts = 1 + (torch.rand(batch, generator=gen) * counts).long()
    else:
        speeds = torch.ones(batch, dtype=torch.long)
        starts = torch.randint(1, 12, (batch,), generator=gen)
    X, Y, mask = gather_windows(vids, starts, speeds)
    if sym_aug:
        draws = torch.randint(0, len(G_ELEMENTS), (batch,), generator=gen).to(device)
        for j, g in enumerate(G_ELEMENTS):
            sel = draws == j
            if sel.any():
                X[sel], Y[sel] = apply_g(X[sel], g), apply_g(Y[sel], g)
    return X, Y, mask


def masked_loss(pred, target, mask):
    """Squared error over pixels, mean over available targets of each video, mean over videos."""
    per_horizon = ((pred - target) ** 2).mean(dim=(2, 3, 4))
    per_video = (per_horizon * mask).sum(1) / mask.sum(1)
    return per_video.mean()


@torch.no_grad()
def validation_error(model, val_float, device, k=1):
    model.eval()
    X, Y, mask = eval_windows(val_float, k)
    total, count = 0.0, 0
    for i in range(0, len(X), 100):
        P = model(X[i:i + 100].to(device))
        h = per_video_mse(P, Y[i:i + 100].to(device))
        m = mask[i:i + 100].to(device)
        total += ((h * m).sum(1) / m.sum(1)).sum().item()
        count += len(P)
    model.train()
    return total / count


def train_model(name, cfg, seed, train_u8, val_u8, device, out_dir, epochs=30, check_every=5, log=print):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = ConvLSTMForecaster(**cfg["arch"]).to(device)
    n_params = count_parameters(model)
    assert n_params <= MAX_PARAMS or name not in ("D", "DB"), n_params
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    val_float = val_u8.float().div(255.0)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    epoch_seconds, train_loss, val_log = [], [], []
    model.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(train_u8), generator=gen)
        sync()
        t0 = time.perf_counter()
        running, seen = 0.0, 0
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            X, Y, mask = sample_batch(train_u8, idx, gen, cfg["speed_aug"], cfg["sym_aug"], device)
            loss = masked_loss(model(X), Y, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
            seen += len(idx)
        sync()
        epoch_seconds.append(time.perf_counter() - t0)
        train_loss.append(running / seen)
        if epoch % check_every == 0 or epoch == epochs:
            val_log.append((epoch, validation_error(model, val_float, device)))
        log(f"{name} seed {seed} epoch {epoch} loss {train_loss[-1]:.4e} time {epoch_seconds[-1]:.1f}s")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"state": model.state_dict(), "cfg": cfg, "seed": seed}, os.path.join(out_dir, f"{name}_s{seed}.pt"))
    return dict(name=name, seed=seed, cfg=cfg, params=n_params, epochs=epochs, epoch_seconds=epoch_seconds,
                train_seconds_per_epoch=sum(epoch_seconds) / len(epoch_seconds), train_loss=train_loss,
                val_E5_1x=val_log, hardware=hardware_string(device))


def load_model(path, device):
    blob = torch.load(path, map_location=device)
    model = ConvLSTMForecaster(**blob["cfg"]["arch"]).to(device)
    model.load_state_dict(blob["state"])
    return model.eval(), blob["cfg"]
