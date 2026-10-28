import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _show(ax, img, title=None):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


def grid_figure(rows, row_labels, col_labels, path, cell=0.9):
    """rows: list of lists of 2D arrays (None leaves a blank cell)."""
    fig, axes = plt.subplots(len(rows), len(rows[0]), figsize=(cell * len(rows[0]) + 0.9, cell * len(rows) + 0.5), squeeze=False)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax = axes[r][c]
            if img is None:
                ax.axis("off")
            else:
                _show(ax, img, col_labels[c] if r == 0 else None)
        axes[r][0].set_ylabel(row_labels[r], fontsize=6, rotation=0, ha="right", va="center")
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _canvas(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def panel_rows(inputs, futures):
    """Four row panel as lists of 10 images. inputs and futures are lists of four (5,64,64) arrays."""
    return [list(i) + list(f) for i, f in zip(inputs, futures)]


COL_LABELS = [f"in {i}" for i in range(1, 6)] + [f"h={h}" for h in range(1, 6)]


def panel_png_and_gif(inputs, futures, row_labels, png_path, gif_path):
    rows = panel_rows(inputs, futures)
    grid_figure(rows, row_labels, COL_LABELS, png_path)
    frames = []
    for h in range(1, 6):
        shown = [list(i) + [f[j] if j < h else None for j in range(5)] for i, f in zip(inputs, futures)]
        fig, axes = plt.subplots(4, 10, figsize=(9.9, 4.1), squeeze=False)
        for r in range(4):
            for c in range(10):
                if shown[r][c] is None:
                    axes[r][c].axis("off")
                else:
                    _show(axes[r][c], shown[r][c], COL_LABELS[c] if r == 0 else None)
            axes[r][0].set_ylabel(row_labels[r], fontsize=6, rotation=0, ha="right", va="center")
        fig.suptitle(f"forecast horizon h = {h}", fontsize=8)
        fig.tight_layout(pad=0.2)
        frames.append(_canvas(fig))
        plt.close(fig)
    imageio.mimsave(gif_path, frames, duration=0.8, loop=0)


def hist_figure(values, marks, path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for a, name in zip(ax, ("CM", "CR")):
        a.hist(values[name].numpy(), bins=50, edgecolor="k")
        for rank, idx in enumerate(marks[name]):
            a.axvline(float(values[name][idx]), color="r", ls="--", lw=1, label="Figure 3 videos" if rank == 0 else None)
        a.set(xlabel=f"per video {name}", ylabel="videos", yscale="log")
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
