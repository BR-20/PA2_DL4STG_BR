import torch


def analytic_reach(patch, kernel, dilation):
    """Longest path inside one step, in original pixels, centre to centre.
    Input and layer 1 state cross both convolutions. Layer 2 state crosses only conv 2."""
    cells = [d * (kernel - 1) // 2 for d in dilation]
    paths = {"input": cells[0] + cells[1], "state_1": cells[0] + cells[1], "state_2": cells[1]}
    return {name: n * patch for name, n in paths.items()}


def _cell_extent(diff, patch, centre_cell):
    cells = torch.nn.functional.pixel_unshuffle(diff.float(), patch).amax(dim=1)[0] > 0
    idx = cells.nonzero()
    return int((idx - centre_cell).abs().max()) if len(idx) else 0


@torch.no_grad()
def empirical_reach(model, seed=0):
    """Perturb one cell at a time and measure how far the output changes in one step."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    side = 64 // model.patch
    centre = side // 2
    pixel = centre * model.patch + model.patch // 2
    states = [(torch.randn(1, model.hidden, side, side, generator=gen), torch.randn(1, model.hidden, side, side, generator=gen)) for _ in model.cells]
    frame = torch.rand(1, 1, 64, 64, generator=gen)
    base, _ = model.step(frame, states)
    result = {}

    moved = frame.clone()
    moved[0, 0, pixel, pixel] += 1.0
    out, _ = model.step(moved, states)
    result["input"] = _cell_extent((out - base).abs() > 0, model.patch, centre) * model.patch

    for name, layer in (("state_1", 0), ("state_2", 1)):
        shifted = [(h.clone(), c.clone()) for h, c in states]
        shifted[layer][0][0, :, centre, centre] += 1.0
        out, _ = model.step(frame, shifted)
        result[name] = _cell_extent((out - base).abs() > 0, model.patch, centre) * model.patch
    return result
