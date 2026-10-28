import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden, kernel, dilation):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv2d(in_channels + hidden, 4 * hidden, kernel, padding=pad, dilation=dilation, bias=True)

    def forward(self, u, h, c):
        i, f, o, g = self.conv(torch.cat([u, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMForecaster(nn.Module):
    def __init__(self, patch=4, hidden=64, kernel=3, dilation=(1, 1)):
        super().__init__()
        self.patch = patch
        self.hidden = hidden
        folded = patch * patch
        self.cells = nn.ModuleList([
            ConvLSTMCell(folded, hidden, kernel, dilation[0]),
            ConvLSTMCell(hidden, hidden, kernel, dilation[1]),
        ])
        self.head = nn.Conv2d(hidden, folded, 1, bias=True)

    def init_states(self, batch, device):
        side = 64 // self.patch
        zeros = lambda: torch.zeros(batch, self.hidden, side, side, device=device)
        return [(zeros(), zeros()) for _ in self.cells]

    def step(self, frame, states, decode=True):
        u = F.pixel_unshuffle(frame, self.patch)
        new_states = []
        for cell, (h, c) in zip(self.cells, states):
            h, c = cell(u, h, c)
            new_states.append((h, c))
            u = h
        out = torch.sigmoid(F.pixel_shuffle(self.head(u), self.patch)) if decode else None
        return out, new_states

    def forward(self, x):
        """x (B,5,1,64,64) -> (B,5,1,64,64). Nine steps, own predictions fed back from step 6."""
        states = self.init_states(x.shape[0], x.device)
        outputs = []
        for tau in range(1, 10):
            frame = x[:, tau - 1] if tau <= 5 else outputs[-1]
            out, states = self.step(frame, states, decode=tau >= 5)
            if tau >= 5:
                outputs.append(out)
        return torch.stack(outputs, dim=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
