import torch

G_ELEMENTS = [(a, m) for m in (0, 1) for a in range(4)]


def mirror(x):
    return torch.flip(x, dims=(-1,))


def rotate(x, a=1):
    return torch.rot90(x, k=a, dims=(-2, -1))


def apply_g(x, g):
    a, m = g
    y = rotate(x, a)
    return mirror(y) if m else y


def invert_g(x, g):
    a, m = g
    y = mirror(x) if m else x
    return rotate(y, -a)
