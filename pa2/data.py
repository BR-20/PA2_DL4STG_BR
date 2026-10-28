import torch

TRAIN = slice(0, 8000)
VAL = slice(8000, 9000)
TEST = slice(9000, 10000)


def load_moving_mnist(root="data"):
    from torchvision.datasets import MovingMNIST
    ds = MovingMNIST(root=root, split=None, download=True)
    data = torch.as_tensor(ds.data)
    if data.ndim == 4:
        data = data.unsqueeze(2)
    assert data.shape == (10000, 20, 1, 64, 64), data.shape
    assert data.dtype == torch.uint8
    return data


def split_videos(data_u8):
    return data_u8[TRAIN], data_u8[VAL], data_u8[TEST]


def to_float(batch_u8):
    return batch_u8.float().div(255.0)
