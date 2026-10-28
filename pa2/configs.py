import json
import os

A_PARAMS = 480784
MAX_PARAMS = int(1.5 * A_PARAMS)

BASE = dict(patch=4, hidden=64, kernel=3, dilation=(1, 1))

# candidates for model D, compared on the validation split only
D_CANDIDATES = {
    "dil13": dict(patch=4, hidden=64, kernel=3, dilation=(1, 3)),
    "dil12": dict(patch=4, hidden=64, kernel=3, dilation=(1, 2)),
    "k5": dict(patch=4, hidden=46, kernel=5, dilation=(1, 1)),
    "patch8": dict(patch=8, hidden=64, kernel=3, dilation=(1, 1)),
}


def d_choice(results_dir="results"):
    path = os.path.join(results_dir, "d_selection.json")
    if os.path.exists(path):
        return json.load(open(path))["chosen"]
    return "dil13"


def get_config(name, results_dir="results"):
    """Architecture plus training switches for a model name A, B, C, D, DB."""
    if name in ("A", "B", "C"):
        arch = BASE
    elif name in ("D", "DB"):
        arch = D_CANDIDATES[d_choice(results_dir)]
    else:
        raise ValueError(name)
    return dict(arch=dict(arch), speed_aug=name in ("C", "D", "DB"), sym_aug=name in ("B", "DB"))
