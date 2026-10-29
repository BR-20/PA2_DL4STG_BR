"""Loads the saved results and formats every number that the report quotes."""
import json
import math
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def fmt(x):
    """Three significant figures, scientific notation below 1e-3."""
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) < 1e-3:
        mant, exp = f"{x:.2e}".split("e")
        return f"{mant} × 10{str(int(exp)).translate(SUP)}".replace("-", "−")
    digits = max(0, 2 - int(math.floor(math.log10(abs(x)))))
    return f"{x:.{digits}f}".replace("-", "−")


def pct(x):
    return f"{100 * x:.0f} percent" if abs(100 * x) >= 9.5 else f"{100 * x:.1f} percent"


def ev(name, seed, sym=False):
    return load(f"eval_{name}{'sym' if sym else ''}_s{seed}.json")


def seeds(name, key, sym=False):
    return [ev(name, s, sym)[key] for s in (0, 1)]


def mean(name, key, sym=False):
    return float(np.mean(seeds(name, key, sym)))


def ms(values):
    return f"{fmt(np.mean(values))} ({fmt(values[0])}, {fmt(values[1])})"


def train_time(name):
    return [load(f"train_{name}_s{s}.json")["train_seconds_per_epoch"] for s in (0, 1)]


def infer_time(name, sym=False):
    return [float(np.median(ev(name, s, sym)["inference_seconds"])) for s in (0, 1)]


BASE = load("eval_baselines.json")
BLANK, COPY = BASE["Blank"], BASE["Copy-last"]
FLOW = load("fig1_stats.json")
REACH = load("reach.json")
SEL = load("selection.json")
DIAG = load("diagnostics.json")
TABLE1 = load("table1.json")
HARDWARE = ev("A", 0)["hardware"]
D_ARCH = load("train_D_s0.json")["cfg"]["arch"]
