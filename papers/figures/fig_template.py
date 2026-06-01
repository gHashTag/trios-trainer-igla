"""Reusable plotting helpers for F2 methodology figures — Loop 53.

Shared between fig1 (bar chart), fig3 (heatmap), fig4 (tipping curves).
All figures expect data that came from F2 CSVs converted via `f2_to_jsonl`.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# F2 canonical fix names (mirrors race::ablation::CANONICAL_FIX_NAMES).
CANONICAL_FIX_NAMES = ["rms", "warmup", "gradclip", "clamp", "smooth", "wd", "dropout"]

# Order of PSE columns in f2_dual_mediation output.
PSE_NAMES = ["NDE", "NIE_M1", "NIE_M2", "NIE_chain"]

# Stratum slot ordering used by f2_stratum_compare.
STRATA_LABELS = ["canonical", "wd0", "warmup0"]

# Color convention: red = harmful direction (lower better → NDE<0 means
# removing helps; we treat that as red "apparent harm" in canonical); green =
# intrinsically helpful; grey = neutral / missing.
COLOR_HARMFUL = "#c0392b"
COLOR_HELPFUL = "#27ae60"
COLOR_NEUTRAL = "#95a5a6"


def load_jsonl(path):
    """Yield JSON objects from a JSONL file, skipping empty lines."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def safe_float(v, default=float("nan")):
    if v is None:
        return default
    try:
        f = float(v)
        return f if not math.isnan(f) else default
    except (TypeError, ValueError):
        return default


def is_real(x):
    return isinstance(x, (int, float)) and not math.isnan(float(x))


def color_for_estimate(est):
    if not is_real(est):
        return COLOR_NEUTRAL
    if est < 0:
        return COLOR_HARMFUL
    return COLOR_HELPFUL


def standard_axes(ax, title, xlabel=None, ylabel=None):
    """Apply uniform axis polish across F2 figures."""
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.6)


def parse_args_with_io(default_input, default_out):
    """Standard CLI for F2 figure scripts: --input JSONL, --out PNG."""
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=default_input)
    p.add_argument("--out", default=default_out)
    return p.parse_args()


def save_figure(fig, out_path):
    """Save figure at workshop-grade DPI (200) and report bytes written."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"# Wrote {out} ({out.stat().st_size} bytes)")
