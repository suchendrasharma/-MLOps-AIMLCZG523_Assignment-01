"""Shared matplotlib styling so EDA and evaluation figures look consistent."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Fixed class colours: identity follows the class, never its rank.
CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}  # blue = no disease, orange = disease
CLASS_NAMES = {0: "No disease", 1: "Disease"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # first three slots, all-pairs safe

# Diverging blue <-> neutral gray <-> red, for signed values such as correlations.
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a82d2d"]
)
# Sequential single hue (blue), for magnitudes such as confusion-matrix counts.
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#f0f6fd", "#86b6ef", "#1c5cab"])


def apply_style() -> None:
    """Set global rcParams: light surface, recessive axes, thin marks."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "patch.linewidth": 0,
    })


def save(fig: plt.Figure, path: Path) -> Path:
    """Save a figure, creating the parent folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path
