"""Shared figure style: recessive axes, thin marks and a fixed, colour-blind-checked palette.

Colours follow the reference palette of the data-viz guidelines (categorical slots in fixed order, a blue-red
diverging pair with a grey midpoint). Benign is always slot 1 and attack slot 2, in every figure.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BENIGN_COLOR, ATTACK_COLOR = SERIES[0], SERIES[1]
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
DIVERGING = (LinearSegmentedColormap.from_list("blue_red", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#9b1c1c"])
             .with_extremes(bad="#d9d8d2"))


def apply_style() -> None:
    """Set the shared matplotlib rcParams: grey recessive axes, light grid, no top or right spine, 200 dpi PNGs."""
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.size": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_SECONDARY,
        "axes.titlesize": 10, "axes.titlecolor": INK, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY, "legend.frameon": False, "lines.linewidth": 2,
    })


def save(fig, target) -> None:
    """PNG without a software tag, so the bytes depend only on the figure."""
    fig.savefig(target, metadata={"Software": None})
    plt.close(fig)
