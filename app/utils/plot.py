"""Shared plotting helpers for handbook lab notebooks."""

from pathlib import Path

import matplotlib.pyplot as plt

ASSETS_DIR = Path(__file__).resolve().parents[2] / "book" / "assets"


def setup_plot_style() -> None:
    """Apply a consistent style for handbook figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "axes.grid": True,
            "font.size": 11,
        }
    )


def save_figure(name: str, dpi: int = 150) -> Path:
    """Save the current figure to book/assets/ and return the path."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSETS_DIR / name
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
