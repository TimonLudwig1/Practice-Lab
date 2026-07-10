"""Shared plotting helpers.

Plot standards (see CLAUDE.md): titles state the *message*, axes are labelled
with units, true values / thresholds are drawn as dashed reference lines, and
uncertainty is made visible. These helpers enforce the recurring pieces so each
project can focus on the substance rather than re-styling from scratch.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def apply_house_style() -> None:
    """Apply a consistent, readable matplotlib style for the whole repo.

    Call once at the top of a notebook. Keeps figures legible and uniform so
    plots across projects read as one body of work.
    """
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def add_truth_line(
    ax: plt.Axes,
    value: float,
    label: str = "Wahrheit",
    orientation: str = "vertical",
    color: str = "crimson",
) -> None:
    """Draw a dashed, labelled reference line for a known true value/threshold.

    In simulations we know the truth and must always show it (CLAUDE.md), so the
    reader can judge every estimate against it.

    Parameters
    ----------
    ax:
        Target axes.
    value:
        Position of the reference line.
    label:
        Legend/annotation text (e.g. ``"wahres μ = 175 cm"``).
    orientation:
        ``"vertical"`` (default) or ``"horizontal"``.
    color:
        Line colour.
    """
    if orientation == "vertical":
        ax.axvline(value, color=color, linestyle="--", linewidth=2, label=label)
    elif orientation == "horizontal":
        ax.axhline(value, color=color, linestyle="--", linewidth=2, label=label)
    else:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")


def plot_null_distribution(
    null_stats: np.ndarray,
    observed: float,
    ax: plt.Axes | None = None,
    stat_name: str = "Teststatistik",
    two_sided: bool = True,
) -> tuple[plt.Axes, float]:
    """Plot a simulated null distribution with the observed statistic marked.

    The backbone of permutation-test / p-value intuition (Projects 6.1, 6.2):
    a histogram of the statistic under H0, the observed value drawn as a line,
    and the tail area (the p-value) shaded. The p-value is returned as the
    fraction of null statistics *at least as extreme* as the observed one.

    Parameters
    ----------
    null_stats:
        Simulated values of the statistic under the null hypothesis.
    observed:
        The statistic actually observed in the data.
    ax:
        Optional target axes; created if omitted.
    stat_name:
        Axis label for the statistic.
    two_sided:
        If True, "extreme" means ``|stat| >= |observed|`` relative to the null
        centre; if False, only the upper tail counts.

    Returns
    -------
    (ax, p_value)
        The axes and the empirical p-value.
    """
    null_stats = np.asarray(null_stats)
    if ax is None:
        _, ax = plt.subplots()

    ax.hist(null_stats, bins=50, color="steelblue", alpha=0.7,
            label="Nullverteilung (simuliert)")

    center = np.mean(null_stats)
    if two_sided:
        threshold = abs(observed - center)
        extreme = np.abs(null_stats - center) >= threshold
    else:
        extreme = null_stats >= observed

    # p-value = fraction of the null distribution at least as extreme as what we saw.
    p_value = float(np.mean(extreme))

    ax.hist(null_stats[extreme], bins=50, color="crimson", alpha=0.6,
            label="mindestens so extrem")
    ax.axvline(observed, color="black", linestyle="--", linewidth=2,
               label=f"beobachtet = {observed:.3g}")
    ax.set_xlabel(stat_name)
    ax.set_ylabel("Häufigkeit")
    ax.set_title(f"p-Wert = {p_value:.4f} (Anteil der Nullverteilung im roten Bereich)")
    ax.legend()
    return ax, p_value
