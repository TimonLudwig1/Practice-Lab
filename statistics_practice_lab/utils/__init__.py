"""Wiederverwendbare Werkzeuge für den Statistik-Lernbegleiter.

Dieses Paket wächst über die Projekte hinweg und wird später der Kern von
Capstone C3 (`statlab`). Grundsatz: Hier leben *Infrastruktur*-Helfer
(Reproduzierbarkeit, Plot-Stil, Simulations-Gerüste) — die statistischen
Kernkonzepte werden in jedem Projekt selbst "von Hand" implementiert.

Beispiel-Import in einem Notebook:

    import sys; sys.path.append("../..")   # Repo-Root in den Pfad
    from utils import make_rng, apply_house_style, plot_null_distribution
"""

from __future__ import annotations

from .plotting import add_truth_line, apply_house_style, plot_null_distribution
from .reproducibility import DEFAULT_SEED, make_rng
from .simulation import coverage_simulation

__all__ = [
    "DEFAULT_SEED",
    "make_rng",
    "apply_house_style",
    "add_truth_line",
    "plot_null_distribution",
    "coverage_simulation",
]
