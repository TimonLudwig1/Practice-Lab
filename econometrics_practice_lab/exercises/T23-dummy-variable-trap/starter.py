"""Starter für T23: Dummy-Variable-Trap und Matrixrang."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720
MODES = ["Bus", "Bike", "Car"]


def make_data(size: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    mode = rng.choice(MODES, size=size, p=[0.4, 0.25, 0.35])
    commute = rng.uniform(10, 60, size=size)
    mode_effect = {"Bus": 0, "Bike": 14, "Car": 7}
    satisfaction = 72 - 0.38 * commute + np.array([mode_effect[value] for value in mode])
    satisfaction += rng.normal(0, 6, size=size)
    return pd.DataFrame(
        {"commuter_id": np.arange(1, size + 1), "mode": mode,
         "commute_minutes": commute, "satisfaction_score": satisfaction}
    )


def main() -> None:
    data = make_data()

    # TODO: Erzeuge ungültige und zwei gültige Designmatrizen.
    # TODO: Berechne Rang, Singulärwerte und Condition Number.
    # TODO: Konstruiere zwei verschiedene Parametervektoren mit identischen Fits.
    # TODO: Vergleiche gültige Parametrisierungen und speichere alle Ergebnisse.


if __name__ == "__main__":
    main()
