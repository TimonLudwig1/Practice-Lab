"""Starter für T08: Pearson-Korrelation und ihre Grenzen."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data() -> pd.DataFrame:
    """Erzeuge lineare, nichtlineare und ausreißergetriebene Szenarien."""
    rng = np.random.default_rng(RNG_SEED)

    x_linear = rng.uniform(0, 10, size=120)
    y_linear = 3 + 1.7 * x_linear + rng.normal(0, 3, size=120)

    x_nonlinear = np.linspace(-3, 3, 120)
    y_nonlinear = x_nonlinear ** 2 + rng.normal(0, 0.65, size=120)

    x_outlier = np.append(rng.normal(size=80), 8.0)
    y_outlier = np.append(rng.normal(size=80), 8.0)

    frames = []
    for scenario, x, y in (
        ("linear", x_linear, y_linear),
        ("nonlinear", x_nonlinear, y_nonlinear),
        ("outlier", x_outlier, y_outlier),
    ):
        frames.append(
            pd.DataFrame(
                {"scenario": scenario, "observation_id": np.arange(1, len(x) + 1),
                 "x": x, "y": y}
            )
        )
    return pd.concat(frames, ignore_index=True)


def pearson_manual(x: np.ndarray, y: np.ndarray) -> float:
    """Berechne Pearson-r als Produkt standardisierter Werte."""
    # TODO: Berechne z-Scores mit Stichproben-SDs und teile die Produktsumme durch n - 1.
    raise NotImplementedError


def main() -> None:
    data = make_data()

    # TODO: Berechne für jedes Szenario Kovarianz, Pearson-r und p-Wert.
    # TODO: Kontrolliere pearson_manual(...) gegen stats.pearsonr(...).
    # TODO: Untersuche im linearen Szenario 1000*x + 50 sowie -x.
    # TODO: Berechne im Ausreißerszenario r mit und ohne die letzte Beobachtung.
    # TODO: Speichere Daten, Ergebnistabelle und eine Small-Multiple-Grafik.


if __name__ == "__main__":
    main()
