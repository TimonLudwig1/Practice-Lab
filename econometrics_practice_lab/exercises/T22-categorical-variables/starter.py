"""Starter für T22: Kategoriale Variablen und Referenzkategorien."""

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
LOCATIONS = ["Center", "Suburb", "University", "Industrial"]


def make_data(size: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    location = rng.choice(LOCATIONS, size=size, p=[0.25, 0.35, 0.25, 0.15])
    marketing = rng.uniform(5, 40, size=size)
    effects = {"Center": 120, "Suburb": 0, "University": 70, "Industrial": -90}
    revenue = 500 + 18 * marketing + np.array([effects[value] for value in location])
    revenue += rng.normal(0, 90, size=size)
    return pd.DataFrame(
        {"store_id": np.arange(1, size + 1), "location": location,
         "marketing_thousand_eur": marketing, "weekly_revenue_thousand_eur": revenue}
    )


def design_matrix(data: pd.DataFrame, reference: str) -> pd.DataFrame:
    # TODO: Zentriere Marketing bei 20 und erzeuge K-1 Dummies mit gewünschter Referenz.
    raise NotImplementedError


def main() -> None:
    data = make_data()

    # TODO: Schätze Modelle mit Suburb und Center als Referenz.
    # TODO: Vergleiche Koeffizienten und prüfe identische Fits.
    # TODO: Berechne adjustierte Standortmittel bei Marketing=20.
    # TODO: Speichere Daten, Tabellen und die Visualisierung.


if __name__ == "__main__":
    main()
