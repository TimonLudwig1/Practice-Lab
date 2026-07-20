"""Starter für T18: Multiple Regression und ceteris paribus."""

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
REGRESSORS = ["area_sqm", "quality_score", "distance_center_km", "building_age_years"]


def make_data(size: int = 350) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    area = rng.uniform(30, 150, size=size)
    distance = rng.uniform(0.5, 20, size=size)
    quality = np.clip(7.2 - 0.16 * distance + rng.normal(0, 1.2, size=size), 1, 10)
    age = rng.uniform(0, 80, size=size)
    rent = 350 + 10.5 * area + 95 * quality - 18 * distance - 2.5 * age
    rent += rng.normal(0, 160, size=size)
    return pd.DataFrame(
        {"apartment_id": np.arange(1, size + 1), "area_sqm": area,
         "quality_score": quality, "distance_center_km": distance,
         "building_age_years": age, "rent_eur": rent}
    )


def main() -> None:
    data = make_data()

    # TODO: Schätze einfaches und multiples Modell.
    # TODO: Reproduziere den Flächenkoeffizienten mit Frisch–Waugh–Lovell.
    # TODO: Schätze ein Modell mit z-standardisierten Variablen.
    # TODO: Vergleiche Modelle und speichere Tabellen sowie Diagnosegrafik.


if __name__ == "__main__":
    main()
