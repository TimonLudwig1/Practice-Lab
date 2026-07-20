"""Starter für T10: Residuen verstehen und diagnostizieren."""

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


def make_data(size: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    area = rng.uniform(25, 140, size=size)
    rent = 320 + 13.5 * area + rng.normal(0, 150, size=size)
    rent[[17, 88, 153]] += np.array([720, -620, 850])
    return pd.DataFrame(
        {"listing_id": np.arange(1, size + 1), "floor_area_sqm": area, "rent_eur": rent}
    )


def main() -> None:
    data = make_data()

    # TODO: Schätze rent_eur auf floor_area_sqm mit Intercept.
    # TODO: Ergänze fitted_rent, residual und absolute_residual.
    # TODO: Prüfe OLS-Eigenschaften und identifiziere die größten Residuen.
    # TODO: Speichere Daten, Residuentabelle, Zusammenfassung und Diagnosegrafik.


if __name__ == "__main__":
    main()
