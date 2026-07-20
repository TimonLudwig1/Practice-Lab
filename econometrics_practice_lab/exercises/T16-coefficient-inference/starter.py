"""Starter für T16: Tests und Konfidenzintervalle für OLS-Koeffizienten."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data(size: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    discount = rng.uniform(0, 20, size=size)
    weekly_units = 120 + 3.0 * discount + rng.normal(0, 25, size=size)
    return pd.DataFrame(
        {"store_id": np.arange(1, size + 1), "discount_percentage": discount,
         "weekly_units": weekly_units}
    )


def test_slope(estimate: float, standard_error: float, null_value: float, df: int) -> dict[str, float]:
    # TODO: Berechne t-Statistik und zweiseitigen p-Wert.
    raise NotImplementedError


def main() -> None:
    data = make_data()

    # TODO: Schätze OLS und leite den Steigungs-SE manuell her.
    # TODO: Teste beta_1=0 und beta_1=3.
    # TODO: Konstruiere 95%- und 99%-Koeffizientenintervalle.
    # TODO: Speichere Daten, Tabellen und die Inferenzgrafik.


if __name__ == "__main__":
    main()
