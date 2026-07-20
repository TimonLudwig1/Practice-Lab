"""Starter für T14: Heteroskedastizität sichtbar machen."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data(size: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    income = rng.uniform(1_500, 9_000, size=size)
    conditional_sd = 100 + 0.12 * income
    consumption = 700 + 0.42 * income + rng.normal(0, conditional_sd, size=size)
    return pd.DataFrame(
        {"household_id": np.arange(1, size + 1), "income_eur": income,
         "consumption_eur": consumption, "true_conditional_sd": conditional_sd}
    )


def main() -> None:
    data = make_data()

    # TODO: Schätze OLS und erzeuge Fits sowie Residuen.
    # TODO: Vergleiche konventionelle mit HC1-robusten Standardfehlern und KIs.
    # TODO: Führe den Breusch–Pagan-Test aus.
    # TODO: Untersuche die Residuen-SD nach Fit-Dezilen.
    # TODO: Speichere Daten, Tabellen und die vierteilige Diagnosegrafik.


if __name__ == "__main__":
    main()
