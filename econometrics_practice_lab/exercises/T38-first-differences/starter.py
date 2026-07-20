"""Starter für T38: First Differences."""

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
TRUE_BETA = 0.6


def make_data(households: int = 150, periods: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    stability = rng.normal(size=households)
    rows = []
    for household in range(households):
        income = 35 + 7 * stability[household] + rng.normal(0, 3)
        for year in range(1, periods + 1):
            income += rng.normal(0, 3.2)
            consumption = 12 + 5 * stability[household] + TRUE_BETA * income + rng.normal(0, 3)
            rows.append((household + 1, year, income, consumption, stability[household]))
    return pd.DataFrame(rows, columns=["household_id", "year", "income_thousand_eur",
                                       "consumption_thousand_eur", "oracle_financial_stability"])


def main() -> None:
    data = make_data()

    # TODO: Berechne First Differences innerhalb jedes Haushalts.
    # TODO: Schätze Pooled-, Within- und FD-Modell.
    # TODO: Prüfe die T=2-Äquivalenz und die Entfernung zeitinvarianter Effekte.
    # TODO: Speichere Daten, Tabellen und die FD-Grafik.


if __name__ == "__main__":
    main()
