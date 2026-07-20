"""Starter für T35: Fixed Effects."""

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
TRUE_BETA = 2.5


def make_data(stores: int = 100, periods: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    quality = rng.normal(0, 8, stores)
    downtown = (quality + rng.normal(0, 5, stores) > 1).astype(int)
    rows = []
    for store in range(stores):
        for month in range(1, periods + 1):
            advertising = 12 + 0.55 * quality[store] + 0.18 * month + rng.normal(0, 2.2)
            sales = 100 + quality[store] + TRUE_BETA * advertising + rng.normal(0, 5)
            rows.append((store + 1, month, advertising, sales, quality[store], downtown[store]))
    return pd.DataFrame(rows, columns=["store_id", "month", "advertising_thousand_eur",
                                       "sales_index", "oracle_store_quality", "downtown_location"])


def main() -> None:
    data = make_data()

    # TODO: Schätze Pooled-, Between- und Filial-FE-Modell.
    # TODO: Verwende geclusterte Standardfehler für Panelbeobachtungen.
    # TODO: Untersuche Within-Variation und Rangdefizienz zeitinvarianter Variablen.
    # TODO: Speichere Daten, Tabellen und die FE-Grafik.


if __name__ == "__main__":
    main()
