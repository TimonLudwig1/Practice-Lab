"""Starter für T37: Within-Estimator."""

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
TRUE_BETA = 1.4


def make_data(firms: int = 90, periods: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    firm_opportunity = rng.normal(0, 4, firms)
    rows = []
    for firm in range(firms):
        for period in range(1, periods + 1):
            cash_flow = 6 + 0.65 * firm_opportunity[firm] + 0.25 * period + rng.normal(0, 1.8)
            investment = 12 + firm_opportunity[firm] + TRUE_BETA * cash_flow + rng.normal(0, 2.5)
            rows.append((firm + 1, period, cash_flow, investment, firm_opportunity[firm]))
    return pd.DataFrame(rows, columns=["firm_id", "period", "cash_flow_million_eur",
                                       "investment_million_eur", "oracle_firm_opportunity"])


def main() -> None:
    data = make_data()

    # TODO: Erzeuge die Within-Transformation.
    # TODO: Schätze Within-, Dummy- und Pooled-Modell.
    # TODO: Berechne den Within-Koeffizienten zusätzlich per Hand.
    # TODO: Speichere Daten, Tabellen und die Within-Grafik.


if __name__ == "__main__":
    main()
