"""Starter für T44: DiD mit individuellen und zeitlichen Fixed Effects."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260724
TRUE_EFFECT = 4.0
WEEK_SHOCKS = np.array([0.0, 2.5, -1.0, 3.0, -2.0, 5.0, 1.5, 4.0, -1.5, 3.5])


def make_data(stores: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    quality = rng.normal(size=stores)
    treated = (quality >= np.median(quality)).astype(int)
    store_intercept = 75 + 8 * quality + rng.normal(0, 1.2, stores)
    rows = []
    for store in range(stores):
        for week, shock in enumerate(WEEK_SHOCKS, start=1):
            post = int(week >= 6)
            did = treated[store] * post
            sales = store_intercept[store] + shock + TRUE_EFFECT * did + rng.normal(0, 1.6)
            rows.append((store + 1, week, treated[store], post, did, sales, quality[store], shock))
    return pd.DataFrame(rows, columns=["store_id", "week", "treated", "post", "did",
                                       "sales_thousand_eur", "oracle_location_quality",
                                       "oracle_week_shock"])


def main() -> None:
    data = make_data()

    # TODO: Vergleiche Modelle ohne FE, mit einer FE-Dimension und mit TWFE.
    # TODO: Reproduziere TWFE durch doppeltes Demeaning und klassisches DiD.
    # TODO: Speichere Daten, Diagnostik und die Ergebnisgrafik.


if __name__ == "__main__":
    main()
