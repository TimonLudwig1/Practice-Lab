"""Starter für T15: Autokorrelation in Regressionsresiduen."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RNG_SEED = 20260720


def make_data(size: int = 365, phi: float = 0.78) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    innovation = rng.normal(0, 6, size=size)
    ar_error = np.empty(size)
    ar_error[0] = innovation[0] / np.sqrt(1 - phi ** 2)
    for day in range(1, size):
        ar_error[day] = phi * ar_error[day - 1] + innovation[day]
    day_index = np.arange(1, size + 1)
    orders = 120 + 0.12 * day_index + ar_error
    return pd.DataFrame(
        {"day": day_index, "orders": orders, "structural_error": ar_error}
    )


def main() -> None:
    data = make_data()

    # TODO: Schätze orders auf day und speichere Fits sowie Residuen.
    # TODO: Berechne Lag-1-Korrelation, ACF, Durbin–Watson und Ljung–Box.
    # TODO: Vergleiche konventionelle mit HAC-Standardfehlern.
    # TODO: Speichere Daten, Tabellen und die vierteilige Zeitreihendiagnose.


if __name__ == "__main__":
    main()
