"""Starter für T31: Balance Tests."""

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
BASELINES = ["age", "prior_spend_eur", "prior_visits", "baseline_conversion",
             "mobile_user", "newsletter_subscriber"]


def make_data(size: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    treatment = np.zeros(size, dtype=int)
    treatment[rng.choice(size, size=size // 2, replace=False)] = 1
    return pd.DataFrame(
        {"user_id": np.arange(1, size + 1), "age": np.clip(rng.normal(39, 11, size), 18, 75),
         "prior_spend_eur": rng.lognormal(np.log(80), 0.65, size),
         "prior_visits": rng.poisson(6, size),
         "baseline_conversion": rng.binomial(1, 0.30, size),
         "mobile_user": rng.binomial(1, 0.66, size),
         "newsletter_subscriber": rng.binomial(1, 0.46, size), "treatment": treatment}
    )


def main() -> None:
    data = make_data()

    # TODO: Erstelle die Balance-Tabelle mit SMDs und Einzeltests.
    # TODO: Führe den gemeinsamen F-Test durch.
    # TODO: Simuliere 800 korrekte Randomisierungen und multiple Tests.
    # TODO: Speichere Daten, Tabellen und die Balance-Grafik.


if __name__ == "__main__":
    main()
