"""Starter für T21: Omitted Variable Bias."""

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


def make_data(size: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    ability = rng.normal(size=size)
    education = np.clip(12 + 1.2 * ability + rng.normal(0, 1.5, size=size), 8, 20)
    log_wage = 2.2 + 0.08 * education + 0.22 * ability + rng.normal(0, 0.20, size=size)
    return pd.DataFrame(
        {"person_id": np.arange(1, size + 1), "education_years": education,
         "ability_score": ability, "log_hourly_wage": log_wage,
         "hourly_wage_eur": np.exp(log_wage)}
    )


def main() -> None:
    data = make_data()

    # TODO: Schätze kurzes, langes und Hilfsmodell.
    # TODO: Berechne beobachteten Bias und OVB-Formel gamma_hat * delta_hat.
    # TODO: Prüfe Richtung und exakte Stichprobenidentität.
    # TODO: Speichere Daten, Tabellen und die OVB-Visualisierung.


if __name__ == "__main__":
    main()
