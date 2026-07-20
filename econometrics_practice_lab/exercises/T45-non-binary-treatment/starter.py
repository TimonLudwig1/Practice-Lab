"""Starter für T45: nicht-binäre Behandlungsintensität."""

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
RNG_SEED = 20260725


def true_dose_effect(dose: np.ndarray | float) -> np.ndarray | float:
    return 7.0 * (1 - np.exp(-np.asarray(dose) / 35.0))


def make_data(cities: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    capacity = rng.normal(size=cities)
    dose = np.zeros(cities)
    funded = np.argsort(capacity)[30:]
    dose[funded] = np.clip(55 + 18 * capacity[funded] + rng.normal(0, 10, len(funded)), 10, 100)
    city_intercept = 48 + 5 * capacity + rng.normal(0, 1.3, cities)
    time_shocks = np.array([0.0, 1.2, -0.8, 2.0, 0.5, 1.6])
    rows = []
    for city in range(cities):
        for year_index, year in enumerate(range(2019, 2025)):
            post = int(year >= 2022)
            effect = float(true_dose_effect(dose[city])) * post
            outcome = city_intercept[city] + time_shocks[year_index] + effect + rng.normal(0, 1.1)
            rows.append((city + 1, year, post, dose[city], outcome, capacity[city], effect))
    return pd.DataFrame(rows, columns=["city_id", "year", "post", "funding_eur_per_resident",
                                       "investment_index", "oracle_capacity", "oracle_treatment_effect"])


def main() -> None:
    data = make_data()

    # TODO: Schätze lineare, quadratische und kategoriale Dosis-DiD-Modelle.
    # TODO: Berechne Gesamt- und marginale Effekte für mehrere Dosiswerte.
    # TODO: Untersuche Modellfit und speichere Daten, Tabellen und Grafik.


if __name__ == "__main__":
    main()
